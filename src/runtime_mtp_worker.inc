static fg_status mtp_small_expert(void *opaque,uint32_t layer,
    const fg_vk_tensor *activation,const fg_vk_tensor *router,uint32_t tokens,
    fg_vk_tensor **output,fg_error *err){
    return fg_expert_decode_small_n(opaque,layer,activation,router,tokens,output,err);
}

static fg_status mtp_slice_partial(fg_output_slice *slice,
    const fg_mtp_head_work *work,uint32_t self,fg_mtp_head_partial *partial,
    fg_error *err){
    if(!slice||fg_output_slice_ways(slice)!=FG_OUTPUT_SPLIT_WAYS_MAX){
        fg_error_set(err,FG_ERR_MISMATCH,"MTP requires the production four-way head");
        return FG_ERR_MISMATCH;
    }
    *partial=(fg_mtp_head_partial){.kind=work->kind,.source_rank=(uint8_t)self,
        .destination_rank=0u,.first_token=work->first_token,.count=work->count};
    fg_status status=FG_OK;
    for(uint32_t i=0;status==FG_OK&&i<work->count;i++)
        status=fg_output_slice_run_hidden(slice,work->hidden+(uint64_t)i*FG_HIDDEN_SIZE,
            &partial->values[i],&partial->ids[i],err);
    return status;
}

static fg_status mtp_send_partial(fg_fabric *fabric,uint64_t session,
    const fg_mtp_head_partial *partial,fg_error *err){
    uint8_t wire[FG_MTP_HEAD_PARTIAL_MAX_BYTES];uint32_t bytes=0u;
    fg_status status=fg_mtp_head_partial_encode(wire,sizeof(wire),&bytes,partial,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
        FG_MSG_MTP_HEAD_PARTIAL,session,
        partial->first_token*FG_LAYER_COUNT+FG_LAYER_COUNT-1u,0u,wire,bytes,err);
    return status;
}

static fg_status mtp_head_fanout(fg_fabric *fabric,const fg_manifest *manifest,
    uint32_t self,uint64_t session,fg_mtp_head_kind kind,uint32_t first,uint32_t count,
    const float *hidden,layer_work_context *context,fg_error *err){
    if(self!=manifest->layer_owner[FG_LAYER_COUNT-1u]||!self){
        fg_error_set(err,FG_ERR_MISMATCH,"MTP head fanout reached the wrong final owner");
        return FG_ERR_MISMATCH;
    }
    fg_status status=FG_OK;
    for(uint32_t way=0;status==FG_OK&&way<FG_OUTPUT_SPLIT_WAYS_MAX;way++){
        uint32_t destination=fg_output_split_rank(FG_OUTPUT_SPLIT_WAYS_MAX,way);
        fg_mtp_head_work work={.kind=kind,.source_rank=(uint8_t)self,
            .destination_rank=(uint8_t)destination,.first_token=first,
            .count=count,.hidden=hidden};
        if(destination==self){
            fg_mtp_head_partial partial;
            status=mtp_slice_partial(context->output_slice,&work,self,&partial,err);
            if(status==FG_OK)status=mtp_send_partial(fabric,session,&partial,err);
        }else{
            uint32_t bytes=0u;
            status=fg_mtp_head_work_encode(context->result_wire,context->result_capacity,
                &bytes,&work,err);
            if(status==FG_OK)status=fg_fabric_send(fabric,destination,FG_FABRIC_BULK,
                FG_MSG_MTP_HEAD_WORK,session,first*FG_LAYER_COUNT+FG_LAYER_COUNT-1u,
                0u,context->result_wire,bytes,err);
        }
    }
    return status;
}

static fg_status handle_mtp_head_work(fg_fabric *fabric,const fg_manifest *manifest,
    uint32_t self,uint64_t session,uint32_t peer,const fg_frame_header *header,
    const uint8_t *payload,uint32_t bytes,layer_work_context *context,fg_error *err){
    fg_mtp_head_work work;
    fg_status status=fg_mtp_head_work_decode(&work,context->mtp_embeddings,
        (uint64_t)FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE,payload,bytes,err);
    uint32_t way=0u;
    if(status==FG_OK&&(!fg_mtp_native_enabled(context->mtp_native)||!session||
       fg_frame_request_id(header)!=session||peer!=work.source_rank||
       peer!=manifest->layer_owner[FG_LAYER_COUNT-1u]||
       work.destination_rank!=self||
       !fg_output_split_way_for_rank(FG_OUTPUT_SPLIT_WAYS_MAX,self,&way)||
       fg_frame_sequence(header)!=work.first_token*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted MTP contracted head work");
        status=FG_ERR_MISMATCH;
    }
    fg_mtp_head_partial partial;
    if(status==FG_OK)status=mtp_slice_partial(context->output_slice,&work,self,&partial,err);
    if(status==FG_OK)status=mtp_send_partial(fabric,session,&partial,err);
    return status;
}

static fg_status mtp_target_memory_prepare(fg_owner_executor *owner,
    fg_expert_executor *expert,uint32_t sessions,fg_vk_context *vk,uint64_t cap,
    fg_error *err){
    fg_status status=fg_expert_prepare_small_n_scratch(expert,err);
    for(uint32_t slot=0;status==FG_OK&&slot<sessions;slot++){
        fg_owner_session_checkpoint checkpoint={0};
        status=fg_owner_session_device_snapshot(owner,slot,&checkpoint,err);
        fg_owner_session_snapshot_release(&checkpoint);
    }
    fg_vk_memory_stats memory;
    fg_vk_get_memory_stats(vk,&memory);
    if(status==FG_OK&&cap&&memory.requested_live_bytes>cap){
        fg_error_set(err,FG_ERR_LIMIT,"MTP target readiness residency: live=%llu cap=%llu",
            (unsigned long long)memory.requested_live_bytes,(unsigned long long)cap);
        status=FG_ERR_LIMIT;
    }
    return status;
}

static fg_status handle_mtp_native_work(fg_fabric *fabric,fg_owner_executor *owner,
    const fg_manifest *manifest,
    uint32_t self,uint64_t session,uint32_t peer,const fg_frame_header *header,
    const uint8_t *payload,uint32_t bytes,layer_work_context *context,fg_error *err){
    fg_mtp_native_work work;char overlay[FG_MTP_OVERLAY_PATH_MAX+1u];
    char embedding_file[FG_MTP_OVERLAY_PATH_MAX+1u];
    fg_status status=fg_mtp_native_work_decode(&work,context->mtp_tokens,
        FG_PREFILL_MAX_TOKENS,overlay,embedding_file,payload,bytes,err);
    uint64_t request=fg_frame_request_id(header);
    bool config=status==FG_OK&&work.operation==FG_MTP_NATIVE_CONFIG;
    bool prepare=status==FG_OK&&work.operation==FG_MTP_NATIVE_PREPARE;
    if(status==FG_OK&&config&&(!context->vk||context->tokens<FG_MTP_MAX_VERIFY)){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"MTP requires a decode owner with N=4 input storage");
        status=FG_ERR_UNAVAILABLE;
    }
    if(status==FG_OK&&(peer!=0u||(!config&&(!session||request!=session))||
       (!config&&!prepare&&self!=manifest->layer_owner[FG_LAYER_COUNT-1u]))){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted native MTP work");
        status=FG_ERR_MISMATCH;
    }
    float hidden[FG_HIDDEN_SIZE];fg_mtp_native_ack ack;
    if(status==FG_OK&&prepare){
        if(!fg_mtp_native_enabled(context->mtp_native)){
            fg_error_set(err,FG_ERR_MISMATCH,"MTP admission before configuration");
            status=FG_ERR_MISMATCH;
        }else status=mtp_target_memory_prepare(owner,context->dispatch.expert,
            work.argument,context->vk,manifest->residency_cap_bytes,err);
    }
    if(status==FG_OK&&prepare&&self!=manifest->layer_owner[FG_LAYER_COUNT-1u])
        ack=(fg_mtp_native_ack){.operation=work.operation};
    else if(status==FG_OK)status=fg_mtp_native_run(context->mtp_native,
        fg_frame_session_slot(header),&work,hidden,&ack,err);
    if(prepare&&
       (status==FG_ERR_LIMIT||status==FG_ERR_OOM)){
        fprintf(stderr,"MTP readiness rejected: %s\n",err?err->message:"memory admission");
        ack=(fg_mtp_native_ack){.operation=work.operation,.status=status};
        status=FG_OK;
    }
    if(status==FG_OK&&work.operation==FG_MTP_NATIVE_DRAFT)
        return mtp_head_fanout(fabric,manifest,self,request,FG_MTP_HEAD_DRAFT,
            work.first_token,1u,hidden,context,err);
    uint8_t wire[FG_MTP_NATIVE_ACK_BYTES];
    if(status==FG_OK)status=fg_mtp_native_ack_encode(wire,&ack,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_CONTROL,
        FG_MSG_MTP_NATIVE_ACK,request,fg_frame_sequence(header),0u,wire,sizeof(wire),err);
    return status;
}

static fg_status handle_mtp_verify_work(fg_fabric *fabric,fg_owner_executor *owner,
    const fg_manifest *manifest,uint32_t self,uint64_t session,uint32_t peer,
    const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,
    layer_work_context *context,depthb_owner_runtime *depthb,fg_error *err){
    uint32_t slot=fg_frame_session_slot(header);
    uint32_t flags=ntohl(header->flags_be)&0x00ffffffu;
    if(!owner||!fg_mtp_native_enabled(context->mtp_native)||
       slot>=FG_OWNER_SESSION_MAX||!depthb->checkpoint_valid[slot]||
       (flags&~FG_MTP_VERIFY_REPLAY)){
        fg_error_set(err,FG_ERR_MISMATCH,"MTP verification requires a prepared owner session");
        return FG_ERR_MISMATCH;
    }
    fg_prefill_layer_work work={0};
    fg_status status=fg_prefill_layer_work_decode(&work,manifest->protocol_version,
        context->positions,context->tokens*3u,context->hyper_in,
        (uint64_t)context->tokens*FG_HYPER_WIDTH,context->ngram,
        (uint64_t)context->tokens*FG_NGRAM_EMBED_VALUES,payload,bytes,err);
    uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&(!session||request!=session||work.source_rank!=peer||
       work.destination_rank!=self||work.token_count>FG_MTP_MAX_VERIFY||
       work.position_mode!=FG_POSITION_TEXT||
       work.first_token>manifest->native_context-work.token_count||
       work.layer>=FG_LAYER_COUNT||manifest->layer_owner[work.layer]!=self||
       (work.layer&&manifest->layer_owner[work.layer-1u]==self)||
       peer!=(work.layer?manifest->layer_owner[work.layer-1u]:0u)||
       ((work.layer<=1u)!=((work.flags&FG_LAYER_WORK_HAS_NGRAM)!=0u))||
       fg_frame_sequence(header)!=work.first_token*FG_LAYER_COUNT+work.layer)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted causal MTP verification");
        status=FG_ERR_MISMATCH;
    }
    uint32_t last=work.layer;
    while(status==FG_OK&&last+1u<FG_LAYER_COUNT&&manifest->layer_owner[last+1u]==self)last++;
    if(status==FG_OK&&!decode_block_chain_eligible(context->decode_dispatch.expert,
        manifest,self,work.layer,last)){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"MTP requires local complete expert slabs");
        status=FG_ERR_UNAVAILABLE;
    }
    if(status==FG_OK)status=fg_owner_set_active_session(owner,slot,err);
    fg_vk_tensor *input=status==FG_OK?fg_owner_prefill_input(owner):NULL;
    if(status==FG_OK&&!input){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"MTP owner input is unavailable");
        status=FG_ERR_UNAVAILABLE;
    }
    if(status==FG_OK)status=fg_vk_tensor_write(input,0u,work.hyper,
        (uint64_t)work.token_count*FG_HYPER_WIDTH*4u,err);
    bool has_ngram=(work.flags&FG_LAYER_WORK_HAS_NGRAM)!=0u;
    if(status==FG_OK&&has_ngram){
        if(!context->ngram_tensor){
            fg_error_set(err,FG_ERR_MISMATCH,"MTP layer one lacks n-gram storage");
            status=FG_ERR_MISMATCH;
        }else status=fg_vk_tensor_write(context->ngram_tensor,0u,work.ngram_embeddings,
            (uint64_t)work.token_count*FG_NGRAM_EMBED_VALUES*4u,err);
    }
    fg_vk_tensor *output=NULL;
    if(status==FG_OK&&work.token_count==1u)
        status=fg_owner_decode_block_chained(owner,work.layer,last,work.first_token,
            work.positions,input,has_ngram?context->ngram_tensor:NULL,chained_decode_expert,
            &context->decode_dispatch,&output,err);
    else if(status==FG_OK)
        status=fg_owner_decode_small_n(owner,work.layer,last,work.first_token,
            work.positions,work.token_count,input,has_ngram?context->ngram_tensor:NULL,
            mtp_small_expert,context->decode_dispatch.expert,&output,err);
    if(status==FG_OK)status=fg_vk_tensor_read(output,0u,context->hyper_out,
        (uint64_t)work.token_count*FG_HYPER_WIDTH*4u,err);
    for(uint32_t i=0;status==FG_OK&&i<work.token_count;i++)
        status=validate_block_hidden(self,work.first_token+i,last,
            context->hyper_out+(uint64_t)i*FG_HYPER_WIDTH,err);
    if(status==FG_OK&&last+1u<FG_LAYER_COUNT){
        fg_prefill_layer_work next={.layer=(uint8_t)(last+1u),.source_rank=(uint8_t)self,
            .destination_rank=manifest->layer_owner[last+1u],.first_token=work.first_token,
            .token_count=work.token_count,.positions=work.positions,.hyper=context->hyper_out};
        uint32_t next_bytes=0u;
        status=fg_prefill_layer_work_encode(context->work_wire,context->work_capacity,
            &next_bytes,manifest->protocol_version,&next,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,next.destination_rank,FG_FABRIC_BULK,
            FG_MSG_MTP_VERIFY_WORK,request,work.first_token*FG_LAYER_COUNT+next.layer,
            flags,context->work_wire,next_bytes,err);
    }else if(status==FG_OK){
        status=fg_mtp_native_observe_verify(context->mtp_native,slot,work.first_token,
            work.token_count,context->hyper_out,err);
        if(status==FG_OK&&!(flags&FG_MTP_VERIFY_REPLAY)){
            float hidden[FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE];
            if(!context->output_hc){
                fg_error_set(err,FG_ERR_MISMATCH,"MTP verifier lacks the target HC head");
                status=FG_ERR_MISMATCH;
            }
            for(uint32_t i=0;status==FG_OK&&i<work.token_count;i++)
                status=fg_output_hc_run(context->output_hc,
                    context->hyper_out+(uint64_t)i*FG_HYPER_WIDTH,
                    hidden+(uint64_t)i*FG_HIDDEN_SIZE,err);
            if(status==FG_OK)status=mtp_head_fanout(fabric,manifest,self,request,
                FG_MTP_HEAD_VERIFY,work.first_token,work.token_count,hidden,context,err);
        }
        fg_mtp_verify_result result={.source_rank=(uint8_t)self,
            .first_token=work.first_token,.count=work.token_count};
        uint8_t wire[FG_MTP_VERIFY_RESULT_BYTES];
        if(status==FG_OK)status=fg_mtp_verify_result_encode(wire,&result,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
            FG_MSG_MTP_VERIFY_RESULT,request,
            work.first_token*FG_LAYER_COUNT+FG_LAYER_COUNT-1u,flags,wire,sizeof(wire),err);
    }
    return status;
}
