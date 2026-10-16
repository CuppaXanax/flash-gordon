struct fg_mtp_serving {
    fg_mtp_native *native;
    uint8_t *native_wire,*receive_wire;
    uint32_t receive_capacity,rpc_sequence,warm_start,prefix_end;
    int32_t *history;
    size_t history_capacity;
};

static fg_status runtime_mtp_policy_init(fg_runtime *runtime,fg_error *err){
    const double ratios[FG_MTP_MAX_DRAFT]={1.5,2.05,2.55};
    return fg_mtp_policy_init(&runtime->mtp_policy,32.0,2.0,ratios,err);
}

static void coordinator_mtp_close(fg_mtp_serving *mtp){
    if(!mtp)return;
    fg_mtp_native_close(mtp->native);
    free(mtp->native_wire);free(mtp->receive_wire);free(mtp->history);free(mtp);
}

static fg_status coordinator_mtp_ack(fg_coordinator *c,uint32_t peer,
    uint64_t request,uint32_t sequence,const fg_mtp_native_work *work,
    fg_mtp_native_ack *ack,fg_error *err){
    uint8_t wire[FG_MTP_NATIVE_ACK_BYTES];fg_frame_header header;uint32_t bytes=0u;
    fg_status status=fg_fabric_recv(c->fabric,peer,FG_FABRIC_CONTROL,&header,wire,
        sizeof(wire),&bytes,err);
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_MTP_NATIVE_ACK||
       fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=sequence||
       fg_frame_session_slot(&header)!=c->output_session)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale native MTP acknowledgement from rank %u",peer);
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_mtp_native_ack_decode(ack,wire,bytes,err);
    if(status==FG_OK&&(ack->operation!=work->operation||
       ack->first_token!=work->first_token||ack->count!=work->count)){
        fg_error_set(err,FG_ERR_MISMATCH,"native MTP acknowledgement operation mismatch");
        status=FG_ERR_MISMATCH;
    }
    return status;
}

static fg_status coordinator_mtp_send_native(fg_coordinator *c,uint32_t peer,
    const fg_mtp_native_work *work,uint32_t *sequence,fg_error *err){
    fg_mtp_serving *m=c->mtp;uint32_t bytes=0u;
    if(m->rpc_sequence==UINT32_MAX){
        fg_error_set(err,FG_ERR_LIMIT,"native MTP RPC sequence exhausted");return FG_ERR_LIMIT;
    }
    *sequence=++m->rpc_sequence;
    fg_status status=fg_mtp_native_work_encode(m->native_wire,FG_MTP_NATIVE_WORK_MAX_BYTES,
        &bytes,work,err);
    uint64_t request=c->session_id?c->session_id:UINT64_C(1);
    if(status==FG_OK)status=fg_fabric_send(c->fabric,peer,FG_FABRIC_BULK,
        FG_MSG_MTP_NATIVE_WORK,request,*sequence,0u,m->native_wire,bytes,err);
    return status;
}
static fg_status coordinator_mtp_rpc(fg_coordinator *c,const fg_mtp_native_work *work,
    uint32_t expected_frontier,fg_error *err){
    uint32_t peer=c->manifest->layer_owner[FG_LAYER_COUNT-1u],sequence=0u;
    fg_status status=coordinator_mtp_send_native(c,peer,work,&sequence,err);
    fg_mtp_native_ack ack;
    if(status==FG_OK)status=coordinator_mtp_ack(c,peer,c->session_id,sequence,work,&ack,err);
    if(status==FG_OK&&ack.frontier!=expected_frontier){
        fg_error_set(err,FG_ERR_MISMATCH,"native MTP frontier %u != expected %u",
                     ack.frontier,expected_frontier);
        status=FG_ERR_MISMATCH;
    }
    if(status!=FG_OK)transport_poison(&c->transport_state);
    return status;
}

static fg_status coordinator_mtp_create(fg_coordinator *c,bool configure,fg_error *err){
    if(!(c->options.experimental_flags&FG_RUNTIME_EXPERIMENTAL_MTP))return FG_OK;
    if(!c->ring_prefill||!c->ring_decode||!c->output_slice||
       fg_output_slice_ways(c->output_slice)!=FG_OUTPUT_SPLIT_WAYS_MAX){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"MTP requires ring prefill/decode and the four-way head");
        return FG_ERR_UNAVAILABLE;
    }
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
        if(fg_expert_local_count(c->manifest,layer,c->manifest->layer_owner[layer])!=FG_EXPERT_COUNT){
            fg_error_set(err,FG_ERR_UNAVAILABLE,"MTP requires whole expert slabs on block owners");
            return FG_ERR_UNAVAILABLE;
        }
    fg_mtp_serving *m=calloc(1,sizeof(*m));
    if(!m){fg_error_set(err,FG_ERR_OOM,"allocate MTP serving stream");return FG_ERR_OOM;}
    c->mtp=m;
    m->receive_capacity=FG_PREFILL_LAYER_HEADER_BYTES+FG_MTP_MAX_VERIFY*3u*4u+
        FG_MTP_MAX_VERIFY*(FG_HYPER_WIDTH+FG_NGRAM_EMBED_VALUES)*4u;
    m->native_wire=malloc(FG_MTP_NATIVE_WORK_MAX_BYTES);
    m->receive_wire=malloc(m->receive_capacity);
    fg_status status=FG_OK;
    if(!m->native_wire||!m->receive_wire){
        fg_error_set(err,FG_ERR_OOM,"allocate MTP serving wire buffers");status=FG_ERR_OOM;
    }
    fg_vk_context *vk=fg_model_vk(c->model);
    if(status==FG_OK)status=fg_mtp_native_create(&m->native,vk,c->manifest,0u,err);
    fg_mtp_native_work work={.operation=FG_MTP_NATIVE_CONFIG,
        .overlay=c->options.mtp_overlay,.count=(uint32_t)strlen(c->options.mtp_overlay),
        .embedding_file=c->options.mtp_embedding,
        .argument=c->options.logical_context_tokens,.limit=c->options.mtp_window_tokens};
    fg_mtp_native_ack ack;
    if(status==FG_OK)status=fg_mtp_native_run(m->native,c->output_session,&work,NULL,&ack,err);
    if(status==FG_OK&&configure){
        uint32_t sequences[FG_RANK_COUNT]={0};
        for(uint32_t peer=1;status==FG_OK&&peer<FG_RANK_COUNT;peer++)
            status=coordinator_mtp_send_native(c,peer,&work,&sequences[peer],err);
        for(uint32_t peer=1;status==FG_OK&&peer<FG_RANK_COUNT;peer++){
            uint64_t request=c->session_id?c->session_id:UINT64_C(1);
            status=coordinator_mtp_ack(c,peer,request,sequences[peer],&work,&ack,err);
            if(status==FG_OK&&ack.frontier!=0u){
                fg_error_set(err,FG_ERR_MISMATCH,"MTP configuration inherited nonzero KV state");
                status=FG_ERR_MISMATCH;
            }
        }
    }
    if(status==FG_OK)c->depthb_work.mtp_native=m->native;
    return status;
}

static fg_status coordinator_mtp_prefix(fg_coordinator *c,const int32_t *history,
    uint32_t offset,uint32_t end,bool warm,fg_error *err){
    if(!c->mtp)return FG_OK;
    fg_mtp_serving *m=c->mtp;
    m->warm_start=warm&&end>c->options.mtp_window_tokens+1u?
        end-1u-c->options.mtp_window_tokens:0u;
    if(!warm)m->warm_start=end-1u;
    m->prefix_end=end;
    uint32_t token=offset?(uint32_t)history[offset]:0u;
    fg_mtp_native_work work={.operation=FG_MTP_NATIVE_PREFIX,.first_token=offset,
        .count=offset&&warm?1u:0u,.argument=m->warm_start,.limit=end,.tokens=&token};
    uint32_t expected=offset>m->warm_start?offset:m->warm_start;
    /* A reused prefix warms cell offset-1 before the first new target input;
     * both the reused and safely reset cases therefore end at max(offset,start). */
    return coordinator_mtp_rpc(c,&work,expected,err);
}

static fg_status coordinator_mtp_advance(fg_coordinator *c,uint32_t token,
    uint32_t index,fg_error *err){
    if(!c->mtp)return FG_OK;
    fg_mtp_native_work work={.operation=FG_MTP_NATIVE_ADVANCE,.first_token=index,
        .count=1u,.tokens=&token};
    return coordinator_mtp_rpc(c,&work,index,err);
}

typedef struct mtp_head_reduction {
    uint32_t received;
    float values[FG_MTP_MAX_VERIFY];
    uint32_t ids[FG_MTP_MAX_VERIFY];
} mtp_head_reduction;

static fg_status mtp_accept_partial(mtp_head_reduction *reduction,
    const fg_mtp_head_partial *p,uint32_t peer,fg_mtp_head_kind kind,
    uint32_t first,uint32_t count,fg_error *err){
    uint32_t way=0u,first_row=0u,rows=0u;
    if(p->source_rank!=peer||p->destination_rank!=0u||p->kind!=kind||
       p->first_token!=first||p->count!=count||
       !fg_output_split_way_for_rank(FG_OUTPUT_SPLIT_WAYS_MAX,peer,&way)||
       (reduction->received&(1u<<peer))){
        fg_error_set(err,FG_ERR_MISMATCH,"duplicate or misrouted MTP head partial");
        return FG_ERR_MISMATCH;
    }
    fg_output_split_span(FG_OUTPUT_SPLIT_WAYS_MAX,way,&first_row,&rows);
    for(uint32_t i=0;i<count;i++){
        if(p->ids[i]<first_row||p->ids[i]-first_row>=rows){
            fg_error_set(err,FG_ERR_MISMATCH,"MTP head partial belongs to another vocabulary slice");
            return FG_ERR_MISMATCH;
        }
        if(!reduction->received){
            reduction->values[i]=p->values[i];reduction->ids[i]=p->ids[i];
        }else fg_output_combine(reduction->values[i],reduction->ids[i],
            p->values[i],p->ids[i],&reduction->values[i],&reduction->ids[i]);
    }
    reduction->received|=1u<<peer;return FG_OK;
}

static uint32_t mtp_head_rank_mask(void){
    uint32_t mask=0u;
    for(uint32_t way=0;way<FG_OUTPUT_SPLIT_WAYS_MAX;way++)
        mask|=1u<<fg_output_split_rank(FG_OUTPUT_SPLIT_WAYS_MAX,way);
    return mask;
}

/* All head messages and causal work/results use BULK so the existing
 * session-aware coroutine receiver can wait on one class without raw polling.
 * Native setup/repair ACKs use CONTROL. Prefill warmfill completes before the
 * ordinary frame result, so there is no cross-channel warm-ACK deadlock. */
static fg_status coordinator_mtp_collect(fg_coordinator *c,fg_mtp_head_kind kind,
    uint32_t first,uint32_t count,bool verify,bool replay,uint32_t *tokens,
    float *logits,fg_error *err){
    fg_mtp_serving *m=c->mtp;
    mtp_head_reduction reduction={0};bool done=!verify;
    uint32_t expected_mask=replay?0u:mtp_head_rank_mask();
    fg_status status=FG_OK;
    uint64_t deadline=monotonic_ms()+output_split_timeout_ms()+FG_OUTPUT_SPLIT_TIMEOUT_GRACE_MS;
    while(status==FG_OK&&(!done||reduction.received!=expected_mask)){
        uint32_t peer=0u,bytes=0u;fg_frame_header header;
        uint64_t now=monotonic_ms();
        int32_t timeout=now<deadline?(int32_t)(deadline-now):0;
        status=fg_fabric_recv_any_timeout(c->fabric,FG_FABRIC_BULK,timeout,&peer,&header,
            m->receive_wire,m->receive_capacity,&bytes,err);
        if(status!=FG_OK)break;
        if(fg_frame_request_id(&header)!=c->session_id||
           fg_frame_session_slot(&header)!=c->output_session){
            fg_error_set(err,FG_ERR_MISMATCH,"MTP receive crossed a stream namespace");
            status=FG_ERR_MISMATCH;break;
        }
        fg_message_type type=fg_frame_type(&header);
        if(type==FG_MSG_MTP_VERIFY_WORK&&verify){
            if(bytes<FG_PREFILL_LAYER_HEADER_BYTES||
               ((uint32_t)m->receive_wire[4]<<24u|(uint32_t)m->receive_wire[5]<<16u|
                (uint32_t)m->receive_wire[6]<<8u|m->receive_wire[7])!=first||
               ((uint32_t)m->receive_wire[8]<<8u|m->receive_wire[9])!=count||
               ((ntohl(header.flags_be)&FG_MTP_VERIFY_REPLAY)!=0u)!=replay){
                fg_error_set(err,FG_ERR_MISMATCH,"MTP root block does not match this verification");
                status=FG_ERR_MISMATCH;break;
            }
            status=handle_mtp_verify_work(c->fabric,c->owner,c->manifest,0u,c->session_id,
                peer,&header,m->receive_wire,bytes,&c->depthb_work,&c->depthb,err);
            if(status==FG_OK)fg_pipeline_checkpoint();
        }else if(type==FG_MSG_MTP_VERIFY_RESULT&&verify&&!done){
            fg_mtp_verify_result result;
            status=fg_mtp_verify_result_decode(&result,m->receive_wire,bytes,err);
            if(status==FG_OK&&(result.source_rank!=peer||
               peer!=c->manifest->layer_owner[FG_LAYER_COUNT-1u]||
               result.first_token!=first||result.count!=count||
               fg_frame_sequence(&header)!=first*FG_LAYER_COUNT+FG_LAYER_COUNT-1u||
               ((ntohl(header.flags_be)&FG_MTP_VERIFY_REPLAY)!=0u)!=replay)){
                fg_error_set(err,FG_ERR_MISMATCH,"MTP final block completion mismatch");
                status=FG_ERR_MISMATCH;
            }
            if(status==FG_OK)done=true;
        }else if(type==FG_MSG_MTP_HEAD_WORK&&!replay){
            float hidden[FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE];fg_mtp_head_work work;
            status=fg_mtp_head_work_decode(&work,hidden,FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE,
                m->receive_wire,bytes,err);
            if(status==FG_OK&&(work.source_rank!=peer||
               peer!=c->manifest->layer_owner[FG_LAYER_COUNT-1u]||
               work.destination_rank!=0u||work.kind!=kind||work.first_token!=first||
               work.count!=count||
               fg_frame_sequence(&header)!=first*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
                fg_error_set(err,FG_ERR_MISMATCH,"MTP local contracted head work mismatch");
                status=FG_ERR_MISMATCH;
            }
            fg_mtp_head_partial partial;
            if(status==FG_OK)status=mtp_slice_partial(c->output_slice,&work,0u,&partial,err);
            if(status==FG_OK)status=mtp_accept_partial(&reduction,&partial,0u,kind,first,count,err);
            if(status==FG_OK)fg_pipeline_checkpoint();
        }else if(type==FG_MSG_MTP_HEAD_PARTIAL&&!replay){
            fg_mtp_head_partial partial;
            status=fg_mtp_head_partial_decode(&partial,m->receive_wire,bytes,err);
            if(status==FG_OK&&fg_frame_sequence(&header)!=first*FG_LAYER_COUNT+FG_LAYER_COUNT-1u){
                fg_error_set(err,FG_ERR_MISMATCH,"MTP head partial sequence mismatch");
                status=FG_ERR_MISMATCH;
            }
            if(status==FG_OK)status=mtp_accept_partial(&reduction,&partial,peer,kind,first,count,err);
        }else{
            fg_error_set(err,FG_ERR_MISMATCH,"unexpected MTP ring message %u",type);
            status=FG_ERR_MISMATCH;
        }
    }
    if(status==FG_OK&&!replay){
        if(tokens)memcpy(tokens,reduction.ids,count*sizeof(*tokens));
        if(logits)memcpy(logits,reduction.values,count*sizeof(*logits));
    }
    if(status!=FG_OK)transport_poison(&c->transport_state);
    return status;
}

typedef struct mtp_round_context {
    fg_runtime *runtime;
    uint32_t first,count,consumed;
    const uint32_t *inputs;
    float logits[FG_MTP_MAX_VERIFY];
} mtp_round_context;

static fg_status coordinator_mtp_verify(mtp_round_context *round,const uint32_t *inputs,
    uint32_t count,bool replay,uint32_t *tokens,fg_error *err){
    fg_coordinator *c=&round->runtime->coordinator;fg_mtp_serving *m=c->mtp;
    uint32_t first=round->first;size_t needed=(size_t)first+count;
    if(needed>m->history_capacity){
        int32_t *history=realloc(m->history,needed*sizeof(*history));
        if(!history){fg_error_set(err,FG_ERR_OOM,"grow MTP verification history");return FG_ERR_OOM;}
        m->history=history;m->history_capacity=needed;
    }
    memcpy(m->history,round->runtime->history,(size_t)first*sizeof(*m->history));
    for(uint32_t i=0;i<count;i++)m->history[first+i]=(int32_t)inputs[i];
    layer_work_context *context=&c->depthb_work;
    prefill_layer_buffers *buffers=&c->prefill_layer[0];
    fg_vk_tensor *embedding=fg_model_tensor(c->model,"token_embd.weight");
    fg_status status=fg_owner_set_active_session(c->owner,c->output_session,err);
    if(status==FG_OK)status=fg_vk_tensor_write(buffers->token_tensor,0u,inputs,(uint64_t)count*4u,err);
    fg_vk_tensor *input=fg_owner_prefill_input(c->owner),*ngram=NULL;
    if(status==FG_OK)status=fg_vk_embedding_q8_0_batch(fg_model_vk(c->model),input,embedding,
        buffers->token_tensor,count,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,FG_Q38_HYPER_COUNT,err);
    if(status==FG_OK)status=fg_ngram_store_lookup_prefill(c->ngram,m->history,needed,
        first,count,&ngram,err);
    if(status==FG_OK)status=fg_vk_tensor_read(input,0u,context->hyper_in,
        (uint64_t)count*FG_HYPER_WIDTH*4u,err);
    float ngram_host[FG_MTP_MAX_VERIFY*FG_NGRAM_EMBED_VALUES];
    if(status==FG_OK)status=fg_vk_tensor_read(ngram,0u,ngram_host,
        (uint64_t)count*FG_NGRAM_EMBED_VALUES*4u,err);
    uint32_t positions[FG_MTP_MAX_VERIFY*3u];
    for(uint32_t i=0;i<count;i++)for(uint32_t axis=0;axis<3u;axis++)
        positions[i*3u+axis]=first+i;
    fg_prefill_layer_work work={.layer=0u,.source_rank=0u,
        .destination_rank=c->manifest->layer_owner[0u],.flags=FG_LAYER_WORK_HAS_NGRAM,
        .first_token=first,.token_count=(uint16_t)count,.positions=positions,
        .hyper=context->hyper_in,.ngram_embeddings=ngram_host};
    uint32_t bytes=0u;
    if(status==FG_OK)status=fg_prefill_layer_work_encode(context->work_wire,
        context->work_capacity,&bytes,c->manifest->protocol_version,&work,err);
    uint32_t flags=(replay?FG_MTP_VERIFY_REPLAY:0u)|
        (c->output_session<<24u);
    if(status==FG_OK&&work.destination_rank==0u){
        fg_frame_header header;
        status=fg_frame_encode_version(&header,c->manifest->protocol_version,
            FG_MSG_MTP_VERIFY_WORK,c->session_id,first*FG_LAYER_COUNT,flags,
            context->work_wire,bytes,err);
        if(status==FG_OK)status=handle_mtp_verify_work(c->fabric,c->owner,c->manifest,
            0u,c->session_id,0u,&header,context->work_wire,bytes,context,&c->depthb,err);
    }else if(status==FG_OK)status=fg_fabric_send(c->fabric,work.destination_rank,FG_FABRIC_BULK,
        FG_MSG_MTP_VERIFY_WORK,c->session_id,first*FG_LAYER_COUNT,
        replay?FG_MTP_VERIFY_REPLAY:0u,context->work_wire,bytes,err);
    if(status==FG_OK)status=coordinator_mtp_collect(c,FG_MTP_HEAD_VERIFY,first,count,true,
        replay,tokens,replay?NULL:round->logits,err);
    if(status==FG_OK){round->consumed=count;round->inputs=inputs;}
    return status;
}

static fg_status mtp_round_checkpoint(void *opaque,fg_error *err){
    mtp_round_context *r=opaque;fg_coordinator *c=&r->runtime->coordinator;
    return coordinator_owner_transaction(c,FG_OWNER_SESSION_PREPARE,
                                         (uint8_t)c->output_session,err);
}
static fg_status mtp_round_verify(void *opaque,const uint32_t *inputs,uint32_t count,
    uint32_t *tokens,fg_error *err){
    return coordinator_mtp_verify(opaque,inputs,count,false,tokens,err);
}
static fg_status mtp_round_restore(void *opaque,fg_error *err){
    mtp_round_context *r=opaque;fg_coordinator *c=&r->runtime->coordinator;
    return coordinator_owner_transaction(c,FG_OWNER_SESSION_RESTORE,
                                         (uint8_t)c->output_session,err);
}
static fg_status mtp_round_replay(void *opaque,const uint32_t *inputs,uint32_t count,
    fg_error *err){
    return coordinator_mtp_verify(opaque,inputs,count,true,NULL,err);
}
static fg_status mtp_round_commit(void *opaque,fg_error *err){
    mtp_round_context *r=opaque;fg_coordinator *c=&r->runtime->coordinator;
    fg_mtp_native_work work={.operation=FG_MTP_NATIVE_FINISH,.first_token=r->first,
        .count=r->consumed,.tokens=r->inputs+1u};
    fg_status status=coordinator_mtp_rpc(c,&work,r->first+r->consumed-1u,err);
    if(status==FG_OK)status=coordinator_owner_transaction(c,FG_OWNER_SESSION_COMMIT,
        (uint8_t)c->output_session,err);
    if(status==FG_OK)status=fg_owner_set_active_session(c->owner,c->output_session,err);
    return status;
}

static fg_status runtime_mtp_round(fg_runtime *runtime,uint32_t pending,uint32_t limit,
    fg_mtp_result *result,float logits[FG_MTP_MAX_VERIFY],fg_error *err){
    fg_coordinator *c=&runtime->coordinator;
    uint32_t max_draft=c->options.mtp_max_draft;
    if(max_draft>limit)max_draft=limit;
    uint32_t k=fg_mtp_policy_draft_width(&runtime->mtp_policy,max_draft,
                                       c->options.mtp_fixed_draft);
    uint32_t first=(uint32_t)runtime->history_count-1u;
    uint32_t drafts[FG_MTP_MAX_DRAFT],input=pending;
    fg_status status=FG_OK;double start=dispatch_ts();
    for(uint32_t j=0;status==FG_OK&&j<k;j++){
        fg_mtp_native_work work={.operation=FG_MTP_NATIVE_DRAFT,.first_token=first+j,
            .count=1u,.argument=j,.tokens=&input};
        uint32_t sequence=0u;
        if(status==FG_OK)status=coordinator_mtp_send_native(c,
            c->manifest->layer_owner[FG_LAYER_COUNT-1u],&work,&sequence,err);
        if(status==FG_OK)status=coordinator_mtp_collect(c,FG_MTP_HEAD_DRAFT,
            first+j,1u,false,false,&drafts[j],NULL,err);
        if(status==FG_OK)input=drafts[j];
    }
    mtp_round_context round={.runtime=runtime,.first=first,.count=k+1u};
    const fg_mtp_ops ops={.checkpoint=mtp_round_checkpoint,.verify=mtp_round_verify,
        .restore=mtp_round_restore,.replay=mtp_round_replay,.commit=mtp_round_commit,
        .context=&round};
    if(status==FG_OK)status=fg_mtp_cycle_greedy(&ops,pending,drafts,k,limit,
        fg_tokenizer_eos(c->tokenizer),result,err);
    if(status==FG_OK){
        memcpy(logits,round.logits,result->count*sizeof(*logits));
        status=fg_mtp_policy_observe(&runtime->mtp_policy,k,result->accepted,
                                     dispatch_ts()-start,err);
        if(status==FG_OK)fprintf(stderr,
            "MTP_ROUND slot=%u first=%u k=%u accepted=%u emitted=%u consumed=%u ms=%.3f mode=%s\n",
            c->output_session,first,k,result->accepted,result->count,result->consumed,
            dispatch_ts()-start,c->options.mtp_fixed_draft?"fixed":"adaptive");
    }
    if(status!=FG_OK)transport_poison(&c->transport_state);
    return status;
}
