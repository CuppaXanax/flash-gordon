/* Exercise the real final-owner output dispatch without GPU/model weights. */
#define fg_fabric_send capture_send
#define fg_output_hc_run capture_hc
#include "../src/runtime.c"

#define CHECK(expression) do { if(!(expression)){ \
    fprintf(stderr,"FAIL line %d: %s (%s)\n",__LINE__,#expression,error.message); \
    exit(1); \
} } while(0)

static struct {
    uint32_t peer,bytes,sequence;
    fg_message_type type;
    uint64_t request;
    uint8_t wire[FG_DECODE_LAYER_RESULT_BYTES];
} sent[8];
static uint32_t sends,hc_runs;

fg_status capture_send(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,
    fg_message_type type,uint64_t request,uint32_t sequence,uint32_t flags,
    const void *payload,uint32_t bytes,fg_error *err){
    (void)fabric;(void)flags;(void)err;
    if(sends>=8u||bytes>sizeof(sent[0].wire)||cls!=FG_FABRIC_BULK)abort();
    sent[sends].peer=peer;sent[sends].type=type;sent[sends].request=request;
    sent[sends].sequence=sequence;sent[sends].bytes=bytes;
    memcpy(sent[sends++].wire,payload,bytes);
    return FG_OK;
}

fg_status capture_hc(fg_output_hc *hc,const void *hyper,float *hidden,fg_error *err){
    (void)hc;(void)err;
    memcpy(hidden,hyper,FG_HIDDEN_SIZE*4u);hc_runs++;
    return FG_OK;
}

static void test_dispatch(void){
    fg_error error={0};
    fg_manifest *manifest=calloc(1,sizeof(*manifest));
    fg_manifest_init(manifest);
    manifest->layer_owner[FG_LAYER_COUNT-1u]=7u;
    CHECK(decode_direct_output_eligible(manifest));
    layer_work_context *context=calloc(1,sizeof(*context));
    context->result_wire=malloc(FG_DECODE_LAYER_RESULT_BYTES);
    context->hyper_out=calloc(FG_HYPER_WIDTH,sizeof(float));
    context->hyper_out[0]=1.25f;
    context->output_hc=(fg_output_hc *)context;
    fg_layer_work *work=calloc(1,sizeof(*work));
    work->token_index=59599u;work->position[0]=50123u;
    const uint64_t session=UINT64_C(7698637730828577217);
    fg_coordinator coordinator={.manifest=manifest,.session_id=session,.output_session=1u};
    fg_sampler_config sampler;
    for(uint32_t pass=0u;pass<2u;pass++){
        uint32_t ways=pass==0u?4u:2u;
        context->output_split_ways=ways;
        for(uint32_t mode=0u;mode<4u;mode++){
            fg_sampler_config_greedy(&sampler);
            if(mode==1u)sampler.temperature=1.0f;
            if(mode==2u)sampler.presence_penalty=0.2f;
            if(mode==3u)sampler.repetition_penalty=1.1f;
            bool split=mode==0u;
            CHECK(output_split_greedy(&sampler)==split);
            work->flags=split?FG_LAYER_WORK_FLAG_OUTPUT_SPLIT_GREEDY:0u;
            uint8_t work_wire[FG_DECODE_LAYER_WORK_MAX_BYTES];uint32_t bytes=0u;
            fg_layer_work *decoded=calloc(1,sizeof(*decoded));
            CHECK(fg_decode_layer_work_encode(work_wire,sizeof(work_wire),&bytes,
                FG_PROTOCOL_VERSION,work,&error)==FG_OK);
            CHECK(fg_decode_layer_work_decode(decoded,FG_PROTOCOL_VERSION,
                work_wire,bytes,&error)==FG_OK);
            CHECK(decoded->flags==work->flags&&decoded->token_index==work->token_index&&
                  decoded->position[0]!=decoded->token_index);
            sends=0u;hc_runs=0u;
            CHECK(worker_decode_output(NULL,manifest,7u,session,decoded,
                FG_LAYER_COUNT-1u,context,&error)==FG_OK);
            free(decoded);
            if(!split&&sends>1u){
                /* Model the legal cross-peer ordering: rank 4 returns its full
                 * sample before rank 0 receives rank 7's unused slice. */
                fg_output_result result={.source_rank=4u,.token_index=work->token_index,
                    .token=123u,.logit=1.0f};
                uint8_t wire[FG_OUTPUT_RESULT_BYTES];fg_frame_header header;
                uint32_t next=0u;
                CHECK(fg_output_result_encode(wire,&result,&error)==FG_OK);
                CHECK(fg_frame_encode(&header,FG_MSG_OUTPUT_RESULT,session,
                    work->token_index*FG_LAYER_COUNT+FG_LAYER_COUNT,1u<<24u,
                    wire,sizeof(wire),&error)==FG_OK);
                CHECK(coordinator_output_result_accept(&coordinator,&header,wire,
                    sizeof(wire),work->token_index,&next,NULL,&error)==FG_OK);
                CHECK(next==123u);
                for(uint32_t i=0u;i<sends;i++)if(sent[i].peer==0u){
                    CHECK(fg_frame_encode(&header,sent[i].type,session,sent[i].sequence,
                        1u<<24u,sent[i].wire,sent[i].bytes,&error)==FG_OK);
                    CHECK(coordinator_output_slice_route(&coordinator,7u,&header,0u,7u,
                        ways==2u?47:-1,work->token_index,work->token_index+1u,&error)==
                        FG_ERR_MISMATCH);
                    fprintf(stderr,"Old late-slice race reproduced: %s\n",error.message);
                }
            }
            CHECK(sends==(split?ways:1u));
            CHECK(hc_runs==(split&&ways==4u?1u:0u));
            for(uint32_t i=0u;i<sends;i++){
                CHECK(sent[i].request==session);
                CHECK(sent[i].sequence==work->token_index*FG_LAYER_COUNT+FG_LAYER_COUNT-1u);
                if(!split){
                    CHECK(sent[i].peer==4u&&sent[i].type==FG_MSG_OUTPUT_HIDDEN);
                    fg_layer_result hidden;
                    CHECK(fg_decode_layer_result_decode(&hidden,sent[i].wire,
                        sent[i].bytes,&error)==FG_OK);
                    CHECK(hidden.token_index==work->token_index&&hidden.hyper[0]==1.25f);
                }else if(sent[i].peer==0u){
                    fg_frame_header header;
                    uint32_t token=0u;
                    if(ways==2u){
                        fg_layer_result slice;
                        CHECK(fg_decode_layer_result_decode(&slice,sent[i].wire,
                            sent[i].bytes,&error)==FG_OK);
                        CHECK(slice.source_rank==7u&&slice.destination_rank==0u);
                        token=slice.token_index;
                    }else{
                        fg_output_slice_hidden slice;
                        CHECK(fg_output_slice_hidden_decode(&slice,sent[i].wire,
                            sent[i].bytes,&error)==FG_OK);
                        CHECK(slice.source_rank==7u&&slice.destination_rank==0u&&
                            slice.hidden[0]==1.25f);
                        token=slice.token_index;
                    }
                    CHECK(token==work->token_index);
                    CHECK(fg_frame_encode(&header,sent[i].type,session,sent[i].sequence,
                        1u<<24u,sent[i].wire,sent[i].bytes,&error)==FG_OK);
                    CHECK(coordinator_output_slice_route(&coordinator,7u,&header,0u,7u,
                        ways==2u?47:-1,work->token_index,work->token_index,&error)==FG_OK);
                    /* Result-before-slice on an unsplit token leaves this stale
                     * frame for the following continuation's first decode. */
                    CHECK(coordinator_output_slice_route(&coordinator,7u,&header,0u,7u,
                        ways==2u?47:-1,work->token_index,work->token_index+1u,&error)==
                        FG_ERR_MISMATCH);
                    CHECK(strstr(error.message,"token=59599/59600"));
                }
            }
        }
    }
    free(work);free(context->hyper_out);free(context->result_wire);free(context);free(manifest);
}

static void test_route_diagnostics(void){
    fg_error error={0};
    fg_manifest *manifest=calloc(1,sizeof(*manifest));
    manifest->layer_owner[47u]=7u;
    fg_coordinator coordinator={.manifest=manifest,.session_id=100u,.output_session=1u};
    for(uint32_t field=0u;field<8u;field++){
        uint32_t destination=field==0u?1u:0u,source=field==2u?6u:7u;
        int32_t layer=field==1u?46:47;
        uint32_t peer=field==3u?6u:7u,token=field==4u?99u:100u;
        uint64_t request=field==5u?99u:100u;
        uint32_t sequence=100u*FG_LAYER_COUNT+47u-(field==6u?1u:0u);
        uint32_t slot=field==7u?0u:1u;
        fg_frame_header header;
        CHECK(fg_frame_encode(&header,FG_MSG_OUTPUT_SLICE,request,sequence,slot<<24u,
            NULL,0u,&error)==FG_OK);
        CHECK(coordinator_output_slice_route(&coordinator,peer,&header,destination,
            source,layer,token,100u,&error)==FG_ERR_MISMATCH);
        const char *expected[]={"dst=1/0","layer=46/47","src=6/7","peer=6/7",
            "token=99/100","request=99/100","seq=4846/4847","slot=0/1"};
        CHECK(strstr(error.message,expected[field]));
        CHECK(strlen(error.message)<sizeof(error.message)-1u);
    }
    free(manifest);
}

int main(void){
    test_dispatch();
    test_route_diagnostics();
    puts("Decode output: unsplit has no late slices; 2/4-way greedy and route diagnostics: PASS");
    return 0;
}
