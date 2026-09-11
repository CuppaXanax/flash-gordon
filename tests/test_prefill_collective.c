#include "fg_expert.h"
#include "fg_fabric.h"
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

static fg_status test_expert(fg_expert_executor *,const fg_prefill_work *,fg_prefill_result *,
    fg_prefill_result_pair *,uint32_t,float *,uint64_t,fg_error *);
static fg_status test_send(fg_fabric *,uint32_t,fg_fabric_class,fg_message_type,
    uint64_t,uint32_t,uint32_t,const void *,uint32_t,fg_error *);
#define fg_expert_prefill test_expert
#define fg_fabric_send test_send
#include "../src/runtime.c"
#undef fg_expert_prefill
#undef fg_fabric_send

/* Real runtime collective and eight TCP/uring peers. Only model arithmetic is
 * replaced by a deterministic oracle. Cached-host merges are the real code. */
static uint32_t self_rank,scenario,work_sends;
static fg_status test_send(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,
    fg_message_type type,uint64_t request,uint32_t sequence,uint32_t flags,
    const void *payload,uint32_t bytes,fg_error *err){
    if(type==FG_MSG_PREFILL_WORK)work_sends++;
    if(scenario==3u&&self_rank==7u&&type==FG_MSG_PREFILL_RESULT&&bytes>=16u&&((const uint8_t *)payload)[7]==16u)sequence++;
    return fg_fabric_send(fabric,peer,cls,type,request,sequence,flags,payload,bytes,err);
}
static float expert_value(uint32_t expert,uint32_t token,uint32_t element){
    return ((int)(expert%13u)-6)*0.031f+(float)token*0.007f+(float)(element%19u)*0.001f;
}
static fg_status test_expert(fg_expert_executor *executor,const fg_prefill_work *work,
    fg_prefill_result *result,fg_prefill_result_pair *pairs,uint32_t capacity,
    float *outputs,uint64_t values,fg_error *err){
    (void)executor;
    if(scenario==1u&&self_rank==7u){fg_error_set(err,FG_ERR_IO,"injected leaf failure");return FG_ERR_IO;}
    if(capacity<work->pair_count||values<(uint64_t)work->token_count*FG_HIDDEN_SIZE)return FG_ERR_LIMIT;
    for(uint32_t i=0;i<work->token_count*FG_Q8K_ACTIVATION_BYTES;i++)
        if(work->activations_q8k[i]!=(uint8_t)(i*13u+5u)){fg_error_set(err,FG_ERR_MISMATCH,"broadcast activation mismatch");return FG_ERR_MISMATCH;}
    memset(outputs,0,(size_t)work->token_count*FG_HIDDEN_SIZE*4u);
    for(uint32_t i=0;i<work->pair_count;i++){
        const fg_prefill_pair *pair=&work->pairs[i];
        pairs[i]=(fg_prefill_result_pair){pair->token_slot,pair->routing_slot};
        for(uint32_t j=0;j<FG_HIDDEN_SIZE;j++){
            uint32_t at=pair->token_slot*FG_HIDDEN_SIZE+j;
            outputs[at]=fmaf(pair->gate,expert_value(pair->expert_id,pair->token_slot,j),outputs[at]);
        }
    }
    *result=(fg_prefill_result){.layer=work->layer,.source_rank=(uint8_t)self_rank,
        .destination_rank=work->source_rank,.contributor_mask=(uint8_t)(1u<<self_rank),
        .first_position=work->first_position,.token_count=work->token_count,
        .pair_count=work->pair_count,.pairs=pairs,.outputs=outputs};
    if(scenario==2u&&self_rank==7u)result->pair_count--; /* Missing route, not a transport error. */
    return FG_OK;
}
static fg_status shared(void *context,fg_error *err){(void)context;if(scenario==4u){fg_error_set(err,FG_ERR_IO,"injected shared failure");return FG_ERR_IO;}return FG_OK;}

static int run_rank(const fg_manifest *manifest,uint32_t rank){
    self_rank=rank;alarm(60);
    fg_error err={0};fg_fabric *fabric=NULL;
    fg_status status=fg_fabric_open(&fabric,manifest,rank,&err);
    if(status==FG_ERR_UNAVAILABLE)return 77;
    if(status!=FG_OK){fprintf(stderr,"collective open rank %u: %s\n",rank,err.message);return 1;}
    prefill_worker_buffers buffers;
    status=prefill_worker_buffers_create(&buffers,128u,rank==0u,&err);
    if(rank==0u){
        atomic_uint transport;atomic_init(&transport,FG_TRANSPORT_READY);
        for(uint32_t c=0;status==FG_OK&&c<(scenario?1u:12u);c++){
            uint32_t layer=scenario?2u:c%8u;
            if(c>=8u)layer=0u;
            uint16_t tokens=scenario?128u:c%3u==0u?1u:c%3u==1u?17u:128u;
            uint32_t n=tokens*FG_TOP_K;
            uint16_t ids[FG_PREFILL_MAX_PAIRS];float gates[FG_PREFILL_MAX_PAIRS];
            uint8_t *activation=malloc((size_t)tokens*FG_Q8K_ACTIVATION_BYTES);
            if(!activation){status=FG_ERR_OOM;break;}
            for(uint32_t i=0;i<tokens*FG_Q8K_ACTIVATION_BYTES;i++)activation[i]=(uint8_t)(i*13u+5u);
            for(uint32_t i=0;i<n;i++){
                ids[i]=(uint16_t)(c==10u?4u*(i%FG_TOP_K)+(i/FG_TOP_K==tokens-1u?1u:0u):c==11u?4u*(i%FG_TOP_K)+(i/FG_TOP_K/16u)%4u:c==8u?4u*(i%FG_TOP_K):c==9u?4u*(i%FG_TOP_K)+1u:i%FG_TOP_K);
                gates[i]=0.01f*(float)(i%FG_TOP_K+1u);
            }
            prefill_dispatch_context context={.fabric=fabric,.manifest=manifest,.self=0u,
                .request_id=9u,.sequence=(c*128u)*FG_LAYER_COUNT+layer,.buffers=&buffers,.transport_state=&transport};
            uint32_t before=work_sends,count=0;fg_prefill_result results[FG_GROUP_SIZE]={0};
            status=dispatch_prefill_experts(&context,layer,c*128u,tokens,ids,gates,
                activation,shared,NULL,results,&count,&err);
            free(activation);
            if(scenario){
                bool rejected=status!=FG_OK&&(transport_ready(&transport)==(scenario==4u));
                fg_fabric_close(fabric);prefill_worker_buffers_destroy(&buffers);
                return rejected?0:1;
            }
            if(status==FG_OK&&(work_sends-before!=(c==8u?0u:1u)||!transport_ready(&transport))){
                fg_error_set(&err,FG_ERR_MISMATCH,"coordinator did not use one broadcast");status=FG_ERR_MISMATCH;
            }
            if(status==FG_OK)status=fg_prefill_results_validate_route(manifest,layer,c*128u,0u,tokens,ids,results,count,&err);
            for(uint32_t t=0;status==FG_OK&&t<tokens;t++)for(uint32_t j=0;j<FG_HIDDEN_SIZE;j++){
                float got=0.0f,expected=0.0f;
                for(uint32_t r=0;r<count;r++)got+=results[r].outputs[t*FG_HIDDEN_SIZE+j];
                for(uint32_t slot=0;slot<FG_TOP_K;slot++)expected=fmaf(gates[t*FG_TOP_K+slot],expert_value(ids[t*FG_TOP_K+slot],t,j),expected);
                if(fabsf(got-expected)>2e-6f*fmaxf(1.0f,fabsf(expected))){fg_error_set(&err,FG_ERR_MISMATCH,"tree sum mismatch");status=FG_ERR_MISMATCH;break;}
            }
        }
        if(status==FG_OK)for(uint32_t peer=1u;peer<FG_RANK_COUNT;peer++)
            status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_CONTROL,9u,0u,0u,NULL,0,&err);
    }else{
        while(status==FG_OK){
            uint32_t peer=0,bytes=0;fg_frame_header header;fg_fabric_class cls;
            status=fg_fabric_wait_ready(fabric,3u,&peer,&cls,&err);
            if(status==FG_OK)status=fg_fabric_recv(fabric,peer,cls,&header,buffers.receive,buffers.receive_capacity,&bytes,&err);
            if(status!=FG_OK)break;
            if(cls==FG_FABRIC_CONTROL&&fg_frame_type(&header)==FG_MSG_CONTROL)break;
            if(cls!=FG_FABRIC_BULK||fg_frame_type(&header)!=FG_MSG_PREFILL_WORK){status=FG_ERR_FORMAT;break;}
            status=handle_prefill_expert_work(fabric,NULL,manifest,rank,9u,peer,&header,buffers.receive,bytes,&buffers,&err);
        }
    }
    fg_fabric_close(fabric);prefill_worker_buffers_destroy(&buffers);
    if(status!=FG_OK&&!scenario){fprintf(stderr,"collective rank %u: %s\n",rank,err.message);return 1;}
    return 0;
}
int main(void){
    for(scenario=0;scenario<5u;scenario++){
        fg_manifest *manifest=malloc(sizeof(*manifest));if(!manifest)return 1;
        fg_manifest_init(manifest);uint32_t base=16000u+(uint32_t)(getpid()%300u)*32u+scenario*16u;
        for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)snprintf(manifest->ranks[rank].endpoint,sizeof(manifest->ranks[rank].endpoint),"127.0.0.1:%u",base+rank*2u);
        pid_t children[FG_RANK_COUNT];
        for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
            children[rank]=fork();if(children[rank]<0)return 1;
            if(!children[rank])_exit(run_rank(manifest,rank));
        }
        uint32_t passed=0,skipped=0;
        for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
            int status;if(waitpid(children[rank],&status,0)<0)return 1;
            if(WIFEXITED(status)&&WEXITSTATUS(status)==0)passed++;
            else if(WIFEXITED(status)&&WEXITSTATUS(status)==77)skipped++;
        }
        free(manifest);
        if(skipped==FG_RANK_COUNT)return 77;
        if(passed!=FG_RANK_COUNT){fprintf(stderr,"collective scenario %u: %u/8 passed\n",scenario,passed);return 1;}
    }
    puts("Eight-process prefill tree: broadcast, reduction, sparse participants, leaf failure, missing coverage, stale reply and streamed failure drain PASS");return 0;
}
