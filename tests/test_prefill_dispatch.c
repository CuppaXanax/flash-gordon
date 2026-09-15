#include "fg_fabric.h"
#include "fg_expert.h"
#include "fg_owner.h"
#include "fg_qsa_replica.h"

static fg_status fake_page_records(const fg_owner_executor *,uint32_t,uint32_t,
    const uint8_t **,fg_error *);
static void fake_page_published(fg_owner_executor *,uint32_t,uint32_t);
static fg_status fake_replica_commit(fg_qsa_replica *,const fg_qsa_replica_item *,
    uint32_t,fg_error *);

static fg_status fake_send(fg_fabric *,uint32_t,fg_fabric_class,fg_message_type,
    uint64_t,uint32_t,uint32_t,const void *,uint32_t,fg_error *);
static fg_status fake_recv(fg_fabric *,uint32_t,fg_fabric_class,fg_frame_header *,
    void *,uint32_t,uint32_t *,fg_error *);
static fg_status fake_expert(fg_expert_executor *,const fg_prefill_work *,
    fg_prefill_result *,fg_prefill_result_pair *,uint32_t,float *,uint64_t,fg_error *);

/* Exercise the real coordinator dispatch and cleanup with deterministic peers. */
#define fg_fabric_send fake_send
#define fg_fabric_recv fake_recv
#define fg_expert_prefill fake_expert
#define fg_owner_qsa_page_records fake_page_records
#define fg_owner_qsa_page_published fake_page_published
#define fg_qsa_replica_commit fake_replica_commit
#include "../src/runtime.c"
#undef fg_fabric_send
#undef fg_fabric_recv
#undef fg_expert_prefill
#undef fg_owner_qsa_page_records
#undef fg_owner_qsa_page_published
#undef fg_qsa_replica_commit

static const fg_manifest *test_manifest;
static unsigned sends,receives,shared_calls,local_calls,fail_send,expected_sends;
static bool fail_shared,fail_local,stale_reply;
static uint8_t *replies[FG_RANK_COUNT];
static uint32_t reply_bytes[FG_RANK_COUNT];
static fg_frame_header headers[FG_RANK_COUNT];
static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);failures++;}}while(0)

static uint32_t released_pages;
static bool fail_commit;
static fg_status fake_page_records(const fg_owner_executor *owner,uint32_t layer,
    uint32_t block,const uint8_t **records,fg_error *err){
    (void)owner;(void)layer;(void)block;(void)err;
    static uint8_t page[FG_QSA_PAGE_RECORD_BYTES];*records=page;return FG_OK;
}
static void fake_page_published(fg_owner_executor *owner,uint32_t layer,uint32_t block){
    (void)owner;(void)layer;(void)block;released_pages++;
}
static fg_status fake_replica_commit(fg_qsa_replica *replica,
    const fg_qsa_replica_item *items,uint32_t count,fg_error *err){
    CHECK(released_pages==0u);
    if(fail_commit){fg_error_set(err,FG_ERR_IO,"injected queue commit failure");return FG_ERR_IO;}
    return fg_qsa_replica_commit(replica,items,count,err);
}
static fg_status accept_pages(void *opaque,uint32_t owner,uint64_t session,
    uint32_t batch,const void *payload,uint32_t bytes,fg_error *err){
    (void)opaque;(void)owner;(void)session;(void)batch;(void)payload;(void)bytes;(void)err;
    return FG_OK;
}
static void test_publication(fg_manifest *manifest){
    for(uint32_t scenario=0;scenario<2u;scenario++){
        fg_coordinator c={.manifest=manifest,.session_id=9u};fg_error err={0};
        for(uint32_t i=0;i<2u;i++){
            c.qsa_pages.append_pages[i]=calloc(FG_QSA_PAGE_APPEND_MAX_PAGES,sizeof(fg_qsa_page));
            CHECK(c.qsa_pages.append_pages[i]!=NULL);
        }
        CHECK(fg_qsa_replica_create(&c.qsa_pages.replica,accept_pages,NULL,
                                    FG_QSA_PAGE_APPEND_MAX_BYTES,&err)==FG_OK);
        released_pages=0u;fail_commit=scenario==0u;
        fg_status status=coordinator_publish_qsa_pages(&c,0u,4u,&err);
        CHECK(status==(fail_commit?FG_ERR_IO:FG_OK));
        CHECK(released_pages==(fail_commit?0u:12u));
        CHECK(c.qsa_pages.append_sequence[0]==(fail_commit?0u:1u));
        CHECK(c.qsa_pages.append_sequence[1]==(fail_commit?0u:1u));
        CHECK(fg_qsa_replica_drain(c.qsa_pages.replica,&err)==FG_OK);
        qsa_page_transport_destroy(&c.qsa_pages);
    }
}

static fg_status fill_result(const fg_prefill_work *work,fg_prefill_result *result,
    fg_prefill_result_pair *pairs,float *outputs){
    for(uint32_t i=0;i<work->pair_count;i++){
        pairs[i]=(fg_prefill_result_pair){work->pairs[i].token_slot,work->pairs[i].routing_slot};
        for(uint32_t j=0;j<FG_HIDDEN_SIZE;j++)outputs[i*FG_HIDDEN_SIZE+j]=(float)work->destination_rank;
    }
    for(uint32_t t=0;t<work->token_count;t++)for(uint32_t j=0;j<FG_HIDDEN_SIZE;j++)outputs[t*FG_HIDDEN_SIZE+j]=0.0f;
    *result=(fg_prefill_result){.layer=work->layer,.source_rank=work->destination_rank,.contributor_mask=(uint8_t)(1u<<work->destination_rank),
        .destination_rank=work->source_rank,.first_position=work->first_position,
        .token_count=work->token_count,.pair_count=work->pair_count,.pairs=pairs,.outputs=outputs};
    return FG_OK;
}

static fg_status fake_send(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,
    fg_message_type type,uint64_t request,uint32_t sequence,uint32_t flags,
    const void *payload,uint32_t bytes,fg_error *err){
    (void)fabric;(void)flags;
    CHECK(cls==FG_FABRIC_BULK&&type==FG_MSG_PREFILL_WORK);
    CHECK(shared_calls==0u&&local_calls==0u);
    sends++;
    if(sends==fail_send){fg_error_set(err,FG_ERR_IO,"send failure");return FG_ERR_IO;}
    uint8_t activations[2u*FG_Q8K_ACTIVATION_BYTES];fg_prefill_pair routes[20];
    fg_prefill_work work={0};
    fg_status status=fg_prefill_work_decode(&work,activations,sizeof(activations),
        routes,20u,payload,bytes,err);
    fg_prefill_result_pair pairs[20];float outputs[20u*FG_HIDDEN_SIZE];
    fg_prefill_result result={0};
    if(status==FG_OK){
        uint8_t ranks[FG_GROUP_SIZE];uint32_t count=0;
        status=prefill_tree_ranks(test_manifest,&work,ranks,&count,err);
        fg_prefill_pair remote[20];uint32_t used=0;
        for(uint32_t i=0;i<work.pair_count;i++)if(test_manifest->expert_rank[work.layer][work.pairs[i].expert_id]!=work.source_rank)remote[used++]=work.pairs[i];
        fg_prefill_work reduced=work;reduced.pairs=remote;reduced.pair_count=(uint16_t)used;
        if(status==FG_OK)status=fill_result(&reduced,&result,pairs,outputs);
        result.contributor_mask=prefill_subtree_mask(ranks,count,0u);
    }
    uint32_t capacity=FG_PREFILL_RESULT_HEADER_BYTES+20u*FG_PREFILL_RESULT_PAIR_BYTES+2u*FG_HIDDEN_SIZE*4u;
    replies[peer]=malloc(capacity);
    if(!replies[peer])return FG_ERR_OOM;
    if(status==FG_OK)status=fg_prefill_result_encode(replies[peer],capacity,
        &reply_bytes[peer],&result,err);
    if(status==FG_OK)status=fg_frame_encode(&headers[peer],FG_MSG_PREFILL_RESULT,
        request+(stale_reply?1u:0u),sequence,0u,replies[peer],reply_bytes[peer],err);
    return status;
}

static fg_status fake_recv(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,
    fg_frame_header *header,void *payload,uint32_t capacity,uint32_t *bytes,fg_error *err){
    (void)fabric;(void)err;
    CHECK(cls==FG_FABRIC_BULK&&replies[peer]&&reply_bytes[peer]<=capacity);
    receives++;*header=headers[peer];*bytes=reply_bytes[peer];
    memcpy(payload,replies[peer],*bytes);return FG_OK;
}

static fg_status fake_expert(fg_expert_executor *executor,const fg_prefill_work *work,
    fg_prefill_result *result,fg_prefill_result_pair *pairs,uint32_t capacity,
    float *outputs,uint64_t output_capacity,fg_error *err){
    (void)executor;
    CHECK(sends==expected_sends&&shared_calls==1u&&receives==0u);
    CHECK(capacity>=work->pair_count&&output_capacity>=work->pair_count*FG_HIDDEN_SIZE);
    local_calls++;
    if(fail_local){fg_error_set(err,FG_ERR_IO,"local failure");return FG_ERR_IO;}
    return fill_result(work,result,pairs,outputs);
}

static fg_status shared(void *context,fg_error *err){
    (void)context;
    CHECK(sends==expected_sends&&local_calls==0u&&receives==0u);shared_calls++;
    if(fail_shared){fg_error_set(err,FG_ERR_IO,"shared failure");return FG_ERR_IO;}
    return FG_OK;
}

int main(void){
    fg_manifest *manifest=malloc(sizeof(*manifest));
    if(!manifest)return 1;
    fg_manifest_init(manifest);test_manifest=manifest;
    for(uint32_t scenario=0;scenario<6u;scenario++){
        sends=receives=shared_calls=local_calls=0u;
        fail_shared=scenario==1u;fail_local=scenario==2u;
        fail_send=scenario==3u?1u:0u;stale_reply=scenario==4u;
        uint32_t layer=scenario==5u?2u:0u;
        expected_sends=1u;
        prefill_worker_buffers buffers;fg_error err={0};
        CHECK(prefill_worker_buffers_create(&buffers,2u,true,&err)==FG_OK);
        atomic_uint transport;atomic_init(&transport,FG_TRANSPORT_READY);
        prefill_dispatch_context context={.manifest=manifest,.self=0u,.request_id=9u,
            .sequence=48u,.buffers=&buffers,.transport_state=&transport};
        uint16_t ids[20];float gates[20];uint8_t activation[2u*FG_Q8K_ACTIVATION_BYTES]={0};
        for(uint32_t i=0;i<20u;i++){ids[i]=(uint16_t)(i%FG_TOP_K);gates[i]=0.1f;}
        fg_prefill_result results[FG_GROUP_SIZE]={0};uint32_t count=0u;
        fg_status status=dispatch_prefill_experts(&context,layer,1u,2u,ids,gates,
            activation,shared,NULL,results,&count,&err);
        CHECK(receives==(scenario==3u?0u:expected_sends));
        CHECK(transport_ready(&transport)==(scenario<3u||scenario==5u));
        if(scenario==0u||scenario==5u){
            CHECK(status==FG_OK&&count==(scenario==0u?2u:1u)&&local_calls==(scenario==0u?1u:0u));
            CHECK(fg_prefill_results_validate_route(manifest,layer,1u,0u,2u,ids,
                results,count,&err)==FG_OK);
        }else CHECK(status!=FG_OK);
        if(scenario==1u)CHECK(local_calls==0u&&!strcmp(err.message,"shared failure"));
        if(scenario==2u)CHECK(!strcmp(err.message,"local failure"));
        if(scenario==3u)CHECK(shared_calls==0u&&local_calls==0u);
        prefill_worker_buffers_destroy(&buffers);
        for(uint32_t peer=0;peer<FG_RANK_COUNT;peer++){free(replies[peer]);replies[peer]=NULL;}
    }
    test_publication(manifest);
    free(manifest);
    if(failures)return 1;
    puts("EP prefill dispatch, failure drain and QSA publication ownership: PASS");return 0;
}
