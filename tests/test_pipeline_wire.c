/* Run the real prefill coroutine and output exchange over socket pairs. GPU
 * operators are identity copies; wire framing, slot routing and scheduling
 * remain production code. No model weights, Vulkan device or io_uring needed. */
#define fg_model_vk wire_model_vk
#define fg_model_tensor wire_model_tensor
#define fg_vk_tensor_read wire_tensor_read
#define fg_vk_tensor_write wire_tensor_write
#define fg_vk_embedding_q8_0_batch wire_embedding
#define fg_vk_batch_active wire_batch_active
#define fg_owner_prefill_input_slot wire_input
#define fg_owner_prefill_layer_begin wire_layer_begin
#define fg_owner_prefill_layer_finish wire_layer_finish
#define fg_owner_qsa_set_tokens wire_qsa_tokens
#define fg_ngram_store_lookup_prefill wire_ngram
#define fg_uring_send_all wire_send_all
#define fg_uring_recv_all wire_recv_all
#include "../src/runtime.c"
#include "../src/fabric.c"

struct fg_vk_tensor {uint8_t *data;uint64_t size;};
static fg_vk_tensor inputs[2],weights;
static uint32_t selected;
static bool inject_fault;
static fg_vk_tensor tensor_new(uint64_t size){
    return (fg_vk_tensor){.data=calloc(1,size),.size=size};
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
fg_vk_context *wire_model_vk(fg_model *model){return NULL;}
fg_vk_tensor *wire_model_tensor(fg_model *model,const char *name){return &weights;}
bool wire_batch_active(const fg_vk_context *context){return false;}
fg_status wire_tensor_read(const fg_vk_tensor *t,uint64_t offset,void *data,
    uint64_t bytes,fg_error *err){
    if(!t||offset>t->size||bytes>t->size-offset)return FG_ERR_LIMIT;
    memcpy(data,t->data+offset,(size_t)bytes);return FG_OK;
}
fg_status wire_tensor_write(fg_vk_tensor *t,uint64_t offset,const void *data,
    uint64_t bytes,fg_error *err){
    if(!t||offset>t->size||bytes>t->size-offset)return FG_ERR_LIMIT;
    memcpy(t->data+offset,data,(size_t)bytes);return FG_OK;
}
fg_vk_tensor *wire_input(fg_owner_executor *owner,uint32_t slot){return &inputs[selected];}
fg_status wire_embedding(fg_vk_context *context,fg_vk_tensor *output,
    const fg_vk_tensor *weight,const fg_vk_tensor *ids,uint32_t count,
    uint32_t width,uint32_t rows,uint32_t copies,fg_error *err){
    float *values=(float *)output->data;
    for(uint64_t i=0;i<(uint64_t)count*FG_HYPER_WIDTH;i++)
        values[i]=(float)(selected+1u);
    return FG_OK;
}
fg_status wire_ngram(fg_ngram_store *store,const int32_t *history,size_t size,
    uint32_t first,uint32_t count,fg_vk_tensor **embedding,fg_error *err){
    *embedding=NULL;return FG_OK;
}
fg_status wire_layer_begin(fg_owner_executor *owner,uint32_t slot,uint32_t layer,
    uint32_t first,const uint32_t *positions,uint16_t count,const fg_vk_tensor *input,
    const fg_vk_tensor *ngram,fg_owner_prefill_fire_fn fire,void *fire_context,
    fg_owner_qsa_prefill_dispatch_fn dispatch,void *dispatch_context,fg_error *err){
    return FG_OK;
}
fg_status wire_layer_finish(fg_owner_executor *owner,uint32_t slot,
    fg_owner_prefill_collect_fn collect,void *context,fg_vk_tensor **output,fg_error *err){
    *output=&inputs[selected];return FG_OK;
}
void wire_qsa_tokens(fg_owner_executor *owner,uint32_t count){}
fg_status wire_send_all(fg_uring *ring,uint32_t fd,const void *data,uint32_t size,
    fg_error *err){
    if(sync_all((int)fd,(void *)data,size,true))return FG_OK;
    fg_error_set(err,FG_ERR_IO,"synthetic send failed");return FG_ERR_IO;
}
fg_status wire_recv_all(fg_uring *ring,uint32_t fd,void *data,uint32_t size,fg_error *err){
    if(sync_all((int)fd,data,size,false))return FG_OK;
    fg_error_set(err,FG_ERR_IO,"synthetic receive failed");return FG_ERR_IO;
}
#pragma GCC diagnostic pop

typedef struct packet {
    struct packet *next;
    fg_frame_header header;
    uint8_t *data;
    uint32_t size;
} packet;
typedef struct synthetic_peer {
    int fd;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    packet *head,*tail;
    bool stop;
} synthetic_peer;
typedef struct synthetic_input {
    int fd;
    uint32_t rank;
    synthetic_peer *destination;
    pthread_t thread;
} synthetic_input;

static void *send_packets(void *argument){
    synthetic_peer *peer=argument;
    for(;;){
        pthread_mutex_lock(&peer->mutex);
        while(!peer->head&&!peer->stop)pthread_cond_wait(&peer->cond,&peer->mutex);
        packet *p=peer->head;
        if(!p){pthread_mutex_unlock(&peer->mutex);break;}
        peer->head=p->next;if(!peer->head)peer->tail=NULL;
        pthread_mutex_unlock(&peer->mutex);
        bool sent=sync_all(peer->fd,&p->header,sizeof(p->header),true)&&
            sync_all(peer->fd,p->data,p->size,true);
        free(p->data);free(p);
        if(!sent)break;
    }
    return NULL;
}

static void *receive_packets(void *argument){
    synthetic_input *input=argument;
    const uint32_t capacity=FG_PREFILL_LAYER_WORK_MAX_BYTES;
    uint8_t *wire=malloc(capacity);
    float *hyper=malloc((size_t)128u*FG_HYPER_WIDTH*4u);
    float *ngram=malloc((size_t)128u*FG_NGRAM_EMBED_VALUES*4u);
    uint32_t positions[128u*3u];
    fg_frame_header header;
    while(sync_all(input->fd,&header,sizeof(header),false)){
        uint32_t size=ntohl(header.bytes_be);
        if(size>capacity||!sync_all(input->fd,wire,size,false))abort();
        packet *p=calloc(1,sizeof(*p));
        p->data=malloc(capacity);
        fg_error error={0};fg_status status;
        fg_message_type type;
        uint32_t sequence=fg_frame_sequence(&header);
        if(input->rank==4u){
            fg_output_work work={0};
            status=fg_output_work_decode(&work,wire,size,&error);
            if(status!=FG_OK||work.hyper[0]!=(float)(fg_frame_session_slot(&header)+1u))
                abort();
            fg_output_result result={.source_rank=4u,.destination_rank=0u,
                .token_index=work.token_index,.token=123u+work.session_slot,.logit=1.0f};
            p->size=FG_OUTPUT_RESULT_BYTES;
            status=fg_output_result_encode(p->data,&result,&error);
            if(inject_fault&&fg_frame_session_slot(&header)==0u){
                memset(p->data+p->size,0,4u);p->size+=4u;
            }
            type=FG_MSG_OUTPUT_RESULT;
        }else{
            fg_prefill_layer_work work={0};
            status=fg_prefill_layer_work_decode(&work,FG_PROTOCOL_VERSION,positions,
                128u*3u,hyper,128u*FG_HYPER_WIDTH,ngram,128u*FG_NGRAM_EMBED_VALUES,
                wire,size,&error);
            if(status!=FG_OK)abort();
            if(input->rank==1u){
                work.source_rank=1u;work.destination_rank=0u;work.layer=6u;
                work.flags=0u;work.ngram_embeddings=NULL;
                status=fg_prefill_layer_work_encode(p->data,capacity,&p->size,
                    FG_PROTOCOL_VERSION,&work,&error);
                type=FG_MSG_PREFILL_LAYER_WORK;
                sequence=work.first_token*FG_LAYER_COUNT+6u;
            }else{
                fg_prefill_layer_result result={.layer=47u,.source_rank=4u,
                    .destination_rank=0u,.first_token=work.first_token,
                    .token_count=work.token_count,.hyper=hyper};
                status=fg_prefill_layer_result_encode(p->data,capacity,&p->size,&result,&error);
                type=FG_MSG_PREFILL_LAYER_RESULT;
                sequence=work.first_token*FG_LAYER_COUNT+47u;
            }
        }
        if(status!=FG_OK||fg_frame_encode_version(&p->header,FG_PROTOCOL_VERSION,type,
            fg_frame_request_id(&header),sequence,ntohl(header.flags_be),
            p->data,p->size,&error)!=FG_OK)abort();
        synthetic_peer *peer=input->destination;
        pthread_mutex_lock(&peer->mutex);
        if(peer->tail)peer->tail->next=p;else peer->head=p;
        peer->tail=p;pthread_cond_signal(&peer->cond);
        pthread_mutex_unlock(&peer->mutex);
    }
    free(ngram);free(hyper);free(wire);return NULL;
}

typedef struct prefill_probe {
    fg_coordinator coordinator;
    uint32_t count,prefilled,next;
    fg_status status;
    fg_error error;
} prefill_probe;
static void prefill_job(void *argument){
    prefill_probe *probe=argument;
    uint32_t *tokens=calloc(probe->count,sizeof(*tokens));
    fg_vk_tensor *output=NULL;
    bool profiled=false;
    probe->status=coordinator_prefill_pipeline_ring(&probe->coordinator,
        (const int32_t *)tokens,probe->count,tokens,0u,probe->count,NULL,0u,
        &profiled,NULL,NULL,&output,&probe->prefilled,&probe->error);
    if(probe->status==FG_OK){
        fg_pipeline_set_phase(FG_PIPELINE_DECODE);
        probe->status=coordinator_output(&probe->coordinator,probe->count-1u,
            output,&probe->next,NULL,&probe->error);
    }
    free(tokens);
}

int main(int argc,char **argv){
    inject_fault=argc==2&&!strcmp(argv[1],"--fault-output");
    alarm(30);
    fg_error error={0};
    fg_manifest *manifest=calloc(1,sizeof(*manifest));
    fg_manifest_init(manifest);
    manifest->protocol_version=FG_PROTOCOL_VERSION;
    manifest->prefill_microbatch=128u;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
        manifest->layer_owner[layer]=layer<6u?1u:layer<12u?0u:layer<18u?2u:4u;
    fg_fabric fabric={.rank=0u,.protocol_version=FG_PROTOCOL_VERSION};
    int peers[FG_RANK_COUNT][2];
    for(uint32_t rank=1;rank<FG_RANK_COUNT;rank++)for(uint32_t cls=0;cls<2;cls++){
        int pair[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,pair))abort();
        peers[rank][cls]=pair[1];
        fabric.peer[rank][cls].fd=pair[0];
        fabric.peer[rank][cls].fixed_slot=(uint32_t)pair[0];
        pthread_mutex_init(&fabric.peer[rank][cls].send_mutex,NULL);
    }
    synthetic_peer outputs[2]={{.fd=peers[1][FG_FABRIC_BULK]},
        {.fd=peers[4][FG_FABRIC_BULK]}};
    for(uint32_t i=0;i<2;i++){
        pthread_mutex_init(&outputs[i].mutex,NULL);pthread_cond_init(&outputs[i].cond,NULL);
        pthread_create(&outputs[i].thread,NULL,send_packets,&outputs[i]);
    }
    synthetic_input sources[3]={
        {.rank=1,.fd=peers[1][FG_FABRIC_BULK],.destination=&outputs[0]},
        {.rank=2,.fd=peers[2][FG_FABRIC_BULK],.destination=&outputs[1]},
        {.rank=4,.fd=peers[4][FG_FABRIC_CONTROL],.destination=&outputs[1]}};
    for(uint32_t i=0;i<3;i++)pthread_create(&sources[i].thread,NULL,receive_packets,&sources[i]);
    fg_pipeline pipeline;
    if(fg_pipeline_init(&pipeline,&fabric,&error)!=FG_OK)abort();
    prefill_probe probes[2]={{.count=14u*128u+28u},
        {.count=inject_fault?64u*128u:20u*128u}};
    for(uint32_t i=0;i<2;i++){
        inputs[i]=tensor_new(128u*FG_HYPER_WIDTH*4u);
        probes[i].coordinator=(fg_coordinator){.manifest=manifest,.fabric=&fabric,
            .session_id=100u+i,.output_session=i,.multi_session=true,
            .ring_prefill=true,.ring_decode=true};
        for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++){
            probes[i].coordinator.prefill_layer[f].token_tensor=
                calloc(1,sizeof(fg_vk_tensor));
            *probes[i].coordinator.prefill_layer[f].token_tensor=tensor_new(128u*4u);
            if(f<2u){
                probes[i].coordinator.ring_output[f]=calloc(1,sizeof(fg_vk_tensor));
                *probes[i].coordinator.ring_output[f]=tensor_new(128u*FG_HYPER_WIDTH*4u);
            }
        }
        if(fg_pipeline_start(&pipeline,i,prefill_job,&probes[i],&error)!=FG_OK)abort();
    }
    while(pipeline.slots[0].active||pipeline.slots[1].active){
        uint32_t slot=fg_pipeline_pick(&pipeline);
        if(slot>=2u){usleep(100);continue;}
        selected=slot;
        fg_pipeline_turn(&pipeline,slot);
        if(probes[slot].status!=FG_OK){
            if(!inject_fault){
                fprintf(stderr,"WIRE_FAIL slot=%u prefilled=%u: %s\n",slot,
                    probes[slot].prefilled,probes[slot].error.message);
                return 1;
            }
            if(!pipeline.failed){
                if(slot!=0u||probes[slot].status!=FG_ERR_FORMAT)abort();
                fg_runtime contaminated={.coordinator=probes[slot].coordinator};
                if(fg_runtime_reset_failure(&contaminated,&error)!=FG_ERR_UNAVAILABLE||
                   !strstr(error.message,"transport is not reusable"))abort();
                puts("Offline old fatal boundary reproduced: output fault with other prefill in flight");
                fg_pipeline_abort(&pipeline,&probes[slot].error);
            }
        }
    }
    for(uint32_t i=0;i<2;i++){
        if(!inject_fault&&(probes[i].prefilled!=probes[i].count||
                           probes[i].next!=123u+i))abort();
        if(inject_fault&&probes[i].status==FG_OK)abort();
    }
    puts(inject_fault?"Contaminated ring: both coroutines unwound without another send/reset: PASS":
        "Real prefill wire: unequal concurrent quanta, short tail -> output, long survivor: PASS");
    /* The process owns these synthetic descriptors and threads; shutdown
     * explicitly rather than asking the fake fabric to destroy an io_uring. */
    fg_pipeline_destroy(&pipeline);
    for(uint32_t rank=1;rank<FG_RANK_COUNT;rank++)for(uint32_t cls=0;cls<2;cls++)
        shutdown(fabric.peer[rank][cls].fd,SHUT_RDWR);
    for(uint32_t i=0;i<3;i++)pthread_join(sources[i].thread,NULL);
    for(uint32_t i=0;i<2;i++){
        pthread_mutex_lock(&outputs[i].mutex);outputs[i].stop=true;
        pthread_cond_signal(&outputs[i].cond);pthread_mutex_unlock(&outputs[i].mutex);
        pthread_join(outputs[i].thread,NULL);
    }
    return 0;
}
