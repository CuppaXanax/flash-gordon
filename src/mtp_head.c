#include "fg_mtp_head.h"
#include "fg_mtp.h"
#include "fg_gguf.h"
#include "fg_owner.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"
#include "fg_sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MTP_LAYER (FG_LAYER_COUNT-1u)
_Static_assert(FG_VK_MTP_VOCAB_STATIC_SLOT>=FG_VK_MTP_STATIC_SLOT+2u*FG_OWNER_SESSION_MAX,
    "local vocabulary replay overlaps native core/warm sessions");
#define MTP_BINDINGS 40u

typedef struct mtp_shared_weights {
    fg_manifest *manifest;
    fg_model *model;
    fg_vk_tensor *weights[MTP_BINDINGS];
    uint32_t count,references,sessions;
} mtp_shared_weights;

struct fg_mtp_head {
    fg_vk_context *vk;
    fg_manifest *manifest;
    fg_model *model;
    fg_owner_executor *owner;
    fg_expert_executor *experts;
    fg_vk_tensor *weights[MTP_BINDINGS];
    uint32_t count,frontier,capacity,context_tokens,window,first_cell;
    uint64_t weight_bytes,residency_cap;
    mtp_shared_weights *shared;
    uint32_t session_slot,core_slot,warm_slot;
    fg_mtp_head_trace_fn trace;
    void *trace_context;
    bool replay_claimed,warm_claimed;
    fg_vk_tensor *replay_input,*replay_embedding,*replay_hc;
    fg_vk_tensor *en,*ep,*hn,*hp,*stream_in[4],*stream_out[4];
    fg_vk_tensor *raw_q,*raw_k,*raw_v,*q,*gate,*key,*attention,*attn_out;
    fg_vk_tensor *keys,*values,*position,*attention_partials;
    fg_vk_tensor *norm,*down,*active,*up,*hidden,*output_partials;
};


struct fg_mtp_vocab {
    fg_vk_context *vk;
    fg_vk_tensor *weights,*tokens,*hidden,*scores[2],*ids[2];
    uint32_t rows,final_bank;
    bool claimed;
};

typedef struct mtp_alias {
    const char *source,*alias;
    uint32_t input,rows,experts;
    bool f32,norm;
    uint32_t half;
} mtp_alias;

static const mtp_alias aliases[]={
    {"fc_embedding.weight","mtp.fc_embedding",2560,2560,1,false,false,0},
    {"fc_hidden.weight","mtp.fc_hidden",2560,2560,1,false,false,0},
    {"pre_fc_norm_embedding.weight","mtp.embedding_norm",2560,1,1,true,true,0},
    {"pre_fc_norm_hidden.weight","mtp.hidden_norm",10240,1,1,true,true,0},
    {"hyper_connection_mixer.hc_norm.weight","mtp.output_norm",10240,1,1,true,true,0},
    {"hyper_connection_mixer.input_mix_weight_down.weight","mtp.output_down",10240,320,1,false,false,0},
    {"hyper_connection_mixer.input_mix_weight_up.weight","mtp.output_up",320,10240,1,false,false,0},
    {"layers.0.attn_hyper_connection.hc_norm.weight","hc_attn_norm.weight",10240,1,1,true,true,0},
    {"layers.0.attn_hyper_connection.input_mix_weight_down.weight","hc_attn_down.weight",10240,320,1,false,false,0},
    {"layers.0.attn_hyper_connection.input_mix_weight_up.weight","hc_attn_up.weight",320,10240,1,false,false,0},
    {"layers.0.attn_hyper_connection.block_inject_weight.weight","hc_attn_inject.weight",10240,4,1,true,false,0},
    {"layers.0.mlp_hyper_connection.hc_norm.weight","hc_ffn_norm.weight",10240,1,1,true,true,0},
    {"layers.0.mlp_hyper_connection.input_mix_weight_down.weight","hc_ffn_down.weight",10240,320,1,false,false,0},
    {"layers.0.mlp_hyper_connection.input_mix_weight_up.weight","hc_ffn_up.weight",320,10240,1,false,false,0},
    {"layers.0.mlp_hyper_connection.block_inject_weight.weight","hc_ffn_inject.weight",10240,4,1,true,false,0},
    {"layers.0.mlp.gate.weight","ffn_gate_inp.weight",2560,512,1,true,false,0},
    {"layers.0.mlp.shared_expert_gate.weight","ffn_gate_inp_shexp.weight",2560,1,1,true,false,0},
    {"layers.0.mlp.shared_expert.gate_proj.weight","ffn_gate_shexp.weight",2560,640,1,false,false,0},
    {"layers.0.mlp.shared_expert.up_proj.weight","ffn_up_shexp.weight",2560,640,1,false,false,0},
    {"layers.0.mlp.shared_expert.down_proj.weight","ffn_down_shexp.weight",640,2560,1,false,false,0},
    {"layers.0.self_attn.q_proj.weight","attn_q.weight",2560,12288,1,false,false,0},
    {"layers.0.self_attn.k_proj.weight","attn_k.weight",2560,512,1,false,false,0},
    {"layers.0.self_attn.v_proj.weight","attn_v.weight",2560,512,1,false,false,0},
    {"layers.0.self_attn.o_proj.weight","attn_output.weight",6144,2560,1,false,false,0},
    {"layers.0.self_attn.q_norm.weight","attn_q_norm.weight",256,1,1,true,true,0},
    {"layers.0.self_attn.k_norm.weight","attn_k_norm.weight",256,1,1,true,true,0},
    {"layers.0.mlp.experts.gate_up_proj","ffn_gate_exps",2560,640,512,false,false,0},
    {"layers.0.mlp.experts.gate_up_proj","ffn_up_exps",2560,640,512,false,false,1},
    {"layers.0.mlp.experts.down_proj","ffn_down_exps",640,2560,512,false,false,0}
};

static fg_vk_tensor *binding(fg_mtp_head *h,const char *name){
    return fg_model_tensor(h->model,name);
}
void fg_mtp_head_set_trace(fg_mtp_head *h,fg_mtp_head_trace_fn trace,void *context){
    if(h){h->trace=trace;h->trace_context=context;}
}
static fg_status trace(fg_mtp_head *h,const char *stage,const fg_vk_tensor *values,
    uint32_t count,fg_error *err){
    return h->trace?h->trace(h->trace_context,stage,values,count,err):FG_OK;
}
static fg_status trace_host(fg_mtp_head *h,const char *stage,const float *values,
    uint32_t count,fg_error *err){
    if(!h->trace)return FG_OK;
    fg_vk_tensor *tensor=NULL;
    fg_status status=fg_vk_tensor_create(h->vk,(uint64_t)count*4u,&tensor,err);
    if(status==FG_OK)status=fg_vk_tensor_write(tensor,0,values,(uint64_t)count*4u,err);
    if(status==FG_OK)status=trace(h,stage,tensor,count,err);
    fg_vk_tensor_destroy(tensor);return status;
}
static fg_status read_exact(int fd,uint64_t offset,void *out,uint64_t bytes,fg_error *err){
    uint8_t *p=out;
    while(bytes){
        size_t amount=bytes>(1u<<20)?(1u<<20):(size_t)bytes;
        ssize_t got=pread(fd,p,amount,(off_t)offset);
        if(got<0&&errno==EINTR)continue;
        if(got<=0){fg_error_set(err,FG_ERR_IO,"read native MTP tensor: %s",
                               got<0?strerror(errno):"short read");return FG_ERR_IO;}
        offset+=(uint64_t)got;p+=got;bytes-=(uint64_t)got;
    }
    return FG_OK;
}

void fg_mtp_vocab_close(fg_mtp_vocab *v){
        if(!v)return;
        if(v->claimed){fg_error ignored={0};fg_vk_static_release(v->vk,FG_VK_MTP_VOCAB_STATIC_SLOT,&ignored);}
        fg_vk_tensor_destroy(v->weights);fg_vk_tensor_destroy(v->tokens);fg_vk_tensor_destroy(v->hidden);
        for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(v->scores[i]);fg_vk_tensor_destroy(v->ids[i]);}
        free(v);
    }
    fg_status fg_mtp_vocab_check(fg_mtp_vocab *v,const fg_tensor_record *r,fg_error *err){
        if(!v||!r||v->rows!=FG_Q38_VOCAB_SIZE||r->ggml_type!=8u||
           r->layout!=FG_TENSOR_LAYOUT_Q8_0_COOKED||r->shape[0]!=FG_HIDDEN_SIZE||
           r->shape[1]!=FG_Q38_VOCAB_SIZE||
           r->bytes!=fg_q8_0_cooked_matrix_bytes(FG_HIDDEN_SIZE,v->rows)){
            fg_error_set(err,FG_ERR_FORMAT,"local MTP head requires the sealed full cooked-Q8 vocabulary");
            return FG_ERR_FORMAT;
        }
        fg_sha256 hash;uint8_t digest[32];fg_sha256_init(&hash);
        fg_sha256_update(&hash,fg_vk_tensor_map(v->weights),(size_t)r->bytes);
        fg_sha256_final(&hash,digest);
        if(memcmp(digest,r->sha256,sizeof(digest))){
            fg_error_set(err,FG_ERR_MISMATCH,"local MTP vocabulary does not match output.weight");
            return FG_ERR_MISMATCH;
        }
        return FG_OK;
    }
    fg_status fg_mtp_vocab_open(fg_mtp_vocab **out,fg_vk_context *vk,const char *path,
        const uint32_t *ids,uint32_t rows,fg_error *err){
        if(!out||!vk||!path||!rows||rows>FG_Q38_VOCAB_SIZE||(!ids&&rows!=FG_Q38_VOCAB_SIZE)){
            fg_error_set(err,FG_ERR_ARGUMENT,"invalid explicit MTP vocabulary");return FG_ERR_ARGUMENT;
        }
        *out=NULL;
        bool seen[FG_Q38_VOCAB_SIZE]={false};
        for(uint32_t i=0;i<rows;i++){
            uint32_t id=ids?ids[i]:i;
            if(id>=FG_Q38_VOCAB_SIZE||seen[id]){
                fg_error_set(err,FG_ERR_ARGUMENT,"invalid or duplicate MTP token ID");return FG_ERR_ARGUMENT;
            }
            seen[id]=true;
        }
        fg_mtp_vocab *v=calloc(1,sizeof(*v));
        if(!v){fg_error_set(err,FG_ERR_OOM,"MTP vocabulary");return FG_ERR_OOM;}
        v->vk=vk;v->rows=rows;
        uint64_t bytes=fg_q8_0_cooked_matrix_bytes(2560u,rows);
        uint32_t partials=(rows+7u)/8u;
        fg_status status=fg_vk_tensor_create(vk,bytes,&v->weights,err);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)rows*4u,&v->tokens,err);
        for(uint32_t i=0;status==FG_OK&&i<2u;i++){
            status=fg_vk_tensor_create(vk,(uint64_t)partials*4u,&v->scores[i],err);
            if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)partials*4u,&v->ids[i],err);
        }
        int fd=status==FG_OK?open(path,O_RDONLY|O_CLOEXEC):-1;
        if(status==FG_OK&&fd<0){status=FG_ERR_IO;fg_error_set(err,status,"open shared MTP vocabulary: %s",strerror(errno));}
        const uint32_t row_bytes=2560u/32u*34u;
        uint8_t raw[16u*2560u/32u*34u];
        uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(2560u);
        for(uint32_t row=0;status==FG_OK&&row<rows;row+=16u){
            uint32_t live=rows-row<16u?rows-row:16u;
            for(uint32_t i=0;status==FG_OK&&i<live;i++){
                uint32_t id=ids?ids[row+i]:row+i;
                status=read_exact(fd,(uint64_t)id*row_bytes,raw+i*row_bytes,row_bytes,err);
            }
            if(status==FG_OK&&!fg_cook_q8_0_rows(raw,(uint8_t *)fg_vk_tensor_map(v->weights)+
                (uint64_t)(row/16u)*tile_bytes,tile_bytes,2560u,live)){
                status=FG_ERR_FORMAT;fg_error_set(err,status,"cook MTP vocabulary tile");
            }
        }
        if(fd>=0)close(fd);
        if(status==FG_OK){
            uint32_t *map=fg_vk_tensor_map(v->tokens);
            for(uint32_t i=0;i<rows;i++)map[i]=ids?ids[i]:i;
            fg_vk_tensor_set_format(v->weights,FG_VK_TENSOR_FORMAT_Q8_0_COOKED);
            status=fg_vk_static_claim(vk,FG_VK_MTP_VOCAB_STATIC_SLOT,err);
            if(status==FG_OK)v->claimed=true;
        }
        if(status!=FG_OK){fg_mtp_vocab_close(v);return status;}
        *out=v;return FG_OK;
    }
    static fg_status record_vocab(fg_mtp_vocab *v,fg_error *err){
        fg_vk_device_write_visible(v->vk);
        fg_status status=fg_vk_mtp_vocab_q8(v->vk,v->scores[0],v->ids[0],v->weights,
            v->hidden,v->tokens,2560u,v->rows,err);
        uint32_t count=(v->rows+7u)/8u,bank=0;
        while(status==FG_OK&&count>1u){
            uint32_t next=0;
            status=fg_vk_argmax_reduce(v->vk,v->scores[bank^1u],v->ids[bank^1u],
                v->scores[bank],v->ids[bank],count,&next,err);
            count=next;bank^=1u;
        }
        v->final_bank=bank;return status;
    }
    fg_status fg_mtp_vocab_argmax(fg_mtp_vocab *v,const fg_vk_tensor *hidden,
        uint32_t *token,float *score,fg_error *err){
        if(!v||!hidden||!token||!score||fg_vk_batch_active(v->vk)){
            fg_error_set(err,FG_ERR_ARGUMENT,"invalid MTP vocabulary input");return FG_ERR_ARGUMENT;
        }
        fg_status status=FG_OK;
        if(fg_vk_tensor_bytes(hidden)<2560u*4u){
            fg_error_set(err,FG_ERR_ARGUMENT,"short MTP vocabulary hidden");return FG_ERR_ARGUMENT;
        }
        if(!v->hidden||fg_vk_tensor_const_map(v->hidden)!=fg_vk_tensor_const_map(hidden)){
            fg_vk_tensor *source=NULL;
            status=fg_vk_tensor_view((fg_vk_tensor *)hidden,0,2560u*4u,&source,err);
            if(status==FG_OK)status=fg_vk_static_reset(v->vk,FG_VK_MTP_VOCAB_STATIC_SLOT,err);
            if(status==FG_OK){fg_vk_tensor_destroy(v->hidden);v->hidden=source;source=NULL;}
            fg_vk_tensor_destroy(source);
        }
        if(status==FG_OK&&fg_vk_profile_active(v->vk)){
            status=fg_vk_begin(v->vk,err);
            if(status==FG_OK)status=record_vocab(v,err);
            if(status==FG_OK)status=fg_vk_end(v->vk,err);
        }else if(status==FG_OK){
            if(!fg_vk_static_recorded(v->vk,FG_VK_MTP_VOCAB_STATIC_SLOT)){
                status=fg_vk_static_begin(v->vk,FG_VK_MTP_VOCAB_STATIC_SLOT,err);
                if(status==FG_OK)status=record_vocab(v,err);
                if(status==FG_OK)status=fg_vk_static_end(v->vk,FG_VK_MTP_VOCAB_STATIC_SLOT,err);
            }
            if(status==FG_OK)status=fg_vk_static_submit(v->vk,FG_VK_MTP_VOCAB_STATIC_SLOT,err);
            if(status==FG_OK)status=fg_vk_static_wait(v->vk,FG_VK_MTP_VOCAB_STATIC_SLOT,err);
        }
        if(status==FG_OK)status=fg_vk_tensor_read(v->scores[v->final_bank],0,score,4u,err);
        if(status==FG_OK)status=fg_vk_tensor_read(v->ids[v->final_bank],0,token,4u,err);
        if(status==FG_OK&&(!isfinite(*score)||*token>=FG_Q38_VOCAB_SIZE)){
            status=FG_ERR_FORMAT;fg_error_set(err,status,"nonfinite MTP vocabulary reduction");
        }
        if(status!=FG_OK){
            fg_error ignored={0};if(fg_vk_batch_active(v->vk))fg_vk_abort(v->vk,&ignored);
            fg_vk_static_drain(v->vk,&ignored);fg_vk_static_reset(v->vk,FG_VK_MTP_VOCAB_STATIC_SLOT,&ignored);
        }
        return status;
    }
static const fg_gguf_tensor *find_source(const fg_gguf *g,const mtp_alias *a){
    char name[160];snprintf(name,sizeof(name),"mtp.%s",a->source);
    for(uint64_t i=0;i<g->tensor_count;i++)
        if(!strcmp(g->tensors[i].name,name))return &g->tensors[i];
    return NULL;
}

static fg_status load_binding(fg_mtp_head *h,const fg_gguf *g,int fd,
    const mtp_alias *a,uint32_t rank,fg_error *err){
    const fg_gguf_tensor *t=find_source(g,a);
    bool fused=a->experts>1u&&strstr(a->source,"gate_up_proj");
    uint32_t rows=a->rows*(fused?2u:1u);
    uint32_t dims=a->experts>1u?3u:a->norm?1u:2u;
    bool quant=t&&!a->f32&&(t->type==12u||t->type==13u||
        (a->experts==1u&&t->type==20u)||(a->experts>1u&&t->type==7u));
    if(!t||t->dims!=dims||t->shape[0]!=a->input||
       (dims>=2u&&t->shape[1]!=rows)||(dims==3u&&t->shape[2]!=a->experts)||
       (t->type!=(a->norm?0u:8u)&&!quant)){
        fg_error_set(err,FG_ERR_FORMAT,"native MTP source shape/type: %s",a->source);
        return FG_ERR_FORMAT;
    }
    uint64_t matrix=(uint64_t)a->input*a->rows;
    uint32_t block=t->type==0u?1u:(t->type==12u||t->type==13u)?256u:32u;
    uint32_t block_bytes=t->type==0u?4u:t->type==12u?144u:t->type==13u?176u:
        t->type==20u?18u:t->type==7u?24u:34u;
    uint64_t raw_bytes=matrix/block*block_bytes;
    uint32_t layout=a->f32?FG_TENSOR_LAYOUT_GGML:a->experts>1u?
        FG_TENSOR_LAYOUT_Q8_0_EXPERT_COOKED:FG_TENSOR_LAYOUT_Q8_0_COOKED;
    uint64_t cooked=a->f32?matrix*4u:fg_q8_0_cooked_matrix_bytes(a->input,a->rows);
    if(quant&&a->experts>1u){
        if(t->type==7u){
            layout=FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED;
            cooked=fg_q5_1_cooked_matrix_bytes(a->input,a->rows);
        }else{
            layout=FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED;
            cooked=fg_k_quant_cooked_matrix_bytes(a->input,a->rows,t->type);
        }
    }else if(quant){
        bool compact=fg_dense_prefer_compact(a->input,a->rows,t->type);
        layout=compact?FG_TENSOR_LAYOUT_DENSE_COMPACT:FG_TENSOR_LAYOUT_DENSE_SOA;
        cooked=compact?fg_dense_compact_bytes(a->input,a->rows,t->type):
            fg_dense_soa_bytes(a->input,a->rows,t->type);
    }
    if(!cooked||h->count>=MTP_BINDINGS||h->weight_bytes>FG_PERSISTENT_CAP_BYTES||
       cooked*a->experts>FG_PERSISTENT_CAP_BYTES-h->weight_bytes){
        fg_error_set(err,FG_ERR_LIMIT,"native MTP weight capacity");return FG_ERR_LIMIT;
    }
    uint32_t index=h->count++;
    fg_status status=fg_vk_tensor_create(h->vk,cooked*a->experts,&h->weights[index],err);
    uint8_t *raw=status==FG_OK?malloc((size_t)raw_bytes):NULL;
    if(status==FG_OK&&!raw){fg_error_set(err,FG_ERR_OOM,"native MTP cooking scratch");status=FG_ERR_OOM;}
    uint8_t *dst=status==FG_OK?fg_vk_tensor_map(h->weights[index]):NULL;
    for(uint32_t e=0;status==FG_OK&&e<a->experts;e++){
        uint64_t at=t->offset+((uint64_t)e*(fused?2u:1u)+a->half)*raw_bytes;
        status=read_exact(fd,at,raw,raw_bytes,err);
        if(status!=FG_OK)break;
        if(a->f32){
            float *f=(float *)(dst+(uint64_t)e*cooked);
            if(t->type==0u)memcpy(f,raw,(size_t)cooked);
            else fg_dequantize_q8_0(raw,f,matrix);
            for(uint64_t j=0;j<matrix;j++){
                if(a->norm)f[j]+=1.0f;
                if(!isfinite(f[j])){fg_error_set(err,FG_ERR_FORMAT,"nonfinite native MTP weight");status=FG_ERR_FORMAT;break;}
            }
        }else if(quant){
            void *out=dst+(uint64_t)e*cooked;
            bool ok=layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED?
                fg_cook_q5_1_rows(raw,out,cooked,a->input,a->rows):
                layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED?
                fg_cook_k_quant_rows(raw,out,cooked,a->input,a->rows,t->type):
                layout==FG_TENSOR_LAYOUT_DENSE_COMPACT?
                fg_cook_dense_compact(raw,out,cooked,a->input,a->rows,t->type):
                fg_cook_dense_soa(raw,out,cooked,a->input,a->rows,t->type);
            if(!ok){fg_error_set(err,FG_ERR_FORMAT,"native MTP Q4-class cooking");status=FG_ERR_FORMAT;}
        }else if(!fg_cook_q8_0_rows(raw,dst+(uint64_t)e*cooked,cooked,a->input,a->rows)){
            fg_error_set(err,FG_ERR_FORMAT,"native MTP Q8 cooking");status=FG_ERR_FORMAT;
        }
    }
    free(raw);
    if(status!=FG_OK)return status;
    fg_tensor_record *r=&h->manifest->tensors[index];
    if(!strncmp(a->alias,"mtp.",4u))snprintf(r->name,sizeof(r->name),"%s",a->alias);
    else if(a->experts>1u)snprintf(r->name,sizeof(r->name),"blk.%u.%s.weight.rank%u",MTP_LAYER,a->alias,rank);
    else snprintf(r->name,sizeof(r->name),"blk.%u.%s",MTP_LAYER,a->alias);
    r->rank=(uint16_t)rank;r->layer=MTP_LAYER;r->ggml_type=a->f32?0u:t->type;
    r->dims=a->experts>1u?3u:2u;r->shape[0]=a->input;r->shape[1]=a->rows;
    if(a->experts>1u)r->shape[2]=a->experts;
    r->bytes=cooked*a->experts;r->kind=a->experts>1u?FG_TENSOR_ROUTED_EXPERT:FG_TENSOR_COMMON;
    r->layout=layout;
    fg_vk_tensor_set_format(h->weights[index],a->f32?FG_VK_TENSOR_FORMAT_DEFAULT:
        quant&&a->experts>1u?(t->type==7u?FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED:
            FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED):
        quant?fg_vk_dense_saved_format(t->type,layout):
        a->experts>1u?FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED:FG_VK_TENSOR_FORMAT_Q8_0_COOKED);
    h->weight_bytes+=r->bytes;h->manifest->tensor_count=h->count;
    return FG_OK;
}

static fg_status scratch(fg_mtp_head *h,uint64_t values,fg_vk_tensor **out,fg_error *err){
    return fg_vk_tensor_create(h->vk,values*4u,out,err);
}
static fg_status allocate_scratch(fg_mtp_head *h,fg_error *err){
    fg_vk_tensor **fields[]={&h->en,&h->ep,&h->hn,&h->hp,
        &h->raw_q,&h->raw_k,&h->raw_v,&h->q,&h->gate,&h->key,
        &h->attention,&h->attn_out,&h->position,&h->norm,&h->down,&h->active,&h->up,&h->hidden,
        &h->output_partials};
    uint32_t counts[]={2560,2560,10240,10240,12288,512,512,6144,6144,512,
        6144,2560,4,10240,320,320,10240,2560,320u*8u};
    fg_status status=FG_OK;
    for(uint32_t i=0;status==FG_OK&&i<sizeof(counts)/sizeof(counts[0]);i++)
        status=scratch(h,counts[i],fields[i],err);
    for(uint32_t i=0;status==FG_OK&&i<4u;i++){
        status=fg_vk_tensor_view(h->hn,(uint64_t)i*2560u*4u,2560u*4u,&h->stream_in[i],err);
        if(status==FG_OK)status=fg_vk_tensor_view(h->hp,(uint64_t)i*2560u*4u,2560u*4u,&h->stream_out[i],err);
    }
    if(status==FG_OK)status=scratch(h,(uint64_t)h->capacity*512u,&h->keys,err);
    if(status==FG_OK)status=scratch(h,(uint64_t)h->capacity*512u,&h->values,err);
    if(status==FG_OK)status=scratch(h,10240u,&h->replay_input,err);
    if(status==FG_OK)status=scratch(h,2560u,&h->replay_embedding,err);
    if(status==FG_OK)status=scratch(h,24u*8u*258u,&h->attention_partials,err);
    return status;
}

static fg_status head_open(fg_mtp_head **out,fg_vk_context *vk,
    const fg_manifest *target,uint32_t rank,uint32_t native_owner,const char *overlay,
    uint32_t context_tokens,uint32_t window,fg_error *err){
    if(!out||!vk||!target||!overlay||!context_tokens||context_tokens>FG_MAX_CONTEXT||
       !target->rank_count||target->rank_count>FG_RANK_COUNT||rank>=target->rank_count||
       target->layer_count!=FG_LAYER_COUNT||rank!=native_owner||
       window>context_tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"native MTP must run on its selected owner");
        return FG_ERR_ARGUMENT;
    }
    uint32_t resolved=0u;
    fg_status placement=fg_mtp_native_owner_resolve(target,native_owner+1u,&resolved,err);
    if(placement!=FG_OK)return placement;
    *out=NULL;
    fg_gguf g={0};const char *paths[]={overlay};
    fg_status status=fg_gguf_open(paths,1u,&g,err);
    if(status!=FG_OK)return status;
    uint32_t capacity=window?window+FG_MTP_MAX_VERIFY:context_tokens;
    uint64_t needed=(uint64_t)capacity*512u*4u*2u+(16u<<20u);
    for(uint32_t i=0;i<sizeof(aliases)/sizeof(aliases[0]);i++){
        const mtp_alias *a=&aliases[i];
        const fg_gguf_tensor *t=find_source(&g,a);
        uint64_t matrix=a->f32?(uint64_t)a->input*a->rows*4u:
            fg_q8_0_cooked_matrix_bytes(a->input,a->rows);
        if(t&&!a->f32&&t->type!=8u){
            matrix=a->experts>1u?(t->type==7u?
                fg_q5_1_cooked_matrix_bytes(a->input,a->rows):
                fg_k_quant_cooked_matrix_bytes(a->input,a->rows,t->type)):
                fg_dense_prefer_compact(a->input,a->rows,t->type)?
                fg_dense_compact_bytes(a->input,a->rows,t->type):
                fg_dense_soa_bytes(a->input,a->rows,t->type);
        }
        if(!t||!matrix){
            fg_gguf_close(&g);fg_error_set(err,FG_ERR_FORMAT,"native MTP source storage: %s",a->source);
            return FG_ERR_FORMAT;
        }
        needed+=matrix*a->experts;
    }
    fg_vk_memory_stats memory={0};fg_vk_get_memory_stats(vk,&memory);
    uint64_t cap=target->residency_cap_bytes;
    if(!cap||cap>FG_RESIDENCY_CAP_BYTES)cap=FG_RESIDENCY_CAP_BYTES;
    if(memory.requested_live_bytes>cap||needed>cap-memory.requested_live_bytes){
        fg_gguf_close(&g);
        fg_error_set(err,FG_ERR_LIMIT,"native MTP weights/KV exceed remaining rank residency cap");
        return FG_ERR_LIMIT;
    }
    fg_mtp_head *h=calloc(1,sizeof(*h));
    if(!h){fg_gguf_close(&g);fg_error_set(err,FG_ERR_OOM,"native MTP head");return FG_ERR_OOM;}
    h->vk=vk;h->capacity=capacity;h->context_tokens=context_tokens;h->window=window;
    h->core_slot=FG_VK_MTP_STATIC_SLOT;h->warm_slot=FG_VK_MTP_STATIC_SLOT+1u;
    h->residency_cap=cap;
    h->manifest=calloc(1,sizeof(*h->manifest));
    if(!h->manifest){status=FG_ERR_OOM;fg_error_set(err,status,"native MTP manifest");}
    if(status==FG_OK){
        fg_manifest_init(h->manifest);
        h->manifest->rank_count=target->rank_count;h->manifest->prefill_microbatch=4u;
        /* Only GR/MoE helpers use this private model. Their scratch geometry
         * need not reserve sparse target-index storage for dense draft KV. */
        h->manifest->native_context=4u;h->manifest->max_context=4u;
        memset(h->manifest->layer_owner,UINT8_MAX,sizeof(h->manifest->layer_owner));
        h->manifest->layer_owner[MTP_LAYER]=(uint8_t)rank;
        for(uint32_t g=0;g<FG_GROUP_SIZE;g++)h->manifest->layer_groups[MTP_LAYER][g]=(uint8_t)rank;
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)h->manifest->expert_rank[MTP_LAYER][e]=(uint16_t)rank;
    }
    int fd=status==FG_OK?open(overlay,O_RDONLY|O_CLOEXEC):-1;
    if(status==FG_OK&&fd<0){status=FG_ERR_IO;fg_error_set(err,status,"open native MTP overlay: %s",strerror(errno));}
    for(uint32_t i=0;status==FG_OK&&i<sizeof(aliases)/sizeof(aliases[0]);i++)
        status=load_binding(h,&g,fd,&aliases[i],rank,err);
    if(fd>=0)close(fd);
    fg_gguf_close(&g);
    if(status==FG_OK)status=fg_model_bind_view(&h->model,vk,h->manifest,rank,h->weights,err);
    if(status==FG_OK)status=fg_owner_executor_create_worker(&h->owner,h->model,err);
    if(status==FG_OK)status=fg_expert_executor_create(&h->experts,h->model,err);
    if(status==FG_OK)status=allocate_scratch(h,err);
    if(status==FG_OK){
        h->shared=calloc(1,sizeof(*h->shared));
        if(!h->shared){status=FG_ERR_OOM;fg_error_set(err,status,"native MTP shared weights");}
        else{
            h->shared->manifest=h->manifest;h->shared->model=h->model;
            memcpy(h->shared->weights,h->weights,sizeof(h->weights));
            h->shared->count=h->count;h->shared->references=1u;
            h->shared->sessions=1u;
        }
    }
    if(status!=FG_OK){fg_mtp_head_close(h);return status;}
    *out=h;return FG_OK;
}

fg_status fg_mtp_head_open(fg_mtp_head **out,fg_vk_context *vk,
    const fg_manifest *target,uint32_t rank,const char *overlay,
    uint32_t context_tokens,fg_error *err){
    return head_open(out,vk,target,rank,target?target->layer_owner[MTP_LAYER]:UINT32_MAX,
                     overlay,context_tokens,0u,err);
}
fg_status fg_mtp_head_open_windowed(fg_mtp_head **out,fg_vk_context *vk,
    const fg_manifest *target,uint32_t rank,const char *overlay,
    uint32_t context_tokens,uint32_t window_tokens,fg_error *err){
    if(!window_tokens){
        if(out)*out=NULL;
        fg_error_set(err,FG_ERR_ARGUMENT,"native MTP window must be nonzero");return FG_ERR_ARGUMENT;
    }
    return head_open(out,vk,target,rank,target?target->layer_owner[MTP_LAYER]:UINT32_MAX,
                     overlay,context_tokens,window_tokens,err);
}

fg_status fg_mtp_head_open_owner_windowed(fg_mtp_head **out,fg_vk_context *vk,
    const fg_manifest *target,uint32_t rank,uint32_t native_owner,const char *overlay,
    uint32_t context_tokens,uint32_t window_tokens,fg_error *err){
    if(!window_tokens){
        if(out)*out=NULL;
        fg_error_set(err,FG_ERR_ARGUMENT,"native MTP window must be nonzero");return FG_ERR_ARGUMENT;
    }
    return head_open(out,vk,target,rank,native_owner,overlay,context_tokens,window_tokens,err);
}

fg_status fg_mtp_head_fork(fg_mtp_head **out,fg_mtp_head *source,
    uint32_t state_slot,fg_error *err){
    if(out)*out=NULL;
    if(!out||!source||!source->shared||state_slot>=FG_OWNER_SESSION_MAX||
       (source->shared->sessions&(1u<<state_slot))||fg_vk_batch_active(source->vk)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid or occupied native MTP session");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_memory_stats memory={0};fg_vk_get_memory_stats(source->vk,&memory);
    uint64_t needed=(uint64_t)source->capacity*512u*4u*2u+(16u<<20u);
    if(memory.requested_live_bytes>source->residency_cap||
       needed>source->residency_cap-memory.requested_live_bytes){
        fg_error_set(err,FG_ERR_LIMIT,
            "native MTP session residency: live=%llu needed=%llu cap=%llu",
            (unsigned long long)memory.requested_live_bytes,
            (unsigned long long)needed,(unsigned long long)source->residency_cap);
        return FG_ERR_LIMIT;
    }
    fg_mtp_head *h=calloc(1,sizeof(*h));
    if(!h){fg_error_set(err,FG_ERR_OOM,"native MTP session");return FG_ERR_OOM;}
    h->vk=source->vk;h->manifest=source->manifest;h->model=source->model;
    h->capacity=source->capacity;h->weight_bytes=source->weight_bytes;
    h->context_tokens=source->context_tokens;h->window=source->window;
    h->residency_cap=source->residency_cap;
    h->shared=source->shared;h->session_slot=state_slot;
    h->core_slot=FG_VK_MTP_STATIC_SLOT+2u*state_slot;
    h->warm_slot=h->core_slot+1u;
    h->shared->references++;h->shared->sessions|=1u<<state_slot;
    fg_status status=fg_owner_executor_create_worker(&h->owner,h->model,err);
    if(status==FG_OK)status=fg_expert_executor_create(&h->experts,h->model,err);
    if(status==FG_OK)status=allocate_scratch(h,err);
    if(status!=FG_OK){fg_mtp_head_close(h);return status;}
    *out=h;return FG_OK;
}

fg_status fg_mtp_head_check_residency(const fg_mtp_head *h,fg_error *err){
    if(!h){fg_error_set(err,FG_ERR_ARGUMENT,"missing native MTP head");return FG_ERR_ARGUMENT;}
    fg_vk_memory_stats memory={0};fg_vk_get_memory_stats(h->vk,&memory);
    if(memory.requested_live_bytes>h->residency_cap){
        fg_error_set(err,FG_ERR_LIMIT,"native MTP readiness: live=%llu cap=%llu",
            (unsigned long long)memory.requested_live_bytes,
            (unsigned long long)h->residency_cap);
        return FG_ERR_LIMIT;
    }
    return FG_OK;
}

void fg_mtp_head_close(fg_mtp_head *h){
    if(!h)return;
    if(h->replay_claimed){
        fg_error ignored={0};
        fg_vk_static_release(h->vk,h->core_slot,&ignored);
    }
    if(h->warm_claimed){
        fg_error ignored={0};
        fg_vk_static_release(h->vk,h->warm_slot,&ignored);
    }
    fg_expert_executor_destroy(h->experts);fg_owner_executor_destroy(h->owner);
    if(!h->shared)fg_model_close(h->model);
    fg_vk_tensor *fields[]={h->en,h->ep,h->hn,h->hp,h->raw_q,h->raw_k,
        h->raw_v,h->q,h->gate,h->key,h->attention,h->attn_out,h->position,h->norm,
        h->down,h->active,h->up,h->hidden,h->keys,h->values,h->output_partials};
    for(uint32_t i=0;i<sizeof(fields)/sizeof(fields[0]);i++)fg_vk_tensor_destroy(fields[i]);
    for(uint32_t i=0;i<4u;i++){fg_vk_tensor_destroy(h->stream_in[i]);fg_vk_tensor_destroy(h->stream_out[i]);}
    fg_vk_tensor_destroy(h->replay_input);fg_vk_tensor_destroy(h->replay_embedding);
    fg_vk_tensor_destroy(h->attention_partials);
    if(h->shared){
        mtp_shared_weights *shared=h->shared;
        shared->sessions&=~(1u<<h->session_slot);
        if(!--shared->references){
            fg_model_close(shared->model);
            for(uint32_t i=0;i<shared->count;i++)fg_vk_tensor_destroy(shared->weights[i]);
            free(shared->manifest);free(shared);
        }
    }else{
        for(uint32_t i=0;i<h->count;i++)fg_vk_tensor_destroy(h->weights[i]);
        free(h->manifest);
    }
    free(h);
}
uint32_t fg_mtp_head_frontier(const fg_mtp_head *h){return h?h->frontier:0u;}
uint64_t fg_mtp_head_weight_bytes(const fg_mtp_head *h){return h?h->weight_bytes:0u;}
fg_status fg_mtp_head_truncate(fg_mtp_head *h,uint32_t cells,fg_error *err){
    if(!h||cells>h->frontier||cells<h->first_cell||
       (h->window&&h->frontier-cells>h->capacity-h->window)||
       fg_vk_batch_active(h->vk)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid MTP draft frontier");return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_vk_static_drain(h->vk,err);
    if(status==FG_OK)h->frontier=cells;
    return status;
}
fg_status fg_mtp_head_reset(fg_mtp_head *h,uint32_t first_cell,fg_error *err){
    if(!h||first_cell>=h->context_tokens||(!h->window&&first_cell)||
       fg_vk_batch_active(h->vk)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid native MTP prefix reset");return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_vk_static_drain(h->vk,err);
    if(status==FG_OK){h->frontier=first_cell;h->first_cell=first_cell;}
    return status;
}
static fg_status dense(fg_mtp_head *h,fg_vk_tensor *out,const char *name,
    const fg_vk_tensor *in,uint32_t k,uint32_t m,fg_error *err){
    return fg_vk_dense_q8_0_f32(h->vk,out,binding(h,name),in,k,m,1u,1.0f,err);
}
static fg_status replay_experts(void *context,uint32_t layer,
    const fg_vk_tensor *activation,const fg_vk_tensor *router,
    fg_vk_tensor **output,fg_error *err){
    fg_mtp_head *h=context;
    return fg_expert_decode_chain(h->experts,layer,activation,router,output,err);
}
static fg_status record_input_mix(fg_mtp_head *h,fg_vk_tensor **mixed,
    const fg_vk_tensor **residual,fg_vk_tensor **injection,fg_error *err){
    fg_status status=fg_vk_group_rms_norm(h->vk,h->en,h->replay_embedding,
        binding(h,"mtp.embedding_norm"),2560u,1u,1u,1e-6f,err);
    if(status==FG_OK)status=dense(h,h->ep,"mtp.fc_embedding",h->en,2560u,2560u,err);
    if(status==FG_OK)status=fg_vk_mtp_hc_norm(h->vk,h->hn,h->replay_input,
        binding(h,"mtp.hidden_norm"),err);
    if(status==FG_OK&&fg_vk_tensor_get_format(binding(h,"mtp.fc_hidden"))==FG_VK_TENSOR_FORMAT_Q8_0_COOKED)
        status=fg_vk_mtp_fc_hidden(h->vk,h->hp,binding(h,"mtp.fc_hidden"),h->hn,err);
    else for(uint32_t i=0;status==FG_OK&&i<4u;i++)
        status=dense(h,h->stream_out[i],"mtp.fc_hidden",h->stream_in[i],2560u,2560u,err);
    for(uint32_t i=0;status==FG_OK&&i<4u;i++)
        status=fg_vk_add_f32(h->vk,h->stream_in[i],h->stream_out[i],h->ep,2560u,err);
    if(status==FG_OK)status=fg_owner_gr_read(h->owner,MTP_LAYER,false,h->hn,
        mixed,residual,injection,err);
    return status;
}
static fg_status record_warm(fg_mtp_head *h,fg_error *err){
    fg_vk_tensor *mixed=NULL,*injection=NULL;
    const fg_vk_tensor *residual=NULL;
    fg_status status=record_input_mix(h,&mixed,&residual,&injection,err);
    if(status==FG_OK)status=dense(h,h->raw_k,"blk.47.attn_k.weight",mixed,2560u,512u,err);
    if(status==FG_OK)status=dense(h,h->raw_v,"blk.47.attn_v.weight",mixed,2560u,512u,err);
    if(status==FG_OK)status=fg_vk_mtp_cache_kv(h->vk,h->keys,h->values,h->raw_k,h->raw_v,
        binding(h,"blk.47.attn_k_norm.weight"),h->position,h->capacity,err);
    return status;
}
static fg_status record_replay(fg_mtp_head *h,fg_error *err){
    fg_vk_tensor *mixed=NULL,*injection=NULL,*after=NULL,*block=NULL;
    const fg_vk_tensor *residual=NULL;
    fg_status status=record_input_mix(h,&mixed,&residual,&injection,err);
    if(status==FG_OK)status=dense(h,h->raw_q,"blk.47.attn_q.weight",mixed,2560u,12288u,err);
    if(status==FG_OK)status=dense(h,h->raw_k,"blk.47.attn_k.weight",mixed,2560u,512u,err);
    if(status==FG_OK)status=dense(h,h->raw_v,"blk.47.attn_v.weight",mixed,2560u,512u,err);
    if(status==FG_OK)status=fg_vk_qsa_prepare(h->vk,h->q,h->gate,h->key,h->raw_q,h->raw_k,
        binding(h,"blk.47.attn_q_norm.weight"),binding(h,"blk.47.attn_k_norm.weight"),h->position,err);
    if(status==FG_OK)status=fg_vk_mtp_attention_window(h->vk,h->attention,h->attention_partials,
        h->keys,h->values,h->q,h->gate,h->key,h->raw_v,h->position,h->capacity,8u,h->window,err);
    if(status==FG_OK)status=dense(h,h->attn_out,"blk.47.attn_output.weight",h->attention,6144u,2560u,err);
    if(status==FG_OK)status=fg_owner_gr_write(h->owner,residual,h->attn_out,injection,&after,err);
    if(status==FG_OK)status=fg_owner_gr_read(h->owner,MTP_LAYER,true,after,&mixed,&residual,&injection,err);
    if(status==FG_OK)status=fg_owner_moe_record(h->owner,MTP_LAYER,mixed,replay_experts,h,&block,err);
    if(status==FG_OK)status=fg_owner_gr_write(h->owner,residual,block,injection,&h->replay_hc,err);
    if(status==FG_OK)status=fg_vk_group_rms_norm(h->vk,h->norm,h->replay_hc,
        binding(h,"mtp.output_norm"),2560u,4u,1u,1e-6f,err);
    fg_vk_tensor *output_down=binding(h,"mtp.output_down");
    if(status==FG_OK&&fg_vk_tensor_get_format(output_down)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED){
        status=fg_vk_dense_q8_0_cooked_split_silu(h->vk,h->down,h->active,h->output_partials,
            output_down,h->norm,10240u,320u,1u,8u,1.0f,0.25f,err);
    }else{
        if(status==FG_OK)status=dense(h,h->down,"mtp.output_down",h->norm,10240u,320u,err);
        if(status==FG_OK)status=fg_vk_silu_scaled(h->vk,h->active,h->down,320u,0.25f,err);
    }
    if(status==FG_OK)status=dense(h,h->up,"mtp.output_up",h->active,320u,10240u,err);
    if(status==FG_OK)status=fg_vk_hc_finalize(h->vk,h->hidden,h->norm,h->up,2560u,4u,1u,err);
    return status;
}
static fg_status run_replay(fg_mtp_head *h,const fg_vk_tensor *hyper,
    const fg_vk_tensor *embedding,uint32_t cell,bool warm,fg_error *err){
    if(!h||!hyper||!embedding||cell!=h->frontier||
       cell>=h->context_tokens||h->trace||fg_vk_batch_active(h->vk)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid native replay input/frontier");return FG_ERR_ARGUMENT;
    }
    fg_status status=FG_OK;
    uint32_t slot=warm?h->warm_slot:h->core_slot;
    bool *claimed=warm?&h->warm_claimed:&h->replay_claimed;
    if(!*claimed){
        status=fg_vk_static_claim(h->vk,slot,err);
        if(status==FG_OK)*claimed=true;
    }
    fg_vk_tensor *source[2]={NULL,NULL};
    if(status==FG_OK)status=fg_vk_tensor_view((fg_vk_tensor *)hyper,0,10240u*4u,&source[0],err);
    if(status==FG_OK)status=fg_vk_tensor_view((fg_vk_tensor *)embedding,0,2560u*4u,&source[1],err);
    fg_vk_tensor *dest[]={h->replay_input,h->replay_embedding};
    const fg_vk_tensor *inputs[]={source[0],source[1]};
    bool profile=fg_vk_profile_active(h->vk);
    if(status==FG_OK&&profile)status=fg_vk_copy_tensors(h->vk,dest,inputs,2u,err);
    uint32_t position[4]={cell,cell,cell,h->first_cell};
    if(status==FG_OK)status=fg_vk_tensor_write(h->position,0,position,sizeof(position),err);
    if(status==FG_OK&&profile){
        status=fg_vk_begin(h->vk,err);
        if(status==FG_OK)status=warm?record_warm(h,err):record_replay(h,err);
        if(status==FG_OK)status=fg_vk_end(h->vk,err);
    }else if(status==FG_OK){
        if(!fg_vk_static_recorded(h->vk,slot)){
            status=fg_vk_static_begin(h->vk,slot,err);
            if(status==FG_OK)status=warm?record_warm(h,err):record_replay(h,err);
            if(status==FG_OK)status=fg_vk_static_end(h->vk,slot,err);
        }
        if(status==FG_OK)status=fg_vk_static_copy_run(h->vk,slot,dest,inputs,2u,err);
    }
    fg_vk_tensor_destroy(source[0]);fg_vk_tensor_destroy(source[1]);
    if(status!=FG_OK){
        fg_error ignored={0};
        if(fg_vk_batch_active(h->vk))fg_vk_abort(h->vk,&ignored);
        fg_vk_static_drain(h->vk,&ignored);
        if(*claimed)fg_vk_static_reset(h->vk,slot,&ignored);
        return status;
    }
    h->frontier=cell+1u;return FG_OK;
}
fg_status fg_mtp_head_step_replay(fg_mtp_head *h,const fg_vk_tensor *hyper,
    const fg_vk_tensor *embedding,uint32_t cell,fg_vk_tensor **next_hyper,
    fg_vk_tensor **contracted,fg_error *err){
    if(!next_hyper||!contracted){
        fg_error_set(err,FG_ERR_ARGUMENT,"native MTP output pointers");return FG_ERR_ARGUMENT;
    }
    *next_hyper=NULL;*contracted=NULL;
    fg_status status=run_replay(h,hyper,embedding,cell,false,err);
    if(status==FG_OK){*next_hyper=h->replay_hc;*contracted=h->hidden;}
    return status;
}
fg_status fg_mtp_head_warm_replay(fg_mtp_head *h,const fg_vk_tensor *hyper,
    const fg_vk_tensor *embedding,uint32_t cell,fg_error *err){
    return run_replay(h,hyper,embedding,cell,true,err);
}

fg_status fg_mtp_head_step(fg_mtp_head *h,const fg_vk_tensor *hyper,
    const fg_vk_tensor *embedding,uint32_t cell,fg_vk_tensor **next_hyper,
    fg_vk_tensor **contracted,fg_error *err){
    if(!h||!hyper||!embedding||!next_hyper||!contracted||cell!=h->frontier||
       cell>=h->capacity||h->window||fg_vk_batch_active(h->vk)){
        fg_error_set(err,FG_ERR_ARGUMENT,"native MTP input/frontier mismatch");return FG_ERR_ARGUMENT;
    }
    *next_hyper=NULL;*contracted=NULL;
    fg_status status=fg_vk_begin(h->vk,err);
    if(status==FG_OK)status=fg_vk_group_rms_norm(h->vk,h->en,embedding,binding(h,"mtp.embedding_norm"),2560u,1u,1u,1e-6f,err);
    if(status==FG_OK)status=dense(h,h->ep,"mtp.fc_embedding",h->en,2560u,2560u,err);
    if(status==FG_OK)status=fg_vk_group_rms_norm(h->vk,h->hn,hyper,binding(h,"mtp.hidden_norm"),10240u,1u,1u,1e-6f,err);
    for(uint32_t i=0;status==FG_OK&&i<4u;i++)
        status=dense(h,h->stream_out[i],"mtp.fc_hidden",h->stream_in[i],2560u,2560u,err);
    /* Reuse the normalized-input views as broadcast-add destinations. */
    for(uint32_t i=0;status==FG_OK&&i<4u;i++)
        status=fg_vk_add_f32(h->vk,h->stream_in[i],h->stream_out[i],h->ep,2560u,err);
    if(status==FG_OK)status=fg_vk_end(h->vk,err);
    if(status==FG_OK)status=trace(h,"combined",h->hn,10240u,err);
    fg_vk_tensor *mixed=NULL,*injection=NULL,*block=NULL,*after=NULL,*hc=NULL;
    const fg_vk_tensor *residual=NULL;
    if(status==FG_OK)status=fg_owner_gr_read(h->owner,MTP_LAYER,false,h->hn,&mixed,&residual,&injection,err);
    if(status==FG_OK)status=trace(h,"attn_mixed",mixed,2560u,err);
    if(status==FG_OK)status=trace(h,"attn_injection",injection,4u,err);
    uint32_t position[3]={cell,cell,cell};
    if(status==FG_OK)status=fg_vk_tensor_write(h->position,0,position,sizeof(position),err);
    if(status==FG_OK)status=fg_vk_begin(h->vk,err);
    if(status==FG_OK)status=dense(h,h->raw_q,"blk.47.attn_q.weight",mixed,2560u,12288u,err);
    if(status==FG_OK)status=dense(h,h->raw_k,"blk.47.attn_k.weight",mixed,2560u,512u,err);
    if(status==FG_OK)status=dense(h,h->raw_v,"blk.47.attn_v.weight",mixed,2560u,512u,err);
    if(status==FG_OK)status=fg_vk_qsa_prepare(h->vk,h->q,h->gate,h->key,h->raw_q,h->raw_k,
        binding(h,"blk.47.attn_q_norm.weight"),binding(h,"blk.47.attn_k_norm.weight"),h->position,err);
    if(status==FG_OK)status=fg_vk_mtp_attention(h->vk,h->attention,h->keys,h->values,h->q,h->gate,h->key,h->raw_v,cell,err);
    if(status==FG_OK)status=dense(h,h->attn_out,"blk.47.attn_output.weight",h->attention,6144u,2560u,err);
    if(status==FG_OK)status=fg_vk_end(h->vk,err);
    if(status==FG_OK)status=trace(h,"raw_q",h->raw_q,12288u,err);
    if(status==FG_OK)status=trace(h,"raw_v",h->raw_v,512u,err);
    if(status==FG_OK)status=trace(h,"attention_output",h->attn_out,2560u,err);
    if(status==FG_OK)status=fg_owner_gr_write(h->owner,residual,h->attn_out,injection,&after,err);
    if(status==FG_OK)status=trace(h,"after_attention",after,10240u,err);
    if(status==FG_OK)status=fg_owner_gr_read(h->owner,MTP_LAYER,true,after,&mixed,&residual,&injection,err);
    if(status==FG_OK)status=trace(h,"ffn_mixed",mixed,2560u,err);
    if(status==FG_OK)status=trace(h,"ffn_injection",injection,4u,err);
    uint16_t ids[FG_TOP_K];float gates[FG_TOP_K];const uint8_t *activation=NULL;
    if(status==FG_OK)status=fg_owner_moe_prepare(h->owner,MTP_LAYER,mixed,ids,gates,&activation,err);
    if(status==FG_OK&&h->trace){
        float routed[FG_TOP_K];for(uint32_t i=0;i<FG_TOP_K;i++)routed[i]=ids[i];
        status=trace_host(h,"router_ids",routed,FG_TOP_K,err);
        if(status==FG_OK)status=trace_host(h,"router_gates",gates,FG_TOP_K,err);
        float quantized[2560];
        for(uint32_t b=0;b<10u;b++){
            float scale;memcpy(&scale,activation+b*296u,4u);
            const int8_t *q=(const int8_t *)(activation+b*296u+8u);
            for(uint32_t j=0;j<256u;j++)quantized[b*256u+j]=q[j]*scale;
        }
        if(status==FG_OK)status=trace_host(h,"quantized",quantized,2560u,err);
    }
    fg_decode_work work={0};fg_expert_result result={0};
    uint32_t rank=h->model?fg_model_rank(h->model):0u;
    if(status==FG_OK){
        work.layer=MTP_LAYER;work.position=cell;work.source_rank=(uint8_t)rank;
        work.destination_rank=(uint8_t)rank;work.selected_count=FG_TOP_K;
        memcpy(work.expert_ids,ids,sizeof(ids));memcpy(work.gates,gates,sizeof(gates));
        for(uint32_t i=0;i<FG_TOP_K;i++)work.routing_slots[i]=(uint8_t)i;
        memcpy(work.activation_q8k,activation,sizeof(work.activation_q8k));
        status=fg_expert_decode(h->experts,&work,&result,err);
    }
    if(status==FG_OK)status=trace_host(h,"expert_result_0",result.outputs[0],2560u,err);
    if(status==FG_OK&&h->trace){
        float mid[FG_TOP_K*640u];
        status=fg_expert_mid_readback(h->experts,FG_TOP_K,mid,err);
        if(status==FG_OK)status=trace_host(h,"expert_mid",mid,FG_TOP_K*640u,err);
    }
    if(status==FG_OK)status=fg_owner_moe_reduce(h->owner,MTP_LAYER,cell,ids,gates,&result,1u,&block,err);
    if(status==FG_OK)status=trace(h,"moe_output",block,2560u,err);
    if(status==FG_OK)status=fg_owner_gr_write(h->owner,residual,block,injection,&hc,err);
    if(status==FG_OK)status=trace(h,"final_hc",hc,10240u,err);
    if(status==FG_OK)status=fg_vk_begin(h->vk,err);
    if(status==FG_OK)status=fg_vk_group_rms_norm(h->vk,h->norm,hc,binding(h,"mtp.output_norm"),2560u,4u,1u,1e-6f,err);
    if(status==FG_OK)status=dense(h,h->down,"mtp.output_down",h->norm,10240u,320u,err);
    if(status==FG_OK)status=fg_vk_silu_scaled(h->vk,h->active,h->down,320u,0.25f,err);
    if(status==FG_OK)status=dense(h,h->up,"mtp.output_up",h->active,320u,10240u,err);
    if(status==FG_OK)status=fg_vk_hc_finalize(h->vk,h->hidden,h->norm,h->up,2560u,4u,1u,err);
    if(status==FG_OK)status=fg_vk_end(h->vk,err);
    if(status!=FG_OK){
        fg_error ignored={0};if(fg_vk_batch_active(h->vk))fg_vk_abort(h->vk,&ignored);
        fg_vk_static_drain(h->vk,&ignored);
        return status;
    }
    h->frontier=cell+1u;*next_hyper=hc;*contracted=h->hidden;return FG_OK;
}
