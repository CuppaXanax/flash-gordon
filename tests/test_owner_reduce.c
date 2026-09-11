#include "../src/owner.c"

/* Real owner scratch layout and GPU reduction; no model weights are needed. */
struct fg_model {fg_manifest manifest;fg_vk_context *vk;};
fg_vk_context *fg_model_vk(fg_model *m){return m->vk;}
const fg_manifest *fg_model_manifest(const fg_model *m){return &m->manifest;}
uint32_t fg_model_rank(const fg_model *m){(void)m;return 0u;}
fg_vk_tensor *fg_model_tensor(fg_model *m,const char *name){(void)m;(void)name;return NULL;}
const fg_tensor_record *fg_model_tensor_record(const fg_model *m,const char *name){(void)m;(void)name;return NULL;}

int main(void){
    fg_error error={0};fg_model *m=calloc(1,sizeof(*m));
    fg_owner_executor *e=calloc(1,sizeof(*e));
    if(!m||!e)return 1;
    fg_manifest_init(&m->manifest);m->manifest.prefill_microbatch=128u;
    if(fg_vk_open(&m->vk,&error)!=FG_OK){free(e);free(m);return 77;}
    e->model=m;e->max_tokens=128u;e->replicated=true;
    e->prefill_slot_outputs=calloc(128u*FG_TOP_K,sizeof(float *));
    int ok=e->prefill_slot_outputs&&fg_vk_tensor_create(m->vk,
        fg_qsa_attention_family_scratch_bytes(128u),&e->attention_family_scratch,&error)==FG_OK&&
        create_transient_views(e,&error)==FG_OK;
    const uint32_t lengths[]={1u,17u,127u,128u};
    for(uint32_t shape=0;ok&&shape<8u;shape++){
        uint32_t n=lengths[shape%4u],pairs=n*FG_TOP_K;
        uint16_t *ids=malloc(pairs*sizeof(*ids));float *gates=malloc(pairs*4u);
        float *values=malloc((uint64_t)pairs*FG_HIDDEN_SIZE*4u);
        fg_prefill_result_pair *slots=calloc(pairs,sizeof(*slots));
        ok=ids&&gates&&values&&slots;
        float *shared=fg_vk_tensor_map(e->shared_output),*logits=fg_vk_tensor_map(e->shared_scalar);
        for(uint32_t t=0;ok&&t<n;t++){
            logits[t]=0.01f*(float)t-0.5f;
            for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++)shared[(uint64_t)t*FG_HIDDEN_SIZE+i]=0.01f*(float)(i%31u)+0.02f*t;
            for(uint32_t slot=0;slot<FG_TOP_K;slot++){
                uint32_t p=t*FG_TOP_K+slot,reverse=pairs-p-1u;
                ids[p]=(uint16_t)slot;m->manifest.expert_rank[0][slot]=0u;
                gates[p]=0.01f*(float)(slot+1u);
                slots[reverse].token_slot=(uint16_t)t;slots[reverse].routing_slot=(uint8_t)slot;
                for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++)values[(uint64_t)reverse*FG_HIDDEN_SIZE+i]=0.001f*(float)(i%17u)+0.01f*(float)p;
            }
        }
        memset(fg_vk_tensor_map(e->injection),0x35,(size_t)fg_vk_tensor_bytes(e->injection));
        uint32_t ranks=shape<4u?1u:FG_GROUP_SIZE,used=0;
        fg_prefill_result results[FG_GROUP_SIZE]={0};
        for(uint32_t r=0;r<ranks;r++){
            uint32_t rank=m->manifest.layer_groups[0][r];
            fg_prefill_result *result=&results[ranks-1u-r];
            *result=(fg_prefill_result){.layer=0u,.source_rank=(uint8_t)rank,.destination_rank=0u,.contributor_mask=(uint8_t)(1u<<rank),
                .first_position=7u,.token_count=(uint16_t)n,.pairs=slots+used,
                .outputs=values+(uint64_t)r*n*FG_HIDDEN_SIZE};
            for(uint32_t t=0;t<n;t++)for(uint32_t slot=0;slot<FG_TOP_K;slot++)if(slot%ranks==r){
                m->manifest.expert_rank[0][slot]=(uint16_t)rank;
                slots[used++]=(fg_prefill_result_pair){(uint16_t)t,(uint8_t)slot};
                result->pair_count++;
            }
            for(uint32_t t=0;t<n;t++)for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++){
                float sum=0.0f;
                for(uint32_t slot=0;slot<FG_TOP_K;slot++)if(slot%ranks==r)
                    sum=fmaf(gates[t*FG_TOP_K+slot],0.001f*(float)(i%17u)+0.01f*(float)(t*FG_TOP_K+slot),sum);
                result->outputs[(uint64_t)t*FG_HIDDEN_SIZE+i]=sum;
            }
        }
        fg_vk_tensor *out=NULL;
        if(ok)ok=fg_owner_moe_reduce_batch(e,0u,7u,(uint16_t)n,ids,gates,results,ranks,&out,&error)==FG_OK;
        const float *actual=fg_vk_tensor_const_map(out);
        for(uint32_t t=0;ok&&t<n;t++)for(uint32_t i=0;ok&&i<FG_HIDDEN_SIZE;i++){
            float original=0.01f*(float)(i%31u)+0.02f*t;
            float expected=original/(1.0f+expf(-logits[t]));
            for(uint32_t slot=0;slot<FG_TOP_K;slot++){
                uint32_t p=t*FG_TOP_K+slot;
                expected=fmaf(gates[p],(0.001f*(float)(i%17u)+0.01f*(float)p),expected);
            }
            if(fabsf(actual[(uint64_t)t*FG_HIDDEN_SIZE+i]-expected)>2e-6f*fmaxf(1.0f,fabsf(expected))||shared[(uint64_t)t*FG_HIDDEN_SIZE+i]!=original)ok=0;
        }
        const uint8_t *injection=fg_vk_tensor_const_map(e->injection);
        for(uint64_t i=0;ok&&i<fg_vk_tensor_bytes(e->injection);i++)if(injection[i]!=0x35u)ok=0;
        if(ok){slots[0]=slots[1];ok=fg_owner_moe_reduce_batch(e,0u,7u,(uint16_t)n,ids,gates,results,ranks,&out,&error)!=FG_OK;}
        free(slots);free(values);free(gates);free(ids);
    }
    fg_owner_executor_destroy(e);fg_vk_close(m->vk);free(m);
    if(!ok){fprintf(stderr,"Owner GPU reduction failed: %s\n",error.message);return 1;}
    puts("Owner GPU reduction: shuffled slots, tile tails, live scratch and invalid routes PASS");return 0;
}
