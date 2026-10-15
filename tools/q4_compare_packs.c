#include "fg_manifest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv){
    if(argc!=3){fprintf(stderr,"usage: q4-compare-packs BASE CANDIDATE\n");return 2;}
    fg_manifest *a=malloc(sizeof(*a)),*b=malloc(sizeof(*b));
    fg_error error={0};int status=1;
    if(!a||!b||fg_manifest_read(argv[1],a,&error)!=FG_OK||
       fg_manifest_read(argv[2],b,&error)!=FG_OK)goto done;
    if(a->tensor_count!=b->tensor_count||a->rank_count!=b->rank_count||
       a->layer_count!=b->layer_count||a->expert_count!=b->expert_count||
       a->hidden_size!=b->hidden_size||a->top_k!=b->top_k||
       a->max_context!=b->max_context||a->prefill_microbatch!=b->prefill_microbatch||
       a->prefill_window!=b->prefill_window||
       memcmp(a->layer_owner,b->layer_owner,sizeof(a->layer_owner))||
       memcmp(a->layer_groups,b->layer_groups,sizeof(a->layer_groups))||
       memcmp(a->expert_rank,b->expert_rank,sizeof(a->expert_rank))){
        fprintf(stderr,"pack geometry differs\n");goto done;
    }
    for(uint32_t rank=0;rank<a->rank_count;rank++)
        if(strcmp(a->ranks[rank].endpoint,b->ranks[rank].endpoint)){
            fprintf(stderr,"rank endpoints differ\n");goto done;
        }
    uint32_t changed=0;
    for(uint32_t i=0;i<a->tensor_count;i++){
        const fg_tensor_record *x=&a->tensors[i],*y=&b->tensors[i];
        if(strcmp(x->name,y->name)||x->ggml_type!=y->ggml_type||
           x->dims!=y->dims||x->bytes!=y->bytes||x->rank!=y->rank||
           x->layer!=y->layer||x->expert!=y->expert||x->kind!=y->kind||
           x->layout!=y->layout||memcmp(x->shape,y->shape,x->dims*sizeof(uint64_t))){
            fprintf(stderr,"tensor metadata/order differs: %s\n",x->name);goto done;
        }
        if(memcmp(x->sha256,y->sha256,sizeof(x->sha256))){
            printf("PACK_CHANGE name=%s\n",x->name);changed++;
        }
    }
    printf("PACK_COMPARE tensors=%u changed=%u metadata_equal=1\n",a->tensor_count,changed);
    status=0;
done:
    if(status&&error.message[0])fprintf(stderr,"%s\n",error.message);
    free(a);free(b);return status;
}
