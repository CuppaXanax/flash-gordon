#include "fg_runtime.h"
#include "fg_mtp_head.h"
#include "fg_mtp.h"
#include "fg_protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static fg_status compare_owner(fg_vk_context *vk,const fg_manifest *m,
    const char *overlay,fg_mtp_head *placed[2],uint32_t *steps,fg_error *err){
    fg_mtp_head *reference[2]={0};
    fg_vk_context *reference_vk=NULL;
    fg_vk_tensor *input[2]={0},*embedding[2]={0};
    fg_status status=fg_vk_open(&reference_vk,err);
    if(status==FG_OK)status=fg_mtp_head_open_windowed(&reference[0],reference_vk,m,
        m->layer_owner[FG_LAYER_COUNT-1u],overlay,m->session.logical_context_tokens,
        8192u,err);
    if(status==FG_OK)status=fg_mtp_head_fork(&reference[1],reference[0],1u,err);
    for(uint32_t i=0;status==FG_OK&&i<2u;i++){
        fg_vk_context *context=i?reference_vk:vk;
        status=fg_vk_tensor_create(context,FG_HYPER_WIDTH*4u,&input[i],err);
        if(status==FG_OK)status=fg_vk_tensor_create(context,FG_HIDDEN_SIZE*4u,&embedding[i],err);
    }
    float hc[FG_HYPER_WIDTH],embed[FG_HIDDEN_SIZE];
    float actual[FG_HYPER_WIDTH],expected[FG_HYPER_WIDTH];
    for(uint32_t cell=0u;status==FG_OK&&cell<4u;cell++)
    for(uint32_t slot=0u;status==FG_OK&&slot<2u;slot++){
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
            hc[i]=(float)((int)(i%71u)-35)*(0.005f+(float)(slot+cell)*0.001f);
        for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++)embed[i]=(float)((int)(i%31u)-15)*0.01f;
        for(uint32_t i=0;status==FG_OK&&i<2u;i++){
            status=fg_vk_tensor_write(input[i],0u,hc,sizeof(hc),err);
            if(status==FG_OK)status=fg_vk_tensor_write(embedding[i],0u,embed,sizeof(embed),err);
        }
        fg_vk_tensor *outputs[2][2]={{0}};
        if(status==FG_OK)status=fg_mtp_head_step_replay(placed[slot],input[0],embedding[0],
            cell,&outputs[0][0],&outputs[0][1],err);
        if(status==FG_OK)status=fg_mtp_head_step_replay(reference[slot],input[1],embedding[1],
            cell,&outputs[1][0],&outputs[1][1],err);
        for(uint32_t kind=0u;status==FG_OK&&kind<2u;kind++){
            uint32_t count=kind?FG_HIDDEN_SIZE:FG_HYPER_WIDTH;
            status=fg_vk_tensor_read(outputs[0][kind],0u,actual,count*4u,err);
            if(status==FG_OK)status=fg_vk_tensor_read(outputs[1][kind],0u,expected,count*4u,err);
            if(status==FG_OK&&memcmp(actual,expected,(size_t)count*4u)){
                fg_error_set(err,FG_ERR_MISMATCH,"placed native output differs from final owner");
                status=FG_ERR_MISMATCH;
            }
            for(uint32_t i=0;status==FG_OK&&i<count;i++)if(!isfinite(actual[i])){
                fg_error_set(err,FG_ERR_FORMAT,"placed native output is non-finite");
                status=FG_ERR_FORMAT;
            }
        }
        if(status==FG_OK)(*steps)++;
    }
    for(uint32_t i=0;i<2u;i++){
        fg_vk_tensor_destroy(embedding[i]);fg_vk_tensor_destroy(input[i]);
    }
    fg_mtp_head_close(reference[1]);fg_mtp_head_close(reference[0]);
    fg_vk_close(reference_vk);return status;
}

static int native_only(const char *path,const char *overlay,uint32_t plus_one){
    fg_manifest *manifest=malloc(sizeof(*manifest));fg_error err={0};
    if(!manifest)return 1;
    fg_vk_context *vk=NULL;fg_mtp_head *head=NULL,*second=NULL;
    fg_status status=fg_manifest_read(path,manifest,&err);
    uint32_t selected=0u,steps=0u;
    fg_runtime_options requested={.experimental_flags=FG_RUNTIME_EXPERIMENTAL_MTP,
        .mtp_overlay=overlay,.mtp_embedding="allocation-probe-unused",
        .mtp_native_rank_plus_one=plus_one},resolved;
    if(status==FG_OK)status=fg_runtime_options_resolve(&resolved,manifest,&requested,&err);
    if(status==FG_OK)status=fg_mtp_native_owner_resolve(manifest,plus_one,&selected,&err);
    if(status==FG_OK)status=fg_vk_open(&vk,&err);
    if(status==FG_OK)status=fg_mtp_head_open_owner_windowed(&head,vk,manifest,
        selected,selected,overlay,
        manifest->session.logical_context_tokens,8192u,&err);
    fg_vk_memory_stats before={0},after={0};
    if(status==FG_OK){
        fg_vk_get_memory_stats(vk,&before);
        status=fg_mtp_head_fork(&second,head,1u,&err);
    }
    if(status==FG_OK){
        fg_vk_get_memory_stats(vk,&after);
        if(selected!=manifest->layer_owner[FG_LAYER_COUNT-1u]){
            fg_mtp_head *placed[2]={head,second};
            status=compare_owner(vk,manifest,overlay,placed,&steps,&err);
        }
    }
    if(status==FG_OK){
        fg_vk_memory_stats warmed;
        fg_vk_get_memory_stats(vk,&warmed);
        printf("{\"kind\":\"isolated_native_allocation_not_full_rank_fit\","
            "\"native_owner\":%u,\"exact_default_equivalence_steps\":%u,"
            "\"weights\":%llu,\"first_total\":%llu,\"both_total\":%llu,"
            "\"private_second\":%llu,\"warm_pair_total\":%llu,\"kv_each\":33570816}\n",
            selected,steps,(unsigned long long)fg_mtp_head_weight_bytes(head),
            (unsigned long long)before.requested_live_bytes,
            (unsigned long long)after.requested_live_bytes,
            (unsigned long long)(after.requested_live_bytes-before.requested_live_bytes),
            (unsigned long long)warmed.requested_live_bytes);
    }else fprintf(stderr,"native-only allocation failed: %s\n",err.message);
    fg_mtp_head_close(second);fg_mtp_head_close(head);fg_vk_close(vk);
    free(manifest);return status==FG_OK?0:1;
}

int main(int argc,char **argv){
    if((argc==4||argc==6)&&!strcmp(argv[1],"--native-only")){
        uint32_t plus_one=0u;
        if(argc==6){
            char *end=NULL;unsigned long rank=strtoul(argv[5],&end,10);
            if(strcmp(argv[4],"--native-rank")||!argv[5][0]||*end||rank>=FG_RANK_COUNT)
                return 2;
            plus_one=(uint32_t)rank+1u;
        }
        return native_only(argv[2],argv[3],plus_one);
    }
    if((argc!=5&&argc!=7)||(strcmp(argv[4],"1")&&strcmp(argv[4],"2"))){
        fprintf(stderr,"usage: mtp-residency MANIFEST OVERLAY EMBEDDING SESSIONS(1|2) [--native-rank N]\n");
        return 2;
    }
    uint32_t plus_one=0u;
    if(argc==7){
        char *end=NULL;unsigned long rank=strtoul(argv[6],&end,10);
        if(strcmp(argv[5],"--native-rank")||!argv[6][0]||*end||rank>=FG_RANK_COUNT)return 2;
        plus_one=(uint32_t)rank+1u;
    }
    fg_error err={0};
    fg_status status=fg_runtime_probe_mtp_residency_owner(
        argv[1],argv[2],argv[3],(uint32_t)(argv[4][0]-'0'),plus_one,&err);
    if(status!=FG_OK)fprintf(stderr,"MTP_RESIDENCY status=%u %s\n",status,err.message);
    else puts("MTP_RESIDENCY ready: all requested target/native sessions pre-provisioned");
    return status==FG_OK?0:status==FG_ERR_LIMIT?3:1;
}
