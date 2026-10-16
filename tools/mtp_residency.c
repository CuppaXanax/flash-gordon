#include "fg_runtime.h"
#include "fg_mtp_head.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int native_only(const char *path,const char *overlay){
    fg_manifest *manifest=malloc(sizeof(*manifest));fg_error err={0};
    if(!manifest)return 1;
    fg_vk_context *vk=NULL;fg_mtp_head *head=NULL,*second=NULL;
    fg_status status=fg_manifest_read(path,manifest,&err);
    if(status==FG_OK)status=fg_vk_open(&vk,&err);
    if(status==FG_OK)status=fg_mtp_head_open_windowed(&head,vk,manifest,
        manifest->layer_owner[FG_LAYER_COUNT-1u],overlay,
        manifest->session.logical_context_tokens,8192u,&err);
    fg_vk_memory_stats before={0},after={0};
    if(status==FG_OK){
        fg_vk_get_memory_stats(vk,&before);
        status=fg_mtp_head_fork(&second,head,1u,&err);
    }
    if(status==FG_OK){
        fg_vk_get_memory_stats(vk,&after);
        printf("{\"kind\":\"isolated_native_allocation_not_full_rank_fit\","
            "\"weights\":%llu,\"first_total\":%llu,\"both_total\":%llu,"
            "\"private_second\":%llu,\"kv_each\":33570816}\n",
            (unsigned long long)fg_mtp_head_weight_bytes(head),
            (unsigned long long)before.requested_live_bytes,
            (unsigned long long)after.requested_live_bytes,
            (unsigned long long)(after.requested_live_bytes-before.requested_live_bytes));
    }else fprintf(stderr,"native-only allocation failed: %s\n",err.message);
    fg_mtp_head_close(second);fg_mtp_head_close(head);fg_vk_close(vk);
    free(manifest);return status==FG_OK?0:1;
}

int main(int argc,char **argv){
    if(argc==4&&!strcmp(argv[1],"--native-only"))
        return native_only(argv[2],argv[3]);
    if(argc!=5||(strcmp(argv[4],"1")&&strcmp(argv[4],"2"))){
        fprintf(stderr,"usage: mtp-residency MANIFEST OVERLAY EMBEDDING SESSIONS(1|2)\n");
        return 2;
    }
    fg_error err={0};
    fg_status status=fg_runtime_probe_mtp_residency(
        argv[1],argv[2],argv[3],(uint32_t)(argv[4][0]-'0'),&err);
    if(status!=FG_OK)fprintf(stderr,"MTP_RESIDENCY status=%u %s\n",status,err.message);
    else puts("MTP_RESIDENCY ready: all requested target/native sessions pre-provisioned");
    return status==FG_OK?0:status==FG_ERR_LIMIT?3:1;
}
