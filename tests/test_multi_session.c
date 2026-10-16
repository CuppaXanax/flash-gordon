#include <sys/stat.h>
#include "../src/runtime.c"

#define REQUIRE(expression) do { \
    if(!(expression)){ \
        fprintf(stderr,"FAIL line %d: %s (%s)\n",__LINE__,#expression,error.message); \
        result=1;goto cleanup; \
    } \
} while(0)

int main(void){
    int result=0;
    fg_error error={0};
    fg_manifest *manifest=malloc(sizeof(*manifest));
    qsa_owner_runtime sessions[2]={0};
    bool mutex_ready[2]={false,false};
    fg_qsa_state *owner_handle=NULL;
    char directory[128];
    snprintf(directory,sizeof(directory),".build/multisess-state-%u",(unsigned)getpid());
    if(mkdir(".build",0700)!=0&&errno!=EEXIST)return 1;
    if(mkdir(directory,0700)!=0){free(manifest);return 1;}
    REQUIRE(manifest!=NULL);
    fg_manifest_init(manifest);
    manifest->protocol_version=FG_PROTOCOL_VERSION;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        manifest->layer_owner[layer]=2u;
    manifest->layer_owner[3u]=1u;
    fg_session_identity identity={0};
    fg_owner_session_control controls[2];
    memset(controls,0,sizeof(controls));
    for(uint32_t slot=0;slot<2u;slot++){
        sessions[slot].enabled=true;
        sessions[slot].layer_count=1u;sessions[slot].layers[0]=3u;
        fg_qsa_owner_guard_init(&sessions[slot].guard,1u);
        REQUIRE(pthread_mutex_init(&sessions[slot].writer_mutex,NULL)==0);
        mutex_ready[slot]=true;
        controls[slot]=(fg_owner_session_control){
            .version=FG_OWNER_SESSION_CONTROL_VERSION,.operation=FG_OWNER_SESSION_BEGIN,
            .rank=1u,.state_slot=(uint8_t)slot,.session_nonce=10u+slot,
            .logical_context_tokens=32768u,.gpu_index_tokens=32768u,
            .qsa_page_cache_bytes=UINT64_C(16)<<20u};
        memcpy(controls[slot].state_format_sha256,
            manifest->session.rank_state_format_sha256[1u],32u);
        fg_status status=qsa_owner_open_session(&sessions[slot],manifest,directory,
            &identity,controls[slot].session_nonce,&controls[slot],false,&error);
        if(status==FG_ERR_UNAVAILABLE){result=77;goto cleanup;}
        REQUIRE(status==FG_OK);
    }
    REQUIRE(strstr(sessions[1].state_path,"-s1.state")!=NULL);
    uint8_t other[FG_QSA_PAGE_RECORD_BYTES],changed[FG_QSA_PAGE_RECORD_BYTES];
    uint8_t actual[FG_QSA_PAGE_RECORD_BYTES];
    memset(other,0x5a,sizeof(other));memset(changed,0xa5,sizeof(changed));
    REQUIRE(fg_qsa_state_write_block(sessions[1].state,0u,0u,other,4u,&error)==FG_OK);
    REQUIRE(fg_qsa_state_open(&owner_handle,sessions[0].state_path,sessions[0].layers,
        1u,32768u,false,&error)==FG_OK);
    struct stat before,after;
    REQUIRE(stat(sessions[0].state_path,&before)==0);
    controls[0].session_nonce++;
    REQUIRE(qsa_owner_open_session(&sessions[0],manifest,directory,&identity,
        controls[0].session_nonce,&controls[0],true,&error)==FG_OK);
    REQUIRE(stat(sessions[0].state_path,&after)==0&&before.st_ino==after.st_ino);
    REQUIRE(fg_qsa_state_write_block(owner_handle,0u,0u,changed,4u,&error)==FG_OK);
    uint32_t committed=0;
    REQUIRE(fg_qsa_state_read_block(sessions[0].state,0u,0u,actual,&committed,&error)==FG_OK);
    REQUIRE(committed==4u&&!memcmp(actual,changed,sizeof(actual)));
    REQUIRE(fg_qsa_state_read_block(sessions[1].state,0u,0u,actual,&committed,&error)==FG_OK);
    REQUIRE(committed==4u&&!memcmp(actual,other,sizeof(actual)));
    REQUIRE(sessions[1].guard.session_nonce==11u);
    REQUIRE(qsa_owner_open_session(&sessions[0],manifest,directory,&identity,
        controls[0].session_nonce,&controls[0],true,&error)==FG_ERR_MISMATCH);
cleanup:
    fg_qsa_state_close(owner_handle);
    for(uint32_t slot=0;slot<2u;slot++){
        if(mutex_ready[slot])pthread_mutex_destroy(&sessions[slot].writer_mutex);
        qsa_owner_runtime_destroy(&sessions[slot]);
    }
    rmdir(directory);free(manifest);
    if(!result)puts("Independent QSA namespaces, retained inode and reset isolation: PASS");
    return result;
}
