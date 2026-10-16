#include "../src/api.c"
#include "mtp_ring_sse.h"

struct fixture_sse {
    int pair[2];
    api_chat_request request;
    api_generation generation;
};
fg_status fixture_sse_open(fixture_sse **out,const char *stop,fg_error *err){
    *out=calloc(1,sizeof(**out));if(!*out)return FG_ERR_OOM;
    fixture_sse *s=*out;
    if(socketpair(AF_UNIX,SOCK_STREAM,0,s->pair)){
        free(s);*out=NULL;return FG_ERR_IO;
    }
    s->request.stream=true;
    if(stop){s->request.stop[0]=strdup(stop);s->request.stop_count=1u;
        if(!s->request.stop[0]){close(s->pair[0]);close(s->pair[1]);free(s);*out=NULL;return FG_ERR_OOM;}}
    s->generation=(api_generation){.sink={.fd=s->pair[0]},.stream=true,
        .think_closed=true,.id="mtp-ring-fixture",.model="synthetic-mtp-fixture",
        .request=&s->request};
    fg_status status=api_send_sse_headers(&s->generation.sink,err);
    if(status==FG_OK)status=send_stream_start(&s->generation,err);
    return status;
}
bool fixture_sse_stopped(fixture_sse *s){return api_interrupted(&s->generation);}
fg_status fixture_sse_token(void *opaque,uint32_t token,const char *text,size_t bytes,
    fg_error *err){
    fixture_sse *s=opaque;return api_token(&s->generation,token,text,bytes,err);
}
fg_status fixture_sse_finish(fixture_sse *s,fg_error *err){
    fg_chat_generated generated={0};
    fg_status status=fg_chat_parse_generated(s->generation.content.data?
        s->generation.content.data:"",false,&generated,err);
    const char *reason=s->generation.stop_matched?"stop":"length";
    if(status==FG_OK)status=send_stream_end(&s->generation,&generated,reason,err);
    shutdown(s->pair[0],SHUT_WR);
    char response[16384];size_t used=0u;
    while(used+1u<sizeof(response)){
        ssize_t bytes=read(s->pair[1],response+used,sizeof(response)-used-1u);
        if(bytes<0&&errno==EINTR)continue;
        if(bytes<=0)break;
        used+=(size_t)bytes;
    }
    response[used]=0;
    api_buffer published={0};
    for(char *frame=strstr(response,"data: ");frame;){
        char *end=strstr(frame,"\n\n");
        if(!end)break;
        if(!strncmp(frame+6u,"[DONE]",6u))break;
        json_value *root=parse_json_body(frame+6u,(size_t)(end-frame-6u),err);
        json_value *choices=root?json_object_get(root,"choices"):NULL;
        json_value *choice=choices&&choices->type==JSON_ARRAY&&choices->as.array.count?
            choices->as.array.items[0]:NULL;
        json_value *delta=choice?json_object_get(choice,"delta"):NULL;
        json_value *content=delta?json_object_get(delta,"content"):NULL;
        if(!root)status=FG_ERR_FORMAT;
        if(content&&content->type==JSON_STRING&&status==FG_OK)
            status=buffer_append(&published,content->as.string,err);
        json_free(root);frame=strstr(end+2u,"data: ");
    }
    if(status==FG_OK&&(!strstr(response,"HTTP/1.1 200 OK")||
       !strstr(response,"text/event-stream")||!strstr(response,"data: [DONE]\n\n")||
       strcmp(published.data?published.data:"",s->generation.content.data?
           s->generation.content.data:"")||
       !strstr(response,s->generation.stop_matched?
           "\"finish_reason\":\"stop\"":"\"finish_reason\":\"length\""))){
        fg_error_set(err,FG_ERR_MISMATCH,"multi-token SSE framing failed");status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK&&s->generation.stop_matched&&
       (strstr(s->generation.content.data?s->generation.content.data:"",s->request.stop[0])||
        s->generation.content.length!=1u)){
        fg_error_set(err,FG_ERR_MISMATCH,"stop sequence or committed suffix leaked");status=FG_ERR_MISMATCH;
    }
    fg_chat_generated_free(&generated);free(published.data);free(s->generation.content.data);
    free(s->generation.visible_pending.data);free(s->generation.stop_pending.data);
    free(s->request.stop[0]);
    close(s->pair[0]);close(s->pair[1]);free(s);return status;
}
