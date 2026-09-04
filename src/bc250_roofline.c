#include "fg_bc250_roofline.h"

#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>

static int append(char *out,size_t cap,size_t *used,const char *format,...){
    if(*used>=cap)return 0;
    va_list args;va_start(args,format);
    int n=vsnprintf(out+*used,cap-*used,format,args);
    va_end(args);
    if(n<0||(size_t)n>=cap-*used)return 0;
    *used+=(size_t)n;return 1;
}

static int append_string(char *out,size_t cap,size_t *used,const char *value){
    if(!append(out,cap,used,"\""))return 0;
    if(value)for(const unsigned char *p=(const unsigned char *)value;*p;p++){
        if(*p=='"'||*p=='\\'){
            if(!append(out,cap,used,"\\%c",*p))return 0;
        }else if(*p<0x20u){
            if(!append(out,cap,used,"\\u%04x",(unsigned)*p))return 0;
        }else if(!append(out,cap,used,"%c",*p))return 0;
    }
    return append(out,cap,used,"\"");
}

static int append_u64(char *out,size_t cap,size_t *used,uint64_t value){
    return value==UINT64_MAX?append(out,cap,used,"null"):
        append(out,cap,used,"%" PRIu64,value);
}

static int append_double(char *out,size_t cap,size_t *used,double value){
    return isfinite(value)?append(out,cap,used,"%.9g",value):append(out,cap,used,"null");
}

int fg_bc250_q8_0_weight_bytes(uint32_t input_width,uint32_t output_width,
                               uint64_t *bytes){
    if(!bytes||!input_width||!output_width||input_width%32u)return 0;
    uint64_t blocks=input_width/32u;
    if(blocks>UINT64_MAX/34u)return 0;
    uint64_t row=blocks*34u;
    if(row>UINT64_MAX/output_width)return 0;
    *bytes=row*output_width;return 1;
}

int fg_bc250_linear_accounting(uint64_t source_bytes,int copy,
                               uint64_t *bytes_read,uint64_t *bytes_written,
                               uint64_t *operations){
    if(!bytes_read||!bytes_written||!operations||source_bytes%4u)return 0;
    *bytes_read=source_bytes;
    *bytes_written=copy?source_bytes:((source_bytes/4u+255u)/256u)*4u;
    *operations=source_bytes/4u;
    return 1;
}

int fg_bc250_q8_0_logical_traffic(uint32_t input_width,uint32_t output_width,
                                 uint32_t tokens,int cooked,
                                 uint64_t *bytes_read,uint64_t *bytes_written,
                                 uint64_t *operations){
    uint64_t weight_bytes=0;
    if(!tokens||!bytes_read||!bytes_written||!operations||
       !fg_bc250_q8_0_weight_bytes(input_width,output_width,&weight_bytes))return 0;
    uint64_t weight_passes=cooked?(tokens+31u)/32u:tokens;
    if(weight_bytes>UINT64_MAX/weight_passes)return 0;
    uint64_t input_bytes=(uint64_t)input_width*tokens*4u;
    uint64_t output_bytes=(uint64_t)output_width*tokens*4u;
    uint64_t flops=UINT64_C(2)*input_width*output_width*tokens;
    if(weight_bytes*weight_passes>UINT64_MAX-input_bytes)return 0;
    *bytes_read=weight_bytes*weight_passes+input_bytes;
    *bytes_written=output_bytes;
    *operations=flops;
    return 1;
}

int fg_bc250_roofline_record_format(char *output,size_t capacity,
                                    const fg_bc250_roofline_record *record){
    if(!output||!capacity||!record||!record->benchmark||!record->name||
       !record->shape||!record->access_pattern||!record->device)return 0;
    size_t used=0;
    if(!append(output,capacity,&used,"{\"schema\":\"%s\",\"benchmark\":",FG_BC250_ROOFLINE_SCHEMA)||
       !append_string(output,capacity,&used,record->benchmark)||
       !append(output,capacity,&used,",\"name\":")||
       !append_string(output,capacity,&used,record->name)||
       !append(output,capacity,&used,",\"shape\":")||
       !append_string(output,capacity,&used,record->shape)||
       !append(output,capacity,&used,",\"access_pattern\":")||
       !append_string(output,capacity,&used,record->access_pattern)||
       !append(output,capacity,&used,",\"batch\":%u,\"tokens\":%u,\"iterations\":%u",
              record->batch,record->tokens,record->iterations)||
       !append(output,capacity,&used,",\"bytes_read\":")||
       !append_u64(output,capacity,&used,record->bytes_read)||
       !append(output,capacity,&used,",\"bytes_written\":")||
       !append_u64(output,capacity,&used,record->bytes_written)||
       !append(output,capacity,&used,",\"operations\":")||
       !append_u64(output,capacity,&used,record->operations)||
       !append(output,capacity,&used,",\"gpu_ms\":")||
       !append_double(output,capacity,&used,record->gpu_ms)||
       !append(output,capacity,&used,",\"wall_ms\":")||
       !append_double(output,capacity,&used,record->wall_ms)||
       !append(output,capacity,&used,",\"derived_gbps\":")||
       !append_double(output,capacity,&used,record->derived_gbps)||
       !append(output,capacity,&used,",\"derived_flops\":")||
       !append_double(output,capacity,&used,record->derived_flops)||
       !append(output,capacity,&used,",\"device\":")||
       !append_string(output,capacity,&used,record->device)||
       !append(output,capacity,&used,"}\n"))return 0;
    return (int)used;
}
