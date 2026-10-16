#include "fg_mtp.h"

#include <math.h>
#include <string.h>

fg_status fg_mtp_cycle_greedy(const fg_mtp_ops *ops,uint32_t pending,
    const uint32_t *drafts,uint32_t draft_count,uint32_t output_limit,
    uint32_t eos_token,fg_mtp_result *result,fg_error *err){
    if(!ops||!ops->checkpoint||!ops->verify||!ops->restore||!ops->replay||
       !ops->commit||!drafts||!draft_count||draft_count>FG_MTP_MAX_DRAFT||
       !output_limit||!result){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid MTP verification cycle");
        return FG_ERR_ARGUMENT;
    }
    memset(result,0,sizeof(*result));
    uint32_t inputs[FG_MTP_MAX_VERIFY],target[FG_MTP_MAX_VERIFY];
    inputs[0]=pending;
    memcpy(inputs+1u,drafts,draft_count*sizeof(*drafts));
    fg_status status=ops->checkpoint(ops->context,err);
    if(status!=FG_OK)return status;
    status=ops->verify(ops->context,inputs,draft_count+1u,target,err);
    fg_mtp_result next={0};
    if(status==FG_OK){
        while(next.accepted<draft_count&&
              drafts[next.accepted]==target[next.accepted]){
            next.tokens[next.count++]=drafts[next.accepted++];
            if(next.count==output_limit||next.tokens[next.count-1u]==eos_token)break;
        }
        if(next.count<output_limit&&
           (!next.count||next.tokens[next.count-1u]!=eos_token))
            next.tokens[next.count++]=target[next.accepted];
        /* The last emitted token is pending, not consumed.  The already
         * emitted input at inputs[0] balances that one-token lag. */
        next.consumed=next.count;
        if(next.consumed<draft_count+1u){
            status=ops->restore(ops->context,err);
            if(status==FG_OK)status=ops->replay(ops->context,inputs,next.consumed,err);
        }
        if(status==FG_OK)status=ops->commit(ops->context,err);
    }
    if(status!=FG_OK){
        fg_error rollback_error={0};
        fg_status rollback=ops->restore(ops->context,&rollback_error);
        if(rollback!=FG_OK){
            fg_error_set(err,rollback,"MTP restore failed after error %d: %s",
                         (int)status,rollback_error.message);
            return rollback;
        }
        return status;
    }
    *result=next;
    return FG_OK;
}

fg_status fg_mtp_rejection_sample(const float *p,const float *q,uint32_t vocab,
    uint32_t draft,float accept_uniform,float residual_uniform,
    bool *accepted,uint32_t *token,fg_error *err){
    if(!p||!q||!vocab||draft>=vocab||!accepted||!token||
       !isfinite(accept_uniform)||accept_uniform<0.0f||accept_uniform>=1.0f||
       !isfinite(residual_uniform)||residual_uniform<0.0f||residual_uniform>=1.0f){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid speculative sampling arguments");
        return FG_ERR_ARGUMENT;
    }
    double psum=0.0,qsum=0.0,residual=0.0;
    for(uint32_t i=0;i<vocab;i++){
        if(!isfinite(p[i])||!isfinite(q[i])||p[i]<0.0f||q[i]<0.0f){
            fg_error_set(err,FG_ERR_ARGUMENT,"invalid speculative probability");
            return FG_ERR_ARGUMENT;
        }
        psum+=p[i];qsum+=q[i];
        residual+=fmax((double)p[i]-q[i],0.0);
    }
    if(fabs(psum-1.0)>1e-5||fabs(qsum-1.0)>1e-5||q[draft]<=0.0f){
        fg_error_set(err,FG_ERR_ARGUMENT,"speculative distributions are not normalized");
        return FG_ERR_ARGUMENT;
    }
    if((double)accept_uniform*q[draft]<p[draft]){
        *accepted=true;*token=draft;
        return FG_OK;
    }
    if(!(residual>0.0)){
        fg_error_set(err,FG_ERR_FORMAT,"rejected speculative draw has no residual mass");
        return FG_ERR_FORMAT;
    }
    double draw=(double)residual_uniform*residual,cumulative=0.0;
    uint32_t last=0u;
    for(uint32_t i=0;i<vocab;i++){
        double mass=fmax((double)p[i]-q[i],0.0);
        if(mass<=0.0)continue;
        last=i;cumulative+=mass;
        if(draw<cumulative){*accepted=false;*token=i;return FG_OK;}
    }
    *accepted=false;*token=last;
    return FG_OK;
}
