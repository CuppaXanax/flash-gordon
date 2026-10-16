#include "fg_mtp.h"

#include <math.h>
#include <string.h>

fg_status fg_mtp_policy_init(fg_mtp_policy *policy,double serial_ms,
    double draft_ms,const double verify_ratios[FG_MTP_MAX_DRAFT],fg_error *err){
    if(!policy||!verify_ratios||!isfinite(serial_ms)||serial_ms<=0.0||
       !isfinite(draft_ms)||draft_ms<0.0){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid MTP policy costs");return FG_ERR_ARGUMENT;
    }
    fg_mtp_policy next={.acceptance={0.80,0.78,0.77}};
    for(uint32_t j=0;j<FG_MTP_MAX_DRAFT;j++){
        double cost=serial_ms*verify_ratios[j]+(j+1u)*draft_ms;
        if(!isfinite(verify_ratios[j])||verify_ratios[j]<=0.0||
           !isfinite(cost)||cost<=0.0){
            fg_error_set(err,FG_ERR_ARGUMENT,"invalid MTP verification cost");
            return FG_ERR_ARGUMENT;
        }
        next.round_ms[j]=cost;
    }
    *policy=next;return FG_OK;
}

uint32_t fg_mtp_policy_choose(const fg_mtp_policy *policy,uint32_t max_draft){
    if(!policy||!max_draft)return 0u;
    if(max_draft>FG_MTP_MAX_DRAFT)max_draft=FG_MTP_MAX_DRAFT;
    /* Periodically sample every shape, including shorter shapes after a
     * workload change. No proposal choice changes the target distribution. */
    if((policy->rounds%32u)==0u)
        return 1u+(uint32_t)((policy->rounds/32u)%max_draft);
    double prefix=1.0,yield=1.0,best=-1.0;
    uint32_t chosen=1u;
    for(uint32_t j=0;j<max_draft;j++){
        prefix*=policy->acceptance[j];yield+=prefix;
        double goodput=yield/policy->round_ms[j];
        if(goodput>best){best=goodput;chosen=j+1u;}
    }
    return chosen;
}

uint32_t fg_mtp_policy_draft_width(const fg_mtp_policy *policy,uint32_t max_draft,
                                  uint32_t fixed_draft){
    if(max_draft>FG_MTP_MAX_DRAFT)max_draft=FG_MTP_MAX_DRAFT;
    return fixed_draft?(fixed_draft<max_draft?fixed_draft:max_draft):
        fg_mtp_policy_choose(policy,max_draft);
}

fg_status fg_mtp_policy_observe(fg_mtp_policy *policy,uint32_t drafted,
    uint32_t accepted,double round_ms,fg_error *err){
    if(!policy||!drafted||drafted>FG_MTP_MAX_DRAFT||accepted>drafted||
       !isfinite(round_ms)||round_ms<=0.0){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid MTP policy observation");return FG_ERR_ARGUMENT;
    }
    const double alpha=1.0/32.0;
    for(uint32_t j=0;j<drafted&&j<=accepted;j++){
        double matched=j<accepted?1.0:0.0;
        policy->acceptance[j]+=alpha*(matched-policy->acceptance[j]);
    }
    uint32_t shape=drafted-1u;
    if(!policy->observations[shape])policy->round_ms[shape]=round_ms;
    else policy->round_ms[shape]+=alpha*(round_ms-policy->round_ms[shape]);
    policy->observations[shape]++;policy->rounds++;
    return FG_OK;
}

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
