#ifndef FLASH_GORDON_MTP_H
#define FLASH_GORDON_MTP_H

#include "fg_sampler.h"
#include "fg_manifest.h"

#define FG_MTP_MAX_DRAFT 3u
#define FG_MTP_MAX_VERIFY (FG_MTP_MAX_DRAFT+1u)

/* Zero preserves the final owner. Relocation is only to its immediate,
 * non-coordinator predecessor, which can grant one downstream frame credit. */
static inline fg_status fg_mtp_native_owner_resolve(const fg_manifest *m,
    uint32_t plus_one,uint32_t *owner,fg_error *err){
    if(!m||!owner||plus_one>m->rank_count||!m->rank_count||
       m->rank_count>FG_RANK_COUNT)goto invalid;
    uint32_t final=m->layer_owner[FG_LAYER_COUNT-1u];
    uint32_t selected=plus_one?plus_one-1u:final;
    if(final>=m->rank_count||selected>=m->rank_count)goto invalid;
    if(selected!=final){
        if(!selected)goto invalid;
        uint32_t first=FG_LAYER_COUNT-1u;
        while(first&&m->layer_owner[first-1u]==final)first--;
        if(!first||m->layer_owner[first-1u]!=selected)goto invalid;
        for(uint32_t l=0;l<first;l++)if(m->layer_owner[l]==final)goto invalid;
        uint32_t predecessor=first;
        while(predecessor&&m->layer_owner[predecessor-1u]==selected)predecessor--;
        for(uint32_t l=0;l<predecessor;l++)if(m->layer_owner[l]==selected)goto invalid;
    }
    *owner=selected;return FG_OK;
invalid:
    fg_error_set(err,FG_ERR_ARGUMENT,"MTP native owner must be the final block or its remote predecessor");
    return FG_ERR_ARGUMENT;
}

typedef struct fg_mtp_policy {
    double acceptance[FG_MTP_MAX_DRAFT];
    double round_ms[FG_MTP_MAX_DRAFT];
    uint64_t observations[FG_MTP_MAX_DRAFT];
    uint64_t rounds;
} fg_mtp_policy;

/* A policy belongs to one stream. Ratios describe N=2,3,4 target service;
 * observed end-to-end round times replace these initialization estimates. */
fg_status fg_mtp_policy_init(fg_mtp_policy *policy,double serial_ms,
    double draft_ms,const double verify_ratios[FG_MTP_MAX_DRAFT],fg_error *err);
uint32_t fg_mtp_policy_choose(const fg_mtp_policy *policy,uint32_t max_draft);
uint32_t fg_mtp_policy_draft_width(const fg_mtp_policy *policy,uint32_t max_draft,
                                  uint32_t fixed_draft);
fg_status fg_mtp_policy_observe(fg_mtp_policy *policy,uint32_t drafted,
    uint32_t accepted,double round_ms,fg_error *err);

typedef struct fg_mtp_result {
    uint32_t tokens[FG_MTP_MAX_VERIFY];
    uint32_t count;
    uint32_t accepted;
    uint32_t consumed;
} fg_mtp_result;

/* Verification advances one causal sequence, not independent depth-B slots.
 * Restore must undo GDN, PLE, QSA and unpublished page/index side effects.
 * Replay consumes only the retained input prefix with the target decoder.
 * No token may be published before commit succeeds. */
typedef struct fg_mtp_ops {
    fg_status (*checkpoint)(void *context,fg_error *err);
    fg_status (*verify)(void *context,const uint32_t *inputs,uint32_t count,
                        uint32_t *target_tokens,fg_error *err);
    fg_status (*restore)(void *context,fg_error *err);
    fg_status (*replay)(void *context,const uint32_t *inputs,uint32_t count,
                        fg_error *err);
    fg_status (*commit)(void *context,fg_error *err);
    void *context;
    /* Optional projected-state commit; count includes the pending input.
     * When present this replaces restore+replay, not the final commit hook. */
    fg_status (*retain)(void *context,uint32_t count,fg_error *err);
} fg_mtp_ops;

fg_status fg_mtp_cycle_greedy(const fg_mtp_ops *ops,uint32_t pending,
                             const uint32_t *drafts,uint32_t draft_count,
                             uint32_t output_limit,uint32_t eos_token,
                             fg_mtp_result *result,fg_error *err);

/* p and q are the complete normalized *sampling* distributions after all
 * controls/penalties, not raw logits or unnormalized top-k scores.
 * On rejection, draw from normalized max(p-q,0); on acceptance emit draft.
 * Uniforms are independent draws in [0,1). */
fg_status fg_mtp_rejection_sample(const float *p,const float *q,uint32_t vocab,
                                  uint32_t draft,float accept_uniform,
                                  float residual_uniform,bool *accepted,
                                  uint32_t *token,fg_error *err);

#endif
