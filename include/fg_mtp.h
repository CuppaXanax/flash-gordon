#ifndef FLASH_GORDON_MTP_H
#define FLASH_GORDON_MTP_H

#include "fg_sampler.h"

#define FG_MTP_MAX_DRAFT 3u
#define FG_MTP_MAX_VERIFY (FG_MTP_MAX_DRAFT+1u)

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
