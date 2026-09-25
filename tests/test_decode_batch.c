#include "fg_decode_batch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);failures++;}}while(0)

/* ------------------------------------------------------------------ policy */

static void test_policy(void){
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    CHECK(policy.max_batch==FG_DECODE_BATCH_MAX);
    CHECK(policy.qsa_layer_count==12u);

    uint64_t short_scan=fg_decode_batch_qsa_bytes_per_token(&policy,4096u);
    uint64_t long_scan=fg_decode_batch_qsa_bytes_per_token(&policy,262144u);
    CHECK(short_scan>94u*1000u*1000u&&short_scan<96u*1000u*1000u);
    CHECK(long_scan>552u*1000u*1000u&&long_scan<554u*1000u*1000u);

    CHECK(fg_decode_batch_choose_batch(&policy,4096u,2u)==2u);
    CHECK(fg_decode_batch_choose_batch(&policy,32768u,2u)==2u);
    CHECK(fg_decode_batch_choose_batch(&policy,262144u,2u)==1u);
    CHECK(fg_decode_batch_choose_batch(&policy,4096u,1u)==1u);

    double speedup2=fg_decode_batch_speedup(&policy,2u,4096u);
    double speedup4=fg_decode_batch_speedup(&policy,4u,4096u);
    CHECK(speedup2>1.8&&speedup2<2.0);
    CHECK(speedup4>3.2&&speedup4<3.5);
    CHECK(fg_decode_batch_speedup(&policy,2u,4096u)>
          fg_decode_batch_speedup(&policy,1u,4096u));
    CHECK(fg_decode_batch_speedup(&policy,1u,262144u)>0.0);
}

/* ------------------------------------------------------- table and schedule */

static uint32_t positions(uint32_t base,uint32_t out[4]){
    for(uint32_t axis=0;axis<4u;axis++)out[axis]=base;
    return base;
}

static void test_table_and_schedule(void){
    fg_error err={0};
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,2u,&err)==FG_OK);
    CHECK(fg_decode_batch_table_init(&table,0u,&err)!=FG_OK);
    CHECK(fg_decode_batch_table_init(&table,3u,&err)!=FG_OK);

    uint32_t pos[4];positions(7u,pos);
    CHECK(fg_decode_batch_sequence_enter(&table,11u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,11u,1u,&err)!=FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,22u,0u,&err)!=FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,22u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,33u,2u,&err)!=FG_OK);
    CHECK(fg_decode_batch_table_find(&table,22u)==1u);
    CHECK(fg_decode_batch_table_find(&table,99u)==FG_DECODE_BATCH_INVALID_SLOT);

    CHECK(fg_decode_batch_sequence_frontier(&table,11u,4096u,4096u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,22u,64u,64u,pos,&err)==FG_OK);
    CHECK(table.sequences[0].qsa_records[3u]==4096u);
    CHECK(table.sequences[0].qsa_records[0u]==0u);

    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    fg_decode_batch batch={0};
    CHECK(fg_decode_batch_schedule(&table,&policy,10u,&batch,&err)==FG_ERR_UNAVAILABLE);
    CHECK(fg_decode_batch_sequence_ready(&table,11u,10u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,22u,20u,&err)==FG_OK);
    CHECK(fg_decode_batch_schedule(&table,&policy,30u,&batch,&err)==FG_OK);
    /* 11 is at 4K (B=2 fits) but 22 is short; worst case 4K, oldest first. */
    CHECK(batch.slot_count==2u);
    CHECK(table.sequences[batch.slots[0u].sequence].sequence_id==11u);
    CHECK(table.sequences[batch.slots[1u].sequence].sequence_id==22u);
    CHECK(batch.slots[0u].state_slot==0u&&batch.slots[1u].state_slot==1u);
    CHECK(batch.slots[0u].token_index==4096u&&batch.slots[1u].token_index==64u);
    CHECK(table.sequences[0].in_flight&&table.sequences[1].in_flight);
    CHECK(!table.sequences[0].ready&&!table.sequences[1].ready);
    CHECK(fg_decode_batch_schedule(&table,&policy,31u,&batch,&err)==FG_ERR_UNAVAILABLE);
}

static void test_shrink_and_fairness(void){
    fg_error err={0};
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,2u,&err)==FG_OK);
    uint32_t pos[4];positions(0u,pos);
    CHECK(fg_decode_batch_sequence_enter(&table,1u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,2u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,3u,0u,&err)!=FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,1u,262144u,262144u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,2u,32u,32u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,1u,100u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,2u,50u,&err)==FG_OK);
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    fg_decode_batch batch={0};
    /* Long context dominates the token budget: B shrinks to 1, oldest ready
     * first (the short sequence), while the long one stays queued. */
    CHECK(fg_decode_batch_schedule(&table,&policy,200u,&batch,&err)==FG_OK);
    CHECK(batch.slot_count==1u);
    CHECK(table.sequences[batch.slots[0u].sequence].sequence_id==2u);
    CHECK(!table.sequences[0].in_flight&&table.sequences[1].in_flight);
    /* The long sequence still owns state slot 0; a third sequence cannot steal
     * a bound slot. */
    CHECK(fg_decode_batch_sequence_enter(&table,3u,0u,&err)!=FG_OK);
}

/* -------------------------------------------------------------- transaction */

typedef struct mock_ops {
    int prepare_calls,commit_calls,restore_calls;
    int fail_prepare_at,fail_commit_at,fail_restore;
    uint64_t last_sequence;
    uint32_t last_state_slot;
    int order[16],order_count;
} mock_ops;

static void mock_record(mock_ops *mock,int code,uint64_t sequence,uint32_t state_slot){
    if(mock->order_count<16)mock->order[mock->order_count++]=code;
    mock->last_sequence=sequence;
    mock->last_state_slot=state_slot;
}

static fg_status mock_prepare(void *context,uint64_t sequence,uint32_t state_slot,
                              fg_error *err){
    mock_ops *mock=context;
    mock->prepare_calls++;
    if(mock->fail_prepare_at==mock->prepare_calls){
        fg_error_set(err,FG_ERR_IO,"injected prepare failure");
        return FG_ERR_IO;
    }
    mock_record(mock,1,sequence,state_slot);
    return FG_OK;
}

static fg_status mock_commit(void *context,uint64_t sequence,uint32_t state_slot,
                             fg_error *err){
    mock_ops *mock=context;
    mock->commit_calls++;
    if(mock->fail_commit_at==mock->commit_calls){
        fg_error_set(err,FG_ERR_IO,"injected commit failure");
        return FG_ERR_IO;
    }
    mock_record(mock,2,sequence,state_slot);
    return FG_OK;
}

static fg_status mock_restore(void *context,uint64_t sequence,uint32_t state_slot,
                              fg_error *err){
    mock_ops *mock=context;
    mock->restore_calls++;
    if(mock->fail_restore){
        fg_error_set(err,FG_ERR_IO,"injected restore failure");
        return FG_ERR_IO;
    }
    mock_record(mock,3,sequence,state_slot);
    return FG_OK;
}

typedef struct sequence_state {
    uint64_t state_generation,sampler_generation;
    uint32_t token_index,committed_tokens,context_tokens;
    uint32_t position[4];
    uint64_t qsa_records[FG_LAYER_COUNT];
    fg_sampler_state sampler;
} sequence_state;

static void capture_state(const fg_decode_batch_sequence *sequence,
                          sequence_state *state){
    memset(state,0,sizeof(*state));
    state->state_generation=sequence->state_generation;
    state->sampler_generation=sequence->sampler_generation;
    state->token_index=sequence->token_index;
    state->committed_tokens=sequence->committed_tokens;
    state->context_tokens=sequence->context_tokens;
    memcpy(state->position,sequence->position,sizeof(state->position));
    memcpy(state->qsa_records,sequence->qsa_records,sizeof(state->qsa_records));
    state->sampler=sequence->sampler;
}

static void check_state(const fg_decode_batch_sequence *sequence,
                        const sequence_state *state,int line){
    if(sequence->state_generation!=state->state_generation||
       sequence->sampler_generation!=state->sampler_generation||
       sequence->token_index!=state->token_index||
       sequence->committed_tokens!=state->committed_tokens||
       sequence->context_tokens!=state->context_tokens||
       memcmp(sequence->position,state->position,sizeof(state->position))||
       memcmp(sequence->qsa_records,state->qsa_records,sizeof(state->qsa_records))||
       memcmp(&sequence->sampler,&state->sampler,sizeof(state->sampler))){
        fprintf(stderr,"FAIL line %d: sequence state changed\n",line);
        failures++;
    }
}

static fg_decode_batch_outcome make_outcome(uint32_t token,uint32_t position){
    fg_decode_batch_outcome outcome;
    memset(&outcome,0,sizeof(outcome));
    outcome.next_token=token;
    for(uint32_t axis=0;axis<4u;axis++)outcome.position[axis]=position;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
        if((layer&3u)==3u)outcome.qsa_records[layer]=position+1u;
    fg_sampler_state_init(&outcome.sampler,1000u+token);
    return outcome;
}

static void test_transaction_commit(void){
    fg_error err={0};
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,2u,&err)==FG_OK);
    uint32_t pos[4];positions(16u,pos);
    CHECK(fg_decode_batch_sequence_enter(&table,7u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,8u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,7u,16u,16u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,8u,16u,16u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,7u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,8u,2u,&err)==FG_OK);
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    mock_ops mock={0};
    fg_decode_batch_ops ops={.prepare=mock_prepare,.commit=mock_commit,
        .restore=mock_restore,.context=&mock};
    fg_decode_batch_step step={0};
    CHECK(fg_decode_batch_step_begin(&table,&policy,&ops,10u,&step,&err)==FG_OK);
    CHECK(step.batch.slot_count==2u&&step.active);
    CHECK(mock.prepare_calls==2&&mock.commit_calls==0&&mock.restore_calls==0);
    fg_decode_batch_outcome first=make_outcome(101u,17u),second=make_outcome(202u,18u);
    CHECK(fg_decode_batch_step_advance(&table,&step,0u,&first,&err)==FG_OK);
    CHECK(fg_decode_batch_step_advance(&table,&step,1u,&second,&err)==FG_OK);
    CHECK(fg_decode_batch_step_commit(&table,&step,&err)==FG_OK);
    CHECK(mock.commit_calls==2&&!step.active);
    CHECK(table.step==1u&&!table.restored);
    CHECK(table.sequences[0].state_generation==1u);
    CHECK(table.sequences[0].sampler_generation==1u);
    CHECK(table.sequences[0].committed_tokens==17u);
    CHECK(table.sequences[0].token_index==17u);
    CHECK(table.sequences[0].position[0]==17u);
    CHECK(table.sequences[0].qsa_records[3u]==18u);
    CHECK(table.sequences[0].steps==1u);
    CHECK(table.sequences[1].committed_tokens==17u);
    CHECK(table.sequences[1].position[0]==18u);
    CHECK(table.sequences[1].qsa_records[47u]==19u);
    CHECK(!table.sequences[0].in_flight&&!table.sequences[1].in_flight);
    CHECK(!table.sequences[0].ready&&!table.sequences[1].ready);
    /* Sampler streams are per sequence: the two commits left distinct states. */
    CHECK(memcmp(&table.sequences[0].sampler,&table.sequences[1].sampler,
                 sizeof(fg_sampler_state))!=0);
}

static void test_transaction_incomplete_commit(void){
    fg_error err={0};
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,2u,&err)==FG_OK);
    uint32_t pos[4];positions(4u,pos);
    CHECK(fg_decode_batch_sequence_enter(&table,1u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,2u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,1u,4u,4u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,2u,4u,4u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,1u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,2u,2u,&err)==FG_OK);
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    mock_ops mock={0};
    fg_decode_batch_ops ops={.prepare=mock_prepare,.commit=mock_commit,
        .restore=mock_restore,.context=&mock};
    fg_decode_batch_step step={0};
    CHECK(fg_decode_batch_step_begin(&table,&policy,&ops,10u,&step,&err)==FG_OK);
    fg_decode_batch_outcome outcome=make_outcome(9u,5u);
    CHECK(fg_decode_batch_step_advance(&table,&step,0u,&outcome,&err)==FG_OK);
    sequence_state before[2];
    capture_state(&table.sequences[0],&before[0]);
    capture_state(&table.sequences[1],&before[1]);
    CHECK(fg_decode_batch_step_commit(&table,&step,&err)==FG_ERR_MISMATCH);
    CHECK(mock.commit_calls==0);
    check_state(&table.sequences[0],&before[0],__LINE__);
    check_state(&table.sequences[1],&before[1],__LINE__);
    CHECK(fg_decode_batch_step_restore(&table,&step,&err)==FG_OK);
    CHECK(table.sequences[0].ready&&table.sequences[1].ready);
    CHECK(!table.sequences[0].in_flight&&!table.sequences[1].in_flight);
}

static void test_transaction_prepare_failure(void){
    fg_error err={0};
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,2u,&err)==FG_OK);
    uint32_t pos[4];positions(4u,pos);
    CHECK(fg_decode_batch_sequence_enter(&table,1u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,2u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,1u,4u,4u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,2u,4u,4u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,1u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,2u,2u,&err)==FG_OK);
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    mock_ops mock={.fail_prepare_at=2};
    fg_decode_batch_ops ops={.prepare=mock_prepare,.commit=mock_commit,
        .restore=mock_restore,.context=&mock};
    fg_decode_batch_step step={0};
    CHECK(fg_decode_batch_step_begin(&table,&policy,&ops,10u,&step,&err)==FG_ERR_IO);
    CHECK(mock.prepare_calls==2&&mock.restore_calls==1);
    CHECK(!step.active);
    CHECK(table.sequences[0].ready&&table.sequences[1].ready);
    CHECK(!table.sequences[0].in_flight&&!table.sequences[1].in_flight);
    CHECK(table.step==0u);
}

static void test_transaction_commit_failure_restores_all(void){
    fg_error err={0};
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,2u,&err)==FG_OK);
    uint32_t pos[4];positions(4u,pos);
    CHECK(fg_decode_batch_sequence_enter(&table,1u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,2u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,1u,4u,4u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,2u,4u,4u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,1u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,2u,2u,&err)==FG_OK);
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    mock_ops mock={.fail_commit_at=2};
    fg_decode_batch_ops ops={.prepare=mock_prepare,.commit=mock_commit,
        .restore=mock_restore,.context=&mock};
    fg_decode_batch_step step={0};
    CHECK(fg_decode_batch_step_begin(&table,&policy,&ops,10u,&step,&err)==FG_OK);
    fg_decode_batch_outcome first=make_outcome(11u,5u),second=make_outcome(12u,6u);
    CHECK(fg_decode_batch_step_advance(&table,&step,0u,&first,&err)==FG_OK);
    CHECK(fg_decode_batch_step_advance(&table,&step,1u,&second,&err)==FG_OK);
    sequence_state before[2];
    capture_state(&table.sequences[0],&before[0]);
    capture_state(&table.sequences[1],&before[1]);
    CHECK(fg_decode_batch_step_commit(&table,&step,&err)==FG_ERR_IO);
    CHECK(mock.commit_calls==2&&mock.restore_calls==2);
    check_state(&table.sequences[0],&before[0],__LINE__);
    check_state(&table.sequences[1],&before[1],__LINE__);
    CHECK(table.sequences[0].ready&&table.sequences[1].ready);
    CHECK(table.restored);
}

static void test_transaction_stale_generation(void){
    fg_error err={0};
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,1u,&err)==FG_OK);
    uint32_t pos[4];positions(2u,pos);
    CHECK(fg_decode_batch_sequence_enter(&table,5u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,5u,2u,2u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,5u,1u,&err)==FG_OK);
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    fg_decode_batch_step step={0};
    CHECK(fg_decode_batch_step_begin(&table,&policy,NULL,10u,&step,&err)==FG_OK);
    table.sequences[0].state_generation++;
    fg_decode_batch_outcome outcome=make_outcome(3u,3u);
    CHECK(fg_decode_batch_step_advance(&table,&step,0u,&outcome,&err)==FG_OK);
    CHECK(fg_decode_batch_step_commit(&table,&step,&err)==FG_ERR_MISMATCH);
    CHECK(table.sequences[0].committed_tokens==2u);
}

/* ---------------------------------------------------------- isolation (B=1) */

static void test_isolation_and_b1(void){
    fg_error err={0};
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,2u,&err)==FG_OK);
    uint32_t pos[4];positions(0u,pos);
    CHECK(fg_decode_batch_sequence_enter(&table,100u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,200u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,100u,10u,10u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,200u,20u,20u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,100u,1u,&err)==FG_OK);
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    /* Only one sequence is ready: the assembled batch is exactly B=1 and the
     * other sequence's state is untouched by the whole step. */
    fg_decode_batch_step step={0};
    CHECK(fg_decode_batch_step_begin(&table,&policy,NULL,5u,&step,&err)==FG_OK);
    CHECK(step.batch.slot_count==1u);
    CHECK(table.sequences[step.batch.slots[0].sequence].sequence_id==100u);
    fg_decode_batch_outcome outcome=make_outcome(7u,11u);
    CHECK(fg_decode_batch_step_advance(&table,&step,0u,&outcome,&err)==FG_OK);
    sequence_state other_before;
    capture_state(&table.sequences[1],&other_before);
    CHECK(fg_decode_batch_step_commit(&table,&step,&err)==FG_OK);
    CHECK(table.sequences[0].committed_tokens==11u);
    CHECK(table.sequences[0].token_index==11u);
    check_state(&table.sequences[1],&other_before,__LINE__);
    CHECK(table.sequences[1].ready_since==0u);
    /* Sequence 100 is not ready now; sequence 200 still is. */
    CHECK(fg_decode_batch_sequence_ready(&table,200u,2u,&err)==FG_OK);
    fg_decode_batch batch={0};
    CHECK(fg_decode_batch_schedule(&table,&policy,6u,&batch,&err)==FG_OK);
    CHECK(batch.slot_count==1u&&batch.slots[0u].sequence==1u);
    /* A sequence that is still in flight cannot be unbound. */
    CHECK(fg_decode_batch_sequence_leave(&table,200u,&err)!=FG_OK);
}

static void test_advance_validation(void){
    fg_error err={0};
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,1u,&err)==FG_OK);
    uint32_t pos[4];positions(0u,pos);
    CHECK(fg_decode_batch_sequence_enter(&table,1u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,1u,0u,0u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,1u,1u,&err)==FG_OK);
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    fg_decode_batch_step step={0};
    CHECK(fg_decode_batch_step_begin(&table,&policy,NULL,2u,&step,&err)==FG_OK);
    fg_decode_batch_outcome outcome=make_outcome(1u,1u);
    CHECK(fg_decode_batch_step_advance(&table,&step,1u,&outcome,&err)==FG_ERR_ARGUMENT);
    outcome.next_token=FG_Q38_VOCAB_SIZE;
    CHECK(fg_decode_batch_step_advance(&table,&step,0u,&outcome,&err)==FG_ERR_FORMAT);
    CHECK(fg_decode_batch_step_restore(&table,&step,&err)==FG_OK);
    CHECK(fg_decode_batch_step_restore(&table,&step,&err)==FG_OK);
}

/* ------------------------------------------------------------------- wire */

static void fill_work(fg_decode_batch_work *work,float *hyper,float *ngram,
                      uint32_t slot_count,uint32_t layer,uint32_t flags){
    memset(work,0,sizeof(*work));
    work->layer=(uint8_t)layer;
    work->source_rank=1u;
    work->destination_rank=2u;
    work->flags=(uint8_t)flags;
    work->position_mode=FG_POSITION_TEXT;
    work->slot_count=(uint16_t)slot_count;
    for(uint32_t slot=0;slot<slot_count;slot++){
        float *h=hyper+(uint64_t)slot*FG_HYPER_WIDTH;
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)h[i]=(float)(slot*1000u+i)*0.5f;
        work->slots[slot].token_index=100u+slot;
        work->slots[slot].state_slot=slot;
        for(uint32_t axis=0;axis<3u;axis++)
            work->slots[slot].position[axis]=200u+slot+axis;
        work->slots[slot].position[3]=0u;
        work->slots[slot].hyper=h;
        if(flags&FG_LAYER_WORK_HAS_NGRAM){
            float *n=ngram+(uint64_t)slot*FG_NGRAM_EMBED_VALUES;
            for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++)n[i]=(float)(slot*7u+i)*-0.25f;
            work->slots[slot].ngram_embedding=n;
        }
    }
}

static void test_wire_roundtrip(void){
    fg_error err={0};
    uint8_t *wire=malloc(FG_DECODE_BATCH_WORK_MAX_BYTES);
    float *hyper=malloc(FG_DECODE_BATCH_MAX*FG_HYPER_WIDTH*sizeof(float));
    float *ngram=malloc(FG_DECODE_BATCH_MAX*FG_NGRAM_EMBED_VALUES*sizeof(float));
    CHECK(wire&&hyper&&ngram);
    fg_decode_batch_work work,decoded;
    fill_work(&work,hyper,ngram,2u,0u,FG_LAYER_WORK_HAS_NGRAM);
    uint32_t bytes=0;
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_VERSION,&work,&err)==FG_OK);
    uint32_t expected=FG_DECODE_BATCH_HEADER_BYTES+2u*(FG_DECODE_BATCH_SLOT_HEADER_BYTES+
        3u*4u+FG_HYPER_WIDTH*4u+FG_NGRAM_EMBED_VALUES*4u);
    CHECK(bytes==expected);
    float *hyper_out=malloc(2u*FG_HYPER_WIDTH*sizeof(float));
    float *ngram_out=malloc(2u*FG_NGRAM_EMBED_VALUES*sizeof(float));
    CHECK(hyper_out&&ngram_out);
    CHECK(fg_decode_batch_work_decode(&decoded,FG_PROTOCOL_VERSION,hyper_out,
        2u*FG_HYPER_WIDTH,ngram_out,2u*FG_NGRAM_EMBED_VALUES,wire,bytes,&err)==FG_OK);
    CHECK(decoded.layer==0u&&decoded.source_rank==1u&&decoded.destination_rank==2u);
    CHECK(decoded.flags==FG_LAYER_WORK_HAS_NGRAM&&decoded.slot_count==2u);
    CHECK(decoded.position_mode==FG_POSITION_TEXT);
    for(uint32_t slot=0;slot<2u;slot++){
        CHECK(decoded.slots[slot].token_index==100u+slot);
        CHECK(decoded.slots[slot].state_slot==slot);
        for(uint32_t axis=0;axis<3u;axis++)
            CHECK(decoded.slots[slot].position[axis]==200u+slot+axis);
        CHECK(decoded.slots[slot].position[3]==0u);
        CHECK(!memcmp(decoded.slots[slot].hyper,hyper+(uint64_t)slot*FG_HYPER_WIDTH,
                      FG_HYPER_WIDTH*sizeof(float)));
        CHECK(!memcmp(decoded.slots[slot].ngram_embedding,
                      ngram+(uint64_t)slot*FG_NGRAM_EMBED_VALUES,
                      FG_NGRAM_EMBED_VALUES*sizeof(float)));
    }

    /* No-ngram, four-axis variant. */
    fill_work(&work,hyper,ngram,2u,2u,0u);
    work.position_mode=FG_POSITION_FOUR_AXIS;
    for(uint32_t slot=0;slot<2u;slot++)work.slots[slot].position[3]=400u+slot;
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_VERSION,&work,&err)==FG_OK);
    CHECK(bytes==FG_DECODE_BATCH_HEADER_BYTES+2u*FG_DECODE_BATCH_SLOT_BYTES);
    CHECK(fg_decode_batch_work_decode(&decoded,FG_PROTOCOL_VERSION,hyper_out,
        2u*FG_HYPER_WIDTH,NULL,0u,wire,bytes,&err)==FG_OK);
    CHECK(decoded.position_mode==FG_POSITION_FOUR_AXIS&&decoded.slot_count==2u);
    CHECK(decoded.slots[1].position[3]==401u);

    fg_decode_batch_result *result=calloc(1,sizeof(*result));
    fg_decode_batch_result *result_out=calloc(1,sizeof(*result_out));
    CHECK(result&&result_out);
    result->layer=47u;result->source_rank=7u;result->destination_rank=0u;
    result->slot_count=2u;
    for(uint32_t slot=0;slot<2u;slot++){
        result->slots[slot].token_index=300u+slot;
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
            result->slots[slot].hyper[i]=(float)(slot*100u+i)*0.125f;
    }
    CHECK(fg_decode_batch_result_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        result,&err)==FG_OK);
    CHECK(bytes==FG_DECODE_BATCH_HEADER_BYTES+2u*(4u+FG_HYPER_WIDTH*4u));
    CHECK(fg_decode_batch_result_decode(result_out,wire,bytes,&err)==FG_OK);
    CHECK(result_out->layer==47u&&result_out->slot_count==2u);
    for(uint32_t slot=0;slot<2u;slot++){
        CHECK(result_out->slots[slot].token_index==300u+slot);
        CHECK(!memcmp(result_out->slots[slot].hyper,result->slots[slot].hyper,
                      sizeof(result->slots[slot].hyper)));
    }
    free(result);free(result_out);free(ngram_out);free(hyper_out);
    free(ngram);free(hyper);free(wire);
}

static void test_output_batch_wire(void){
    fg_error err={0};
    uint8_t wire[FG_OUTPUT_BATCH_WORK_MAX_BYTES];
    fg_output_batch_work work,decoded;
    memset(&work,0,sizeof(work));
    work.source_rank=0u;work.destination_rank=4u;work.slot_count=2u;
    for(uint32_t slot=0;slot<2u;slot++){
        work.slots[slot].session_slot=(uint8_t)slot;
        work.slots[slot].token_index=700u+slot;
        work.slots[slot].uniform=0.25f+0.5f*(float)slot;
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
            work.slots[slot].hyper[i]=(float)(slot*10u+i)*0.03125f;
    }
    uint32_t bytes=0;
    CHECK(fg_output_batch_work_encode(wire,FG_OUTPUT_BATCH_WORK_MAX_BYTES,&bytes,
        &work,&err)==FG_OK);
    CHECK(bytes==FG_OUTPUT_BATCH_HEADER_BYTES+2u*FG_OUTPUT_BATCH_SLOT_BYTES);
    CHECK(fg_output_batch_work_decode(&decoded,wire,bytes,&err)==FG_OK);
    CHECK(decoded.source_rank==0u&&decoded.destination_rank==4u&&
          decoded.slot_count==2u);
    for(uint32_t slot=0;slot<2u;slot++){
        CHECK(decoded.slots[slot].session_slot==slot);
        CHECK(decoded.slots[slot].token_index==700u+slot);
        CHECK(decoded.slots[slot].uniform==work.slots[slot].uniform);
        CHECK(!memcmp(decoded.slots[slot].hyper,work.slots[slot].hyper,
                      sizeof(work.slots[slot].hyper)));
    }
    /* One-slot message is exactly one slot shorter. */
    work.slot_count=1u;
    CHECK(fg_output_batch_work_encode(wire,FG_OUTPUT_BATCH_WORK_MAX_BYTES,&bytes,
        &work,&err)==FG_OK);
    CHECK(bytes==FG_OUTPUT_BATCH_HEADER_BYTES+FG_OUTPUT_BATCH_SLOT_BYTES);
    /* Rejects: reserved header byte, truncated size, slot count over the bound. */
    work.slot_count=2u;
    CHECK(fg_output_batch_work_encode(wire,FG_OUTPUT_BATCH_WORK_MAX_BYTES,&bytes,
        &work,&err)==FG_OK);
    uint8_t saved=wire[5];
    wire[5]=1u;
    CHECK(fg_output_batch_work_decode(&decoded,wire,bytes,&err)!=FG_OK);
    wire[5]=saved;
    CHECK(fg_output_batch_work_decode(&decoded,wire,bytes-1u,&err)!=FG_OK);
    work.slot_count=3u;
    CHECK(fg_output_batch_work_encode(wire,FG_OUTPUT_BATCH_WORK_MAX_BYTES,&bytes,
        &work,&err)!=FG_OK);

    fg_output_batch_result result,result_out;
    memset(&result,0,sizeof(result));
    result.source_rank=4u;result.destination_rank=0u;result.slot_count=2u;
    for(uint32_t slot=0;slot<2u;slot++){
        result.slots[slot].token_index=800u+slot;
        result.slots[slot].token=900u+slot;
        result.slots[slot].logit=(float)slot*0.5f;
    }
    uint8_t result_wire[FG_OUTPUT_BATCH_RESULT_BYTES];
    CHECK(fg_output_batch_result_encode(result_wire,&result,&err)==FG_OK);
    CHECK(fg_output_batch_result_decode(&result_out,result_wire,
        sizeof(result_wire),&err)==FG_OK);
    CHECK(result_out.slot_count==2u);
    for(uint32_t slot=0;slot<2u;slot++){
        CHECK(result_out.slots[slot].token_index==800u+slot);
        CHECK(result_out.slots[slot].token==900u+slot);
        CHECK(result_out.slots[slot].logit==result.slots[slot].logit);
    }
    result.slots[0].token=FG_Q38_VOCAB_SIZE;
    CHECK(fg_output_batch_result_encode(result_wire,&result,&err)!=FG_OK);
}

static void test_wire_rejects(void){
    fg_error err={0};
    uint8_t *wire=malloc(FG_DECODE_BATCH_WORK_MAX_BYTES);
    float *hyper=malloc(FG_DECODE_BATCH_MAX*FG_HYPER_WIDTH*sizeof(float));
    float *ngram=malloc(FG_DECODE_BATCH_MAX*FG_NGRAM_EMBED_VALUES*sizeof(float));
    float *hyper_out=malloc(FG_DECODE_BATCH_MAX*FG_HYPER_WIDTH*sizeof(float));
    CHECK(wire&&hyper&&ngram&&hyper_out);
    fg_decode_batch_work work,decoded;
    fill_work(&work,hyper,ngram,2u,0u,FG_LAYER_WORK_HAS_NGRAM);
    uint32_t bytes=0;
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_VERSION,&work,&err)==FG_OK);
    /* Protocol 5 cannot carry the batch contract. */
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_MIN_VERSION,&work,&err)!=FG_OK);
    /* Capacity and slot-count bounds. */
    CHECK(fg_decode_batch_work_encode(wire,16u,&bytes,FG_PROTOCOL_VERSION,&work,&err)
        !=FG_OK);
    work.slot_count=0u;
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_VERSION,&work,&err)!=FG_OK);
    work.slot_count=3u;
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_VERSION,&work,&err)!=FG_OK);
    /* Duplicate owner state slots are a state-corruption bug, not a wire quirk. */
    fill_work(&work,hyper,ngram,2u,0u,FG_LAYER_WORK_HAS_NGRAM);
    work.slots[1].state_slot=work.slots[0].state_slot;
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_VERSION,&work,&err)!=FG_OK);
    /* Unknown flags and non-finite hidden. */
    fill_work(&work,hyper,ngram,2u,2u,0u);
    work.flags=0x80u;
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_VERSION,&work,&err)!=FG_OK);
    fill_work(&work,hyper,ngram,2u,2u,0u);
    hyper[5]=NAN;
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_VERSION,&work,&err)!=FG_OK);

    /* Decode-side rejections on a valid buffer. */
    fill_work(&work,hyper,ngram,2u,0u,FG_LAYER_WORK_HAS_NGRAM);
    CHECK(fg_decode_batch_work_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        FG_PROTOCOL_VERSION,&work,&err)==FG_OK);
    CHECK(fg_decode_batch_work_decode(&decoded,FG_PROTOCOL_VERSION,hyper_out,
        2u*FG_HYPER_WIDTH,ngram,2u*FG_NGRAM_EMBED_VALUES,wire,bytes,&err)==FG_OK);
    CHECK(fg_decode_batch_work_decode(&decoded,FG_PROTOCOL_VERSION,hyper_out,
        2u*FG_HYPER_WIDTH,NULL,0u,wire,bytes,&err)!=FG_OK);
    CHECK(fg_decode_batch_work_decode(&decoded,FG_PROTOCOL_VERSION,hyper_out,
        FG_HYPER_WIDTH,ngram,2u*FG_NGRAM_EMBED_VALUES,wire,bytes,&err)!=FG_OK);
    uint8_t *truncated=malloc(bytes-1u);
    memcpy(truncated,wire,bytes-1u);
    CHECK(fg_decode_batch_work_decode(&decoded,FG_PROTOCOL_VERSION,hyper_out,
        2u*FG_HYPER_WIDTH,ngram,2u*FG_NGRAM_EMBED_VALUES,truncated,bytes-1u,&err)!=FG_OK);
    free(truncated);
    uint8_t *reserved=malloc(bytes);
    memcpy(reserved,wire,bytes);
    reserved[5]=1u;
    CHECK(fg_decode_batch_work_decode(&decoded,FG_PROTOCOL_VERSION,hyper_out,
        2u*FG_HYPER_WIDTH,ngram,2u*FG_NGRAM_EMBED_VALUES,reserved,bytes,&err)!=FG_OK);
    free(reserved);
    /* Result decode rejects reserved bytes and truncation. */
    fg_decode_batch_result *result=calloc(1,sizeof(*result));
    fg_decode_batch_result *result_out=calloc(1,sizeof(*result_out));
    CHECK(result&&result_out);
    result->layer=47u;result->slot_count=1u;
    result->slots[0].token_index=9u;
    CHECK(fg_decode_batch_result_encode(wire,FG_DECODE_BATCH_WORK_MAX_BYTES,&bytes,
        result,&err)==FG_OK);
    CHECK(fg_decode_batch_result_decode(result_out,wire,bytes,&err)==FG_OK);
    wire[3]=1u;
    CHECK(fg_decode_batch_result_decode(result_out,wire,bytes,&err)!=FG_OK);
    wire[3]=0u;
    CHECK(fg_decode_batch_result_decode(result_out,wire,bytes-1u,&err)!=FG_OK);
    free(result);free(result_out);free(hyper_out);free(ngram);free(hyper);free(wire);
}

/* ------------------------------------------------- batched vs sequential sim */

/* A deterministic stand-in for the owner-side per-session state and the ring
 * wire: the "device" state is advanced once per decoded token per state slot,
 * the "worker" executes a whole batch message, and the "coordinator" samples.
 * Two runs through the same simulation - B=2 batched and B=1 sequential - must
 * produce byte-identical per-session streams and device state, which is the
 * host-side half of the B=2 parity gate. */
#define SIM_STREAM_MAX 8u
typedef struct sim_model {
    float state[FG_DECODE_BATCH_MAX][FG_HYPER_WIDTH];
    float backup[FG_DECODE_BATCH_MAX][FG_HYPER_WIDTH];
    uint32_t tokens[FG_DECODE_BATCH_MAX][SIM_STREAM_MAX];
    uint32_t token_count[FG_DECODE_BATCH_MAX];
    uint8_t wire[FG_DECODE_BATCH_WORK_MAX_BYTES];
    float hyper[FG_DECODE_BATCH_MAX*FG_HYPER_WIDTH];
    float ngram[FG_DECODE_BATCH_MAX*FG_NGRAM_EMBED_VALUES];
    float result_hyper[FG_DECODE_BATCH_MAX*FG_HYPER_WIDTH];
    int fail_slot;
    int fail_at_step;
    int steps;
    int worker_calls;
    int restores;
} sim_model;

static void sim_fill_hyper(float *hyper,uint32_t state_slot,uint32_t token_index){
    for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
        hyper[i]=(float)((state_slot+1u)*100000u+token_index*31u+i%97u)*0.001f;
}

static void sim_fill_ngram(float *ngram,uint32_t state_slot,uint32_t token_index){
    for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++)
        ngram[i]=(float)((state_slot+3u)*7u+token_index+i%13u)*-0.002f;
}

static void sim_append(sim_model *model,uint32_t state_slot,uint32_t token){
    if(model->token_count[state_slot]>=SIM_STREAM_MAX)return;
    model->tokens[state_slot][model->token_count[state_slot]++]=token;
}

static fg_status sim_prepare(void *context,uint64_t sequence_id,uint32_t state_slot,
                             fg_error *err){
    (void)sequence_id;(void)err;
    sim_model *model=context;
    memcpy(model->backup[state_slot],model->state[state_slot],
           sizeof(model->state[state_slot]));
    return FG_OK;
}

static fg_status sim_commit(void *context,uint64_t sequence_id,uint32_t state_slot,
                            fg_error *err){
    (void)context;(void)sequence_id;(void)state_slot;(void)err;
    return FG_OK;
}

static fg_status sim_restore(void *context,uint64_t sequence_id,uint32_t state_slot,
                             fg_error *err){
    (void)sequence_id;(void)err;
    sim_model *model=context;
    memcpy(model->state[state_slot],model->backup[state_slot],
           sizeof(model->state[state_slot]));
    model->restores++;
    return FG_OK;
}

/* Execute one ring step exactly as the runtime will: schedule + PREPARE, build
 * and encode one batch message, run the worker slots, decode the result,
 * sample, then COMMIT (or RESTORE on any failure). */
static fg_status sim_ring_step(fg_decode_batch_table *table,
                               const fg_decode_batch_policy *policy,sim_model *model,
                               uint64_t now,fg_error *err){
    fg_decode_batch_ops hooks={.prepare=sim_prepare,.commit=sim_commit,
        .restore=sim_restore,.context=model};
    fg_decode_batch_step step;
    memset(&step,0,sizeof(step));
    fg_status status=fg_decode_batch_step_begin(table,policy,&hooks,now,&step,err);
    if(status!=FG_OK)return status;
    fg_decode_batch_work work;
    memset(&work,0,sizeof(work));
    work.layer=0u;work.source_rank=1u;work.destination_rank=2u;
    work.flags=FG_LAYER_WORK_HAS_NGRAM;
    work.position_mode=FG_POSITION_TEXT;
    work.slot_count=(uint16_t)step.batch.slot_count;
    for(uint32_t slot=0;slot<step.batch.slot_count;slot++){
        const fg_decode_batch_slot *entry=&step.batch.slots[slot];
        float *hyper=model->hyper+(uint64_t)slot*FG_HYPER_WIDTH;
        float *ngram=model->ngram+(uint64_t)slot*FG_NGRAM_EMBED_VALUES;
        sim_fill_hyper(hyper,entry->state_slot,entry->token_index);
        sim_fill_ngram(ngram,entry->state_slot,entry->token_index);
        work.slots[slot].token_index=entry->token_index;
        work.slots[slot].state_slot=entry->state_slot;
        for(uint32_t axis=0;axis<3u;axis++)
            work.slots[slot].position[axis]=entry->position[axis];
        work.slots[slot].hyper=hyper;
        work.slots[slot].ngram_embedding=ngram;
    }
    uint32_t wire_bytes=0;
    status=fg_decode_batch_work_encode(model->wire,sizeof(model->wire),&wire_bytes,
        FG_PROTOCOL_VERSION,&work,err);
    if(status!=FG_OK){
        fg_decode_batch_step_restore(table,&step,err);
        return status;
    }
    /* Worker side: decode the batch message and run each slot against its own
     * owner state slot; a slot failure aborts the whole step. */
    fg_decode_batch_work received;
    float *hyper_out=malloc(FG_DECODE_BATCH_MAX*FG_HYPER_WIDTH*sizeof(float));
    float *ngram_out=malloc(FG_DECODE_BATCH_MAX*FG_NGRAM_EMBED_VALUES*sizeof(float));
    if(!hyper_out||!ngram_out){
        free(hyper_out);free(ngram_out);
        fg_decode_batch_step_restore(table,&step,err);
        fg_error_set(err,FG_ERR_OOM,"simulate worker storage");
        return FG_ERR_OOM;
    }
    status=fg_decode_batch_work_decode(&received,FG_PROTOCOL_VERSION,hyper_out,
        FG_DECODE_BATCH_MAX*FG_HYPER_WIDTH,ngram_out,
        FG_DECODE_BATCH_MAX*FG_NGRAM_EMBED_VALUES,model->wire,wire_bytes,err);
    fg_decode_batch_result result;
    memset(&result,0,sizeof(result));
    if(status==FG_OK){
        model->steps++;
        model->worker_calls++;
        result.layer=47u;result.source_rank=7u;result.destination_rank=0u;
        result.slot_count=received.slot_count;
        for(uint32_t slot=0;slot<received.slot_count;slot++){
            const fg_decode_batch_slot_work *entry=&received.slots[slot];
            if(model->fail_slot==(int)slot&&
               (model->fail_at_step<0||model->fail_at_step==model->steps)){
                fg_error_set(err,FG_ERR_IO,"simulated worker failure on slot %u",slot);
                status=FG_ERR_IO;
                break;
            }
            float *state=model->state[entry->state_slot];
            float *out=model->result_hyper+(uint64_t)slot*FG_HYPER_WIDTH;
            for(uint32_t i=0;i<FG_HYPER_WIDTH;i++){
                state[i]=state[i]*0.5f+entry->hyper[i];
                out[i]=state[i];
            }
            result.slots[slot].token_index=entry->token_index;
            memcpy(result.slots[slot].hyper,out,sizeof(result.slots[slot].hyper));
        }
    }
    free(hyper_out);free(ngram_out);
    if(status!=FG_OK){
        fg_decode_batch_step_restore(table,&step,err);
        return status;
    }
    uint32_t result_bytes=0;
    status=fg_decode_batch_result_encode(model->wire,sizeof(model->wire),&result_bytes,
        &result,err);
    fg_decode_batch_result sampled;
    if(status==FG_OK)status=fg_decode_batch_result_decode(&sampled,model->wire,
        result_bytes,err);
    if(status==FG_OK){
        for(uint32_t slot=0;slot<sampled.slot_count;slot++){
            fg_decode_batch_outcome outcome;
            memset(&outcome,0,sizeof(outcome));
            const fg_decode_batch_slot *entry=&step.batch.slots[slot];
            uint32_t token=(uint32_t)fabsf(sampled.slots[slot].hyper[0]*1000.0f)%
                FG_Q38_VOCAB_SIZE;
            outcome.next_token=token;
            for(uint32_t axis=0;axis<4u;axis++)
                outcome.position[axis]=entry->position[axis]+1u;
            for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
                if((layer&3u)==3u)
                    outcome.qsa_records[layer]=entry->token_index+1u;
            fg_sampler_state_init(&outcome.sampler,entry->token_index+token);
            sim_append(model,entry->state_slot,token);
            status=fg_decode_batch_step_advance(table,&step,slot,&outcome,err);
            if(status!=FG_OK)break;
        }
    }
    if(status!=FG_OK){
        fg_decode_batch_step_restore(table,&step,err);
        return status;
    }
    return fg_decode_batch_step_commit(table,&step,err);
}

static void test_batched_equals_sequential(void){
    fg_error err={0};
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    uint32_t pos[4];positions(0u,pos);
    const uint32_t steps=4u;

    sim_model *batched=calloc(1,sizeof(*batched));
    sim_model *sequential=calloc(1,sizeof(*sequential));
    CHECK(batched&&sequential);
    if(!batched||!sequential)return;
    batched->fail_slot=-1;batched->fail_at_step=-1;
    sequential->fail_slot=-1;sequential->fail_at_step=-1;

    /* Run A: both sequences ready, one B=2 ring step per round. */
    fg_decode_batch_table table_a;
    CHECK(fg_decode_batch_table_init(&table_a,2u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table_a,11u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table_a,22u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table_a,11u,100u,100u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table_a,22u,50u,50u,pos,&err)==FG_OK);
    for(uint32_t round=0;round<steps;round++){
        CHECK(fg_decode_batch_sequence_ready(&table_a,11u,2u*round+1u,&err)==FG_OK);
        CHECK(fg_decode_batch_sequence_ready(&table_a,22u,2u*round+2u,&err)==FG_OK);
        CHECK(sim_ring_step(&table_a,&policy,batched,3u*round,&err)==FG_OK);
        CHECK(table_a.sequences[0].state_generation==round+1u);
        CHECK(table_a.sequences[1].state_generation==round+1u);
    }

    /* Run B: same prompts, strictly sequential B=1 ring steps. */
    fg_decode_batch_table table_b;
    CHECK(fg_decode_batch_table_init(&table_b,2u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table_b,11u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table_b,22u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table_b,11u,100u,100u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table_b,22u,50u,50u,pos,&err)==FG_OK);
    for(uint32_t round=0;round<steps;round++){
        CHECK(fg_decode_batch_sequence_ready(&table_b,11u,2u*round+1u,&err)==FG_OK);
        CHECK(sim_ring_step(&table_b,&policy,sequential,3u*round,&err)==FG_OK);
        CHECK(fg_decode_batch_sequence_ready(&table_b,22u,2u*round+2u,&err)==FG_OK);
        CHECK(sim_ring_step(&table_b,&policy,sequential,3u*round+1u,&err)==FG_OK);
    }
    for(uint32_t slot=0;slot<FG_DECODE_BATCH_MAX;slot++){
        CHECK(table_a.sequences[slot].committed_tokens==
              table_b.sequences[slot].committed_tokens);
        CHECK(table_a.sequences[slot].token_index==table_b.sequences[slot].token_index);
        CHECK(!memcmp(table_a.sequences[slot].qsa_records,
                      table_b.sequences[slot].qsa_records,
                      sizeof(table_a.sequences[slot].qsa_records)));
        CHECK(!memcmp(&table_a.sequences[slot].sampler,
                      &table_b.sequences[slot].sampler,
                      sizeof(fg_sampler_state)));
        CHECK(!memcmp(batched->state[slot],sequential->state[slot],
                      sizeof(batched->state[slot])));
        CHECK(batched->token_count[slot]==sequential->token_count[slot]);
        CHECK(!memcmp(batched->tokens[slot],sequential->tokens[slot],
                      sizeof(uint32_t)*batched->token_count[slot]));
        CHECK(batched->worker_calls==steps&&sequential->worker_calls==2u*steps);
        CHECK(batched->restores==0&&sequential->restores==0);
    }
    free(batched);free(sequential);
}

static void test_failure_rolls_back_batch(void){
    fg_error err={0};
    fg_decode_batch_policy policy;
    fg_decode_batch_policy_default(&policy);
    uint32_t pos[4];positions(0u,pos);
    sim_model *model=calloc(1,sizeof(*model));
    CHECK(model!=NULL);
    if(!model)return;
    model->fail_slot=-1;model->fail_at_step=-1;
    fg_decode_batch_table table;
    CHECK(fg_decode_batch_table_init(&table,2u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,11u,0u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_enter(&table,22u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,11u,100u,100u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_frontier(&table,22u,50u,50u,pos,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,11u,1u,&err)==FG_OK);
    CHECK(fg_decode_batch_sequence_ready(&table,22u,2u,&err)==FG_OK);
    float state_before[FG_DECODE_BATCH_MAX][FG_HYPER_WIDTH];
    memcpy(state_before,model->state,sizeof(state_before));
    /* Slot 1 fails after slot 0 already advanced its state: the whole step
     * must roll back and both sequences must be ready to retry. */
    model->fail_slot=1;
    CHECK(sim_ring_step(&table,&policy,model,3u,&err)==FG_ERR_IO);
    CHECK(model->restores==2);
    CHECK(!memcmp(model->state,state_before,sizeof(state_before)));
    CHECK(table.sequences[0].committed_tokens==100u);
    CHECK(table.sequences[1].committed_tokens==50u);
    CHECK(table.sequences[0].ready&&table.sequences[1].ready);
    CHECK(table.restored);
    /* Retry succeeds and produces the same result as a clean run. */
    model->fail_slot=-1;
    CHECK(sim_ring_step(&table,&policy,model,4u,&err)==FG_OK);
    CHECK(table.sequences[0].committed_tokens==101u);
    CHECK(table.sequences[1].committed_tokens==51u);
    CHECK(!table.restored);
    free(model);
}

int main(void){
    test_policy();
    test_table_and_schedule();
    test_shrink_and_fairness();
    test_transaction_commit();
    test_transaction_incomplete_commit();
    test_transaction_prepare_failure();
    test_transaction_commit_failure_restores_all();
    test_transaction_stale_generation();
    test_isolation_and_b1();
    test_advance_validation();
    test_wire_roundtrip();
    test_output_batch_wire();
    test_wire_rejects();
    test_batched_equals_sequential();
    test_failure_rolls_back_batch();
    if(failures){
        fprintf(stderr,"decode batch: %d failure(s)\n",failures);
        return 1;
    }
    puts("depth-B decode batch scheduler, transaction and wire: PASS");
    return 0;
}
