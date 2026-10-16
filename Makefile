CC ?= cc
CFLAGS ?= -O3 -g -std=c17 -D_GNU_SOURCE -Wall -Wextra -Wpedantic -Werror -MMD -MP
CPPFLAGS ?= -Iinclude -Ivendor
LDLIBS ?= -pthread -lm -lvulkan
GLSLC ?= sh ./glslc
FG_SHADER_SRC := $(wildcard shaders/*.comp)
FG_SHADER_OUT := $(patsubst shaders/%.comp,vulkan/%.spv,$(FG_SHADER_SRC))

SRC := src/main.c vendor/stb_image_impl.c src/util.c src/tower.c src/tower_vk.c src/video.c src/manifest.c src/session.c src/topology.c src/sha256.c src/gguf.c src/q38_schema.c src/q38_math.c src/quant.c src/vk.c src/model.c src/expert.c src/owner.c src/output.c src/tokenizer.c src/pack.c src/uring.c src/loader.c src/ngram.c src/qsa_state.c src/qsa_locality.c src/qsa_cache.c src/qsa.c src/qsa_owner.c src/qsa_replica.c src/protocol.c src/fabric.c src/pipeline.c src/prefix.c src/runtime_options.c src/sampler.c src/ledger.c src/decode_batch.c src/runtime.c src/chat.c src/api.c
SRC += src/mtp.c
SRC += src/mtp_head.c
SRC += src/mtp_capture.c
SRC += src/mtp_wire.c src/mtp_native.c
SRC += src/mtp_embedding.c
OBJ := $(SRC:.c=.o)
DEP := tests/test_prefill_collective.d tests/test_owner_reduce.d $(OBJ:.o=.d) tests/test_expert_prefill.d tests/test_prefill_dispatch.d tests/test_qsa_prefill.d tests/test_core.d tests/test_session.d tests/test_prefix.d tests/test_chat.d tests/test_chat_runtime.d tests/test_api.d tests/test_ngram_deployment.d tests/test_fg_vk.d tests/test_hc_down_split.d tests/test_model_load.d tests/test_qsa_model_load.d tests/test_tokenizer.d tests/test_fabric.d tests/test_tower.d tests/test_decode_batch.d
TEST_COMMON := src/util.o src/manifest.o src/session.o src/topology.o src/sha256.o src/gguf.o src/q38_schema.o src/q38_math.o src/quant.o src/vk.o src/tokenizer.o src/pack.o src/uring.o src/loader.o src/ngram.o src/qsa_state.o src/qsa_locality.o src/qsa_cache.o src/qsa_owner.o src/qsa_replica.o src/protocol.o src/runtime_options.o src/sampler.o src/ledger.o
DEP += tests/test_multi_session.d
DEP += tests/test_mtp_wire.d tests/test_mtp_native.d tests/test_mtp_embedding.d tools/mtp-embedding.d
DEP += tests/test_mtp_ring.d
DEP += tests/mtp_ring_sse.d

MTP_RING_WRAPS = fg_vk_tensor_create fg_vk_tensor_destroy fg_vk_tensor_write fg_vk_tensor_read \
 fg_owner_mtp_prepare fg_owner_mtp_retain \
 fg_model_vk fg_model_rank fg_model_tensor fg_vk_embedding_q8_0_batch \
 fg_ngram_store_lookup_prefill fg_tokenizer_eos fg_tokenizer_token fg_tokenizer_decode_token \
 fg_output_slice_ways fg_output_hc_run \
 fg_output_slice_run_hidden fg_owner_session_count fg_owner_active_session \
 fg_owner_set_active_session fg_owner_prefill_input fg_owner_prefill_input_slot \
 fg_owner_decode_results fg_owner_qsa_frontier fg_owner_session_device_snapshot \
 fg_owner_session_rollback fg_owner_session_snapshot_release fg_expert_decode_chain_ready \
 fg_owner_decode_small_n fg_owner_decode_block_chained fg_owner_prefill_layer_begin \
 fg_owner_prefill_layer_finish fg_mtp_embedding_open fg_mtp_embedding_close \
 fg_mtp_embedding_row fg_mtp_head_open_windowed fg_mtp_head_open_owner_windowed fg_mtp_head_fork fg_mtp_head_close \
 fg_mtp_head_check_residency \
 fg_expert_prepare_small_n_scratch fg_vk_get_memory_stats \
 fg_mtp_head_frontier fg_mtp_head_reset fg_mtp_head_truncate fg_mtp_head_warm_replay \
 fg_mtp_head_step_replay fg_mtp_vocab_open fg_mtp_vocab_close fg_mtp_vocab_check fg_mtp_vocab_argmax
comma := ,

.PHONY: all clean test test-sampler test-mtp test-mtp-capture test-ngram-deployment test-vulkan test-qsa-fleet test-gdn-fleet test-fg-profile-analysis test-tower test-decode-batch shaders
all: flash-gordon shaders

flash-gordon: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_mtp_ring: tests/test_mtp_ring.c src/runtime.c src/runtime_mtp_coordinator.inc src/runtime_mtp_worker.inc tests/mtp_ring_sse.o $(filter-out src/main.o src/runtime.o src/api.o,$(OBJ))
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(filter %.o,$^) \
	 $(addprefix -Wl$(comma)--wrap=,$(MTP_RING_WRAPS)) $(LDLIBS)

-include tests/test_mtp_ring.d

.PHONY: test-mtp-ring
test-mtp-ring: tests/test_mtp_ring
	@for native in 7 6; do for reject in 0 1 2 3; do ./tests/test_mtp_ring 127.0.0.1 127.0.0.1 $$((29600+reject*20+(7-native)*100)) $$reject 2 $$native || exit; done; done

tests/test_core: tests/test_core.o $(TEST_COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_tower: tests/test_tower.o src/tower.o src/tower_vk.o src/video.o vendor/stb_image_impl.o $(TEST_COMMON) | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-tower: tests/test_tower
	./tests/test_tower

tests/test_video: tests/test_video.o src/video.o src/tower.o vendor/stb_image_impl.o $(TEST_COMMON) | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-video: tests/test_video
	./tests/test_video || test $$? -eq 77

tests/test_session: tests/test_session.o $(TEST_COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_decode_batch: tests/test_decode_batch.o src/decode_batch.o $(TEST_COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-decode-batch: tests/test_decode_batch
	./tests/test_decode_batch

tests/test_prefix: tests/test_prefix.c src/prefix.c include/fg_prefix.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tests/test_prefix.c src/prefix.c src/util.c $(LDLIBS)

tests/test_chat: tests/test_chat.c src/chat.c src/prefix.c src/util.c include/fg_chat.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DFG_CHAT_RENDER_ONLY -o $@ tests/test_chat.c src/chat.c src/prefix.c src/util.c $(LDLIBS)

tests/test_chat_runtime: tests/test_chat_runtime.c src/chat.c src/prefix.c src/util.c include/fg_chat.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tests/test_chat_runtime.c src/prefix.c src/util.c $(LDLIBS)

tests/test_api: tests/test_api.c src/api.c src/chat.c src/prefix.c src/util.c src/sampler.o src/sha256.o include/fg_api.h include/fg_chat.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DFG_CHAT_RENDER_ONLY -DFG_API_TEST_BUILD -o $@ tests/test_api.c src/chat.c src/prefix.c src/util.c src/sampler.o src/sha256.o $(LDLIBS)

tests/test_sampler: tests/test_sampler.c src/sampler.o src/util.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-sampler: tests/test_sampler
	./tests/test_sampler

tests/test_mtp: tests/test_mtp.c src/mtp.o $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tests/test_mtp_wire: tests/test_mtp_wire.c src/mtp_wire.o src/fabric.o $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tests/test_mtp_native: tests/test_mtp_native.c src/mtp_native.o src/mtp_wire.o src/util.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) -lm

tests/test_mtp_embedding: tests/test_mtp_embedding.c src/mtp_embedding.o $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

test-mtp: tests/test_mtp tests/test_mtp_wire tests/test_mtp_native tests/test_mtp_embedding
	./tests/test_mtp
	./tests/test_mtp_wire
	./tests/test_mtp_native
	./tests/test_mtp_native 6
	mkdir -p .build
	./tests/test_mtp_embedding .build/mtp-embedding-test

tools/bench-mtp-head: tools/bench_mtp_head.c src/mtp_head.o src/model.o src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< src/mtp_head.o src/model.o src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) $(LDLIBS)

-include tools/bench-mtp-head.d

tools/mtp-acceptance: tools/mtp_acceptance.c src/mtp_head.o src/model.o src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< src/mtp_head.o src/model.o src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) $(LDLIBS)

tools/mtp-embedding: tools/mtp_embedding.c src/mtp_embedding.o $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/mtp-real-cook: tools/mtp_real_cook.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/mtp-real-place: tools/mtp_real_place.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/mtp-residency: tools/mtp_residency.c $(filter-out src/main.o,$(OBJ)) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/mtp-native-replay: tools/mtp_native_replay.c $(filter-out src/main.o,$(OBJ)) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/bench-mtp-chain: tools/bench_mtp_chain.c $(filter-out src/main.o,$(OBJ)) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

-include tools/bench-mtp-chain.d

tools/mtp-tokens: tools/mtp_tokens.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/mtp-offline: tools/mtp_offline.c $(filter-out src/main.o,$(OBJ)) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

-include tools/mtp-offline.d

-include tools/mtp-tokens.d

-include tools/mtp-native-replay.d

-include tools/mtp-residency.d

-include tools/mtp-real-place.d

-include tools/mtp-real-cook.d

-include tools/mtp-acceptance.d

tests/test_mtp_capture_io: tests/test_mtp_capture_io.c src/mtp_capture.o src/util.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

-include tests/test_mtp_capture_io.d

test-mtp-capture: tests/test_mtp_capture_io
	mkdir -p .build
	./tests/test_mtp_capture_io .build/capture-io-test
	python3 -m unittest discover -s tests -p test_mtp_capture.py

test: tests/test_q4_repack
test: tests/test_multi_session tests/test_prefill_collective tests/test_core tests/test_session tests/test_prefix tests/test_chat tests/test_chat_runtime tests/test_api tests/test_ngram_deployment tests/test_prefill_dispatch tests/test_decode_batch tests/test_video tests/test_mtp
	@bash tools/check-flags.sh
	./tests/test_core
	./tests/test_q4_repack
	./tests/test_session
	./tests/test_multi_session || test $$? -eq 77
	./tests/test_prefix
	./tests/test_chat
	./tests/test_chat_runtime
	./tests/test_api
	./tests/test_ngram_deployment
	./tests/test_prefill_dispatch
	./tests/test_decode_batch
	./tests/test_mtp
	./tests/test_prefill_collective || test $$? -eq 77
	./tests/test_video
	bash tests/test_ep_trace_validator.sh

shaders: $(FG_SHADER_OUT)

vulkan/%.spv: shaders/%.comp
	@mkdir -p vulkan
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/fg_dense_q4_k.spv vulkan/fg_dense_q5_k.spv vulkan/fg_dense_q6_k.spv vulkan/fg_dense_iq4_nl.spv: shaders/fg_dense_quant_impl.glsl
vulkan/fg_router_top10.spv vulkan/fg_router_top10_tiles.spv: shaders/fg_router_top10.glsl
vulkan/fg_dense_q8_0_cooked_r8.spv vulkan/fg_shared_swiglu_q8.spv: shaders/fg_dense_q8_0_cooked_r8.glsl
vulkan/fg_shared_swiglu_q4.spv: shaders/fg_dense_soa_impl.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_roof_gdn_gate_controls*.comp)): shaders/fg_dense_soa_impl.glsl shaders/fg_dense_compact_gemv.glsl
vulkan/fg_moe_decode_gate_up.spv vulkan/fg_roof_expert.spv: shaders/fg_moe_decode_gate_up.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_roof_expert*.comp)): shaders/fg_moe_decode_gate_up.glsl
vulkan/fg_moe_decode_down_reduce.spv vulkan/fg_roof_down_w32.spv vulkan/fg_roof_down_w64.spv vulkan/fg_small_down_q8_exact.spv: shaders/fg_moe_decode_down_reduce.glsl
vulkan/fg_roof_down_split.spv vulkan/fg_roof_down_split_w64.spv: shaders/fg_moe_decode_down_reduce.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_roof_down_slots*.comp)): shaders/fg_moe_decode_down_reduce.glsl
vulkan/fg_roof_down_rows1.spv vulkan/fg_roof_down_rows2.spv: shaders/fg_roof_down_rows.glsl
vulkan/fg_roof_hc_up_mix_iq4.spv vulkan/fg_roof_hc_up_mix_q8.spv: shaders/fg_roof_hc_up_mix.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_roof_hc_down[48]*.comp shaders/fg_roof_hc_down_single*.comp)): shaders/fg_dense_soa_impl.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_roof_stream*.comp)): shaders/fg_roof_stream.glsl
vulkan/fg_qsa_decode_attention_split.spv vulkan/fg_roof_qsa_h1.spv vulkan/fg_roof_qsa_h3.spv vulkan/fg_roof_qsa_h6.spv: shaders/fg_qsa_decode_attention_split.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_roof_dense*.comp)): shaders/fg_dense_soa_impl.glsl shaders/fg_dense_compact_gemv.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_roof_dense_rows*.comp)): shaders/fg_roof_dense_rows.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_small_q4*.comp shaders/fg_small_q5*.comp shaders/fg_small_iq4*.comp)): shaders/fg_small_dense.glsl shaders/fg_dense_soa_impl.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_small_reuse*.comp)): shaders/fg_small_reuse.glsl shaders/fg_small_dense.glsl shaders/fg_dense_soa_impl.glsl
vulkan/fg_small_gate_up.spv vulkan/fg_small_gate_up_pairs.spv vulkan/fg_small_gate_up_four.spv: shaders/fg_small_gate_up.glsl
vulkan/fg_small_gate_up_subgroup.spv vulkan/fg_small_gate_up_four_subgroup.spv vulkan/fg_small_gate_up_prefetch.spv vulkan/fg_small_gate_up_four_prefetch.spv: shaders/fg_small_gate_up.glsl
vulkan/fg_small_down.spv vulkan/fg_small_down_union.spv: shaders/fg_small_down_decode.glsl
vulkan/fg_small_gate_up_fma.spv vulkan/fg_small_gate_up_four_fma.spv: shaders/fg_small_gate_up.glsl shaders/fg_small_fma.glsl
$(foreach type,q4 q5 iq4,$(foreach n,2 3 4,vulkan/fg_small_reuse_$(type)_n$(n).spv)): shaders/fg_small_fma.glsl
vulkan/fg_small_q8_reuse_n2.spv vulkan/fg_small_q8_reuse_n3.spv vulkan/fg_small_q8_reuse_n4.spv: shaders/fg_small_fma.glsl
vulkan/fg_small_down.spv vulkan/fg_small_down_fma.spv: shaders/fg_small_down.glsl shaders/fg_small_down_decode.glsl shaders/fg_small_fma.glsl
$(foreach rows,4 8,$(foreach type,q4 q5 iq4,$(foreach n,2 3 4,vulkan/fg_small_reuse$(rows)_$(type)_n$(n).spv))): shaders/fg_small_reuse_rows.glsl shaders/fg_small_reuse_decode.glsl shaders/fg_small_fma.glsl shaders/fg_small_dense.glsl shaders/fg_dense_soa_impl.glsl
$(foreach type,q4 q5 iq4,$(foreach n,2 3 4,vulkan/fg_small_reuse_$(type)_n$(n).spv)): shaders/fg_small_reuse_decode.glsl
vulkan/fg_small_gate_up_half.spv vulkan/fg_small_gate_up_four_half.spv: shaders/fg_small_gate_up.glsl
vulkan/fg_small_gate_up_packed.spv vulkan/fg_small_gate_up_four_packed.spv: shaders/fg_small_gate_up.glsl shaders/fg_small_packed_dot.glsl
vulkan/fg_gdn_recurrent_algebraic.spv vulkan/fg_gdn_small.spv: shaders/fg_gdn_algebraic.glsl
vulkan/fg_dense_q8_0_cooked_r8.spv vulkan/fg_mtp_vocab_q8.spv vulkan/fg_mtp_fc_hidden.spv: shaders/fg_dense_q8_0_cooked_r8.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_small_f32*.comp shaders/fg_small_router_bf16*.comp shaders/fg_small_index_bf16*.comp)): shaders/fg_small_plain.glsl
vulkan/fg_dense_q8_0_cooked.spv $(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_small_q8*.comp)): shaders/fg_small_q8_parallel.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_small_q8_reuse*.comp)): shaders/fg_small_q8_reuse.glsl
vulkan/fg_moe_decode_down_reduce.spv vulkan/fg_small_down.spv: shaders/fg_moe_decode_down_reduce.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_dense_soa*.comp)): shaders/fg_dense_soa_impl.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_dense_soa*tile*.comp)): shaders/fg_dense_soa_tile_impl.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_dense_compact*.comp)): shaders/fg_dense_soa_impl.glsl shaders/fg_dense_compact_gemv.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_dense_compact*tile*.comp)): shaders/fg_dense_soa_tile_impl.glsl

tests/test_fg_vk: tests/test_fg_vk.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/uring.o src/sha256.o src/q38_schema.o src/gguf.o src/topology.o src/util.o src/output.o src/model.o src/loader.o src/manifest.o src/runtime_options.o src/sampler.o src/protocol.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lvulkan

test-fg-profile-analysis:
	python3 -m unittest tools/test_analyze_fg_profile.py

test-fg-dense-rates:
	python3 -m unittest tools/test_fg_dense_rates.py

.PHONY: test-fusion-ledger
test-fusion-ledger:
	python3 -m unittest tools/test_fusion_ledger.py

tools/q4-repack: tools/q4_repack.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/bench-ngram-latency: tools/bench_ngram_latency.c $(TEST_COMMON) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/bench-fabric-latency: tools/bench_fabric_latency.c src/fabric.o $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/q4-extract-pack: tools/q4_extract_pack.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/q4-compare-packs: tools/q4_compare_packs.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/q4-eval-batch: tools/q4_eval_batch.c $(filter-out src/main.o,$(OBJ))
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/bench-decode-fusion: tools/bench_decode_fusion.c src/model.c src/owner.o src/expert.o src/qsa.o $(TEST_COMMON) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tools/bench_decode_fusion.c src/owner.o src/expert.o src/qsa.o $(TEST_COMMON) $(LDLIBS)

tests/test_q4_repack: tests/test_q4_repack.c tools/q4_repack.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tests/test_q4_repack.c $(TEST_COMMON) $(LDLIBS)

tests/test_hc_down_split: tests/test_hc_down_split.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/uring.o src/sha256.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lvulkan

tests/test_model_load: tests/test_model_load.o src/model.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/loader.o src/uring.o src/sha256.o src/manifest.o src/runtime_options.o src/topology.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_expert_prefill: tests/test_expert_prefill.o src/expert.o $(TEST_COMMON) | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_owner_reduce: tests/test_owner_reduce.o src/qsa.o $(TEST_COMMON) | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_qsa_prefill: tests/test_qsa_prefill.o $(TEST_COMMON) | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_prefill_collective: tests/test_prefill_collective.o $(filter-out src/main.o src/runtime.o,$(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_multi_session: tests/test_multi_session.o $(filter-out src/main.o src/runtime.o,$(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-multi-session
test-multi-session: tests/test_multi_session tests/test_fabric tests/test_api tests/test_session
	./tests/test_multi_session || test $$? -eq 77
	./tests/test_fabric --pipeline-only || test $$? -eq 77
	./tests/test_api
	./tests/test_session

tests/test_prefill_dispatch: tests/test_prefill_dispatch.o $(filter-out src/main.o src/runtime.o,$(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_qsa_model_load: tests/test_qsa_model_load.o src/model.o src/expert.o src/owner.o src/qsa.o src/qsa_cache.o src/qsa_locality.o src/qsa_state.o src/protocol.o src/sampler.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/loader.o src/uring.o src/sha256.o src/manifest.o src/runtime_options.o src/topology.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_tokenizer: tests/test_tokenizer.o src/tokenizer.o src/prefix.o src/quant.o src/q38_math.o src/uring.o src/sha256.o src/manifest.o src/runtime_options.o src/topology.o src/q38_schema.o src/gguf.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_fabric: tests/test_fabric.o src/fabric.o src/pipeline.o src/quant.o src/q38_math.o src/uring.o src/protocol.o src/sampler.o src/manifest.o src/runtime_options.o src/sha256.o src/topology.o src/q38_schema.o src/gguf.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_ngram_deployment: tests/test_ngram_deployment.o $(TEST_COMMON)
	$(CC) $(CFLAGS) -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc \
		-o $@ $^ $(LDLIBS)

test-ngram-deployment: tests/test_ngram_deployment
	./tests/test_ngram_deployment

test-vulkan: tests/test_owner_reduce tests/test_fg_vk tests/test_hc_down_split tests/test_model_load tests/test_qsa_model_load tests/test_tokenizer tests/test_fabric tests/test_expert_prefill tests/test_qsa_prefill
	./tests/test_fg_vk
	./tests/test_hc_down_split || test $$? -eq 77
	./tests/test_model_load || test $$? -eq 77
	./tests/test_qsa_model_load || test $$? -eq 77
	./tests/test_tokenizer || test $$? -eq 77
	./tests/test_fabric || test $$? -eq 77
	./tests/test_expert_prefill || test $$? -eq 77
	./tests/test_qsa_prefill || test $$? -eq 77
	./tests/test_owner_reduce || test $$? -eq 77

QSA_FLEET_FILTERS := qsa_resident_commit_exact qsa_resident_early_exhaustive qsa_resident_hierarchical_topk qsa_resident_causal_batch_attention qsa_resident_t1_compat
test-qsa-fleet: tests/test_fg_vk tests/test_qsa_model_load
	DS4_REMOTE_TEST_FILTER=qsa_resident_geometry ./tests/test_qsa_model_load
	@for filter in $(QSA_FLEET_FILTERS); do \
		echo "BC250 QSA fleet filter: $$filter"; \
		DS4_REMOTE_TEST_FILTER=$$filter ./tests/test_fg_vk || exit $$?; \
	done
	DS4_REMOTE_TEST_FILTER=qsa_resident_lifecycle ./tests/test_qsa_model_load

GDN_FLEET_FILTERS := gdn_chunked_prefill_parity_zero gdn_chunked_prefill_parity_random gdn_chunked_prefill_parity_extreme gdn_chunked_prefill_composition gdn_chunked_prefill_decode_compat
test-gdn-fleet: tests/test_fg_vk
	@for filter in $(GDN_FLEET_FILTERS); do \
		echo "BC250 GDN fleet filter: $$filter"; \
		DS4_REMOTE_TEST_FILTER=$$filter ./tests/test_fg_vk || exit $$?; \
	done

clean:
	rm -f tools/bench-mtp-chain tools/bench-mtp-chain.d
	rm -f tools/mtp-offline tools/mtp-offline.d
	rm -f tools/mtp-native-replay tools/mtp-native-replay.d
	rm -f tools/mtp-tokens tools/mtp-tokens.d
	rm -f tools/mtp-residency tools/mtp-residency.d
	rm -f tools/mtp-real-place tools/mtp-real-place.d
	rm -f tools/mtp-real-cook tools/mtp-real-cook.d
	rm -f tests/test_mtp_ring
	rm -f tests/test_mtp_wire tests/test_mtp_native tests/test_mtp_embedding tools/mtp-embedding tools/mtp-embedding.d
	rm -f tests/test_mtp_capture_io tests/test_mtp_capture_io.d tools/mtp-acceptance tools/mtp-acceptance.d
	rm -f tests/test_mtp tests/test_mtp.d tools/bench-decode-fusion tools/bench-decode-fusion.d tools/bench-mtp-head tools/bench-mtp-head.d
	rm -f tests/test_q4_repack tests/test_q4_repack.d tools/q4-repack tools/q4-repack.d tools/q4-extract-pack tools/q4-extract-pack.d tools/q4-compare-packs tools/q4-compare-packs.d tools/q4-eval-batch tools/q4-eval-batch.d
	rm -f tests/test_multi_session
	rm -f tests/test_prefill_collective flash-gordon $(OBJ) $(DEP) vendor/*.o vendor/*.d tests/*.o tests/test_core tests/test_session tests/test_prefix tests/test_chat tests/test_chat_runtime tests/test_api tests/test_sampler tests/test_ngram_deployment tests/test_expert_prefill tests/test_qsa_prefill tests/test_owner_reduce tests/test_prefill_dispatch tests/test_decode_batch tests/test_expert_prefill.d tests/test_prefill_dispatch.d tests/test_decode_batch.d tests/test_fg_vk tests/test_hc_down_split tests/test_model_load tests/test_qsa_model_load tests/test_tokenizer tests/test_fabric tests/test_video vulkan/*.spv

-include $(DEP) tests/test_mtp.d tools/bench-decode-fusion.d tests/test_expert_prefill.d tests/test_prefill_dispatch.d tests/test_decode_batch.d tests/test_q4_repack.d tools/q4-repack.d tools/q4-extract-pack.d tools/q4-compare-packs.d tools/q4-eval-batch.d
