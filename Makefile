CC ?= cc
CFLAGS ?= -O3 -g -std=c17 -D_GNU_SOURCE -Wall -Wextra -Wpedantic -Werror -MMD -MP
CPPFLAGS ?= -Iinclude -Ivendor
LDLIBS ?= -pthread -lm -lvulkan
GLSLC ?= sh ./glslc
FG_SHADER_SRC := $(wildcard shaders/*.comp)
FG_SHADER_OUT := $(patsubst shaders/%.comp,vulkan/%.spv,$(FG_SHADER_SRC))

SRC := src/main.c vendor/stb_image_impl.c src/util.c src/tower.c src/tower_vk.c src/video.c src/manifest.c src/session.c src/topology.c src/sha256.c src/gguf.c src/q38_schema.c src/q38_math.c src/quant.c src/vk.c src/model.c src/expert.c src/owner.c src/output.c src/tokenizer.c src/pack.c src/uring.c src/loader.c src/ngram.c src/qsa_state.c src/qsa_locality.c src/qsa_cache.c src/qsa.c src/qsa_owner.c src/qsa_replica.c src/protocol.c src/fabric.c src/prefix.c src/runtime_options.c src/sampler.c src/mtp.c src/ledger.c src/decode_batch.c src/runtime.c src/chat.c src/api.c
SRC += src/mtp_head.c
SRC += src/mtp_capture.c
OBJ := $(SRC:.c=.o)
DEP := tests/test_prefill_collective.d tests/test_owner_reduce.d $(OBJ:.o=.d) tests/test_expert_prefill.d tests/test_prefill_dispatch.d tests/test_qsa_prefill.d tests/test_core.d tests/test_session.d tests/test_prefix.d tests/test_chat.d tests/test_chat_runtime.d tests/test_api.d tests/test_ngram_deployment.d tests/test_fg_vk.d tests/test_hc_down_split.d tests/test_model_load.d tests/test_qsa_model_load.d tests/test_tokenizer.d tests/test_fabric.d tests/test_tower.d tests/test_decode_batch.d
TEST_COMMON := src/util.o src/manifest.o src/session.o src/topology.o src/sha256.o src/gguf.o src/q38_schema.o src/q38_math.o src/quant.o src/vk.o src/tokenizer.o src/pack.o src/uring.o src/loader.o src/ngram.o src/qsa_state.o src/qsa_locality.o src/qsa_cache.o src/qsa_owner.o src/qsa_replica.o src/protocol.o src/runtime_options.o src/sampler.o src/ledger.o

.PHONY: all clean test test-sampler test-mtp test-mtp-capture test-ngram-deployment test-vulkan test-qsa-fleet test-gdn-fleet test-fg-profile-analysis test-tower test-decode-batch shaders
all: flash-gordon shaders

flash-gordon: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

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
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-mtp: tests/test_mtp
	./tests/test_mtp

tools/bench-decode-fusion: tools/bench_decode_fusion.c src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) $(LDLIBS)

tools/bench-mtp-head: tools/bench_mtp_head.c src/mtp_head.o src/model.o src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< src/mtp_head.o src/model.o src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) $(LDLIBS)

-include tools/bench-mtp-head.d

tools/mtp-acceptance: tools/mtp_acceptance.c src/mtp_head.o src/model.o src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) | shaders
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< src/mtp_head.o src/model.o src/expert.o src/owner.o src/qsa.o $(TEST_COMMON) $(LDLIBS)

-include tools/mtp-acceptance.d

tests/test_mtp_capture_io: tests/test_mtp_capture_io.c src/mtp_capture.o src/util.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

-include tests/test_mtp_capture_io.d

test-mtp-capture: tests/test_mtp_capture_io
	mkdir -p .build
	./tests/test_mtp_capture_io .build/capture-io-test
	python3 -m unittest discover -s tests -p test_mtp_capture.py

test: tests/test_q4_repack
test: tests/test_prefill_collective tests/test_core tests/test_session tests/test_prefix tests/test_chat tests/test_chat_runtime tests/test_api tests/test_ngram_deployment tests/test_prefill_dispatch tests/test_decode_batch tests/test_video tests/test_mtp
	@bash tools/check-flags.sh
	./tests/test_core
	./tests/test_q4_repack
	./tests/test_session
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
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_small_q4*.comp shaders/fg_small_q5*.comp shaders/fg_small_iq4*.comp)): shaders/fg_small_dense.glsl shaders/fg_dense_soa_impl.glsl
vulkan/fg_small_gate_up.spv vulkan/fg_small_gate_up_pairs.spv: shaders/fg_small_gate_up.glsl
vulkan/fg_gdn_recurrent_algebraic.spv vulkan/fg_gdn_small.spv: shaders/fg_gdn_algebraic.glsl
vulkan/fg_dense_q8_0_cooked_r8.spv vulkan/fg_mtp_vocab_q8.spv vulkan/fg_mtp_fc_hidden.spv: shaders/fg_dense_q8_0_cooked_r8.glsl
$(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_small_f32*.comp shaders/fg_small_router_bf16*.comp shaders/fg_small_index_bf16*.comp)): shaders/fg_small_plain.glsl
vulkan/fg_dense_q8_0_cooked.spv $(patsubst shaders/%.comp,vulkan/%.spv,$(wildcard shaders/fg_small_q8*.comp)): shaders/fg_small_q8_parallel.glsl
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

tools/q4-repack: tools/q4_repack.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/q4-extract-pack: tools/q4_extract_pack.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/q4-compare-packs: tools/q4_compare_packs.c $(TEST_COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

tools/q4-eval-batch: tools/q4_eval_batch.c $(filter-out src/main.o,$(OBJ))
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDLIBS)

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

tests/test_prefill_dispatch: tests/test_prefill_dispatch.o $(filter-out src/main.o src/runtime.o,$(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_qsa_model_load: tests/test_qsa_model_load.o src/model.o src/expert.o src/owner.o src/qsa.o src/qsa_cache.o src/qsa_locality.o src/qsa_state.o src/protocol.o src/sampler.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/loader.o src/uring.o src/sha256.o src/manifest.o src/runtime_options.o src/topology.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_tokenizer: tests/test_tokenizer.o src/tokenizer.o src/prefix.o src/quant.o src/q38_math.o src/uring.o src/sha256.o src/manifest.o src/runtime_options.o src/topology.o src/q38_schema.o src/gguf.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_fabric: tests/test_fabric.o src/fabric.o src/quant.o src/q38_math.o src/uring.o src/protocol.o src/sampler.o src/manifest.o src/runtime_options.o src/sha256.o src/topology.o src/q38_schema.o src/gguf.o src/util.o
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
	rm -f tests/test_mtp_capture_io tests/test_mtp_capture_io.d tools/mtp-acceptance tools/mtp-acceptance.d
	rm -f tests/test_mtp tests/test_mtp.d tools/bench-decode-fusion tools/bench-decode-fusion.d tools/bench-mtp-head tools/bench-mtp-head.d
	rm -f tests/test_q4_repack tests/test_q4_repack.d tools/q4-repack tools/q4-repack.d tools/q4-extract-pack tools/q4-extract-pack.d tools/q4-compare-packs tools/q4-compare-packs.d tools/q4-eval-batch tools/q4-eval-batch.d
	rm -f tests/test_prefill_collective flash-gordon $(OBJ) $(DEP) vendor/*.o vendor/*.d tests/*.o tests/test_core tests/test_session tests/test_prefix tests/test_chat tests/test_chat_runtime tests/test_api tests/test_sampler tests/test_ngram_deployment tests/test_expert_prefill tests/test_qsa_prefill tests/test_owner_reduce tests/test_prefill_dispatch tests/test_decode_batch tests/test_expert_prefill.d tests/test_prefill_dispatch.d tests/test_decode_batch.d tests/test_fg_vk tests/test_hc_down_split tests/test_model_load tests/test_qsa_model_load tests/test_tokenizer tests/test_fabric tests/test_video vulkan/*.spv

-include $(DEP) tests/test_mtp.d tools/bench-decode-fusion.d tests/test_expert_prefill.d tests/test_prefill_dispatch.d tests/test_decode_batch.d tests/test_q4_repack.d tools/q4-repack.d tools/q4-extract-pack.d tools/q4-compare-packs.d tools/q4-eval-batch.d
