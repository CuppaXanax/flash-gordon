CC ?= cc
CFLAGS ?= -O3 -g -std=c17 -D_GNU_SOURCE -Wall -Wextra -Wpedantic -Werror -MMD -MP
CPPFLAGS ?= -Iinclude
LDLIBS ?= -pthread -lm -lvulkan
GLSLC ?= sh ./glslc
FG_SHADER_SRC := $(wildcard shaders/*.comp)
FG_SHADER_OUT := $(patsubst shaders/%.comp,vulkan/%.spv,$(FG_SHADER_SRC))

SRC := src/main.c src/util.c src/manifest.c src/session.c src/topology.c src/sha256.c src/gguf.c src/q38_schema.c src/q38_math.c src/quant.c src/vk.c src/model.c src/expert.c src/owner.c src/output.c src/stage.c src/embedding.c src/tokenizer.c src/pack.c src/uring.c src/loader.c src/ngram.c src/qsa_state.c src/qsa_locality.c src/qsa_cache.c src/qsa.c src/qsa_owner.c src/qsa_replica.c src/protocol.c src/fabric.c src/pipeline.c src/pipeline_runtime.c src/prefix.c src/runtime_options.c src/sampler.c src/runtime.c src/chat.c src/api.c
OBJ := $(SRC:.c=.o)
DEP := $(OBJ:.o=.d) tests/test_core.d tests/test_session.d tests/test_prefix.d tests/test_chat.d tests/test_chat_runtime.d tests/test_api.d tests/test_embedding.d tests/test_ngram_deployment.d tests/test_fg_vk.d tests/test_hc_down_split.d tests/test_model_load.d tests/test_qsa_model_load.d tests/test_tokenizer.d tests/test_fabric.d tests/test_pipeline.d tests/test_stage.d tests/bench_bc250_roofline.d tests/test_bc250_roofline.d tests/bench_bc250_qsa_curve.d tests/test_bc250_qsa_curve.d src/bc250_roofline.d src/bc250_qsa_curve.d
TEST_COMMON := src/util.o src/manifest.o src/session.o src/topology.o src/sha256.o src/gguf.o src/q38_schema.o src/q38_math.o src/quant.o src/vk.o src/tokenizer.o src/pack.o src/uring.o src/loader.o src/ngram.o src/qsa_state.o src/qsa_locality.o src/qsa_cache.o src/qsa_owner.o src/qsa_replica.o src/protocol.o src/runtime_options.o src/sampler.o

.PHONY: all clean test test-sampler test-embedding test-ngram-deployment test-vulkan test-qsa-fleet test-gdn-fleet test-pipeline-decode-fleet test_stage bench-bc250-roofline test-bc250-roofline bench-bc250-qsa-curve test-bc250-qsa-curve shaders
all: flash-gordon

flash-gordon: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_core: tests/test_core.o $(TEST_COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session: tests/test_session.o $(TEST_COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_prefix: tests/test_prefix.c src/prefix.c include/fg_prefix.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tests/test_prefix.c src/prefix.c src/util.c $(LDLIBS)

tests/test_chat: tests/test_chat.c src/chat.c src/prefix.c src/util.c include/fg_chat.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DFG_CHAT_RENDER_ONLY -o $@ tests/test_chat.c src/chat.c src/prefix.c src/util.c $(LDLIBS)

tests/test_chat_runtime: tests/test_chat_runtime.c src/chat.c src/prefix.c src/util.c include/fg_chat.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tests/test_chat_runtime.c src/prefix.c src/util.c $(LDLIBS)

tests/test_api: tests/test_api.c src/api.c src/chat.c src/prefix.c src/util.c src/sampler.o include/fg_api.h include/fg_chat.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DFG_CHAT_RENDER_ONLY -o $@ tests/test_api.c src/chat.c src/prefix.c src/util.c src/sampler.o $(LDLIBS)

tests/test_sampler: tests/test_sampler.c src/sampler.o src/util.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-sampler: tests/test_sampler
	./tests/test_sampler

tests/test_stage: tests/test_stage.o src/stage.o src/expert.o src/sampler.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_stage: tests/test_stage
	./tests/test_stage

tests/test_embedding: tests/test_embedding.o src/embedding.o src/quant.o src/sha256.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

test-embedding: tests/test_embedding
	./tests/test_embedding

test: tests/test_core tests/test_session tests/test_prefix tests/test_chat tests/test_chat_runtime tests/test_api tests/test_embedding tests/test_ngram_deployment tests/test_pipeline
	./tests/test_core
	./tests/test_session
	./tests/test_prefix
	./tests/test_chat
	./tests/test_chat_runtime
	./tests/test_api
	./tests/test_embedding
	./tests/test_ngram_deployment
	./tests/test_pipeline
	bash tests/test_ep_trace_validator.sh

shaders: $(FG_SHADER_OUT)

vulkan/%.spv: shaders/%.comp
	@mkdir -p vulkan
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

tests/test_fg_vk: tests/test_fg_vk.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/uring.o src/sha256.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lvulkan

tests/bench_bc250_roofline: tests/bench_bc250_roofline.o src/bc250_roofline.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/uring.o src/sha256.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lvulkan

bench-bc250-roofline: tests/bench_bc250_roofline
	./tests/bench_bc250_roofline

tests/test_bc250_roofline: tests/test_bc250_roofline.o src/bc250_roofline.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ -lm

test-bc250-roofline: tests/test_bc250_roofline
	./tests/test_bc250_roofline

tests/bench_bc250_qsa_curve: tests/bench_bc250_qsa_curve.o src/bc250_qsa_curve.o src/bc250_roofline.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/uring.o src/sha256.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lvulkan

bench-bc250-qsa-curve: tests/bench_bc250_qsa_curve
	./tests/bench_bc250_qsa_curve

tests/test_bc250_qsa_curve: tests/test_bc250_qsa_curve.o src/bc250_qsa_curve.o src/bc250_roofline.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ -lm

test-bc250-qsa-curve: tests/test_bc250_qsa_curve
	./tests/test_bc250_qsa_curve

tests/test_hc_down_split: tests/test_hc_down_split.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/uring.o src/sha256.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lvulkan

tests/test_model_load: tests/test_model_load.o src/model.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/loader.o src/uring.o src/sha256.o src/manifest.o src/runtime_options.o src/topology.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_qsa_model_load: tests/test_qsa_model_load.o src/model.o src/expert.o src/owner.o src/qsa.o src/qsa_cache.o src/qsa_locality.o src/qsa_state.o src/protocol.o src/sampler.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/loader.o src/uring.o src/sha256.o src/manifest.o src/runtime_options.o src/topology.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_tokenizer: tests/test_tokenizer.o src/tokenizer.o src/prefix.o src/quant.o src/q38_math.o src/uring.o src/sha256.o src/manifest.o src/runtime_options.o src/topology.o src/q38_schema.o src/gguf.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_fabric: tests/test_fabric.o src/fabric.o src/quant.o src/q38_math.o src/uring.o src/protocol.o src/sampler.o src/manifest.o src/runtime_options.o src/sha256.o src/topology.o src/q38_schema.o src/gguf.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_pipeline: tests/test_pipeline.o src/pipeline_runtime.o src/pipeline.o src/fabric.o src/quant.o src/q38_math.o src/uring.o src/protocol.o src/sampler.o src/manifest.o src/runtime_options.o src/sha256.o src/topology.o src/q38_schema.o src/gguf.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_ngram_deployment: tests/test_ngram_deployment.o $(TEST_COMMON)
	$(CC) $(CFLAGS) -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc \
		-o $@ $^ $(LDLIBS)

test-ngram-deployment: tests/test_ngram_deployment
	./tests/test_ngram_deployment

test-vulkan: tests/test_fg_vk tests/test_hc_down_split tests/test_model_load tests/test_qsa_model_load tests/test_tokenizer tests/test_fabric
	./tests/test_fg_vk
	./tests/test_hc_down_split || test $$? -eq 77
	./tests/test_model_load || test $$? -eq 77
	./tests/test_qsa_model_load || test $$? -eq 77
	./tests/test_tokenizer || test $$? -eq 77
	./tests/test_fabric || test $$? -eq 77

QSA_FLEET_FILTERS := qsa_resident_commit_exact qsa_resident_early_exhaustive qsa_resident_hierarchical_topk qsa_resident_causal_batch_attention qsa_resident_t1_compat
test-qsa-fleet: tests/test_fg_vk tests/test_qsa_model_load
	DS4_REMOTE_TEST_FILTER=qsa_resident_geometry ./tests/test_qsa_model_load
	@for filter in $(QSA_FLEET_FILTERS); do \
		echo "BC250 QSA fleet filter: $$filter"; \
		DS4_REMOTE_TEST_FILTER=$$filter ./tests/test_fg_vk || exit $$?; \
	done
	DS4_REMOTE_TEST_FILTER=qsa_resident_lifecycle ./tests/test_qsa_model_load

GDN_FLEET_FILTERS := gdn_pipeline_prefill_parity_zero gdn_pipeline_prefill_parity_random gdn_pipeline_prefill_parity_extreme gdn_pipeline_prefill_composition gdn_pipeline_prefill_decode_compat
test-gdn-fleet: tests/test_fg_vk
	@for filter in $(GDN_FLEET_FILTERS); do \
		echo "BC250 GDN fleet filter: $$filter"; \
		DS4_REMOTE_TEST_FILTER=$$filter ./tests/test_fg_vk || exit $$?; \
	done

PIPELINE_DECODE_VK_FILTERS := batch_submission_parity decode_tile_schedule kquant_cooked pipeline_decode_down_formats moe_reduce ple_decode gdn_algebraic gdn_decode qsa_resident_t1_compat expert_graph_replay
PIPELINE_DECODE_MOCK_FILTERS := pipeline_decode_mapping pipeline_decode_geometry pipeline_decode_down_formats pipeline_decode_batch pipeline_decode_failure_cleanup pipeline_decode_terminal_submissions
test-pipeline-decode-fleet: tests/test_fg_vk tests/test_stage
	@for filter in $(PIPELINE_DECODE_VK_FILTERS); do \
		echo "BC250 pipeline decode Vulkan filter: $$filter"; \
		DS4_REMOTE_TEST_FILTER=$$filter ./tests/test_fg_vk || exit $$?; \
	done
	@for filter in $(PIPELINE_DECODE_MOCK_FILTERS); do \
		echo "Pipeline decode stage filter: $$filter"; \
		DS4_REMOTE_TEST_FILTER=$$filter ./tests/test_stage || exit $$?; \
	done

clean:
	rm -f flash-gordon $(OBJ) $(DEP) tests/*.o tests/test_core tests/test_session tests/test_prefix tests/test_chat tests/test_chat_runtime tests/test_api tests/test_sampler tests/test_embedding tests/test_ngram_deployment tests/test_fg_vk tests/bench_bc250_roofline tests/test_bc250_roofline tests/bench_bc250_qsa_curve tests/test_bc250_qsa_curve tests/test_hc_down_split tests/test_model_load tests/test_qsa_model_load tests/test_tokenizer tests/test_fabric tests/test_pipeline tests/test_stage vulkan/*.spv

-include $(DEP)
