CC ?= cc
CFLAGS ?= -O3 -g -std=c17 -D_GNU_SOURCE -Wall -Wextra -Wpedantic -Werror -MMD -MP
CPPFLAGS ?= -Iinclude
LDLIBS ?= -pthread -lm -lvulkan
GLSLC ?= sh ./glslc
FG_SHADER_SRC := $(wildcard shaders/*.comp)
FG_SHADER_OUT := $(patsubst shaders/%.comp,vulkan/%.spv,$(FG_SHADER_SRC))

SRC := src/main.c src/util.c src/manifest.c src/session.c src/topology.c src/sha256.c src/gguf.c src/q38_schema.c src/q38_math.c src/quant.c src/vk.c src/model.c src/expert.c src/owner.c src/output.c src/tokenizer.c src/pack.c src/uring.c src/loader.c src/ngram.c src/qsa_state.c src/qsa.c src/protocol.c src/fabric.c src/prefix.c src/runtime_options.c src/runtime.c src/chat.c src/api.c
OBJ := $(SRC:.c=.o)
DEP := $(OBJ:.o=.d) tests/test_core.d tests/test_session.d tests/test_prefix.d tests/test_chat.d tests/test_chat_runtime.d tests/test_api.d tests/test_fg_vk.d tests/test_hc_down_split.d tests/test_model_load.d tests/test_tokenizer.d tests/test_fabric.d
TEST_COMMON := src/util.o src/manifest.o src/session.o src/topology.o src/sha256.o src/gguf.o src/q38_schema.o src/q38_math.o src/quant.o src/vk.o src/tokenizer.o src/pack.o src/uring.o src/loader.o src/ngram.o src/qsa_state.o src/protocol.o src/runtime_options.o

.PHONY: all clean test test-vulkan shaders
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

tests/test_api: tests/test_api.c src/api.c src/chat.c src/prefix.c src/util.c include/fg_api.h include/fg_chat.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DFG_CHAT_RENDER_ONLY -o $@ tests/test_api.c src/chat.c src/prefix.c src/util.c $(LDLIBS)

test: tests/test_core tests/test_session tests/test_prefix tests/test_chat tests/test_chat_runtime tests/test_api
	./tests/test_core
	./tests/test_session
	./tests/test_prefix
	./tests/test_chat
	./tests/test_chat_runtime
	./tests/test_api
	bash tests/test_ep_trace_validator.sh

shaders: $(FG_SHADER_OUT)

vulkan/%.spv: shaders/%.comp
	@mkdir -p vulkan
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

tests/test_fg_vk: tests/test_fg_vk.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/uring.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lvulkan

tests/test_hc_down_split: tests/test_hc_down_split.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/uring.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lvulkan

tests/test_model_load: tests/test_model_load.o src/model.o src/vk.o src/quant.o src/q38_math.o src/ngram.o src/loader.o src/uring.o src/sha256.o src/manifest.o src/topology.o src/q38_schema.o src/gguf.o src/util.o | shaders
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_tokenizer: tests/test_tokenizer.o src/tokenizer.o src/prefix.o src/quant.o src/q38_math.o src/uring.o src/sha256.o src/manifest.o src/topology.o src/q38_schema.o src/gguf.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_fabric: tests/test_fabric.o src/fabric.o src/quant.o src/q38_math.o src/uring.o src/protocol.o src/manifest.o src/sha256.o src/topology.o src/q38_schema.o src/util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-vulkan: tests/test_fg_vk tests/test_hc_down_split tests/test_model_load tests/test_tokenizer tests/test_fabric
	./tests/test_fg_vk
	./tests/test_hc_down_split || test $$? -eq 77
	./tests/test_model_load || test $$? -eq 77
	./tests/test_tokenizer || test $$? -eq 77
	./tests/test_fabric || test $$? -eq 77

clean:
	rm -f flash-gordon $(OBJ) $(DEP) tests/*.o tests/test_core tests/test_session tests/test_prefix tests/test_chat tests/test_chat_runtime tests/test_api tests/test_fg_vk tests/test_hc_down_split tests/test_model_load tests/test_tokenizer tests/test_fabric vulkan/*.spv

-include $(DEP)
