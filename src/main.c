#include "fg.h"
#include "fg_api.h"
#include "fg_chat.h"
#include "fg_manifest.h"
#include "fg_pack.h"
#include "fg_runtime.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *file) {
    fprintf(file,
            "Flash Gordon %u.%u - Qwen3.8-Flash-Next BC250 appliance\n\n"
            "Usage:\n"
            "  flash-gordon pack --output DIR --source FILE [--source FILE ...] "
            "[--router-profile FILE | --expert-map FILE] [--dry-run]\n"
            "  flash-gordon verify --manifest FILE --pack-dir DIR --source FILE "
            "[--source FILE ...]\n"
            "  flash-gordon schema --source FILE [--source FILE ...]\n"
            "  flash-gordon rank --manifest FILE --rank N\n"
            "  flash-gordon serve --manifest FILE\n"
            "  flash-gordon bench --manifest FILE\n"
            "  flash-gordon eval --manifest FILE --prompt TEXT [--generate N]\n"
            "  flash-gordon chat --manifest FILE [--max-tokens N] [RUNTIME OPTIONS]\n"
            "  flash-gordon api --manifest FILE [--host HOST] [--port PORT] "
            "[RUNTIME OPTIONS]\n"
            "\nRuntime options:\n"
            "  --context-tokens N --gpu-index-tokens N --qsa-hot-tokens N "
            "--qsa-page-cache-mib N\n"
            "  --experimental-context N --experimental-mtp --experimental-vision\n"
            "  flash-gordon inspect --manifest FILE\n",
            FG_VERSION_MAJOR, FG_VERSION_MINOR);
}

static const char *arg_value(int *index, int argc, char **argv, const char *flag,
                             fg_error *err) {
    if (*index + 1 >= argc) {
        fg_error_set(err, FG_ERR_ARGUMENT, "%s requires a value", flag);
        return NULL;
    }
    return argv[++*index];
}

static fg_status parse_u32(const char *text, const char *flag, uint32_t minimum,
                           uint32_t maximum, uint32_t *value, fg_error *err) {
    if (!text) return err->code;
    errno = 0;
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);
    if (errno == ERANGE || !end || *end || parsed < minimum || parsed > maximum) {
        fg_error_set(err, FG_ERR_ARGUMENT, "%s must be an integer from %u to %u", flag,
                     minimum, maximum);
        return FG_ERR_ARGUMENT;
    }
    *value = (uint32_t)parsed;
    return FG_OK;
}

static fg_status parse_runtime_option(int *index, int argc, char **argv,
                                      fg_runtime_options *options, bool *handled,
                                      fg_error *err) {
    const char *flag = argv[*index];
    *handled = true;
    if (!strcmp(flag, "--context-tokens")) {
        return parse_u32(arg_value(index, argc, argv, flag, err), flag, 1u, FG_MAX_CONTEXT,
                         &options->logical_context_tokens, err);
    }
    if (!strcmp(flag, "--gpu-index-tokens")) {
        return parse_u32(arg_value(index, argc, argv, flag, err), flag, 1u, FG_MAX_CONTEXT,
                         &options->gpu_index_tokens, err);
    }
    if (!strcmp(flag, "--qsa-hot-tokens")) {
        return parse_u32(arg_value(index, argc, argv, flag, err), flag, 1u, FG_MAX_CONTEXT,
                         &options->qsa_hot_tokens, err);
    }
    if (!strcmp(flag, "--qsa-page-cache-mib")) {
        uint32_t mib = 0;
        fg_status status =
            parse_u32(arg_value(index, argc, argv, flag, err), flag, 0u, 4096u, &mib, err);
        if (status == FG_OK) options->qsa_page_cache_bytes = (uint64_t)mib << 20u;
        return status;
    }
    if (!strcmp(flag, "--experimental-context")) {
        fg_status status =
            parse_u32(arg_value(index, argc, argv, flag, err), flag, 1u, FG_MAX_CONTEXT,
                      &options->logical_context_tokens, err);
        if (status == FG_OK) options->experimental_flags |= FG_RUNTIME_EXPERIMENTAL_CONTEXT;
        return status;
    }
    if (!strcmp(flag, "--experimental-mtp")) {
        options->experimental_flags |= FG_RUNTIME_EXPERIMENTAL_MTP;
        return FG_OK;
    }
    if (!strcmp(flag, "--experimental-vision")) {
        options->experimental_flags |= FG_RUNTIME_EXPERIMENTAL_VISION;
        return FG_OK;
    }
    *handled = false;
    return FG_OK;
}

static fg_status pack_cmd(int argc, char **argv, fg_error *err) {
    fg_pack_options options = {0};
    const char **sources = calloc((size_t)argc, sizeof(*sources));
    if (!sources) {
        fg_error_set(err, FG_ERR_OOM, "allocate source arguments");
        return FG_ERR_OOM;
    }
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--output"))
            options.output_dir = arg_value(&i, argc, argv, "--output", err);
        else if (!strcmp(argv[i], "--source"))
            sources[options.source_count++] = arg_value(&i, argc, argv, "--source", err);
        else if (!strcmp(argv[i], "--router-profile"))
            options.router_profile_path =
                arg_value(&i, argc, argv, "--router-profile", err);
        else if (!strcmp(argv[i], "--expert-map"))
            options.expert_map_path = arg_value(&i, argc, argv, "--expert-map", err);
        else if (!strcmp(argv[i], "--dry-run"))
            options.dry_run = true;
        else {
            fg_error_set(err, FG_ERR_ARGUMENT, "unknown pack option: %s", argv[i]);
            free(sources);
            return FG_ERR_ARGUMENT;
        }
        if (err->code != FG_OK) {
            free(sources);
            return err->code;
        }
    }
    options.source_paths = sources;
    fg_status status = fg_pack_run(&options, err);
    free(sources);
    return status;
}

static fg_status verify_cmd(int argc, char **argv, fg_error *err) {
    fg_verify_options options = {0};
    const char **sources = calloc((size_t)argc, sizeof(*sources));
    if (!sources) {
        fg_error_set(err, FG_ERR_OOM, "allocate verify source arguments");
        return FG_ERR_OOM;
    }
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--manifest"))
            options.manifest_path = arg_value(&i, argc, argv, "--manifest", err);
        else if (!strcmp(argv[i], "--pack-dir"))
            options.pack_dir = arg_value(&i, argc, argv, "--pack-dir", err);
        else if (!strcmp(argv[i], "--source"))
            sources[options.source_count++] = arg_value(&i, argc, argv, "--source", err);
        else {
            fg_error_set(err, FG_ERR_ARGUMENT, "unknown verify option: %s", argv[i]);
            free(sources);
            return FG_ERR_ARGUMENT;
        }
        if (err->code != FG_OK) {
            free(sources);
            return err->code;
        }
    }
    options.source_paths = sources;
    fg_status status = fg_pack_verify(&options, err);
    free(sources);
    return status;
}

static fg_status schema_cmd(int argc, char **argv, fg_error *err) {
    const char **sources = calloc((size_t)argc, sizeof(*sources));
    uint32_t count = 0;
    if (!sources) {
        fg_error_set(err, FG_ERR_OOM, "allocate schema source arguments");
        return FG_ERR_OOM;
    }
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--source"))
            sources[count++] = arg_value(&i, argc, argv, "--source", err);
        else {
            fg_error_set(err, FG_ERR_ARGUMENT, "unknown schema option: %s", argv[i]);
            free(sources);
            return FG_ERR_ARGUMENT;
        }
    }
    fg_gguf gguf;
    fg_status status = fg_gguf_open(sources, count, &gguf, err);
    if (status == FG_OK) {
        fg_gguf_print_schema(&gguf);
        fg_gguf_close(&gguf);
    }
    free(sources);
    return status;
}

static fg_status chat_cmd(int argc, char **argv, fg_error *err) {
    const char *manifest = NULL;
    uint32_t max_tokens = 512u;
    fg_runtime_options runtime_options;
    fg_runtime_options_init(&runtime_options);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--manifest")) {
            manifest = arg_value(&i, argc, argv, "--manifest", err);
        } else if (!strcmp(argv[i], "--max-tokens")) {
            const char *text = arg_value(&i, argc, argv, "--max-tokens", err);
            if (text && parse_u32(text, "--max-tokens", 1u, 4096u, &max_tokens, err) !=
                            FG_OK)
                return err->code;
        } else {
            bool handled = false;
            fg_status status =
                parse_runtime_option(&i, argc, argv, &runtime_options, &handled, err);
            if (status != FG_OK) return status;
            if (handled) continue;
            fg_error_set(err, FG_ERR_ARGUMENT, "unknown chat option: %s", argv[i]);
            return FG_ERR_ARGUMENT;
        }
    }
    if (!manifest) {
        fg_error_set(err, FG_ERR_ARGUMENT, "chat requires --manifest");
        return FG_ERR_ARGUMENT;
    }
    return fg_chat_main_with_options(manifest, max_tokens, &runtime_options, err);
}

static fg_status api_cmd(int argc, char **argv, fg_error *err) {
    const char *manifest = NULL;
    const char *host = "127.0.0.1";
    uint32_t port = 8000u;
    fg_runtime_options runtime_options;
    fg_runtime_options_init(&runtime_options);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--manifest")) {
            manifest = arg_value(&i, argc, argv, "--manifest", err);
        } else if (!strcmp(argv[i], "--host")) {
            host = arg_value(&i, argc, argv, "--host", err);
        } else if (!strcmp(argv[i], "--port")) {
            const char *text = arg_value(&i, argc, argv, "--port", err);
            if (text && parse_u32(text, "--port", 1u, UINT16_MAX, &port, err) != FG_OK)
                return err->code;
        } else {
            bool handled = false;
            fg_status status =
                parse_runtime_option(&i, argc, argv, &runtime_options, &handled, err);
            if (status != FG_OK) return status;
            if (handled) continue;
            fg_error_set(err, FG_ERR_ARGUMENT, "unknown api option: %s", argv[i]);
            return FG_ERR_ARGUMENT;
        }
    }
    if (!manifest) {
        fg_error_set(err, FG_ERR_ARGUMENT, "api requires --manifest");
        return FG_ERR_ARGUMENT;
    }
    return fg_api_main_with_options(manifest, host, (uint16_t)port, &runtime_options, err);
}

static fg_status manifest_cmd(const char *command, int argc, char **argv, fg_error *err) {
    const char *path = NULL;
    const char *prompt = NULL;
    uint32_t rank = UINT32_MAX;
    uint32_t generate = 1u;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--manifest")) {
            path = arg_value(&i, argc, argv, "--manifest", err);
        } else if (!strcmp(argv[i], "--rank")) {
            const char *text = arg_value(&i, argc, argv, "--rank", err);
            if (text && parse_u32(text, "--rank", 0u, UINT32_MAX, &rank, err) != FG_OK)
                return err->code;
        } else if (!strcmp(argv[i], "--prompt")) {
            prompt = arg_value(&i, argc, argv, "--prompt", err);
        } else if (!strcmp(argv[i], "--generate")) {
            const char *text = arg_value(&i, argc, argv, "--generate", err);
            if (text &&
                parse_u32(text, "--generate", 0u, UINT32_MAX, &generate, err) != FG_OK)
                return err->code;
        } else {
            fg_error_set(err, FG_ERR_ARGUMENT, "unknown %s option: %s", command, argv[i]);
            return FG_ERR_ARGUMENT;
        }
    }
    if (!path) {
        fg_error_set(err, FG_ERR_ARGUMENT, "%s requires --manifest", command);
        return FG_ERR_ARGUMENT;
    }
    if (!strcmp(command, "rank")) return fg_rank_main(path, rank, err);
    if (!strcmp(command, "serve")) return fg_serve_main(path, err);
    if (!strcmp(command, "bench")) return fg_bench_main(path, err);
    if (!strcmp(command, "eval")) {
        if (!prompt) {
            fg_error_set(err, FG_ERR_ARGUMENT, "eval requires --prompt");
            return FG_ERR_ARGUMENT;
        }
        return fg_eval_main(path, prompt, generate, err);
    }
    if (!strcmp(command, "inspect")) {
        fg_manifest *manifest = malloc(sizeof(*manifest));
        if (!manifest) {
            fg_error_set(err, FG_ERR_OOM, "allocate manifest");
            return FG_ERR_OOM;
        }
        fg_status status = fg_manifest_read(path, manifest, err);
        if (status == FG_OK) fg_manifest_print(manifest);
        free(manifest);
        return status;
    }
    return FG_ERR_ARGUMENT;
}

int main(int argc, char **argv) {
    fg_error err = {0};
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "help")) {
        usage(stdout);
        return 0;
    }
    fg_status status;
    if (!strcmp(argv[1], "pack"))
        status = pack_cmd(argc, argv, &err);
    else if (!strcmp(argv[1], "verify"))
        status = verify_cmd(argc, argv, &err);
    else if (!strcmp(argv[1], "schema"))
        status = schema_cmd(argc, argv, &err);
    else if (!strcmp(argv[1], "chat"))
        status = chat_cmd(argc, argv, &err);
    else if (!strcmp(argv[1], "api"))
        status = api_cmd(argc, argv, &err);
    else if (!strcmp(argv[1], "rank") || !strcmp(argv[1], "serve") ||
             !strcmp(argv[1], "bench") || !strcmp(argv[1], "eval") ||
             !strcmp(argv[1], "inspect"))
        status = manifest_cmd(argv[1], argc, argv, &err);
    else {
        usage(stderr);
        fprintf(stderr, "\nunknown command: %s\n", argv[1]);
        return FG_ERR_ARGUMENT;
    }
    if (status != FG_OK)
        fprintf(stderr, "flash-gordon: %s\n",
                err.message[0] ? err.message : "operation failed");
    return (int)status;
}
