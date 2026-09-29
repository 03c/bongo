// route_capture — dump per-(layer, token, expert) MoE router selections from llama.cpp.
//
// Built against the headers at commit 4da6337767f973e2b4d0797e5b323d77d8565e4a
// (the same commit as the bongo vulkan build, "build 11223"). It links the
// *existing* libllama.so and installs a ggml_backend_sched_eval_callback that
// copies the `ffn_moe_topk-<layer>` tensor (I32 view, shape
// [n_expert_used, n_tokens]) once the graph node has been computed.
//
// Two phases:
//   prefill — one llama_decode over tokens[0 .. n_keep), all tokens routed.
//   decode  — teacher-forced single-token decodes over the tail tokens, which
//             produces genuine per-token decode routing (including the final
//             layer, which prefill only evaluates at the last position).
//
// Output is TSV. One line per layer per phase:
//   TOPK<TAB>ffn_moe_topk-<il><TAB>ne0<TAB>ne1<TAB>ne2<TAB>ne3<TAB>idx...
// Flat order is token-major: element (k, token) is at k + token*ne0.
// Before every decode step a `STEP<TAB><i>` marker line is written to the
// decode file, so token boundaries are recoverable even if a step is skipped.
//
// Usage:
//   route_capture <model.gguf> <prompt.txt> <prefill.tsv> <n_ctx> [decode_steps] [decode.tsv]

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE * g_out = NULL;
static long   g_hits = 0;

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (ask) {
        return true;  // we want every node's data
    }
    if (t == NULL || g_out == NULL) {
        return true;
    }
    const char * name = t->name;
    if (name != NULL && strncmp(name, "ffn_moe_topk", 12) == 0 && t->type == GGML_TYPE_I32) {
        // ffn_moe_topk is a strided view of the layer's argsort output, so
        // ggml_nbytes() over-counts. Walk the real strides and emit exactly
        // ne[0]*ne[1]*ne[2]*ne[3] values in token-major order.
        fprintf(g_out, "TOPK\t%s", name);
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            fprintf(g_out, "\t%lld", (long long) t->ne[d]);
        }
        for (int i3 = 0; i3 < t->ne[3]; i3++) {
            for (int i2 = 0; i2 < t->ne[2]; i2++) {
                for (int i1 = 0; i1 < t->ne[1]; i1++) {
                    for (int i0 = 0; i0 < t->ne[0]; i0++) {
                        const size_t off = (size_t) i0 * t->nb[0]
                                         + (size_t) i1 * t->nb[1]
                                         + (size_t) i2 * t->nb[2]
                                         + (size_t) i3 * t->nb[3];
                        int32_t v = -1;
                        ggml_backend_tensor_get(t, &v, off, sizeof(v));
                        fprintf(g_out, "\t%d", v);
                    }
                }
            }
        }
        fputc('\n', g_out);
        g_hits++;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <model.gguf> <prompt.txt> <prefill.tsv> <n_ctx> [decode_steps] [decode.tsv]\n",
                argv[0]);
        return 2;
    }
    const char * model_path   = argv[1];
    const char * prompt_path  = argv[2];
    const char * out_prefill  = argv[3];
    const int    n_ctx        = atoi(argv[4]);
    const int    decode_steps = argc > 5 ? atoi(argv[5]) : 0;
    const char * out_decode   = argc > 6 ? argv[6] : NULL;

    FILE * pf = fopen(prompt_path, "rb");
    if (pf == NULL) { perror("prompt"); return 2; }
    fseek(pf, 0, SEEK_END);
    long psz = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    char * text = (char *) malloc((size_t) psz + 1);
    if (fread(text, 1, (size_t) psz, pf) != (size_t) psz) { perror("fread"); return 2; }
    text[psz] = '\0';
    fclose(pf);

    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;      // router capture is backend-independent; keep it on host RAM
    mp.load_mode    = LLAMA_LOAD_MODE_MMAP;

    struct llama_model * model = llama_model_load_from_file(model_path, mp);
    if (model == NULL) { fprintf(stderr, "model load failed\n"); return 1; }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    llama_token * tokens = (llama_token *) malloc(sizeof(llama_token) * ((size_t) psz + 16));
    int32_t nt = llama_tokenize(vocab, text, (int32_t) psz, tokens, (int32_t) (psz + 16), true, true);
    if (nt <= 0) { fprintf(stderr, "tokenize failed: %d\n", nt); return 1; }

    int n_prefill = nt;
    if (decode_steps > 0) {
        if (decode_steps >= nt) { fprintf(stderr, "decode_steps >= tokens\n"); return 1; }
        n_prefill = nt - decode_steps;
    }
    if (n_prefill > n_ctx) { fprintf(stderr, "prefill %d > n_ctx %d\n", n_prefill, n_ctx); return 1; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = (uint32_t) (n_ctx < nt ? nt + 8 : n_ctx);
    cp.n_batch           = cp.n_ctx;
    cp.n_ubatch          = cp.n_ctx;   // one graph, every prefill token routes in one pass
    cp.cb_eval           = eval_cb;
    cp.cb_eval_user_data = NULL;

    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == NULL) { fprintf(stderr, "context init failed\n"); return 1; }

    fprintf(stderr, "route_capture: prompt tokens=%d prefill=%d decode_steps=%d n_ctx=%d\n",
            nt, n_prefill, decode_steps, cp.n_ctx);

    g_out = fopen(out_prefill, "w");
    if (g_out == NULL) { perror("prefill out"); return 1; }
    struct llama_batch batch = llama_batch_get_one(tokens, n_prefill);
    int rc = llama_decode(ctx, batch);
    fprintf(stderr, "route_capture: prefill decode rc=%d topk_nodes=%ld\n", rc, g_hits);
    fclose(g_out);
    g_out = NULL;
    if (rc != 0) { return 1; }

    if (decode_steps > 0 && out_decode != NULL) {
        FILE * df = fopen(out_decode, "w");
        if (df == NULL) { perror("decode out"); return 1; }
        g_out = df;
        g_hits = 0;
        for (int i = 0; i < decode_steps; i++) {
            fprintf(df, "STEP\t%d\n", i);
            llama_token tok = tokens[n_prefill + i];
            struct llama_batch one = llama_batch_get_one(&tok, 1);
            rc = llama_decode(ctx, one);
            if (rc != 0) { fprintf(stderr, "route_capture: decode step %d rc=%d\n", i, rc); break; }
            if ((i + 1) % 32 == 0) {
                fprintf(stderr, "route_capture: decode step %d/%d\n", i + 1, decode_steps);
            }
        }
        fprintf(stderr, "route_capture: decode topk_nodes=%ld\n", g_hits);
        fclose(df);
        g_out = NULL;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(tokens);
    free(text);
    return 0;
}
