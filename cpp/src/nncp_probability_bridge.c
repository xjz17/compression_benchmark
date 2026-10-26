/* The model, optimizer and arithmetic coding below are the unmodified upstream
 * NNCP 3.3 implementation, Copyright Fabrice Bellard, MIT. LibNC is its original
 * separately licensed binary dependency. See docs/nncp_probability.md.
 */
#define main nncp_upstream_main
#include "../../external/prediction_probability/nncp-2024-06-05/nncp.c"
#undef main
#include "nncp_probability.h"

struct NNCPProbability {
    NNCPModelState *model;
    NCTensor *input;
    NCTensor *expected;
    float probabilities[256];
    unsigned previous;
    unsigned observed;
    int pending;
    int failed;
};

NNCPProbability *nncp_probability_create(void)
{
    NNCPModelParams params;
    NNCPProbability *state = calloc(1, sizeof(*state));
    if (!state)
        return NULL;
    nncp_init_params(&params, "default");
    /* This is an explicit streaming profile of the unchanged upstream model.
     * No future symbol is present in the one-symbol training segment. */
    params.batch_size = 1;
    params.seg_len = 1;
    params.seq_eval = FALSE;
    for (int i = 0; i < params.u.trf.n_layer; ++i)
        params.u.trf.attn_len[i] = params.u.trf.mem_len + params.seg_len;
    state->model = nncp_init(&params);
    state->input = nc_new_tensor_2d(state->model->cpu_device, NC_TYPE_I32, 1, 1);
    state->expected = nc_new_tensor_2d(state->model->cpu_device, NC_TYPE_I32, 1, 1);
    state->model->model_class->model_reset(state->model);
    return state;
}

int nncp_probability_predict(NNCPProbability *state, float probabilities[256])
{
    if (!state || !probabilities || state->failed || state->observed >= NNCP_PROBABILITY_SYMBOL_LIMIT)
        return -1;
    if (!state->pending) {
        size_t stride;
        nc_set1_i32_2d(state->input, 0, 0, state->previous);
        NCTensor *output = state->model->model_class->model_eval(state->model, -1, state->input);
        output = nc_tensor_to_cpu_device(output);
        const float *values = nc_tensor_get_ptr(output, &stride);
        memcpy(state->probabilities, values, sizeof(state->probabilities));
        nc_free_tensor(output);
        state->pending = 1;
        double sum = 0;
        for (int i = 0; i < 256; ++i) {
            if (!isfinite(state->probabilities[i]) || state->probabilities[i] < 0 || state->probabilities[i] > 1) {
                state->failed = 1;
                return -1;
            }
            sum += state->probabilities[i];
        }
        if (fabs(sum - 1.0) > 0.0001) {
            state->failed = 1;
            return -1;
        }
    }
    memcpy(probabilities, state->probabilities, sizeof(state->probabilities));
    return 0;
}

int nncp_probability_observe(NNCPProbability *state, unsigned symbol)
{
    if (!state || !state->pending || state->failed || symbol > 255 || state->observed >= NNCP_PROBABILITY_SYMBOL_LIMIT)
        return -1;
    NNCPModelState *model = state->model;
    nc_set1_i32_2d(state->expected, 0, 0, symbol);
    model->model_class->model_set_lr(model, get_interp_param(&model->lr, model->train_step));
    model->model_class->model_eval_gradient(model, state->expected);
    model->model_class->model_update(model);
    model->train_step++;
    state->previous = symbol;
    state->observed++;
    state->pending = 0;
    return 0;
}

void nncp_probability_destroy(NNCPProbability *state)
{
    if (!state)
        return;
    if (state->pending)
        state->model->model_class->model_eval_end(state->model);
    nc_free_tensor(state->input);
    nc_free_tensor(state->expected);
    nncp_end(state->model);
    free(state);
}

int nncp_probability_encode(FILE *input, FILE *output, uint64_t count)
{
    if (!input || !output || count > NNCP_PROBABILITY_SYMBOL_LIMIT)
        return -1;
    uint8_t buffer[4096];
    PutBitState writer;
    NNCPProbability *state = nncp_probability_create();
    if (!state)
        return -1;
    put_bit_init(&writer, buffer, sizeof(buffer), arith_write_buf, output);
    for (uint64_t i = 0; i < count; ++i) {
        float probabilities[256];
        int symbol = fgetc(input);
        if (symbol == EOF || nncp_probability_predict(state, probabilities) < 0) {
            nncp_probability_destroy(state);
            return -1;
        }
        write_sym(&writer, probabilities, 256, symbol);
        if (nncp_probability_observe(state, symbol) < 0) {
            nncp_probability_destroy(state);
            return -1;
        }
    }
    put_bit_flush(&writer);
    nncp_probability_destroy(state);
    return ferror(output) ? -1 : 0;
}

int nncp_probability_decode(FILE *input, FILE *output, uint64_t count)
{
    if (!input || !output || count > NNCP_PROBABILITY_SYMBOL_LIMIT)
        return -1;
    const long body_start = ftell(input);
    if (body_start < 0)
        return -1;
    FILE *canonical = tmpfile();
    if (!canonical)
        return -1;
    uint8_t buffer[4096], encoded_buffer[4096];
    GetBitState reader;
    PutBitState writer;
    NNCPProbability *state = nncp_probability_create();
    if (!state) {
        fclose(canonical);
        return -1;
    }
    get_bit_init(&reader, buffer, sizeof(buffer), arith_read_buf, input);
    put_bit_init(&writer, encoded_buffer, sizeof(encoded_buffer), arith_write_buf, canonical);
    for (uint64_t i = 0; i < count; ++i) {
        float probabilities[256];
        if (nncp_probability_predict(state, probabilities) < 0) {
            nncp_probability_destroy(state);
            fclose(canonical);
            return -1;
        }
        const int symbol = read_sym(&reader, probabilities, 256);
        write_sym(&writer, probabilities, 256, symbol);
        if (fputc(symbol, output) == EOF || nncp_probability_observe(state, symbol) < 0) {
            nncp_probability_destroy(state);
            fclose(canonical);
            return -1;
        }
    }
    put_bit_flush(&writer);
    nncp_probability_destroy(state);
    /* The upstream reader pads EOF with zero bits. Compare the canonical body
     * reconstructed with the same predictions to reject hidden tail/truncation. */
    if (ferror(input) || ferror(canonical) || fseek(input, body_start, SEEK_SET) != 0 ||
        fseek(canonical, 0, SEEK_SET) != 0) {
        fclose(canonical);
        return -1;
    }
    int original, encoded;
    do {
        original = fgetc(input);
        encoded = fgetc(canonical);
        if (original != encoded) {
            fclose(canonical);
            return -1;
        }
    } while (original != EOF);
    const int error = ferror(input) || ferror(canonical);
    fclose(canonical);
    return error ? -1 : 0;
}
