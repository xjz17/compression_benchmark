#ifndef COMPRESSION_BENCHMARK_NNCP_PROBABILITY_H
#define COMPRESSION_BENCHMARK_NNCP_PROBABILITY_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NNCP 3.3 default Transformer, CPU float32, batch=1, segment=1, seed=123.
 * Each handle owns an independent upstream model. Probabilities precede observe.
 */
typedef struct NNCPProbability NNCPProbability;
#define NNCP_PROBABILITY_SYMBOL_LIMIT 4096
NNCPProbability *nncp_probability_create(void);
int nncp_probability_predict(NNCPProbability *state, float probabilities[256]);
int nncp_probability_observe(NNCPProbability *state, unsigned symbol);
void nncp_probability_destroy(NNCPProbability *state);

/* Matched upstream arithmetic coding, used to verify cold model reconstruction. */
int nncp_probability_encode(FILE *input, FILE *output, uint64_t count);
int nncp_probability_decode(FILE *input, FILE *output, uint64_t count);

#ifdef __cplusplus
}
#endif
#endif
