/*
 * SPDP v1.1 — Burtscher et al., DCC 2018.
 *
 * Algorithm reference (unmodified upstream snapshot):
 *   benchmark/reference/upstream/spdp/standalone/SPDP_11.c
 * Upstream: https://userweb.cs.txstate.edu/~burtscher/research/SPDPcompressor/
 *
 * This build copy omits main(); exposes spdp_compress_block / spdp_decompress_block
 * for benchmark_cpp (see spdp_codec.cpp). Block size limit matches upstream BUFFER_SIZE.
 */

#include <stdlib.h>
#include <string.h>

/*
SPDP code: SPDP is a unified compression/decompression algorithm that works
well on both binary 32-bit single-precision (float) and binary 64-bit double-
precision (double) floating-point data.

Copyright (c) 2015-2020, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Steven Claggett and Martin Burtscher
*/

#define SPDP_MAX_BLOCK (1 << 23)
#define MAX_TABLE_SIZE (1 << 18)

typedef unsigned char spdp_byte_t;
typedef unsigned int spdp_word_t;

static size_t spdp_compress_core(const spdp_byte_t level, const size_t length,
                                 spdp_byte_t *const buf1, spdp_byte_t *const buf2,
                                 const int ablation_mode) {
    spdp_word_t *in = (spdp_word_t *)buf1;
    spdp_word_t *out = (spdp_word_t *)buf2;
    size_t len = length / sizeof(spdp_word_t);

    size_t pos;
    if (ablation_mode == 1) {
        memcpy(buf2, buf1, length);
    } else {
        spdp_word_t prev2 = 0;
        spdp_word_t prev1 = 0;
        for (pos = 0; pos < len; pos++) {
            spdp_word_t curr = in[pos];
            out[pos] = curr - prev2;
            prev2 = prev1;
            prev1 = curr;
        }
        for (pos = len * sizeof(spdp_word_t); pos < length; pos++) {
            buf2[pos] = buf1[pos];
        }
    }

    spdp_byte_t prev = 0;
    size_t wpos = 0;
    if (ablation_mode == 2) {
        memcpy(buf1, buf2, length);
        wpos = length;
    } else {
        size_t d;
        for (d = 0; d < 8; d++) {
            size_t rpos;
            for (rpos = d; rpos < length; rpos += 8) {
                spdp_byte_t curr = buf2[rpos];
                buf1[wpos] = ablation_mode == 1 ? curr : (spdp_byte_t)(curr - prev);
                prev = curr;
                wpos++;
            }
        }
    }

    if (ablation_mode == 3) {
        memcpy(buf2, buf1, length);
        return length;
    }

    size_t predtabsize = 1u << (level + 9);
    if (predtabsize > MAX_TABLE_SIZE) {
        predtabsize = MAX_TABLE_SIZE;
    }
    const size_t predtabsizem1 = predtabsize - 1;

    unsigned int lastpos[MAX_TABLE_SIZE];
    memset(lastpos, 0, predtabsize * sizeof(unsigned int));

    size_t rpos = 0;
    wpos = 0;
    unsigned int hist = 0;
    while (rpos < length) {
        spdp_byte_t val = buf1[rpos];
        unsigned int lpos = lastpos[hist];
        if (lpos >= 6) {
            if ((buf1[lpos - 6] == buf1[rpos - 6]) && (buf1[lpos - 5] == buf1[rpos - 5]) &&
                (buf1[lpos - 4] == buf1[rpos - 4]) && (buf1[lpos - 3] == buf1[rpos - 3]) &&
                (buf1[lpos - 2] == buf1[rpos - 2]) && (buf1[lpos - 1] == buf1[rpos - 1])) {
                spdp_byte_t cnt = 0;
                while ((val == buf1[lpos]) && (cnt < 255) && (rpos < (length - 1))) {
                    lastpos[hist] = rpos;
                    hist = ((hist << 2) ^ val) & predtabsizem1;
                    rpos++;
                    lpos++;
                    cnt++;
                    val = buf1[rpos];
                }
                buf2[wpos] = cnt;
                wpos++;
            }
        }
        buf2[wpos] = val;
        wpos++;
        lastpos[hist] = rpos;
        hist = ((hist << 2) ^ val) & predtabsizem1;
        rpos++;
    }

    return wpos;
}

static void spdp_decompress_core(const spdp_byte_t level, const size_t length,
                                 spdp_byte_t *const buf2, spdp_byte_t *const buf1,
                                 const int ablation_mode) {
    unsigned int predtabsize = 1u << (level + 9);
    if (predtabsize > MAX_TABLE_SIZE) {
        predtabsize = MAX_TABLE_SIZE;
    }
    const unsigned int predtabsizem1 = predtabsize - 1;

    unsigned int lastpos[MAX_TABLE_SIZE];
    memset(lastpos, 0, predtabsize * sizeof(unsigned int));

    size_t rpos = 0;
    size_t wpos = 0;
    if (ablation_mode == 3) {
        memcpy(buf1, buf2, length);
        wpos = length;
    } else {
        unsigned int hist = 0;
        while (rpos < length) {
            unsigned int lpos = lastpos[hist];
            if (lpos >= 6) {
                if ((buf1[lpos - 6] == buf1[wpos - 6]) && (buf1[lpos - 5] == buf1[wpos - 5]) &&
                    (buf1[lpos - 4] == buf1[wpos - 4]) && (buf1[lpos - 3] == buf1[wpos - 3]) &&
                    (buf1[lpos - 2] == buf1[wpos - 2]) && (buf1[lpos - 1] == buf1[wpos - 1])) {
                    spdp_byte_t cnt = buf2[rpos];
                    rpos++;
                    spdp_byte_t j;
                    for (j = 0; j < cnt; j++) {
                        spdp_byte_t val = buf1[wpos] = buf1[lpos];
                        lastpos[hist] = wpos;
                        hist = ((hist << 2) ^ val) & predtabsizem1;
                        wpos++;
                        lpos++;
                    }
                }
            }
            spdp_byte_t val = buf1[wpos] = buf2[rpos];
            lastpos[hist] = wpos;
            hist = ((hist << 2) ^ val) & predtabsizem1;
            wpos++;
            rpos++;
        }
    }
    const size_t usize = wpos;

    if (ablation_mode == 2) {
        memcpy(buf2, buf1, usize);
    } else {
        spdp_byte_t val = 0;
        rpos = 0;
        size_t d;
        for (d = 0; d < 8; d++) {
            size_t wpos2;
            for (wpos2 = d; wpos2 < usize; wpos2 += 8) {
                if (ablation_mode == 1) val = buf1[rpos];
                else val = (spdp_byte_t)(val + buf1[rpos]);
                buf2[wpos2] = val;
                rpos++;
            }
        }
    }

    if (ablation_mode == 1) {
        memcpy(buf1, buf2, usize);
    } else {
        spdp_word_t *in = (spdp_word_t *)buf2;
        spdp_word_t *out = (spdp_word_t *)buf1;
        const size_t len = usize / sizeof(spdp_word_t);
        spdp_word_t prev2 = 0;
        spdp_word_t prev1 = 0;
        size_t pos;
        for (pos = 0; pos < len; pos++) {
            spdp_word_t curr = in[pos] + prev2;
            out[pos] = curr;
            prev2 = prev1;
            prev1 = curr;
        }
        for (pos = len * sizeof(spdp_word_t); pos < usize; pos++) buf1[pos] = buf2[pos];
    }
}

size_t spdp_compress_block(unsigned char level, const unsigned char *in, size_t len, unsigned char *out,
                           size_t out_cap) {
    if (level > 9 || len == 0 || len > SPDP_MAX_BLOCK || out_cap < len * 2 + 9) {
        return 0;
    }
    unsigned char *buf1 = (unsigned char *)malloc(len);
    unsigned char *buf2 = (unsigned char *)malloc(len * 2 + 9);
    if (!buf1 || !buf2) {
        free(buf1);
        free(buf2);
        return 0;
    }
    memcpy(buf1, in, len);
    size_t csize = spdp_compress_core((spdp_byte_t)level, len, buf1, buf2, 0);
    if (csize > out_cap) {
        free(buf1);
        free(buf2);
        return 0;
    }
    memcpy(out, buf2, csize);
    free(buf1);
    free(buf2);
    return csize;
}

int spdp_decompress_block(unsigned char level, const unsigned char *comp, size_t csize, unsigned char *out,
                          size_t out_len) {
    if (level > 9 || csize == 0 || out_len == 0 || out_len > SPDP_MAX_BLOCK) {
        return -1;
    }
    /* Second phase indexes buf2 up to intermediate length (≤ out_len); may exceed csize. */
    size_t buf2_cap = out_len * 2 + 9;
    if (buf2_cap < csize) {
        buf2_cap = csize;
    }
    unsigned char *buf2 = (unsigned char *)malloc(buf2_cap);
    unsigned char *buf1 = (unsigned char *)malloc(out_len);
    if (!buf2 || !buf1) {
        free(buf2);
        free(buf1);
        return -2;
    }
    memcpy(buf2, comp, csize);
    spdp_decompress_core((spdp_byte_t)level, csize, buf2, buf1, 0);
    memcpy(out, buf1, out_len);
    free(buf2);
    free(buf1);
    return 0;
}

size_t spdp_compress_block_ablation(unsigned char level, const unsigned char *in, size_t len,
                                    unsigned char *out, size_t out_cap, int ablation_mode) {
    if (level > 9 || len == 0 || len > SPDP_MAX_BLOCK || out_cap < len * 2 + 9 ||
        ablation_mode < 0 || ablation_mode > 3) return 0;
    unsigned char *buf1 = (unsigned char *)malloc(len);
    unsigned char *buf2 = (unsigned char *)malloc(len * 2 + 9);
    if (!buf1 || !buf2) {
        free(buf1);
        free(buf2);
        return 0;
    }
    memcpy(buf1, in, len);
    size_t csize = spdp_compress_core((spdp_byte_t)level, len, buf1, buf2, ablation_mode);
    if (csize <= out_cap) memcpy(out, buf2, csize);
    else csize = 0;
    free(buf1);
    free(buf2);
    return csize;
}

int spdp_decompress_block_ablation(unsigned char level, const unsigned char *comp, size_t csize,
                                   unsigned char *out, size_t out_len, int ablation_mode) {
    if (level > 9 || csize == 0 || out_len == 0 || out_len > SPDP_MAX_BLOCK ||
        ablation_mode < 0 || ablation_mode > 3) return -1;
    size_t buf2_cap = out_len * 2 + 9;
    if (buf2_cap < csize) buf2_cap = csize;
    unsigned char *buf2 = (unsigned char *)malloc(buf2_cap);
    unsigned char *buf1 = (unsigned char *)malloc(out_len);
    if (!buf2 || !buf1) {
        free(buf2);
        free(buf1);
        return -2;
    }
    memcpy(buf2, comp, csize);
    spdp_decompress_core((spdp_byte_t)level, csize, buf2, buf1, ablation_mode);
    memcpy(out, buf1, out_len);
    free(buf2);
    free(buf1);
    return 0;
}
