/*
Copyright (c) 2008 The University of Texas.  All rights reserved.
Author: Dr. Martin Burtscher

For Software License Terms and Conditions, see http://users.ices.utexas.edu/~burtscher/research/pFPC/pFPC_license.pdf.
*/


#include "pfpc.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>


#define ALIGNSIZE (4 * 1024)  /* should be equal to the system's page size */
#define BUFSIZE (8 * 1024 * 1024)  /* has to be a large power of two */
#define MAXTHREADS 128  /* must be less than 256 */

#define HLSHIFT 6
#define HRSHIFT 48
#define DLSHIFT 2
#define DRSHIFT 40

#define ull unsigned long long

static const ull mask[8] =
{0x0000000000000000LL,
 0x00000000000000ffLL,
 0x000000000000ffffLL,
 0x0000000000ffffffLL,
 0x000000ffffffffffLL,
 0x0000ffffffffffffLL,
 0x00ffffffffffffffLL,
 0xffffffffffffffffLL};


static volatile int g_predsizem1;
static volatile int g_chunksize;
static volatile int g_threads;
static volatile struct CompStruct {
  ull *ibuf;
  unsigned char *obuf;
  ull *fcm;
  ull *dfcm;
  int end;
} g_comp[MAXTHREADS];
static volatile struct DecompStruct {
  unsigned char *ibuf;
  ull *obuf;
  ull *fcm;
  ull *dfcm;
  int end;
} g_decomp[MAXTHREADS];
static pthread_t thread[MAXTHREADS];



/*
This is the function that should be called by each parallel thread to compress the chunks of data assigned to this thread.

Inputs
------
threads: the number of parallel threads
chunksize: the number of consecutive doubles that make up a chunk
end: the number of doubles to the end of ibuf
ibuf: a pointer to the first double in the input buffer that this thread should compress
obuf: a pointer to the first byte of the output buffer that this thread will write (make sure the buffer is large enough)
predsizem1: the fcm and dfcm predictor size (which must be a power of two) minus one [in number of doubles]
fcm: a pointer to the fcm predictor table
dfcm: a pointer to the dfcm predictor table

Output
------
The compressed data in obuf (the first four bytes specify the total number of bytes)
*/

static void CompressChunk(const int threads, int chunksize, const int end, const ull * restrict const ibuf, unsigned char * restrict const obuf, const int predsizem1, ull * restrict const fcm, ull * restrict const dfcm)
{
  register int i, o, out, hash, dhash, code, bcode, save, offset, increment, full, part;
  register ull val, lastval, stride, pred1, pred2, xor1, xor2;

  full = end / (chunksize * threads);
  part = end - full * chunksize * threads;
  if (part > chunksize) part = chunksize;

  out = full * chunksize + part;
  out = ((out + 1) >> 1) + 4;
  *((ull *)&obuf[out & -8]) = 0;

  hash = 0;
  dhash = 0;
  lastval = 0;
  pred1 = 0;
  pred2 = 0;
  save = 0;

  o = 8;
  offset = 0;
  increment = threads * chunksize;
  while (0 < (end - offset)) {
    if (chunksize > (end - offset)) {
      chunksize = end - offset;
    }
    for (i = 0; i < chunksize; i++) {
      val = ibuf[offset + i];
      xor1 = val ^ pred1;
      fcm[hash] = val;
      hash = ((hash << HLSHIFT) ^ (val >> HRSHIFT)) & predsizem1;
      pred1 = fcm[hash];

      stride = val - lastval;
      xor2 = val ^ (lastval + pred2);
      lastval = val;
      dfcm[dhash] = stride;
      dhash = ((dhash << DLSHIFT) ^ (stride >> DRSHIFT)) & predsizem1;
      pred2 = dfcm[dhash];

      code = 0;
      if (xor1 > xor2) {
        code = 0x8;
        xor1 = xor2;
      }
      bcode = 7;
      if (0 == (xor1 >> 56)) bcode = 6;
      if (0 == (xor1 >> 48)) bcode = 5;
      if (0 == (xor1 >> 40)) bcode = 4;
      if (0 == (xor1 >> 24)) bcode = 3;
      if (0 == (xor1 >> 16)) bcode = 2;
      if (0 == (xor1 >> 8)) bcode = 1;
      if (0 == xor1) bcode = 0;

      *((ull *)&obuf[out & -8]) |= xor1 << ((out & 0x7) << 3);
      *((ull *)&obuf[(out & -8) + 8]) = (xor1 >> (63 - ((out & 0x7) << 3))) >> 1;

      code |= bcode;
      out += bcode + (bcode >> 2);
      obuf[o >> 1] = save | code;
      save = code << 4;
      o++;
    }
    offset += increment;
  }
  if (0 != (o & 1)) {
    obuf[o >> 1] = save;
  }

  *((int *)&obuf[0]) = out;  // number of (compressed) bytes
}


static void *CompressChunks(void *arg)
{
  register int id = (int)arg;

  // retrieve the 'parameters' passed via global memory
  CompressChunk(g_threads, g_chunksize, g_comp[id].end, g_comp[id].ibuf, g_comp[id].obuf, g_predsizem1, g_comp[id].fcm, g_comp[id].dfcm);
  pthread_exit(NULL);
}


static int CompressToMemory(int num, ull *inbuf, unsigned char **inter, int predsizem1, int threads, int chunksize,
                            ull **fcm, ull **dfcm, unsigned char *out, size_t out_cap, size_t *out_off)
{
  register int i, cnt;
  int status;
  pthread_attr_t attr;

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  g_predsizem1 = predsizem1;
  g_chunksize = chunksize;
  g_threads = threads;
  for (i = 0; i < threads; i++) {
    g_comp[i].ibuf = &inbuf[i * chunksize];
    g_comp[i].obuf = inter[i];
    g_comp[i].fcm = fcm[i];
    g_comp[i].dfcm = dfcm[i];
    g_comp[i].end = num - (i * chunksize);
    if (0 > g_comp[i].end) g_comp[i].end = 0;
  }

  for (i = 0; i < threads; i++) {
    pthread_create(&thread[i], &attr, CompressChunks, (void *)(long)i);
  }

  if (*out_off + 4 > out_cap) {
    for (i = 0; i < threads; i++) {
      pthread_join(thread[i], (void **)&status);
    }
    pthread_attr_destroy(&attr);
    return -1;
  }
  memcpy(out + *out_off, &num, 4);
  *out_off += 4;

  for (i = 0; i < threads; i++) {
    pthread_join(thread[i], (void **)&status);
    assert(0 == status);

    cnt = *((int *)inter[i]);
    if (*out_off + (size_t)cnt > out_cap) {
      pthread_attr_destroy(&attr);
      return -1;
    }
    memcpy(out + *out_off, inter[i], (size_t)cnt);
    *out_off += (size_t)cnt;
  }

  pthread_attr_destroy(&attr);
  return 0;
}


/*
This is the function that should be called by each parallel thread to compress the chunks of data assigned to this thread.

Inputs
------
threads: the number of parallel threads
chunksize: the number of consecutive doubles that make up a chunk
end: the number of doubles to the end of the obuf
ibuf: a pointer to the first byte in the input buffer that this thread should decompress
obuf: a pointer to the first double of the output buffer that this thread will write (make sure the buffer is large enough)
predsizem1: the fcm and dfcm predictor size (which must be a power of two) minus one [in number of doubles]
fcm: a pointer to the fcm predictor table
dfcm: a pointer to the dfcm predictor table

Output
------
The decompressed data in obuf
*/

static void DecompressChunk(const int threads, int chunksize, const int end, const unsigned char * restrict const ibuf, ull * restrict const obuf, const int predsizem1, ull * restrict const fcm, ull * restrict const dfcm)
{
  register int i, o, alt, in, hash, dhash, bcode, tmp, offset, increment, full, part;
  register ull val, lastval, stride, pred1, pred2, next;

  hash = 0;
  dhash = 0;
  lastval = 0;
  pred1 = 0;
  pred2 = 0;

  full = end / (chunksize * threads);
  part = end - full * chunksize * threads;
  if (part > chunksize)
    part = chunksize;
  in = full * chunksize + part;
  in = ((in + 1) >> 1) + 4;
  i = 8;
  alt = 4;

  offset = 0;
  increment = threads * chunksize;
  while (0 < (end - offset)) {
    if (chunksize > (end - offset)) {
      chunksize = end - offset;
    }
    for (o = 0; o < chunksize; o++) {
      bcode = ibuf[i >> 1] >> alt;
      i++;
      alt ^= 4;

      val = *((ull *)&ibuf[in & -8]);
      next = *((ull *)&ibuf[(in & -8) + 8]);
      tmp = (in & 0x7) << 3;
      val >>= tmp;
      next <<= 63 - tmp;
      val |= next + next;

      if (0 != (bcode & 0x8))
        pred1 = pred2;
      bcode &= 0x7;
      val &= mask[bcode];
      in += bcode + (bcode >> 2);
      val ^= pred1;

      fcm[hash] = val;
      hash = ((hash << HLSHIFT) ^ (val >> HRSHIFT)) & predsizem1;
      pred1 = fcm[hash];

      stride = val - lastval;
      dfcm[dhash] = stride;
      dhash = ((dhash << DLSHIFT) ^ (stride >> DRSHIFT)) & predsizem1;
      pred2 = val + dfcm[dhash];
      lastval = val;

      obuf[offset + o] = val;
    }
    offset += increment;
  }
}


static void *DecompressChunks(void *arg)
{
  register int id = (int)arg;

  // retrieve the 'parameters' passed via global memory
  DecompressChunk(g_threads, g_chunksize, g_decomp[id].end, g_decomp[id].ibuf, g_decomp[id].obuf, g_predsizem1, g_decomp[id].fcm, g_decomp[id].dfcm);
  pthread_exit(NULL);
}


static int DecompressToMemory(int num, unsigned char **inter, ull *outbuf, int predsizem1, int threads, int chunksize,
                              ull **fcm, ull **dfcm)
{
  register int i;
  int status;
  pthread_attr_t attr;

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  g_predsizem1 = predsizem1;
  g_chunksize = chunksize;
  g_threads = threads;
  for (i = 0; i < threads; i++) {
    g_decomp[i].ibuf = inter[i];
    g_decomp[i].obuf = &outbuf[i * chunksize];
    g_decomp[i].fcm = fcm[i];
    g_decomp[i].dfcm = dfcm[i];
    g_decomp[i].end = num - (i * chunksize);
  }

  for (i = 0; i < threads; i++) {
    pthread_create(&thread[i], &attr, DecompressChunks, (void *)(long)i);
  }

  for (i = 0; i < threads; i++) {
    pthread_join(thread[i], (void **)&status);
    assert(0 == status);
  }

  pthread_attr_destroy(&attr);
  return 0;
}


static unsigned char *AlignedAlloc(int size, int factor)
{
  register int diff;
  register unsigned char *buf;

  // allocate memory and zero out
  buf = (unsigned char *)calloc(size + ALIGNSIZE - 1, factor);
  assert(NULL != buf);

  // adjust buffer beginning to align size
  diff = ((long)buf) % ALIGNSIZE;
  if (0 != diff) {
    buf += ALIGNSIZE - diff;
  }

  return buf;
}


static int pfpc_chunkrepeat(int threads, int chunksize)
{
  int chunkrepeat = BUFSIZE / (8 * threads * chunksize);
  if ((chunkrepeat * 8 * threads * chunksize) < BUFSIZE) {
    chunkrepeat++;
  }
  return chunkrepeat;
}

static int pfpc_alloc_work(int threads, int chunk_size, int predsizem1, unsigned char ***inter_out, ull ***fcm_out,
                           ull ***dfcm_out, ull **output_out)
{
  register int i;
  const int chunkrepeat = pfpc_chunkrepeat(threads, chunk_size);
  const size_t inter_bytes =
      (size_t)(8 + ((chunk_size * chunkrepeat + 1) / 2) + (chunk_size * chunkrepeat * 8) + 7);
  unsigned char **inter = (unsigned char **)calloc((size_t)threads, sizeof(unsigned char *));
  ull **fcm = (ull **)calloc((size_t)threads, sizeof(ull *));
  ull **dfcm = (ull **)calloc((size_t)threads, sizeof(ull *));
  ull *output = (ull *)calloc(BUFSIZE >> 3, sizeof(ull));
  if (inter == NULL || fcm == NULL || dfcm == NULL || output == NULL) {
    free(inter);
    free(fcm);
    free(dfcm);
    free(output);
    return -1;
  }
  for (i = 0; i < threads; i++) {
    inter[i] = (unsigned char *)calloc(inter_bytes, 1);
    fcm[i] = (ull *)calloc((size_t)(predsizem1 + 1), sizeof(ull));
    dfcm[i] = (ull *)calloc((size_t)(predsizem1 + 1), sizeof(ull));
    if (inter[i] == NULL || fcm[i] == NULL || dfcm[i] == NULL) {
      for (int j = 0; j <= i; j++) {
        free(inter[j]);
        free(fcm[j]);
        free(dfcm[j]);
      }
      free(inter);
      free(fcm);
      free(dfcm);
      free(output);
      return -1;
    }
  }
  *inter_out = inter;
  *fcm_out = fcm;
  *dfcm_out = dfcm;
  *output_out = output;
  return 0;
}

static void pfpc_free_work(unsigned char **inter, ull **fcm, ull **dfcm, ull *output, int threads)
{
  if (inter != NULL) {
    for (int i = 0; i < threads; i++) {
      free(inter[i]);
    }
    free(inter);
  }
  if (fcm != NULL) {
    for (int i = 0; i < threads; i++) {
      free(fcm[i]);
    }
    free(fcm);
  }
  if (dfcm != NULL) {
    for (int i = 0; i < threads; i++) {
      free(dfcm[i]);
    }
    free(dfcm);
  }
  free(output);
}

size_t pfpc_compress_memory(const void *in_stream, size_t in_bytes, void *out_stream, size_t out_cap,
                            unsigned char pred_size_lg2, unsigned char threads, int chunk_size)
{
  if (in_bytes == 0 || (in_bytes % 8) != 0 || out_stream == NULL || out_cap < 6) {
    return 0;
  }
  if (threads < 1 || threads >= 256 || chunk_size < 1 || pred_size_lg2 >= 256) {
    return 0;
  }

  unsigned char *out = (unsigned char *)out_stream;
  size_t out_off = 0;
  out[0] = pred_size_lg2;
  out[1] = threads;
  memcpy(out + 2, &chunk_size, 4);
  out_off = 6;

  const int predsizem1 = (1 << pred_size_lg2) - 1;
  unsigned char **inter = NULL;
  ull **fcm = NULL;
  ull **dfcm = NULL;
  ull *output = NULL;
  if (pfpc_alloc_work(threads, chunk_size, predsizem1, &inter, &fcm, &dfcm, &output) != 0) {
    return 0;
  }

  const ull *input = (const ull *)in_stream;
  size_t remaining = in_bytes / 8;
  const size_t max_block = BUFSIZE >> 3;

  while (remaining > 0) {
    const int num = (int)((remaining > max_block) ? max_block : remaining);
    if (CompressToMemory((int)num, (ull *)input, inter, predsizem1, threads, chunk_size, fcm, dfcm, out, out_cap,
                         &out_off) != 0) {
      pfpc_free_work(inter, fcm, dfcm, output, threads);
      return 0;
    }
    input += num;
    remaining -= (size_t)num;
  }

  pfpc_free_work(inter, fcm, dfcm, output, threads);
  return out_off;
}

size_t pfpc_decompress_memory(const void *in_stream, size_t in_bytes, void *out_stream, size_t out_cap)
{
  if (in_stream == NULL || out_stream == NULL || in_bytes < 6) {
    return 0;
  }

  const unsigned char *in = (const unsigned char *)in_stream;
  const unsigned char pred_lg2 = in[0];
  const unsigned char threads = in[1];
  int chunksize = 0;
  memcpy(&chunksize, in + 2, 4);
  if (threads < 1 || threads >= 256 || chunksize < 1 || pred_lg2 >= 256) {
    return 0;
  }

  size_t in_off = 6;
  const int predsizem1 = (1 << pred_lg2) - 1;
  unsigned char **inter = NULL;
  ull **fcm = NULL;
  ull **dfcm = NULL;
  ull *output = NULL;
  if (pfpc_alloc_work(threads, chunksize, predsizem1, &inter, &fcm, &dfcm, &output) != 0) {
    return 0;
  }

  unsigned char *out = (unsigned char *)out_stream;
  size_t out_written = 0;

  while (in_off + 4 <= in_bytes) {
    int num = 0;
    memcpy(&num, in + in_off, 4);
    in_off += 4;
    if (num <= 0) {
      break;
    }

    register int i;
    for (i = 0; i < threads; i++) {
      if (in_off + 4 > in_bytes) {
        pfpc_free_work(inter, fcm, dfcm, output, threads);
        return 0;
      }
      int chunk_total = 0;
      memcpy(&chunk_total, in + in_off, 4);
      if (chunk_total < 4 || in_off + (size_t)chunk_total > in_bytes) {
        pfpc_free_work(inter, fcm, dfcm, output, threads);
        return 0;
      }
      memcpy(inter[i], in + in_off, (size_t)chunk_total);
      in_off += (size_t)chunk_total;
    }

    if (DecompressToMemory(num, inter, output, predsizem1, threads, chunksize, fcm, dfcm) != 0) {
      pfpc_free_work(inter, fcm, dfcm, output, threads);
      return 0;
    }

    const size_t block_bytes = (size_t)num * 8;
    if (out_written + block_bytes > out_cap) {
      pfpc_free_work(inter, fcm, dfcm, output, threads);
      return 0;
    }
    memcpy(out + out_written, output, block_bytes);
    out_written += block_bytes;
  }

  pfpc_free_work(inter, fcm, dfcm, output, threads);
  return in_off;
}

static int pfpc_ablation_byte_count(ull value)
{
  int bytes = 8;
  while (bytes > 0 && ((value >> ((bytes - 1) * 8)) & 0xffULL) == 0) {
    bytes--;
  }
  return bytes;
}

static void pfpc_ablation_update_predictors(ull value, int predsizem1,
                                             ull *fcm, ull *dfcm,
                                             int *hash, int *dhash,
                                             ull *lastval, ull *pred1,
                                             ull *pred2)
{
  const ull stride = value - *lastval;
  fcm[*hash] = value;
  *hash = ((*hash << HLSHIFT) ^ (int)(value >> HRSHIFT)) & predsizem1;
  *pred1 = fcm[*hash];
  dfcm[*dhash] = stride;
  *dhash = ((*dhash << DLSHIFT) ^ (int)(stride >> DRSHIFT)) & predsizem1;
  *pred2 = value + dfcm[*dhash];
  *lastval = value;
}

size_t pfpc_compress_memory_ablation(const void *in_stream, size_t in_bytes,
                                     void *out_stream, size_t out_cap,
                                     unsigned char pred_size_lg2,
                                     enum pfpc_ablation_mode mode)
{
  static const unsigned char magic[4] = {'P', 'F', 'A', '1'};
  if (in_stream == NULL || out_stream == NULL || in_bytes == 0 ||
      (in_bytes % 8) != 0 || pred_size_lg2 > 24 ||
      (mode != PFPC_ABLATE_XOR && mode != PFPC_ABLATE_TRAILING_ZERO_PACK) ||
      out_cap < 14) {
    return 0;
  }
  const size_t count = in_bytes / 8;
  unsigned char *out = (unsigned char *)out_stream;
  memcpy(out, magic, sizeof(magic));
  out[4] = (unsigned char)mode;
  out[5] = pred_size_lg2;
  memcpy(out + 6, &count, 8);
  size_t pos = 14;

  const int predsizem1 = (1 << pred_size_lg2) - 1;
  ull *fcm = (ull *)calloc((size_t)(predsizem1 + 1), sizeof(ull));
  ull *dfcm = (ull *)calloc((size_t)(predsizem1 + 1), sizeof(ull));
  if (fcm == NULL || dfcm == NULL) {
    free(fcm);
    free(dfcm);
    return 0;
  }
  int hash = 0, dhash = 0;
  ull lastval = 0, pred1 = 0, pred2 = 0;
  const ull *input = (const ull *)in_stream;
  for (size_t i = 0; i < count; i++) {
    const ull value = input[i];
    if (mode == PFPC_ABLATE_XOR) {
      const int bytes = pfpc_ablation_byte_count(value);
      if (pos + 1 + (size_t)bytes > out_cap) {
        free(fcm);
        free(dfcm);
        return 0;
      }
      out[pos++] = (unsigned char)bytes;
      for (int byte = 0; byte < bytes; byte++) {
        out[pos++] = (unsigned char)((value >> (byte * 8)) & 0xffULL);
      }
    } else {
      ull residual = value ^ pred1;
      unsigned char selector = 0;
      const ull residual2 = value ^ pred2;
      if (residual2 < residual) {
        residual = residual2;
        selector = 1;
      }
      if (pos + 9 > out_cap) {
        free(fcm);
        free(dfcm);
        return 0;
      }
      out[pos++] = selector;
      memcpy(out + pos, &residual, 8);
      pos += 8;
    }
    pfpc_ablation_update_predictors(value, predsizem1, fcm, dfcm,
                                    &hash, &dhash, &lastval, &pred1, &pred2);
  }
  free(fcm);
  free(dfcm);
  return pos;
}

size_t pfpc_decompress_memory_ablation(const void *in_stream, size_t in_bytes,
                                       void *out_stream, size_t out_cap)
{
  static const unsigned char magic[4] = {'P', 'F', 'A', '1'};
  if (in_stream == NULL || out_stream == NULL || in_bytes < 14 ||
      memcmp(in_stream, magic, sizeof(magic)) != 0) {
    return 0;
  }
  const unsigned char *in = (const unsigned char *)in_stream;
  const enum pfpc_ablation_mode mode = (enum pfpc_ablation_mode)in[4];
  const unsigned char pred_size_lg2 = in[5];
  size_t count = 0;
  memcpy(&count, in + 6, 8);
  if ((mode != PFPC_ABLATE_XOR && mode != PFPC_ABLATE_TRAILING_ZERO_PACK) ||
      pred_size_lg2 > 24 || count > out_cap / 8) {
    return 0;
  }

  const int predsizem1 = (1 << pred_size_lg2) - 1;
  ull *fcm = (ull *)calloc((size_t)(predsizem1 + 1), sizeof(ull));
  ull *dfcm = (ull *)calloc((size_t)(predsizem1 + 1), sizeof(ull));
  if (fcm == NULL || dfcm == NULL) {
    free(fcm);
    free(dfcm);
    return 0;
  }
  int hash = 0, dhash = 0;
  ull lastval = 0, pred1 = 0, pred2 = 0;
  ull *output = (ull *)out_stream;
  size_t pos = 14;
  for (size_t i = 0; i < count; i++) {
    ull value = 0;
    if (mode == PFPC_ABLATE_XOR) {
      if (pos >= in_bytes) {
        free(fcm);
        free(dfcm);
        return 0;
      }
      const int bytes = in[pos++];
      if (bytes < 0 || bytes > 8 || pos + (size_t)bytes > in_bytes) {
        free(fcm);
        free(dfcm);
        return 0;
      }
      for (int byte = 0; byte < bytes; byte++) {
        value |= ((ull)in[pos++]) << (byte * 8);
      }
    } else {
      if (pos + 9 > in_bytes || in[pos] > 1) {
        free(fcm);
        free(dfcm);
        return 0;
      }
      const unsigned char selector = in[pos++];
      ull residual = 0;
      memcpy(&residual, in + pos, 8);
      pos += 8;
      value = residual ^ (selector == 0 ? pred1 : pred2);
    }
    output[i] = value;
    pfpc_ablation_update_predictors(value, predsizem1, fcm, dfcm,
                                    &hash, &dhash, &lastval, &pred1, &pred2);
  }
  free(fcm);
  free(dfcm);
  return pos == in_bytes ? pos : 0;
}
