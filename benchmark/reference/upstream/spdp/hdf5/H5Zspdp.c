/*
SPDP compression filter for HDF5.
SPDP is a fast, lossless, unified compression/decompression algorithm designed
for both 32-bit single-precision (float) and 64-bit double-precision (double)
floating-point data.  It also works on other data.

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

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/SPDP/.

Publication: This work is described in detail in the following paper.
Steven Claggett, Sahar Azimi, and Martin Burtscher. SPDP: An Automatically
Synthesized Lossless Compression Algorithm for Floating-Point Data. Proceedings
of the 2018 Data Compression Conference, pp. 337-346. March 2018.
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include "H5PLextern.h"

#define maxpredtabsize (1 << 18)

typedef unsigned char byte_t;
typedef unsigned int word_t;

static size_t H5Z_filter_spdp(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes, size_t* buf_size, void** buf);

#define H5Z_filter_SPDP 32009

const H5Z_class2_t H5Z_SPDP[1] = {{
  H5Z_CLASS_T_VERS,              /* H5Z_class_t version */
  (H5Z_filter_t)H5Z_filter_SPDP, /* Filter id number */
  1,              /* encoder_present flag (set to true) */
  1,              /* decoder_present flag (set to true) */
  "SPDP compressor/decompressor for SP and DP floating-point data.", /* Filter name for debugging */
  NULL,                          /* The "can apply" callback */
  NULL,                          /* The "set local" callback */
  (H5Z_func_t)H5Z_filter_spdp,   /* The actual filter function */
}};

H5PL_type_t H5PLget_plugin_type(void) {return H5PL_TYPE_FILTER;}
const void *H5PLget_plugin_info(void) {return H5Z_SPDP;}


static size_t H5Z_filter_spdp(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes, size_t* buf_size, void** buf)
{
  assert(1 == sizeof(byte_t));
  assert(4 == sizeof(word_t));
  assert((((size_t)*buf) % sizeof(word_t)) == 0);  /* check alignment */
  if (nbytes <= 0) return 0;

  size_t size = 0;
  if (flags & H5Z_FLAG_REVERSE) {  /* decompress data */
    byte_t* buf2 = (byte_t*)*buf;
    nbytes--;
    byte_t i, cnt = buf2[nbytes];
    unsigned int level = (cnt >> 4) & 15;
    cnt &= 15;
    for (i = 0; i < cnt; i++) {
      nbytes--;
      size = (size << 8) + buf2[nbytes];
    }

    buf2 = (byte_t*)malloc(size);  assert(buf2 != NULL);
    byte_t* buf1 = (byte_t*)malloc(size);  assert(buf1 != NULL);
    *buf_size = size;
    byte_t* tin = (byte_t*)*buf;
    unsigned int predtabsize = 1 << (level + 9);
    if (predtabsize > maxpredtabsize) predtabsize = maxpredtabsize;
    const int predtabsizem1 = predtabsize - 1;

    int lastpos[maxpredtabsize];
    memset(lastpos, 0, predtabsize * sizeof(int));

    size_t rpos = 0;
    size_t wpos = 0;
    int hist = 0;
    while (rpos < nbytes) {
      int lpos = lastpos[hist];
      if (lpos >= 6) {
        if ((buf1[lpos - 6] == buf1[wpos - 6]) && (buf1[lpos - 5] == buf1[wpos - 5]) &&
            (buf1[lpos - 4] == buf1[wpos - 4]) && (buf1[lpos - 3] == buf1[wpos - 3]) &&
            (buf1[lpos - 2] == buf1[wpos - 2]) && (buf1[lpos - 1] == buf1[wpos - 1])) {
          byte_t cnt = tin[rpos];
          rpos++;
          int j;
          for (j = 0; j < cnt; j++) {
            byte_t val = buf1[wpos] = buf1[lpos];
            lastpos[hist] = wpos;
            hist = ((hist << 2) ^ val) & predtabsizem1;
            wpos++;
            lpos++;
          }
        }
      }
      byte_t val = buf1[wpos] = tin[rpos];
      lastpos[hist] = wpos;
      hist = ((hist << 2) ^ val) & predtabsizem1;
      wpos++;
      rpos++;
    }
    assert(wpos == size);
    free(*buf);
    const size_t usize = wpos;

    byte_t val = 0;
    rpos = 0;
    int d;
    for (d = 0; d < 8; d++) {
      size_t wpos;
      for (wpos = d; wpos < usize; wpos += 8) {
        val += buf1[rpos];
        buf2[wpos] = val;
        rpos++;
      }
    }

    word_t* in = (word_t*)buf2;
    word_t* out = (word_t*)buf1;
    const size_t len = size / sizeof(word_t);

    word_t prev2 = 0;
    word_t prev1 = 0;
    size_t pos;
    for (pos = 0; pos < len; pos++) {
      word_t curr = in[pos] + prev2;
      out[pos] = curr;
      prev2 = prev1;
      prev1 = curr;
    }
    for (pos = len * sizeof(word_t); pos < usize; pos++) {
      buf1[pos] = buf2[pos];
    }

    free(buf2);
    *buf = buf1;
  } else {  /* compress data */
    word_t* in = (word_t*)*buf;
    word_t* out = (word_t*)malloc(nbytes * 2 + 9);  assert(out != NULL);
    *buf_size = nbytes * 2 + 9;
    size_t len = nbytes / sizeof(word_t);

    word_t prev2 = 0;
    word_t prev1 = 0;
    size_t pos;
    for (pos = 0; pos < len; pos++) {
      word_t curr = in[pos];
      out[pos] = curr - prev2;
      prev2 = prev1;
      prev1 = curr;
    }

    byte_t* buf1 = (byte_t*)in;
    byte_t* buf2 = (byte_t*)out;
    for (pos = len * sizeof(word_t); pos < nbytes; pos++) {
      buf2[pos] = buf1[pos];
    }

    byte_t prev = 0;
    size_t wpos = 0;
    int d;
    for (d = 0; d < 8; d++) {
      size_t rpos;
      for (rpos = d; rpos < nbytes; rpos += 8) {
        byte_t curr = buf2[rpos];
        buf1[wpos] = curr - prev;
        prev = curr;
        wpos++;
      }
    }

    unsigned int level = 4;
    if (cd_nelmts > 0) level = cd_values[0];
    if (level > 9) level = 9;
    unsigned int predtabsize = 1 << (level + 9);
    if (predtabsize > maxpredtabsize) predtabsize = maxpredtabsize;
    const int predtabsizem1 = predtabsize - 1;

    int lastpos[maxpredtabsize];
    memset(lastpos, 0, predtabsize * sizeof(int));

    size_t rpos = 0;
    wpos = 0;
    int hist = 0;
    while (rpos < nbytes) {
      byte_t val = buf1[rpos];
      int lpos = lastpos[hist];
      if (lpos >= 6) {
        if ((buf1[lpos - 6] == buf1[rpos - 6]) && (buf1[lpos - 5] == buf1[rpos - 5]) &&
            (buf1[lpos - 4] == buf1[rpos - 4]) && (buf1[lpos - 3] == buf1[rpos - 3]) &&
            (buf1[lpos - 2] == buf1[rpos - 2]) && (buf1[lpos - 1] == buf1[rpos - 1])) {
          byte_t cnt = 0;
          while ((val == buf1[lpos]) && (cnt < 255) && (rpos < (nbytes - 1))) {
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

    size_t tmp = nbytes;
    byte_t cnt = 0;
    while (tmp > 0) {
      buf2[wpos] = tmp & 255;
      tmp >>= 8;
      wpos++;
      cnt++;
    }
    buf2[wpos] = cnt + (level << 4);
    wpos++;

    free(buf1);
    *buf = buf2;
    size = wpos;
  }

  return size;
}
