#include "shrinker.h"
#include <string.h>

// https://code.google.com/p/data-shrinker/

#if defined(_MSC_VER)
typedef unsigned __int8 u8;
typedef unsigned __int16 u16;
typedef unsigned __int32 u32;
typedef unsigned __int64 u64;
#else
#include <stdint.h>
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#endif

//the code from LZ4
#if (GCC_VERSION >= 302) || (__INTEL_COMPILER >= 800) || defined(__clang__)
# define expect(expr,value)    (__builtin_expect ((expr),(value)) )
#else
# define expect(expr,value)    (expr)
#endif
#define likely(expr)     expect((expr) != 0, 1)
#define unlikely(expr)   expect((expr) != 0, 0)
////////////////////

#define HASH_BITS 15
#define HASH(a) ((a*21788233) >> (32 - HASH_BITS))
#define MINMATCH 4

static int shrinker_ablation_mode = SHRINKER_ABLATION_NONE;

static void shrinker_put_u32(u8 **dst, u32 value)
{
    memcpy(*dst, &value, sizeof(value));
    *dst += sizeof(value);
}

static u32 shrinker_get_u32(u8 **src)
{
    u32 value;
    memcpy(&value, *src, sizeof(value));
    *src += sizeof(value);
    return value;
}

void shrinker_set_ablation_mode(int mode)
{
    shrinker_ablation_mode = mode;
}

#include <limits.h>
#if ULONG_MAX == UINT_MAX //32bit
#define MEMCPY_NOOVERLAP(a, b, c) do{do{*(u32*)a = *(u32*)b;a+=4;b+=4;}while(b<c);a-=(b-c);b=c;}while(0)
#define MEMCPY_NOOVERLAP_NOSURPASS(a, b, c) do{c-=4;while(b<c){*(u32*)(a) = *(u32*)(b);a+=4;b+=4;}c+=4;while(b<c)*a++=*b++;}while(0)
#define MEMCPY(a, b, c) do{if (likely(a>b+4)) MEMCPY_NOOVERLAP(a, b, c); else while(b<c) *a++=*b++;}while(0)
#else
#define MEMCPY_NOOVERLAP(a, b, c) do{do{*(u64*)a = *(u64*)b;a+=8;b+=8;}while(b<c);a-=(b-c);b=c;}while(0)
#define MEMCPY_NOOVERLAP_NOSURPASS(a, b, c) do{c-=8;while(b<c){*(u64*)(a) = *(u64*)(b);a+=8;b+=8;}c+=8;while(b<c)*a++=*b++;}while(0)
#define MEMCPY(a, b, c) do{if (likely(a>b+8)) MEMCPY_NOOVERLAP(a, b, c); else while(b<c) *a++=*b++;}while(0)
#endif

int shrinker_compress_with_capacity(void *in, void *out, size_t out_capacity, int size)
{
    /* Deterministic initial match state; upstream left this stack table unset. */
    u32 ht[(1<<HASH_BITS)] = {0};
    u8 *src = (u8*)in, *dst = (u8*)out;
    u8 *src_end = src + size - MINMATCH - 8;
    u8 *dst_end = dst + out_capacity - MINMATCH - 8;
    u8 *pfind, *pcur;

    u32 cur_hash;
    u8  *p_last_lit = src;
    u32 cpy_len, match_dist;
    u8 flag, *pflag, cache;
    u32 cur_u32;

    if (size < 32) return -1;
    if (size > (1 << 27) - 1) return -1;
    if (out_capacity < 32) return -1;

    if (shrinker_ablation_mode == SHRINKER_ABLATION_FIXED_TOKENS) {
        memcpy(dst, "SHT1", 4);
        dst += 4;
    }

    if (shrinker_ablation_mode == SHRINKER_ABLATION_NO_LZ_MATCH) {
        u32 literal_len = (u32)size;
        pflag = dst++;
        if (literal_len < 7) flag = literal_len << 5;
        else {
            literal_len -= 7;
            flag = (7 << 5);
            while (literal_len >= 255) {
                *dst++ = 255;
                literal_len -= 255;
            }
            *dst++ = literal_len;
        }
        flag |= 7 + 16;
        *pflag = flag;
        *dst++ = 0xff;
        *dst++ = 0xff;
        memcpy(dst, in, size);
        dst += size;
        return (int)(dst - (u8*)out);
    }

    cur_u32 = *(u32*)src;
    while(likely(src < src_end) && likely(dst < dst_end))
    {
        u32 tmp = src - (u8*)in;
        pcur = src;
        cur_u32 = *(u32*)pcur;
        cur_hash = HASH(cur_u32);
        cache = ht[cur_hash] >> 27;
        pfind = (u8*)in + (ht[cur_hash] & 0x07ffffff);
        ht[cur_hash] = tmp|(*src<<27);

        if (unlikely(cache == (*pcur & 0x1f))
            && pfind + 0xffff >= (u8*)pcur
            && pfind < pcur
            &&*(u32*)pfind == *(u32*)pcur) 
        {  
            pfind += 4; pcur += 4;
            while(likely(pcur < src_end) && *(u32*)pfind == *(u32*)pcur)
            { pfind += 4; pcur += 4;}

            if (likely(pcur < src_end))
            if (*(u16*)pfind == *(u16*)pcur) {pfind += 2; pcur += 2;}
            if (*pfind == *pcur) {pfind++; pcur++;}

            cpy_len = src - p_last_lit;
            if (shrinker_ablation_mode == SHRINKER_ABLATION_FIXED_TOKENS) {
                u32 match_len = pcur - src;
                u32 match_offset = pcur - pfind;
                if ((size_t)(dst_end - dst) < (size_t)cpy_len + 13) return -1;
                *dst++ = 1;
                shrinker_put_u32(&dst, cpy_len);
                shrinker_put_u32(&dst, match_len);
                shrinker_put_u32(&dst, match_offset);
                memcpy(dst, p_last_lit, cpy_len);
                dst += cpy_len;
            } else {
                pflag = dst++;
                if (likely(cpy_len < 7)) flag = cpy_len << 5;
                else {
                    cpy_len -= 7;flag = (7<<5);
                    while (cpy_len >= 255)
                    { *dst++ = 255;cpy_len -= 255;}
                    *dst++ = cpy_len;
                }

                cpy_len = pcur - src  - MINMATCH;
                if (likely(cpy_len < 15))  flag |= cpy_len;
                else {
                    cpy_len -= 15; flag |= 15;
                    while (cpy_len >= 255)
                    { *dst++ = 255;cpy_len -= 255;}
                    *dst++ = cpy_len;
                }
                match_dist = pcur - pfind - 1;
                *pflag = flag;
                *dst++ = match_dist & 0xff;
                if (match_dist > 0xff) {
                    *pflag |= 16;
                    *dst++ = match_dist >> 8;
                }
                MEMCPY_NOOVERLAP(dst, p_last_lit, src);
            }

            cur_u32 = *(u32*)(src+1);
            ht[HASH(cur_u32)] = (src - (u8*)in + 1)|(*(src+1)<<27);
            cur_u32 = *(u32*)(src+3);
            ht[HASH(cur_u32)] = (src - (u8*)in + 3)|(*(src+3)<<27);
            src = pcur;
            p_last_lit = src;
            continue;
        }
        src++;
    }


    if (shrinker_ablation_mode != SHRINKER_ABLATION_FIXED_TOKENS &&
        dst - (u8*)out + 3 >= src - (u8*)in) return -1;
    src = (u8*)in + size;
    if (shrinker_ablation_mode == SHRINKER_ABLATION_FIXED_TOKENS) {
        cpy_len = src - p_last_lit;
        if ((size_t)(dst_end - dst) < (size_t)cpy_len + 5) return -1;
        *dst++ = 0;
        shrinker_put_u32(&dst, cpy_len);
        memcpy(dst, p_last_lit, cpy_len);
        dst += cpy_len;
        return dst - (u8*)out;
    }
    pflag = dst++;
    cpy_len = src - p_last_lit;
    if (likely(cpy_len < 7)) flag = cpy_len << 5;
    else {
        cpy_len -= 7; flag = (7<<5);
        while (cpy_len >= 255)
        { *dst++ = 255; cpy_len -= 255;}
        *dst++ = cpy_len;
    }

    flag |= 7 + 16; // any number
    *pflag = flag;
    *dst++ = 0xff; *dst++ = 0xff;
    MEMCPY_NOOVERLAP_NOSURPASS(dst, p_last_lit, src);

    if (dst > dst_end) return -1;
    else return dst - (u8*)out;
}

int shrinker_compress(void *in, void *out, int size)
{
    return shrinker_compress_with_capacity(in, out, (size_t)size, size);
}



int shrinker_decompress(void *in, void *out, int size)
{
    u8 *src = (u8*)in, *dst = (u8*)out;
    u8 *end = dst + size;
    u8 *pcpy, *pend;
    u8 flag, long_dist;
    u32 literal_len;
    u32 match_len, match_dist;

    if (memcmp(src, "SHT1", 4) == 0) {
        src += 4;
        while (dst < end) {
            u8 tag = *src++;
            u32 literal_count = shrinker_get_u32(&src);
            if (tag == 1) {
                match_len = shrinker_get_u32(&src);
                match_dist = shrinker_get_u32(&src);
            }
            if ((size_t)(end - dst) < literal_count) return -1;
            memcpy(dst, src, literal_count);
            src += literal_count;
            dst += literal_count;
            if (tag == 0) break;
            if (tag != 1) return -1;
            if (match_dist == 0 || match_dist > (u32)(dst - (u8*)out) ||
                (size_t)(end - dst) < match_len) return -1;
            pcpy = dst - match_dist;
            while (match_len--) *dst++ = *pcpy++;
        }
        return dst == end ? size : -1;
    }

    for(;;) {
        flag = *src++;
        literal_len = flag >> 5; // 3-bits
        match_len = flag & 0xf; // 4-bits
        long_dist = flag & 0x10; // 1-bit

        if (unlikely(literal_len == 7)) {
            while((flag = *src++) == 255)
                literal_len += 255;
            literal_len += flag;
        }

        if (unlikely(match_len == 15)) {
            while((flag = *src++) == 255)
                match_len += 255;
            match_len += flag;
        }

        match_dist = *src++;
        if (long_dist) 
        {
            match_dist |= ((*src++) << 8);
            if (unlikely(match_dist == 0xffff)) {
                pend = src + literal_len;
                if (unlikely(dst + literal_len > end)) return -1;
                MEMCPY_NOOVERLAP_NOSURPASS(dst, src, pend);
                break;
            }
        }

        pend = src + literal_len;
        if (unlikely(dst + literal_len > end)) return -1;
        MEMCPY_NOOVERLAP(dst, src, pend);
        pcpy = dst - match_dist - 1;
        pend = pcpy + match_len + MINMATCH;
        if (unlikely(pcpy < (u8*)out || dst + match_len + MINMATCH > end)) return -1;
        MEMCPY(dst, pcpy, pend);
    }
    return dst - (u8*)out;
}
