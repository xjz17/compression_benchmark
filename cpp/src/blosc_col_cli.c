/*
 * Column-wise Blosc1 (blosclz + shuffle) CLI for Java benchmark subprocesses.
 *
 *   blosc_col_bench -c -i <raw.bin> -o <out.b2> -s <typesize>
 *   blosc_col_bench -d -i <out.b2> -o <raw.bin>
 */
#include <blosc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(const char *argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s -c -i <raw.bin> -o <out.b2> -s 1|4|8\n"
            "  %s -d -i <out.b2> -o <raw.bin>\n",
            argv0, argv0);
    return 2;
}

int main(int argc, char **argv) {
    int compress = -1;
    const char *in_path = NULL;
    const char *out_path = NULL;
    int typesize = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-c") == 0) {
            compress = 1;
        } else if (strcmp(argv[i], "-d") == 0) {
            compress = 0;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            in_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            typesize = atoi(argv[++i]);
        } else {
            return usage(argv[0]);
        }
    }
    if (compress < 0 || in_path == NULL || out_path == NULL) {
        return usage(argv[0]);
    }

    blosc_init();
    blosc_set_nthreads(1);
    if (blosc_set_compressor(BLOSC_BLOSCLZ_COMPNAME) < 0) {
        blosc_destroy();
        return 1;
    }

    int rc = 1;
    if (compress) {
        if (typesize != 1 && typesize != 4 && typesize != 8) {
            blosc_destroy();
            return usage(argv[0]);
        }
        FILE *fin = fopen(in_path, "rb");
        if (fin == NULL) {
            blosc_destroy();
            return 1;
        }
        if (fseek(fin, 0, SEEK_END) != 0) {
            fclose(fin);
            blosc_destroy();
            return 1;
        }
        long raw_len_l = ftell(fin);
        if (raw_len_l <= 0 || raw_len_l > BLOSC_MAX_BUFFERSIZE) {
            fclose(fin);
            blosc_destroy();
            return 1;
        }
        if (fseek(fin, 0, SEEK_SET) != 0) {
            fclose(fin);
            blosc_destroy();
            return 1;
        }
        const int raw_len = (int)raw_len_l;
        void *raw = malloc((size_t)raw_len);
        if (raw == NULL) {
            fclose(fin);
            blosc_destroy();
            return 1;
        }
        if ((int)fread(raw, 1, (size_t)raw_len, fin) != raw_len) {
            free(raw);
            fclose(fin);
            blosc_destroy();
            return 1;
        }
        fclose(fin);

        void *dest = malloc((size_t)raw_len + BLOSC_MAX_OVERHEAD);
        if (dest == NULL) {
            free(raw);
            blosc_destroy();
            return 1;
        }
        const int csize = blosc_compress(5, BLOSC_SHUFFLE, (size_t)typesize, raw_len, raw, dest,
                                         (size_t)raw_len + BLOSC_MAX_OVERHEAD);
        if (csize > 0) {
            FILE *fout = fopen(out_path, "wb");
            if (fout != NULL) {
                if (fwrite(dest, 1, (size_t)csize, fout) == (size_t)csize) {
                    rc = 0;
                }
                fclose(fout);
            }
        }
        free(dest);
        free(raw);
    } else {
        FILE *fcomp = fopen(in_path, "rb");
        if (fcomp == NULL) {
            blosc_destroy();
            return 1;
        }
        if (fseek(fcomp, 0, SEEK_END) != 0) {
            fclose(fcomp);
            blosc_destroy();
            return 1;
        }
        long comp_len_l = ftell(fcomp);
        if (comp_len_l <= 0) {
            fclose(fcomp);
            blosc_destroy();
            return 1;
        }
        if (fseek(fcomp, 0, SEEK_SET) != 0) {
            fclose(fcomp);
            blosc_destroy();
            return 1;
        }
        void *comp = malloc((size_t)comp_len_l);
        if (comp == NULL) {
            fclose(fcomp);
            blosc_destroy();
            return 1;
        }
        if (fread(comp, 1, (size_t)comp_len_l, fcomp) != (size_t)comp_len_l) {
            free(comp);
            fclose(fcomp);
            blosc_destroy();
            return 1;
        }
        fclose(fcomp);

        size_t nbytes = 0;
        size_t cbytes = 0;
        size_t blocksize = 0;
        blosc_cbuffer_sizes(comp, &nbytes, &cbytes, &blocksize);
        if (nbytes == 0 || nbytes > (size_t)BLOSC_MAX_BUFFERSIZE) {
            free(comp);
            blosc_destroy();
            return 1;
        }
        void *decoded = malloc(nbytes);
        if (decoded == NULL) {
            free(comp);
            blosc_destroy();
            return 1;
        }
        const int dsize = blosc_decompress(comp, decoded, nbytes);
        if (dsize > 0) {
            FILE *fout = fopen(out_path, "wb");
            if (fout != NULL) {
                if (fwrite(decoded, 1, (size_t)dsize, fout) == (size_t)dsize) {
                    rc = 0;
                }
                fclose(fout);
            }
        }
        free(decoded);
        free(comp);
    }

    blosc_destroy();
    return rc;
}
