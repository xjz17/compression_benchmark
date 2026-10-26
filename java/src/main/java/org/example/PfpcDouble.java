package org.example;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;

/**
 * Burtscher pFPC v1.0 — Java port aligned with {@code cpp/src/pfpc.c}.
 * Upstream reference: {@code benchmark/reference/upstream/pfpc/pFPC.c}.
 */
public final class PfpcDouble {

    public static final String ROW_NAME = "pfpc";
    public static final int DEFAULT_PRED_SIZE_LG2 = 20;
    public static final int DEFAULT_THREADS = 8;
    public static final int DEFAULT_CHUNKSIZE = 2048;

    private static final int BUFSIZE = 8 * 1024 * 1024;
    private static final int MAXTHREADS = 128;

    private static final int HLSHIFT = 6;
    private static final int HRSHIFT = 48;
    private static final int DLSHIFT = 2;
    private static final int DRSHIFT = 40;

    private static final long[] MASK = {
        0x0000000000000000L,
        0x00000000000000ffL,
        0x000000000000ffffL,
        0x0000000000ffffffL,
        0x000000ffffffffffL,
        0x0000ffffffffffffL,
        0x00ffffffffffffffL,
        0xffffffffffffffffL
    };

    private PfpcDouble() {}

    /** Override default with {@code WEB_COMPRESSION_PFPC_PRED_SIZE_LG2} (0..24). */
    public static int predSizeLg2ToBench() {
        String env = System.getenv("WEB_COMPRESSION_PFPC_PRED_SIZE_LG2");
        if (env != null && !env.isEmpty()) {
            try {
                int v = Integer.parseInt(env.trim());
                if (v >= 0 && v <= 24) {
                    return v;
                }
            } catch (NumberFormatException ignored) {
                // fall through
            }
        }
        return DEFAULT_PRED_SIZE_LG2;
    }

    /** Override default with {@code WEB_COMPRESSION_PFPC_THREADS} (1..128). */
    public static int threadsToBench() {
        String env = System.getenv("WEB_COMPRESSION_PFPC_THREADS");
        if (env != null && !env.isEmpty()) {
            try {
                int v = Integer.parseInt(env.trim());
                if (v >= 1 && v <= MAXTHREADS) {
                    return v;
                }
            } catch (NumberFormatException ignored) {
                // fall through
            }
        }
        return DEFAULT_THREADS;
    }

    /** Override default with {@code WEB_COMPRESSION_PFPC_CHUNKSIZE} (1..65536). */
    public static int chunkSizeToBench() {
        String env = System.getenv("WEB_COMPRESSION_PFPC_CHUNKSIZE");
        if (env != null && !env.isEmpty()) {
            try {
                int v = Integer.parseInt(env.trim());
                if (v >= 1 && v <= 65536) {
                    return v;
                }
            } catch (NumberFormatException ignored) {
                // fall through
            }
        }
        return DEFAULT_CHUNKSIZE;
    }

    private static long unsignedShiftRight(long v, int bits) {
        return v >>> bits;
    }

    private static int readIntLE(byte[] b, int off) {
        return (b[off] & 0xFF)
                | ((b[off + 1] & 0xFF) << 8)
                | ((b[off + 2] & 0xFF) << 16)
                | ((b[off + 3] & 0xFF) << 24);
    }

    private static void writeIntLE(byte[] b, int off, int v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >>> 8);
        b[off + 2] = (byte) (v >>> 16);
        b[off + 3] = (byte) (v >>> 24);
    }

    private static long readU64LE(byte[] b, int off) {
        return (b[off] & 0xFFL)
                | ((b[off + 1] & 0xFFL) << 8)
                | ((b[off + 2] & 0xFFL) << 16)
                | ((b[off + 3] & 0xFFL) << 24)
                | ((b[off + 4] & 0xFFL) << 32)
                | ((b[off + 5] & 0xFFL) << 40)
                | ((b[off + 6] & 0xFFL) << 48)
                | ((b[off + 7] & 0xFFL) << 56);
    }

    private static void writeU64LE(byte[] b, int off, long v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >>> 8);
        b[off + 2] = (byte) (v >>> 16);
        b[off + 3] = (byte) (v >>> 24);
        b[off + 4] = (byte) (v >>> 32);
        b[off + 5] = (byte) (v >>> 40);
        b[off + 6] = (byte) (v >>> 48);
        b[off + 7] = (byte) (v >>> 56);
    }

    private static int bcodeFromXor(long xor1) {
        int bcode = 7;
        if (xor1 >> 56 == 0) {
            bcode = 6;
        }
        if (xor1 >> 48 == 0) {
            bcode = 5;
        }
        if (xor1 >> 40 == 0) {
            bcode = 4;
        }
        if (xor1 >> 24 == 0) {
            bcode = 3;
        }
        if (xor1 >> 16 == 0) {
            bcode = 2;
        }
        if (xor1 >> 8 == 0) {
            bcode = 1;
        }
        if (xor1 == 0) {
            bcode = 0;
        }
        return bcode;
    }

    private static void writeXorAtOffset(byte[] outbuf, int out, long xor1) {
        int base = (out >> 3) << 3;
        int bitShift = (out & 7) << 3;
        long lo = readU64LE(outbuf, base);
        lo |= xor1 << bitShift;
        writeU64LE(outbuf, base, lo);
        long hi = unsignedShiftRight(xor1, 63 - bitShift);
        hi = unsignedShiftRight(hi, 1);
        writeU64LE(outbuf, base + 8, hi);
    }

    private static long readXorAtOffset(byte[] inbuf, int in) {
        int base = (in >> 3) << 3;
        int tmp = (in & 7) << 3;
        long val = readU64LE(inbuf, base);
        long next = readU64LE(inbuf, base + 8);
        val = unsignedShiftRight(val, tmp);
        if (tmp != 0) {
            next = next << (63 - tmp);
            val |= next + next;
        }
        return val;
    }

    private static final class WorkBuffers {
        final byte[][] inter;
        final long[][] fcm;
        final long[][] dfcm;
        final long[] output;

        WorkBuffers(byte[][] inter, long[][] fcm, long[][] dfcm, long[] output) {
            this.inter = inter;
            this.fcm = fcm;
            this.dfcm = dfcm;
            this.output = output;
        }
    }

    private static int chunkRepeat(int threads, int chunksize) {
        int chunkrepeat = BUFSIZE / (8 * threads * chunksize);
        if (chunkrepeat * 8L * threads * chunksize < BUFSIZE) {
            chunkrepeat++;
        }
        return chunkrepeat;
    }

    private static WorkBuffers allocWork(int threads, int chunksize, int predsizem1) {
        if (threads < 1 || threads > MAXTHREADS) {
            return null;
        }
        int chunkrepeat = chunkRepeat(threads, chunksize);
        byte[][] inter = new byte[threads][];
        long[][] fcm = new long[threads][];
        long[][] dfcm = new long[threads][];
        long[] output = new long[BUFSIZE >> 3];
        int interSize = 8 + ((chunksize * chunkrepeat + 1) / 2) + (chunksize * chunkrepeat * 8) + 7;
        int predSize = predsizem1 + 1;
        for (int i = 0; i < threads; i++) {
            inter[i] = new byte[interSize];
            fcm[i] = new long[predSize];
            dfcm[i] = new long[predSize];
        }
        return new WorkBuffers(inter, fcm, dfcm, output);
    }

    /**
     * Matches C {@code CompressChunk}.
     */
    private static void compressChunk(int threads, int chunksize, int end, long[] ibuf, int ibufBase,
            byte[] obuf, int predsizem1, long[] fcm, long[] dfcm) {
        int full = end / (chunksize * threads);
        int part = end - full * chunksize * threads;
        if (part > chunksize) {
            part = chunksize;
        }

        int out = full * chunksize + part;
        out = ((out + 1) >> 1) + 4;
        writeU64LE(obuf, (out >> 3) << 3, 0);

        int hash = 0;
        int dhash = 0;
        long lastval = 0;
        long pred1 = 0;
        long pred2 = 0;
        int save = 0;

        int o = 8;
        int offset = 0;
        int increment = threads * chunksize;
        int workChunksize = chunksize;
        while (0 < (end - offset)) {
            if (workChunksize > (end - offset)) {
                workChunksize = end - offset;
            }
            for (int i = 0; i < workChunksize; i++) {
                long val = ibuf[ibufBase + offset + i];
                long xor1 = val ^ pred1;
                fcm[hash] = val;
                hash = ((hash << HLSHIFT) ^ (int) unsignedShiftRight(val, HRSHIFT)) & predsizem1;
                pred1 = fcm[hash];

                long stride = val - lastval;
                long xor2 = val ^ (lastval + pred2);
                lastval = val;
                dfcm[dhash] = stride;
                dhash = ((dhash << DLSHIFT) ^ (int) unsignedShiftRight(stride, DRSHIFT)) & predsizem1;
                pred2 = dfcm[dhash];

                int code = 0;
                if (Long.compareUnsigned(xor1, xor2) > 0) {
                    code = 0x8;
                    xor1 = xor2;
                }
                int bcode = bcodeFromXor(xor1);

                writeXorAtOffset(obuf, out, xor1);

                code |= bcode;
                out += bcode + (bcode >> 2);
                obuf[o >> 1] = (byte) (save | code);
                save = code << 4;
                o++;
            }
            offset += increment;
        }
        if ((o & 1) != 0) {
            obuf[o >> 1] = (byte) save;
        }

        writeIntLE(obuf, 0, out);
    }

    private static void compressToMemory(int num, long[] inbuf, WorkBuffers work, int predsizem1, int threads,
            int chunksize) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(threads);
        for (int i = 0; i < threads; i++) {
            final int id = i;
            final int end = num - (id * chunksize);
            final int threadEnd = end > 0 ? end : 0;
            Thread t = new Thread(() -> {
                try {
                    compressChunk(threads, chunksize, threadEnd, inbuf, id * chunksize, work.inter[id], predsizem1,
                            work.fcm[id], work.dfcm[id]);
                } finally {
                    latch.countDown();
                }
            });
            t.start();
        }
        latch.await();
    }

    /**
     * Matches C {@code DecompressChunk}.
     */
    private static void decompressChunk(int threads, int chunksize, int end, byte[] ibuf, long[] obuf, int obufBase,
            int predsizem1, long[] fcm, long[] dfcm) {
        int hash = 0;
        int dhash = 0;
        long lastval = 0;
        long pred1 = 0;
        long pred2 = 0;

        int full = end / (chunksize * threads);
        int part = end - full * chunksize * threads;
        if (part > chunksize) {
            part = chunksize;
        }
        int in = full * chunksize + part;
        in = ((in + 1) >> 1) + 4;
        int i = 8;
        int alt = 4;

        int offset = 0;
        int increment = threads * chunksize;
        int workChunksize = chunksize;
        while (0 < (end - offset)) {
            if (workChunksize > (end - offset)) {
                workChunksize = end - offset;
            }
            for (int o = 0; o < workChunksize; o++) {
                int bcode = (ibuf[i >> 1] & 0xFF) >> alt;
                i++;
                alt ^= 4;

                long val = readXorAtOffset(ibuf, in);
                if ((bcode & 0x8) != 0) {
                    pred1 = pred2;
                }
                bcode &= 0x7;
                val &= MASK[bcode];
                in += bcode + (bcode >> 2);
                val ^= pred1;

                fcm[hash] = val;
                hash = ((hash << HLSHIFT) ^ (int) unsignedShiftRight(val, HRSHIFT)) & predsizem1;
                pred1 = fcm[hash];

                long stride = val - lastval;
                dfcm[dhash] = stride;
                dhash = ((dhash << DLSHIFT) ^ (int) unsignedShiftRight(stride, DRSHIFT)) & predsizem1;
                pred2 = val + dfcm[dhash];
                lastval = val;

                obuf[obufBase + offset + o] = val;
            }
            offset += increment;
        }
    }

    private static void decompressToMemory(int num, WorkBuffers work, int predsizem1, int threads, int chunksize)
            throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(threads);
        for (int i = 0; i < threads; i++) {
            final int id = i;
            final int end = num - (id * chunksize);
            Thread t = new Thread(() -> {
                try {
                    decompressChunk(threads, chunksize, end, work.inter[id], work.output, id * chunksize,
                            predsizem1, work.fcm[id], work.dfcm[id]);
                } finally {
                    latch.countDown();
                }
            });
            t.start();
        }
        latch.await();
    }

    /**
     * Matches C {@code pfpc_compress_memory}. Returns compressed bytes, or {@code null} on failure.
     */
    public static byte[] compressMemory(byte[] in, int predSizeLg2, int threads, int chunkSize) {
        if (in == null || in.length == 0 || (in.length & 7) != 0) {
            return null;
        }
        if (threads < 1 || threads >= 256 || chunkSize < 1 || predSizeLg2 >= 256) {
            return null;
        }

        int predsizem1 = (1 << predSizeLg2) - 1;
        WorkBuffers work = allocWork(threads, chunkSize, predsizem1);
        if (work == null) {
            return null;
        }

        byte[] outStream = new byte[in.length * 2 + 1024];
        int outOff = 0;
        outStream[outOff++] = (byte) predSizeLg2;
        outStream[outOff++] = (byte) threads;
        writeIntLE(outStream, outOff, chunkSize);
        outOff += 4;

        int inCursor = 0;
        int remaining = in.length / 8;
        int maxBlock = BUFSIZE >> 3;
        long[] input = new long[maxBlock];

        try {
            while (remaining > 0) {
                int num = Math.min(remaining, maxBlock);
                for (int i = 0; i < num; i++) {
                    input[i] = readU64LE(in, inCursor + i * 8);
                }
                inCursor += num * 8;

                if (outOff + 4 > outStream.length) {
                    return null;
                }
                writeIntLE(outStream, outOff, num);
                outOff += 4;

                compressToMemory(num, input, work, predsizem1, threads, chunkSize);

                for (int i = 0; i < threads; i++) {
                    int cnt = readIntLE(work.inter[i], 0);
                    if (cnt < 4 || outOff + cnt > outStream.length) {
                        if (outOff + cnt > outStream.length) {
                            outStream = Arrays.copyOf(outStream, Math.max(outStream.length * 2, outOff + cnt));
                        } else {
                            return null;
                        }
                    }
                    System.arraycopy(work.inter[i], 0, outStream, outOff, cnt);
                    outOff += cnt;
                }
                remaining -= num;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }

        return Arrays.copyOf(outStream, outOff);
    }

    /**
     * Decompress into {@code out}. Returns bytes consumed from {@code comp} (matching C return value),
     * or {@code -1} on failure.
     */
    public static int decompressMemory(byte[] comp, byte[] out) {
        if (comp == null || out == null || comp.length < 6) {
            return -1;
        }

        int predLg2 = comp[0] & 0xFF;
        int threads = comp[1] & 0xFF;
        int chunksize = readIntLE(comp, 2);
        if (threads < 1 || threads >= 256 || chunksize < 1 || predLg2 >= 256) {
            return -1;
        }

        int predsizem1 = (1 << predLg2) - 1;
        WorkBuffers work = allocWork(threads, chunksize, predsizem1);
        if (work == null) {
            return -1;
        }

        int inOff = 6;
        int outWritten = 0;

        try {
            while (inOff + 4 <= comp.length) {
                int num = readIntLE(comp, inOff);
                inOff += 4;
                if (num <= 0) {
                    break;
                }

                for (int i = 0; i < threads; i++) {
                    if (inOff + 4 > comp.length) {
                        return -1;
                    }
                    int chunkTotal = readIntLE(comp, inOff);
                    if (chunkTotal < 4 || inOff + chunkTotal > comp.length) {
                        return -1;
                    }
                    if (chunkTotal > work.inter[i].length) {
                        work.inter[i] = Arrays.copyOf(work.inter[i], chunkTotal);
                    }
                    System.arraycopy(comp, inOff, work.inter[i], 0, chunkTotal);
                    inOff += chunkTotal;
                }

                decompressToMemory(num, work, predsizem1, threads, chunksize);

                int blockBytes = num * 8;
                if (outWritten + blockBytes > out.length) {
                    return -1;
                }
                for (int i = 0; i < num; i++) {
                    writeU64LE(out, outWritten + i * 8, work.output[i]);
                }
                outWritten += blockBytes;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return -1;
        }

        return inOff;
    }

    public static void benchPfpcDouble(List<? extends List<Double>> columns, long[] out) {
        benchPfpcDouble(columns, out, null);
    }

    public static void benchPfpcDouble(List<? extends List<Double>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        int predSizeLg2 = predSizeLg2ToBench();
        int threads = threadsToBench();
        int chunkSize = chunkSizeToBench();
        try {
            for (List<? extends Double> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int nbytes = n * 8;
                if ((nbytes & 7) != 0) {
                    continue;
                }
                byte[] raw = new byte[nbytes];
                int p = 0;
                for (Double dv : col) {
                    long bits = Double.doubleToRawLongBits(dv == null ? Double.NaN : dv);
                    raw[p++] = (byte) bits;
                    raw[p++] = (byte) (bits >>> 8);
                    raw[p++] = (byte) (bits >>> 16);
                    raw[p++] = (byte) (bits >>> 24);
                    raw[p++] = (byte) (bits >>> 32);
                    raw[p++] = (byte) (bits >>> 40);
                    raw[p++] = (byte) (bits >>> 48);
                    raw[p++] = (byte) (bits >>> 56);
                }

                long t0 = System.nanoTime();
                byte[] compBytes = compressMemory(raw, predSizeLg2, threads, chunkSize);
                long t1 = System.nanoTime();
                if (compBytes == null || compBytes.length == 0) {
                    continue;
                }

                byte[] dec = new byte[nbytes];
                long t2 = System.nanoTime();
                int consumed = decompressMemory(compBytes, dec);
                long t3 = System.nanoTime();
                if (consumed != compBytes.length || !Arrays.equals(raw, dec)) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Double> decVals = new ArrayList<>(n);
                    ByteBuffer db = ByteBuffer.wrap(dec).order(ByteOrder.LITTLE_ENDIAN);
                    for (int i = 0; i < n; i++) {
                        decVals.add(db.getDouble());
                    }
                    List<Double> origCol = new ArrayList<>(col.size());
                    for (Double dv : col) {
                        origCol.add(dv);
                    }
                    BenchmarkLoss.accumulateDoubleListLoss(origCol, decVals, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += compBytes.length;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }

    public static void benchPfpcLong(List<? extends List<Long>> columns, long[] out) {
        benchPfpcLong(columns, out, null);
    }

    public static void benchPfpcLong(List<? extends List<Long>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        int predSizeLg2 = predSizeLg2ToBench();
        int threads = threadsToBench();
        int chunkSize = chunkSizeToBench();
        try {
            for (List<? extends Long> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int nbytes = n * 8;
                if ((nbytes & 7) != 0) {
                    continue;
                }
                byte[] raw = new byte[nbytes];
                int p = 0;
                for (Long lv : col) {
                    long v = lv == null ? 0L : lv;
                    raw[p++] = (byte) v;
                    raw[p++] = (byte) (v >>> 8);
                    raw[p++] = (byte) (v >>> 16);
                    raw[p++] = (byte) (v >>> 24);
                    raw[p++] = (byte) (v >>> 32);
                    raw[p++] = (byte) (v >>> 40);
                    raw[p++] = (byte) (v >>> 48);
                    raw[p++] = (byte) (v >>> 56);
                }

                long t0 = System.nanoTime();
                byte[] compBytes = compressMemory(raw, predSizeLg2, threads, chunkSize);
                long t1 = System.nanoTime();
                if (compBytes == null || compBytes.length == 0) {
                    continue;
                }

                byte[] dec = new byte[nbytes];
                long t2 = System.nanoTime();
                int consumed = decompressMemory(compBytes, dec);
                long t3 = System.nanoTime();
                if (consumed != compBytes.length || !Arrays.equals(raw, dec)) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Long> decVals = new ArrayList<>(n);
                    ByteBuffer db = ByteBuffer.wrap(dec).order(ByteOrder.LITTLE_ENDIAN);
                    for (int i = 0; i < n; i++) {
                        decVals.add(db.getLong());
                    }
                    List<Long> origCol = new ArrayList<>(col.size());
                    for (Long lv : col) {
                        origCol.add(lv);
                    }
                    BenchmarkLoss.accumulateLongListLoss(origCol, decVals, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += compBytes.length;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }
}
