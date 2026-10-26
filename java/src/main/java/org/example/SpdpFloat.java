package org.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Burtscher SPDP v1.1 (DCC 2018) — Java port aligned with {@code cpp/src/spdp.c}.
 * Upstream reference: {@code benchmark/reference/upstream/spdp/standalone/SPDP_11.c}.
 *
 * <p>Levels 0..9 only change the prediction-table size ({@code 1 << (level + 9)}); the pipeline is
 * otherwise identical (second-order delta → byte-plane delta → history compression).
 */
public final class SpdpFloat {

    public static final int MIN_LEVEL = 0;
    public static final int MAX_LEVEL = 9;

    private static final int BUFFER_SIZE = 1 << 23;
    private static final int MAX_TABLE_SIZE = 1 << 18;

    private SpdpFloat() {}

    public static String rowName(int level) {
        return "spdp_level_" + level;
    }

    /** Levels to benchmark (always all 0..9; use {@code --algorithms spdp_level_N} to refresh one level). */
    public static List<Integer> levelsToBench() {
        List<Integer> all = new ArrayList<>(MAX_LEVEL - MIN_LEVEL + 1);
        for (int level = MIN_LEVEL; level <= MAX_LEVEL; level++) {
            all.add(level);
        }
        return all;
    }

    private static int readU32(byte[] b, int off) {
        return (b[off] & 0xFF)
                | ((b[off + 1] & 0xFF) << 8)
                | ((b[off + 2] & 0xFF) << 16)
                | ((b[off + 3] & 0xFF) << 24);
    }

    private static void writeU32(byte[] b, int off, int v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >>> 8);
        b[off + 2] = (byte) (v >>> 16);
        b[off + 3] = (byte) (v >>> 24);
    }

    private static long readU64(byte[] b, int off) {
        return (b[off] & 0xFFL)
                | ((b[off + 1] & 0xFFL) << 8)
                | ((b[off + 2] & 0xFFL) << 16)
                | ((b[off + 3] & 0xFFL) << 24)
                | ((b[off + 4] & 0xFFL) << 32)
                | ((b[off + 5] & 0xFFL) << 40)
                | ((b[off + 6] & 0xFFL) << 48)
                | ((b[off + 7] & 0xFFL) << 56);
    }

    private static void writeU64(byte[] b, int off, long v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >>> 8);
        b[off + 2] = (byte) (v >>> 16);
        b[off + 3] = (byte) (v >>> 24);
        b[off + 4] = (byte) (v >>> 32);
        b[off + 5] = (byte) (v >>> 40);
        b[off + 6] = (byte) (v >>> 48);
        b[off + 7] = (byte) (v >>> 56);
    }

    private static int compressCore(int level, int length, byte[] buf1, byte[] buf2, int wordBytes) {
        int len = length / wordBytes;
        if (wordBytes == 4) {
            int prev2 = 0;
            int prev1 = 0;
            for (int pos = 0; pos < len; pos++) {
                int curr = readU32(buf1, pos * wordBytes);
                writeU32(
                        buf2,
                        pos * wordBytes,
                        (int)
                                ((Integer.toUnsignedLong(curr) - Integer.toUnsignedLong(prev2))
                                        & 0xFFFF_FFFFL));
                prev2 = prev1;
                prev1 = curr;
            }
        } else {
            long prev2 = 0L;
            long prev1 = 0L;
            for (int pos = 0; pos < len; pos++) {
                long curr = readU64(buf1, pos * wordBytes);
                writeU64(buf2, pos * wordBytes, curr - prev2);
                prev2 = prev1;
                prev1 = curr;
            }
        }
        for (int pos = len * wordBytes; pos < length; pos++) {
            buf2[pos] = buf1[pos];
        }
        int prev = 0;
        int wpos = 0;
        for (int d = 0; d < 8; d++) {
            for (int rpos = d; rpos < length; rpos += 8) {
                int curr = buf2[rpos] & 0xFF;
                buf1[wpos] = (byte) ((curr - prev) & 0xFF);
                prev = curr;
                wpos++;
            }
        }
        int predtabsize = 1 << (level + 9);
        if (predtabsize > MAX_TABLE_SIZE) {
            predtabsize = MAX_TABLE_SIZE;
        }
        int predtabsizem1 = predtabsize - 1;
        int[] lastpos = new int[predtabsize];
        Arrays.fill(lastpos, 0);
        int rpos = 0;
        wpos = 0;
        int hist = 0;
        while (rpos < length) {
            int val = buf1[rpos] & 0xFF;
            int lpos = lastpos[hist];
            if (lpos >= 6) {
                if ((buf1[lpos - 6] == buf1[rpos - 6]) && (buf1[lpos - 5] == buf1[rpos - 5])
                        && (buf1[lpos - 4] == buf1[rpos - 4]) && (buf1[lpos - 3] == buf1[rpos - 3])
                        && (buf1[lpos - 2] == buf1[rpos - 2]) && (buf1[lpos - 1] == buf1[rpos - 1])) {
                    int cnt = 0;
                    while ((val == (buf1[lpos] & 0xFF)) && (cnt < 255) && (rpos < (length - 1))) {
                        lastpos[hist] = rpos;
                        hist = ((hist << 2) ^ val) & predtabsizem1;
                        rpos++;
                        lpos++;
                        cnt++;
                        val = buf1[rpos] & 0xFF;
                    }
                    buf2[wpos] = (byte) cnt;
                    wpos++;
                }
            }
            buf2[wpos] = (byte) val;
            wpos++;
            lastpos[hist] = rpos;
            hist = ((hist << 2) ^ val) & predtabsizem1;
            rpos++;
        }
        return wpos;
    }

    private static void decompressCore(int level, int length, byte[] buf2, byte[] buf1, int wordBytes) {
        int predtabsize = 1 << (level + 9);
        if (predtabsize > MAX_TABLE_SIZE) {
            predtabsize = MAX_TABLE_SIZE;
        }
        int predtabsizem1 = predtabsize - 1;
        int[] lastpos = new int[predtabsize];
        Arrays.fill(lastpos, 0);
        int rpos = 0;
        int wpos = 0;
        int hist = 0;
        while (rpos < length) {
            int lpos = lastpos[hist];
            if (lpos >= 6) {
                if ((buf1[lpos - 6] == buf1[wpos - 6]) && (buf1[lpos - 5] == buf1[wpos - 5])
                        && (buf1[lpos - 4] == buf1[wpos - 4]) && (buf1[lpos - 3] == buf1[wpos - 3])
                        && (buf1[lpos - 2] == buf1[wpos - 2]) && (buf1[lpos - 1] == buf1[wpos - 1])) {
                    int cnt = buf2[rpos] & 0xFF;
                    rpos++;
                    for (int j = 0; j < cnt; j++) {
                        byte val = buf1[lpos];
                        buf1[wpos] = val;
                        lastpos[hist] = wpos;
                        hist = ((hist << 2) ^ (val & 0xFF)) & predtabsizem1;
                        wpos++;
                        lpos++;
                    }
                }
            }
            byte val = buf2[rpos];
            buf1[wpos] = val;
            lastpos[hist] = wpos;
            hist = ((hist << 2) ^ (val & 0xFF)) & predtabsizem1;
            wpos++;
            rpos++;
        }
        int usize = wpos;
        int acc = 0;
        rpos = 0;
        for (int d = 0; d < 8; d++) {
            for (int wpos2 = d; wpos2 < usize; wpos2 += 8) {
                acc = (acc + (buf1[rpos] & 0xFF)) & 0xFF;
                buf2[wpos2] = (byte) acc;
                rpos++;
            }
        }
        int wlen = usize / wordBytes;
        if (wordBytes == 4) {
            int p2 = 0;
            int p1 = 0;
            for (int i = 0; i < wlen; i++) {
                int curr =
                        (int)
                                ((Integer.toUnsignedLong(readU32(buf2, i * wordBytes))
                                                + Integer.toUnsignedLong(p2))
                                        & 0xFFFF_FFFFL);
                writeU32(buf1, i * wordBytes, curr);
                p2 = p1;
                p1 = curr;
            }
        } else {
            long p2 = 0L;
            long p1 = 0L;
            for (int i = 0; i < wlen; i++) {
                long curr = readU64(buf2, i * wordBytes) + p2;
                writeU64(buf1, i * wordBytes, curr);
                p2 = p1;
                p1 = curr;
            }
        }
        for (int i = wlen * wordBytes; i < usize; i++) {
            buf1[i] = buf2[i];
        }
    }

    public static int compressBlock(int level, byte[] in, int len, byte[] out, int outCap) {
        return compressBlock(level, in, len, out, outCap, 4);
    }

    public static int compressBlock(
            int level, byte[] in, int len, byte[] out, int outCap, int wordBytes) {
        if (level < MIN_LEVEL
                || level > MAX_LEVEL
                || len <= 0
                || len > BUFFER_SIZE
                || outCap < len * 2 + 9
                || (wordBytes != 4 && wordBytes != 8)
                || (len % wordBytes) != 0) {
            return 0;
        }
        byte[] buf1 = Arrays.copyOf(in, len);
        byte[] buf2 = new byte[len * 2 + 9];
        int csize = compressCore(level, len, buf1, buf2, wordBytes);
        if (csize > outCap) {
            return 0;
        }
        System.arraycopy(buf2, 0, out, 0, csize);
        return csize;
    }

    public static boolean decompressBlock(int level, byte[] comp, int csize, byte[] out, int outLen) {
        return decompressBlock(level, comp, csize, out, outLen, 4);
    }

    public static boolean decompressBlock(
            int level, byte[] comp, int csize, byte[] out, int outLen, int wordBytes) {
        if (level < MIN_LEVEL
                || level > MAX_LEVEL
                || csize <= 0
                || outLen <= 0
                || outLen > BUFFER_SIZE
                || (wordBytes != 4 && wordBytes != 8)
                || (outLen % wordBytes) != 0) {
            return false;
        }
        int buf2Cap = Math.max(csize, outLen * 2 + 9);
        byte[] buf2 = Arrays.copyOf(comp, buf2Cap);
        byte[] buf1 = new byte[outLen];
        decompressCore(level, csize, buf2, buf1, wordBytes);
        System.arraycopy(buf1, 0, out, 0, outLen);
        return true;
    }

    public static void benchSpdpFloat(List<? extends List<Float>> columns, int level, long[] out) {
        benchSpdpFloat(columns, level, out, 4);
    }

    public static void benchSpdpFloat(
            List<? extends List<Float>> columns, int level, long[] out, int wireValueBytes) {
        benchSpdpFloat(columns, level, out, wireValueBytes, null);
    }

    public static void benchSpdpFloat(
            List<? extends List<Float>> columns,
            int level,
            long[] out,
            int wireValueBytes,
            double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        if (level < MIN_LEVEL || level > MAX_LEVEL) {
            out[0] = 0;
            return;
        }
        try {
            for (List<? extends Float> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int nbytes = n * 4;
                byte[] raw = new byte[nbytes];
                int p = 0;
                for (float f : col) {
                    int bits = Float.floatToRawIntBits(f);
                    raw[p++] = (byte) bits;
                    raw[p++] = (byte) (bits >>> 8);
                    raw[p++] = (byte) (bits >>> 16);
                    raw[p++] = (byte) (bits >>> 24);
                }
                ArrayList<byte[]> chunks = new ArrayList<>();
                long t0 = System.nanoTime();
                for (int off = 0; off < nbytes; off += BUFFER_SIZE) {
                    int chunk = Math.min(BUFFER_SIZE, nbytes - off);
                    byte[] slice = Arrays.copyOfRange(raw, off, off + chunk);
                    byte[] cbuf = new byte[chunk * 2 + 9];
                    int cs = compressBlock(level, slice, chunk, cbuf, cbuf.length, 4);
                    if (cs == 0) {
                        chunks = null;
                        break;
                    }
                    chunks.add(Arrays.copyOf(cbuf, cs));
                }
                long t1 = System.nanoTime();
                if (chunks == null) {
                    continue;
                }
                long totalC = 0;
                for (byte[] ch : chunks) {
                    totalC += ch.length;
                }
                byte[] dec = new byte[nbytes];
                long t2 = System.nanoTime();
                int di = 0;
                boolean ok = true;
                for (byte[] ch : chunks) {
                    int chunk = Math.min(BUFFER_SIZE, nbytes - di);
                    byte[] block = new byte[chunk];
                    if (!decompressBlock(level, ch, ch.length, block, chunk, 4)) {
                        ok = false;
                        break;
                    }
                    System.arraycopy(block, 0, dec, di, chunk);
                    di += chunk;
                }
                long t3 = System.nanoTime();
                if (!ok || di != nbytes || !Arrays.equals(raw, dec)) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Double> origVals = new ArrayList<>(n);
                    List<Double> decVals = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        origVals.add((double) col.get(i));
                    }
                    java.nio.ByteBuffer db =
                            java.nio.ByteBuffer.wrap(dec).order(java.nio.ByteOrder.LITTLE_ENDIAN);
                    for (int i = 0; i < n; i++) {
                        decVals.add((double) db.getFloat());
                    }
                    BenchmarkLoss.accumulateDoubleListLoss(origVals, decVals, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, wireValueBytes == BenchmarkWireFormat.FLOAT_BYTES);
                comp += totalC;
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

    public static void benchSpdpDouble(List<? extends List<Double>> columns, int level, long[] out) {
        benchSpdpDouble(columns, level, out, null);
    }

    public static void benchSpdpDouble(
            List<? extends List<Double>> columns, int level, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        if (level < MIN_LEVEL || level > MAX_LEVEL) {
            out[0] = 0;
            return;
        }
        try {
            for (List<? extends Double> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int nbytes = n * 8;
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
                ArrayList<byte[]> chunks = new ArrayList<>();
                long t0 = System.nanoTime();
                for (int off = 0; off < nbytes; off += BUFFER_SIZE) {
                    int chunk = Math.min(BUFFER_SIZE, nbytes - off);
                    byte[] slice = Arrays.copyOfRange(raw, off, off + chunk);
                    byte[] cbuf = new byte[chunk * 2 + 9];
                    int cs = compressBlock(level, slice, chunk, cbuf, cbuf.length, 8);
                    if (cs == 0) {
                        chunks = null;
                        break;
                    }
                    chunks.add(Arrays.copyOf(cbuf, cs));
                }
                long t1 = System.nanoTime();
                if (chunks == null) {
                    continue;
                }
                long totalC = 0;
                for (byte[] ch : chunks) {
                    totalC += ch.length;
                }
                byte[] dec = new byte[nbytes];
                long t2 = System.nanoTime();
                int di = 0;
                boolean ok = true;
                for (byte[] ch : chunks) {
                    int chunk = Math.min(BUFFER_SIZE, nbytes - di);
                    byte[] block = new byte[chunk];
                    if (!decompressBlock(level, ch, ch.length, block, chunk, 8)) {
                        ok = false;
                        break;
                    }
                    System.arraycopy(block, 0, dec, di, chunk);
                    di += chunk;
                }
                long t3 = System.nanoTime();
                if (!ok || di != nbytes || !Arrays.equals(raw, dec)) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Double> decVals = new ArrayList<>(n);
                    java.nio.ByteBuffer db =
                            java.nio.ByteBuffer.wrap(dec).order(java.nio.ByteOrder.LITTLE_ENDIAN);
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
                comp += totalC;
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

    public static void benchSpdpLong(List<? extends List<Long>> columns, int level, long[] out) {
        benchSpdpLong(columns, level, out, null);
    }

    public static void benchSpdpLong(
            List<? extends List<Long>> columns, int level, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        if (level < MIN_LEVEL || level > MAX_LEVEL) {
            out[0] = 0;
            return;
        }
        try {
            for (List<? extends Long> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int nbytes = n * 8;
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
                ArrayList<byte[]> chunks = new ArrayList<>();
                long t0 = System.nanoTime();
                for (int off = 0; off < nbytes; off += BUFFER_SIZE) {
                    int chunk = Math.min(BUFFER_SIZE, nbytes - off);
                    byte[] slice = Arrays.copyOfRange(raw, off, off + chunk);
                    byte[] cbuf = new byte[chunk * 2 + 9];
                    int cs = compressBlock(level, slice, chunk, cbuf, cbuf.length, 8);
                    if (cs == 0) {
                        chunks = null;
                        break;
                    }
                    chunks.add(Arrays.copyOf(cbuf, cs));
                }
                long t1 = System.nanoTime();
                if (chunks == null) {
                    continue;
                }
                long totalC = 0;
                for (byte[] ch : chunks) {
                    totalC += ch.length;
                }
                byte[] dec = new byte[nbytes];
                long t2 = System.nanoTime();
                int di = 0;
                boolean ok = true;
                for (byte[] ch : chunks) {
                    int chunk = Math.min(BUFFER_SIZE, nbytes - di);
                    byte[] block = new byte[chunk];
                    if (!decompressBlock(level, ch, ch.length, block, chunk, 8)) {
                        ok = false;
                        break;
                    }
                    System.arraycopy(block, 0, dec, di, chunk);
                    di += chunk;
                }
                long t3 = System.nanoTime();
                if (!ok || di != nbytes || !Arrays.equals(raw, dec)) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Long> decVals = new ArrayList<>(n);
                    java.nio.ByteBuffer db =
                            java.nio.ByteBuffer.wrap(dec).order(java.nio.ByteOrder.LITTLE_ENDIAN);
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
                comp += totalC;
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
