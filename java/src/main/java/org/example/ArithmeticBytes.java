package org.example;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Static order-0 range coding (carry-free / Subbotin-style) for benchmark row {@code arithmetic}.
 *
 * <p>Same header and payload bit layout as {@link HuffmanBytes}: LE u32 {@code orig_len}, 256×LE u32
 * raw byte counts, LE u32 {@code num_bits}, then ceil(num_bits/8) bytes (MSB-first bit order; payload
 * is byte-aligned so {@code num_bits} is a multiple of 8). Single distinct raw byte uses
 * {@code num_bits = 0} and repeats that byte {@code orig_len} times.
 *
 * <p>Frequency model matches Python {@code arithmetic_bytes_python}: Laplace +1 on raw counts,
 * scaled to sum {@link #M} ({@code 1<<15}), same tie-breaking as {@link AnsEntropyRans}.
 */
public final class ArithmeticBytes {

    private static final int HEADER = 4 + 1024 + 4;
    public static final int M = 1 << 15;
    private static final long MAX_RANGE = 0xffffffffL;
    private static final long MIN_RANGE = 0x10000L;
    private static final long MASK = 0xffffffffL;
    private static final long MASK1 = 0xff000000L;
    private static final int SHIFT = 24;

    private ArithmeticBytes() {}

    static long[] countRaw(byte[] data) {
        long[] c = new long[256];
        if (data != null) {
            for (byte b : data) {
                c[b & 0xff]++;
            }
        }
        return c;
    }

    static int[] buildFreqFromRawCounts(long[] raw) {
        int[] adj = new int[256];
        long tot = 0;
        for (int i = 0; i < 256; i++) {
            adj[i] = (int) Math.min(raw[i], 0xffffffffL) + 1;
            tot += adj[i];
        }
        int[] f = new int[256];
        for (int i = 0; i < 256; i++) {
            f[i] = (int) Math.max(1L, (adj[i] * (long) M) / tot);
        }
        int sum = 0;
        for (int v : f) {
            sum += v;
        }
        while (sum < M) {
            int best = 0;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < 256; i++) {
                double score = (adj[i] * (double) M) / (double) tot - (double) f[i];
                if (score > bestScore) {
                    bestScore = score;
                    best = i;
                }
            }
            f[best]++;
            sum++;
        }
        while (sum > M) {
            int best = -1;
            int bestVal = -1;
            for (int i = 0; i < 256; i++) {
                if (f[i] > 1 && f[i] > bestVal) {
                    bestVal = f[i];
                    best = i;
                }
            }
            if (best < 0) {
                throw new IllegalStateException("arithmetic: cannot normalize frequency table");
            }
            f[best]--;
            sum--;
        }
        return f;
    }

    static int[] buildCum(int[] freq) {
        int[] cum = new int[257];
        cum[0] = 0;
        for (int i = 0; i < 256; i++) {
            cum[i + 1] = cum[i] + freq[i];
        }
        if (cum[256] != M) {
            throw new IllegalStateException("arithmetic frequency sum");
        }
        return cum;
    }

    static int searchSym(int[] cum, int value) {
        int i = 0;
        int j = 255;
        while (i < j) {
            int k = (i + j) >>> 1;
            if (cum[k + 1] <= value) {
                i = k + 1;
            } else {
                j = k;
            }
        }
        return i;
    }

    private static long[] encodeNormalize(long low, long range, ByteArrayOutputStream out) {
        low &= MASK;
        while ((low & MASK1) == (((low + range) & MASK) & MASK1)) {
            out.write((int) ((low >>> SHIFT) & 0xff));
            low = (low << 8) & MASK;
            range = (range << 8) & MASK;
        }
        while (range < MIN_RANGE) {
            range = (MIN_RANGE - (low & (MIN_RANGE - 1))) << 8;
            out.write((int) ((low >>> SHIFT) & 0xff));
            low = (low << 8) & MASK;
        }
        return new long[]{low, range & MASK};
    }

    private static long[] decodeNormalize(long low, long range, long code, byte[] payload, int[] pos) {
        low &= MASK;
        code &= MASK;
        while ((low & MASK1) == (((low + range) & MASK) & MASK1)) {
            if (pos[0] >= payload.length) {
                throw new IllegalArgumentException("arithmetic: truncated payload (normalize)");
            }
            code = (((code << 8) & MASK) + (payload[pos[0]] & 0xffL)) & MASK;
            pos[0]++;
            low = (low << 8) & MASK;
            range = (range << 8) & MASK;
        }
        while (range < MIN_RANGE) {
            range = (MIN_RANGE - (low & (MIN_RANGE - 1))) << 8;
            if (pos[0] >= payload.length) {
                throw new IllegalArgumentException("arithmetic: truncated payload (normalize 2)");
            }
            code = (((code << 8) & MASK) + (payload[pos[0]] & 0xffL)) & MASK;
            pos[0]++;
            low = (low << 8) & MASK;
        }
        return new long[]{low, range & MASK, code};
    }

    public static byte[] encode(byte[] data) {
        int origLen = data == null ? 0 : data.length;
        long[] raw = countRaw(data);
        ByteBuffer hdr = ByteBuffer.allocate(HEADER).order(ByteOrder.LITTLE_ENDIAN);
        hdr.putInt(origLen);
        for (int i = 0; i < 256; i++) {
            hdr.putInt((int) Math.min(raw[i], 0xffffffffL));
        }
        if (origLen == 0) {
            hdr.putInt(0);
            return hdr.array();
        }
        int nz = 0;
        for (int s = 0; s < 256; s++) {
            if (raw[s] > 0) {
                nz++;
            }
        }
        if (nz == 1) {
            hdr.putInt(0);
            return hdr.array();
        }
        int[] freq = buildFreqFromRawCounts(raw);
        int[] cum = buildCum(freq);

        ByteArrayOutputStream body = new ByteArrayOutputStream();
        long low = 0;
        long range = MAX_RANGE;
        for (int i = 0; i < origLen; i++) {
            int sym = data[i] & 0xff;
            long temp = range / M;
            if (temp == 0) {
                temp = 1;
            }
            low = (low + (long) cum[sym] * temp) & MASK;
            range = freq[sym] * temp;
            long[] st = encodeNormalize(low, range, body);
            low = st[0];
            range = st[1];
        }
        body.write((int) ((low >>> 24) & 0xff));
        body.write((int) ((low >>> 16) & 0xff));
        body.write((int) ((low >>> 8) & 0xff));
        body.write((int) (low & 0xff));

        byte[] bodyBytes = body.toByteArray();
        int numBits = bodyBytes.length * 8;
        hdr.putInt(numBits);
        byte[] out = new byte[HEADER + bodyBytes.length];
        System.arraycopy(hdr.array(), 0, out, 0, HEADER);
        System.arraycopy(bodyBytes, 0, out, HEADER, bodyBytes.length);
        return out;
    }

    public static byte[] decode(byte[] comp) {
        if (comp.length < HEADER) {
            throw new IllegalArgumentException("arithmetic: truncated header");
        }
        ByteBuffer bb = ByteBuffer.wrap(comp, 0, HEADER).order(ByteOrder.LITTLE_ENDIAN);
        int origLen = bb.getInt();
        long[] raw = new long[256];
        for (int i = 0; i < 256; i++) {
            raw[i] = bb.getInt() & 0xffffffffL;
        }
        int numBits = bb.getInt();
        if (origLen == 0) {
            return new byte[0];
        }
        int nz = 0;
        int onlySym = -1;
        for (int s = 0; s < 256; s++) {
            if (raw[s] > 0) {
                nz++;
                onlySym = s;
            }
        }
        if (nz == 1) {
            byte[] out = new byte[origLen];
            Arrays.fill(out, (byte) onlySym);
            return out;
        }
        int need = HEADER + (numBits + 7) / 8;
        if (numBits < 0 || comp.length < need) {
            throw new IllegalArgumentException("arithmetic: truncated payload");
        }
        int payloadLen = (numBits + 7) / 8;
        byte[] payload = Arrays.copyOfRange(comp, HEADER, HEADER + payloadLen);
        if (payload.length < 4) {
            throw new IllegalArgumentException("arithmetic: payload too short");
        }
        long code = payload[0] & 0xffL;
        code = (code << 8) | (payload[1] & 0xffL);
        code = (code << 8) | (payload[2] & 0xffL);
        code = (code << 8) | (payload[3] & 0xffL);
        code &= MASK;

        int[] pos = {4};
        long low = 0;
        long range = MAX_RANGE;
        int[] freq = buildFreqFromRawCounts(raw);
        int[] cum = buildCum(freq);

        byte[] out = new byte[origLen];
        for (int op = 0; op < origLen; op++) {
            long temp = range / M;
            if (temp == 0) {
                temp = 1;
            }
            long diff = (code - low) / temp;
            int value = (int) diff;
            if (value < 0) {
                value = 0;
            }
            if (value >= M) {
                value = M - 1;
            }
            int s = searchSym(cum, value);
            out[op] = (byte) s;
            low = (low + (long) cum[s] * temp) & MASK;
            range = freq[s] * temp;
            long[] st = decodeNormalize(low, range, code, payload, pos);
            low = st[0];
            range = st[1];
            code = st[2];
        }
        return out;
    }

    public static void bench(byte[] data, long[] result) {
        bench(data, result, null);
    }

    public static void bench(byte[] data, long[] result, double[] lossAcc) {
        long t0 = System.nanoTime();
        byte[] c = encode(data);
        long t1 = System.nanoTime();
        byte[] d = decode(c);
        long t2 = System.nanoTime();
        if (data == null) {
            data = new byte[0];
        }
        if (!Arrays.equals(data, d)) {
            BenchmarkLoss.accumulateByteLoss(data, d, lossAcc);
            throw new IllegalStateException("arithmetic roundtrip failed len=" + data.length);
        }
        BenchmarkLoss.accumulateByteLoss(data, d, lossAcc);
        result[0] = data.length;
        result[1] = c.length;
        result[2] = t1 - t0;
        result[3] = t2 - t1;
    }
}
