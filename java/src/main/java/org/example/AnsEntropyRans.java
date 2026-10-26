package org.example;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Order-0 rANS (range variant of ANS) entropy coder over bytes.
 * Used for benchmark rows keyed {@code ans}; layout is self-contained (freq table + bitstream).
 */
public final class AnsEntropyRans {

    public static final int PROB_BITS = 15;
    public static final int M = 1 << PROB_BITS;

    private AnsEntropyRans() {}

    static int[] buildFreq(byte[] data) {
        int[] c = new int[256];
        for (byte b : data) {
            c[b & 0xff]++;
        }
        for (int i = 0; i < 256; i++) {
            c[i]++;
        }
        long tot = 0;
        for (int v : c) {
            tot += v;
        }
        int[] f = new int[256];
        for (int i = 0; i < 256; i++) {
            f[i] = (int) Math.max(1L, (c[i] * (long) M) / tot);
        }
        int sum = 0;
        for (int v : f) {
            sum += v;
        }
        while (sum < M) {
            int best = 0;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < 256; i++) {
                double score = (c[i] * (double) M) / (double) tot - (double) f[i];
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
                throw new IllegalStateException("ANS: cannot normalize frequency table");
            }
            f[best]--;
            sum--;
        }
        return f;
    }

    /** Uniform frequencies (empty payload). */
    static int[] uniformFreq() {
        int[] f = new int[256];
        int base = M / 256;
        int rem = M - base * 256;
        Arrays.fill(f, base);
        f[0] += rem;
        return f;
    }

    static int[] buildCum(int[] freq) {
        int[] cum = new int[257];
        cum[0] = 0;
        for (int i = 0; i < 256; i++) {
            cum[i + 1] = cum[i] + freq[i];
        }
        if (cum[256] != M) {
            throw new IllegalStateException("ANS frequency table must sum to M");
        }
        return cum;
    }

    /** Largest s with cum[s] <= slot (slot in [0, M-1]). */
    static int symFromSlot(int[] cum, int slot) {
        int lo = 0;
        int hi = 255;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (cum[mid] <= slot) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    public static byte[] encode(byte[] data) {
        int[] freq = (data == null || data.length == 0) ? uniformFreq() : buildFreq(data);
        int[] cum = buildCum(freq);
        long state = 1L << 23;
        ArrayList<Byte> emit = new ArrayList<>();
        if (data != null) {
            for (int i = data.length - 1; i >= 0; i--) {
                int s = data[i] & 0xff;
                int fs = freq[s];
                int cs = cum[s];
                long max = (((1L << 23) >> PROB_BITS) << 8) * fs;
                while (state >= max) {
                    emit.add((byte) (state & 0xFFL));
                    state >>>= 8;
                }
                state = (state / fs) * M + (state % fs) + cs;
            }
        }
        state &= 0xffffffffL;
        for (int k = 0; k < 4; k++) {
            emit.add((byte) (state & 0xFFL));
            state >>>= 8;
        }
        ByteBuffer hdr = ByteBuffer.allocate(4 + 512).order(ByteOrder.LITTLE_ENDIAN);
        hdr.putInt(data == null ? 0 : data.length);
        for (int i = 0; i < 256; i++) {
            hdr.putShort((short) freq[i]);
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream(516 + emit.size());
        try {
            out.write(hdr.array());
            for (int i = emit.size() - 1; i >= 0; i--) {
                out.write(emit.get(i));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return out.toByteArray();
    }

    public static byte[] decode(byte[] comp) {
        if (comp.length < 4 + 512 + 4) {
            throw new IllegalArgumentException("ANS: truncated blob");
        }
        ByteBuffer hdr = ByteBuffer.wrap(comp, 0, 4 + 512).order(ByteOrder.LITTLE_ENDIAN);
        int origLen = hdr.getInt();
        int[] freq = new int[256];
        for (int i = 0; i < 256; i++) {
            freq[i] = hdr.getShort() & 0xffff;
        }
        int sumf = 0;
        for (int f : freq) {
            sumf += f;
        }
        if (sumf != M) {
            throw new IllegalArgumentException("ANS: bad frequency sum");
        }
        int[] cum = buildCum(freq);
        int bodyStart = 4 + 512;
        // Encoder reverses emit[]: first body bytes are flush as fb3,fb2,fb1,fb0 (big-endian state).
        int n = comp.length;
        long state = ((long) (comp[bodyStart] & 0xff) << 24)
                | ((long) (comp[bodyStart + 1] & 0xff) << 16)
                | ((long) (comp[bodyStart + 2] & 0xff) << 8)
                | (comp[bodyStart + 3] & 0xffL);
        int ptr = bodyStart + 4;
        if (origLen == 0) {
            if (state != (1L << 23) || ptr != n) {
                throw new IllegalArgumentException("ANS: invalid empty stream");
            }
            return new byte[0];
        }
        byte[] outb = new byte[origLen];
        int op = 0;
        while (op < origLen) {
            while (state < (1L << 23)) {
                if (ptr >= n) {
                    throw new IllegalArgumentException("ANS: truncated body");
                }
                state = (state << 8) | (comp[ptr] & 0xffL);
                ptr++;
            }
            int slot = (int) (state % (long) M);
            int s = symFromSlot(cum, slot);
            outb[op++] = (byte) s;
            int fs = freq[s];
            int cs = cum[s];
            state = fs * (state / M) + (state % M) - cs;
            state &= 0xffffffffL;
        }
        while (state < (1L << 23)) {
            if (ptr >= n) {
                throw new IllegalArgumentException("ANS: truncated final state");
            }
            state = (state << 8) | (comp[ptr] & 0xffL);
            ptr++;
        }
        if (ptr != n || state != (1L << 23)) {
            throw new IllegalArgumentException("ANS: trailing or corrupt body");
        }
        return outb;
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
        if (!Arrays.equals(data, d)) {
            BenchmarkLoss.accumulateByteLoss(data, d, lossAcc);
            throw new IllegalStateException("ANS roundtrip failed len=" + (data == null ? 0 : data.length));
        }
        BenchmarkLoss.accumulateByteLoss(data, d, lossAcc);
        result[0] = data == null ? 0 : data.length;
        result[1] = c.length;
        result[2] = t1 - t0;
        result[3] = t2 - t1;
    }
}
