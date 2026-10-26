package org.example;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;

import me.lemire.integercompression.Composition;
import me.lemire.integercompression.FastPFOR;
import me.lemire.integercompression.IntWrapper;
import me.lemire.integercompression.IntegerCODEC;
import me.lemire.integercompression.VariableByte;

/**
 * Lossless decimal-scaled {@code int64} limbs (lo/hi int32) for double-wire research codecs.
 * Simple8b and Stream VByte operate on 32-bit words; full 64-bit columns are split into two
 * little-endian int32 streams without narrowing quantized magnitudes to int32.
 */
public final class Int64ColumnCodec {

    private static final IntegerCODEC FAST_PFOR = new Composition(new FastPFOR(), new VariableByte());

    private Int64ColumnCodec() {}

    static long[] quantizeDoubleColumn(List<? extends Double> col, int colScale) {
        return quantizeDoubleColumn(col, colScale, true);
    }

    static long[] quantizeDoubleColumn(
            List<? extends Double> col, int colScale, boolean requireBitExactRoundTrip) {
        int n = col.size();
        double pow = Math.pow(10.0, colScale);
        long[] data = new long[n];
        for (int i = 0; i < n; i++) {
            Double dv = col.get(i);
            double v = dv == null ? Double.NaN : dv;
            if (!Double.isFinite(v)) {
                data[i] = 0L;
                continue;
            }
            if (requireBitExactRoundTrip && v == 0.0) {
                data[i] = Double.doubleToRawLongBits(v);
                continue;
            }
            double scaled = v * pow;
            if (!Double.isFinite(scaled)) {
                return null;
            }
            if (scaled > Long.MAX_VALUE || scaled < Long.MIN_VALUE) {
                // Overflow here would silently saturate in Math.round and distort
                // downstream delta/varint stats; force caller to try a lower scale.
                return null;
            }
            long m = Math.round(scaled);
            if (requireBitExactRoundTrip) {
                double r = ((double) m) / pow;
                if (Double.doubleToRawLongBits(r) != Double.doubleToRawLongBits(v)) {
                    return null;
                }
            }
            data[i] = m;
        }
        return data;
    }

    static int[] lo32(long[] data) {
        int[] lo = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            lo[i] = (int) data[i];
        }
        return lo;
    }

    static int[] hi32(long[] data) {
        int[] hi = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            hi[i] = (int) (data[i] >>> 32);
        }
        return hi;
    }

    static long reconstruct(int lo, int hi) {
        return ((long) hi << 32) | (lo & 0xFFFFFFFFL);
    }

    static int[] fastPForCompress(int[] data, int n) {
        int[] buf = new int[Math.max(n * 2, n + 4096)];
        IntWrapper inputOffset = new IntWrapper(0);
        IntWrapper outputOffset = new IntWrapper(0);
        FAST_PFOR.compress(data, inputOffset, n, buf, outputOffset);
        return Arrays.copyOf(buf, outputOffset.get());
    }

    static int[] fastPForUncompress(int[] compressed, int n) {
        int[] recovered = new int[n];
        IntWrapper uIn = new IntWrapper(0);
        IntWrapper uOut = new IntWrapper(0);
        FAST_PFOR.uncompress(compressed, uIn, compressed.length, recovered, uOut);
        if (uOut.get() != n) {
            throw new IllegalStateException("FastPFOR int64 limb: length mismatch");
        }
        return recovered;
    }

    /** Wire: LE u32 {@code loWordCount}, then lo FastPFOR words, then hi FastPFOR words. */
    static byte[] fastPForEncodeInt64(long[] data, int n) {
        int[] lo = lo32(data);
        int[] hi = hi32(data);
        int[] cLo = fastPForCompress(lo, n);
        int[] cHi = fastPForCompress(hi, n);
        ByteBuffer bb =
                ByteBuffer.allocate(4 + (cLo.length + cHi.length) * 4)
                        .order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(cLo.length);
        for (int w : cLo) {
            bb.putInt(w);
        }
        for (int w : cHi) {
            bb.putInt(w);
        }
        return bb.array();
    }

    static long[] fastPForDecodeInt64(byte[] wire, int n) {
        if (wire.length < 4) {
            throw new IllegalArgumentException("FastPFOR int64: truncated header");
        }
        ByteBuffer bb = ByteBuffer.wrap(wire).order(ByteOrder.LITTLE_ENDIAN);
        int loWords = bb.getInt();
        if (loWords < 0 || bb.remaining() < loWords * 4) {
            throw new IllegalArgumentException("FastPFOR int64: truncated body");
        }
        int[] cLo = new int[loWords];
        for (int i = 0; i < loWords; i++) {
            cLo[i] = bb.getInt();
        }
        int hiWords = bb.remaining() / 4;
        int[] cHi = new int[hiWords];
        for (int i = 0; i < hiWords; i++) {
            cHi[i] = bb.getInt();
        }
        int[] lo = fastPForUncompress(cLo, n);
        int[] hi = fastPForUncompress(cHi, n);
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            out[i] = reconstruct(lo[i], hi[i]);
        }
        return out;
    }

    /** Wire: LE u32 {@code loPayloadLen}, then Stream VByte delta-1 on lo, then on hi. */
    static byte[] streamVByteEncodeInt64(long[] data, int n) {
        int[] lo = lo32(data);
        int[] hi = hi32(data);
        byte[] eLo = StreamVByteDelta1.encode(lo, n);
        byte[] eHi = StreamVByteDelta1.encode(hi, n);
        ByteBuffer bb =
                ByteBuffer.allocate(4 + eLo.length + eHi.length).order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(eLo.length);
        bb.put(eLo);
        bb.put(eHi);
        return bb.array();
    }

    static long[] streamVByteDecodeInt64(byte[] wire, int n) {
        if (wire.length < 4) {
            throw new IllegalArgumentException("streamvbyte int64: truncated header");
        }
        ByteBuffer bb = ByteBuffer.wrap(wire).order(ByteOrder.LITTLE_ENDIAN);
        int loLen = bb.getInt();
        if (loLen < 0 || bb.remaining() < loLen) {
            throw new IllegalArgumentException("streamvbyte int64: truncated lo payload");
        }
        byte[] eLo = new byte[loLen];
        bb.get(eLo);
        byte[] eHi = new byte[bb.remaining()];
        bb.get(eHi);
        int[] lo = new int[n];
        int[] hi = new int[n];
        StreamVByteDelta1.decode(eLo, lo, n);
        StreamVByteDelta1.decode(eHi, hi, n);
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            out[i] = reconstruct(lo[i], hi[i]);
        }
        return out;
    }

    /** Simple8b on sequential lo then hi int32 limbs ({@code 2*n} words). */
    static byte[] simple8bEncodeInt64(long[] data, int n) {
        int[] lo = lo32(data);
        int[] hi = hi32(data);
        int[] u32 = new int[2 * n];
        System.arraycopy(lo, 0, u32, 0, n);
        System.arraycopy(hi, 0, u32, n, n);
        byte[] raw = new byte[2 * n * 4];
        ByteBuffer wb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int w : u32) {
            wb.putInt(w);
        }
        return Simple8bIntColumn.encodeColumnBytes(raw);
    }

    static long[] simple8bDecodeInt64(byte[] body, int n) {
        byte[] raw = Simple8bIntColumn.decodeColumnBytes(body);
        if (raw.length != 2 * n * 4) {
            throw new IllegalStateException("simple8b int64: limb length mismatch");
        }
        ByteBuffer rb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int[] lo = new int[n];
        int[] hi = new int[n];
        for (int i = 0; i < n; i++) {
            lo[i] = rb.getInt();
        }
        for (int i = 0; i < n; i++) {
            hi[i] = rb.getInt();
        }
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            out[i] = reconstruct(lo[i], hi[i]);
        }
        return out;
    }

    static boolean doublesMatchColumn(List<? extends Double> col, long[] data, int colScale) {
        double pow = Math.pow(10.0, colScale);
        for (int i = 0; i < col.size(); i++) {
            double v = col.get(i);
            if (v == 0.0) {
                if (data[i] != Double.doubleToRawLongBits(v)) {
                    return false;
                }
                continue;
            }
            double r = ((double) data[i]) / pow;
            if (Double.doubleToRawLongBits(r) != Double.doubleToRawLongBits(v)) {
                return false;
            }
        }
        return true;
    }
}
