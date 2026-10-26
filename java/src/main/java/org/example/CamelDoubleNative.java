package org.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Pure-Java Camel (double API) aligned with {@code backend/myapp/camel_python.py} and
 * {@code cpp/src/camel_float_codec.cpp}. Upstream Camel caps fractional digits at 3, which is lossy for
 * columns like Cpu-usage ({@code max_point_number=9}). This port keeps the standard decimal path when it
 * is bit-exact and falls back to a 64-bit XOR literal per value otherwise, so {@code originalSize} always
 * reflects the full double column ({@code n * 8} bytes).
 */
public final class CamelDoubleNative {

    private static final int DECIMAL_MAX_COUNT = 3;
    private static final int[] M_VALUE_BITS = {3, 5, 7, 10, 15};
    private static final int[] THRESHOLD = {5, 25, 125, 625};
    private static final long[] POWERS = {1L, 10L, 100L, 1000L, 10_000L, 100_000L};
    private static final int CHUNK_POINTS = 1000;
    private static final int BUF_SIZE = 100_000;
    private static final int INTEGER_DELTA_MAX = 0xFFFF;

    private CamelDoubleNative() {}

    private static final class DecimalCount {
        final int dc;
        final int decimalValue;

        DecimalCount(int dc, int decimalValue) {
            this.dc = dc;
            this.decimalValue = decimalValue;
        }
    }

    private static final class DecimalParts {
        final int dc;
        final double decimalVal;

        DecimalParts(int dc, double decimalVal) {
            this.dc = dc;
            this.decimalVal = decimalVal;
        }
    }

    private static long javaRoundDouble(double x) {
        if (x >= 0.0) {
            return Math.round(x);
        }
        return (long) Math.ceil(x - 0.5);
    }

    private static int lowBitsMask32(int len) {
        if (len <= 0) {
            return 0;
        }
        if (len >= 32) {
            return 0xFFFFFFFF;
        }
        return (1 << len) - 1;
    }

    private static long lowBitsMask64(int len) {
        if (len <= 0) {
            return 0L;
        }
        if (len >= 64) {
            return 0xFFFFFFFFFFFFFFFFL;
        }
        return (1L << len) - 1L;
    }

    private static final class VignaOut {
        final byte[] buf = new byte[BUF_SIZE];
        long writtenBits;
        int current;
        int freeBits = 8;
        int pos;

        void writeByte(int b) {
            if (pos >= buf.length) {
                throw new IllegalStateException("camel: output overflow");
            }
            buf[pos++] = (byte) (b & 0xFF);
        }

        void writeInCurrent(int b, int len) {
            if (len == 0) {
                return;
            }
            freeBits -= len;
            current |= (b & lowBitsMask32(len)) << freeBits;
            if (freeBits == 0) {
                writeByte(current);
                freeBits = 8;
                current = 0;
            }
            writtenBits += len;
        }

        void writeBit(int bit) {
            writeInCurrent(bit != 0 ? 1 : 0, 1);
        }

        void writeInt(long x, int len) {
            x &= 0xFFFFFFFFL;
            if (len == 0) {
                return;
            }
            if (len <= freeBits) {
                writeInCurrent((int) x, len);
                return;
            }
            int i = len - freeBits;
            int queue = i & 7;
            if (freeBits != 0) {
                writeInCurrent((int) (x >> i), freeBits);
            }
            if (queue != 0) {
                i -= queue;
                writeInCurrent((int) (x & lowBitsMask32(queue)), queue);
                x >>= queue;
            }
            if (i == 32) {
                writeByte((int) ((x >> 24) & 0xFF));
            }
            if (i > 23) {
                writeByte((int) ((x >> 16) & 0xFF));
            }
            if (i > 15) {
                writeByte((int) ((x >> 8) & 0xFF));
            }
            if (i > 7) {
                writeByte((int) (x & 0xFF));
            }
            writtenBits += i;
        }

        void writeLong(long x, int len) {
            if (len <= 0) {
                return;
            }
            if (len < 64) {
                x &= (1L << len) - 1L;
            }
            if (len <= freeBits) {
                writeInCurrent((int) x, len);
                return;
            }
            int i = len - freeBits;
            int queue = i & 7;
            if (freeBits != 0) {
                writeInCurrent((int) (x >> i), freeBits);
            }
            if (queue != 0) {
                i -= queue;
                writeInCurrent((int) (x & lowBitsMask64(queue)), queue);
                x >>= queue;
            }
            if (i >= 56) {
                writeByte((int) ((x >> 56) & 0xFF));
            }
            if (i >= 48) {
                writeByte((int) ((x >> 48) & 0xFF));
            }
            if (i >= 40) {
                writeByte((int) ((x >> 40) & 0xFF));
            }
            if (i >= 32) {
                writeByte((int) ((x >> 32) & 0xFF));
            }
            if (i > 23) {
                writeByte((int) ((x >> 24) & 0xFF));
            }
            if (i > 15) {
                writeByte((int) ((x >> 16) & 0xFF));
            }
            if (i > 7) {
                writeByte((int) ((x >> 8) & 0xFF));
            }
            if (i > 0) {
                writeByte((int) (x & 0xFF));
            }
            writtenBits += i;
        }

        void alignFlush() {
            if (freeBits != 8) {
                writeInCurrent(0, freeBits);
            }
        }
    }

    private static final class VignaIn {
        final byte[] buffer;
        int pos;
        int fill;
        int current;

        VignaIn(byte[] buffer) {
            this.buffer = buffer;
        }

        int read() {
            if (pos >= buffer.length) {
                return -1;
            }
            return buffer[pos++] & 0xFF;
        }

        int readFromCurrent(int l) {
            if (fill == 0) {
                int b = read();
                if (b < 0) {
                    throw new IllegalStateException("camel: truncated bit stream");
                }
                current = b;
                fill = 8;
            }
            fill -= l;
            return (current >> fill) & lowBitsMask32(l);
        }

        int readBit() {
            return readFromCurrent(1);
        }

        int readInt(int l) {
            if (l < 0 || l > 32) {
                throw new IllegalStateException("camel: bad read_int");
            }
            if (l == 0) {
                return 0;
            }
            int x = 0;
            for (int i = 0; i < l; i++) {
                x = (x << 1) | (readBit() & 1);
            }
            return x & 0xFFFFFFFF;
        }

        long readLongBits(int l) {
            if (l < 0 || l > 64) {
                throw new IllegalStateException("camel: bad read_long");
            }
            if (l == 0) {
                return 0L;
            }
            long x = 0L;
            for (int i = 0; i < l; i++) {
                x = (x << 1) | (readBit() & 1);
            }
            return l == 64 ? x : (x & ((1L << l) - 1L));
        }
    }

    private static DecimalCount calDecimalCount(double value) {
        double factor = 1.0;
        int decimalCount = 0;
        value = Math.abs(value);
        if (!Double.isFinite(value)) {
            return new DecimalCount(DECIMAL_MAX_COUNT, 0);
        }
        final double epsilon = 1e-7;
        int guard = 0;
        while (Math.abs(value * factor - javaRoundDouble(value * factor)) > epsilon) {
            factor *= 10.0;
            decimalCount++;
            if (++guard > 64) {
                break;
            }
        }
        if (decimalCount == 0) {
            decimalCount = 1;
        }
        int dc;
        if (decimalCount > 0 && decimalCount <= DECIMAL_MAX_COUNT) {
            dc = decimalCount;
        } else {
            dc = DECIMAL_MAX_COUNT;
        }
        long powI = POWERS[dc];
        int decimalValue = (int) (javaRoundDouble(value * powI) % powI);
        return new DecimalCount(dc, decimalValue);
    }

    private static double adjustedFirstValue(double value, DecimalCount dec) {
        double sign = value < 0.0 ? -1.0 : 1.0;
        double p = POWERS[dec.dc];
        return sign * ((int) Math.abs(value) * POWERS[dec.dc] + dec.decimalValue) / p;
    }

    private static double reconstructDecimalPath(double value) {
        int intSignal = value < 0.0 ? 0 : 1;
        int intVal = (int) value;
        DecimalCount dec = calDecimalCount(value);
        double scale = POWERS[dec.dc];
        double sign = intSignal == 0 ? -1.0 : 1.0;
        double decimalVal = dec.decimalValue / scale;
        decimalVal = javaRoundDouble(decimalVal * scale) / scale;
        return sign * ((intVal * scale + decimalVal * scale) / scale);
    }

    private static boolean decimalPathLossless(double value) {
        return Double.doubleToRawLongBits(value)
                == Double.doubleToRawLongBits(reconstructDecimalPath(value));
    }

    /**
     * Integer-part delta branch can only represent absolute deltas up to 16 bits.
     * When the integer jump is larger, fall back to literal XOR path.
     */
    private static boolean integerDeltaEncodable(long previousIntegerPart, int currentIntegerPart) {
        long diff = (long) currentIntegerPart - previousIntegerPart;
        return diff >= -INTEGER_DELTA_MAX && diff <= INTEGER_DELTA_MAX;
    }

    private static final class CamelCompressor {
        final VignaOut out = new VignaOut();
        long storedVal;
        long storedDoubleBits;
        boolean first = true;

        void writeFirst(long valueBits) {
            first = false;
            storedDoubleBits = valueBits;
            storedVal = (long) Double.longBitsToDouble(valueBits);
            out.writeInt(valueBits >>> 32, 32);
            out.writeInt(valueBits & 0xFFFFFFFFL, 32);
        }

        void compressDecimalValue(int decimalValue, int decimalCount) {
            if (decimalCount == 0) {
                return;
            }
            out.writeInt(decimalCount - 1, 2);
            int thread = THRESHOLD[decimalCount - 1];
            int m = decimalValue;
            if (decimalValue - thread >= 0) {
                out.writeBit(1);
                m = decimalValue % thread;
                double p = POWERS[decimalCount];
                long xor = Double.doubleToRawLongBits(decimalValue / p + 1.0)
                        ^ Double.doubleToRawLongBits((double) m / p + 1.0);
                out.writeLong(xor >> (52 - decimalCount), decimalCount);
            } else {
                out.writeBit(0);
            }

            if (decimalCount == 1) {
                out.writeInt(m, 3);
            } else if (decimalCount == 2) {
                if (m < 8) {
                    out.writeInt(0, 1);
                    out.writeInt(m, 3);
                } else {
                    out.writeInt(1, 1);
                    out.writeInt(m, 5);
                }
            } else if (decimalCount == 3) {
                if (m < 2) {
                    out.writeInt(0, 2);
                    out.writeInt(m, 1);
                } else if (m < 8) {
                    out.writeInt(1, 2);
                    out.writeInt(m, 3);
                } else if (m < 32) {
                    out.writeInt(2, 2);
                    out.writeInt(m, 5);
                } else {
                    out.writeInt(3, 2);
                    out.writeInt(m, M_VALUE_BITS[decimalCount - 1]);
                }
            } else {
                if (m < 16) {
                    out.writeInt(0, 2);
                    out.writeInt(m, 4);
                } else if (m < 64) {
                    out.writeInt(1, 2);
                    out.writeInt(m, 6);
                } else if (m < 256) {
                    out.writeInt(2, 2);
                    out.writeInt(m, 8);
                } else {
                    out.writeInt(3, 2);
                    out.writeInt(m, M_VALUE_BITS[decimalCount - 1]);
                }
            }
        }

        void compressIntegerValue(int intValue, int intSignal) {
            int diff = intValue - (int) storedVal;
            out.writeInt(intSignal, 1);
            if (-1 <= diff && diff <= 1) {
                out.writeInt(diff + 1, 2);
            } else {
                out.writeInt(3, 2);
                if (diff < 0) {
                    out.writeBit(0);
                    diff = -diff;
                } else {
                    out.writeBit(1);
                }
                if (2 <= diff && diff < 8) {
                    out.writeInt(0, 1);
                    out.writeInt(diff, 3);
                } else {
                    out.writeInt(1, 1);
                    out.writeInt(diff, 16);
                }
            }
            storedVal = intValue;
        }

        void compressLiteral(double value) {
            out.writeBit(1);
            long bits = Double.doubleToRawLongBits(value);
            long xor = bits ^ storedDoubleBits;
            out.writeInt(xor >>> 32, 32);
            out.writeInt(xor & 0xFFFFFFFFL, 32);
            storedDoubleBits = bits;
            storedVal = (long) value;
        }

        void compressValue(double value) {
            int intValue = (int) value;
            if (!decimalPathLossless(value) || !integerDeltaEncodable(storedVal, intValue)) {
                compressLiteral(value);
                return;
            }
            out.writeBit(0);
            int intSignal = value < 0.0 ? 0 : 1;
            compressIntegerValue(intValue, intSignal);
            DecimalCount dec = calDecimalCount(value);
            compressDecimalValue(dec.decimalValue, dec.dc);
            storedDoubleBits = Double.doubleToRawLongBits(value);
        }

        void addValue(double value) {
            if (first) {
                DecimalCount dec = calDecimalCount(value);
                double adjusted = adjustedFirstValue(value, dec);
                if (Double.doubleToRawLongBits(adjusted) == Double.doubleToRawLongBits(value)) {
                    writeFirst(Double.doubleToRawLongBits(adjusted));
                } else {
                    writeFirst(Double.doubleToRawLongBits(value));
                }
                return;
            }
            compressValue(value);
        }

        void close() {
            out.writeBit(0);
            out.alignFlush();
        }

        byte[] finish() {
            long wb = out.writtenBits;
            int compBytes = (int) ((wb + 7L) / 8L);
            if (compBytes <= 0 || compBytes > out.buf.length) {
                throw new IllegalStateException("camel: bad size");
            }
            return Arrays.copyOf(out.buf, compBytes);
        }
    }

    private static final class CamelDecompressor {
        final VignaIn in;
        long storedVal;
        long storedDoubleBits;
        boolean first = true;
        boolean eos;
        int readNum;

        CamelDecompressor(byte[] data) {
            in = new VignaIn(data);
        }

        int[] readIntPart() {
            int intSignal = in.readInt(1);
            int integerNum = in.readInt(2);
            int diffVal;
            if (integerNum < 3) {
                diffVal = integerNum - 1;
            } else {
                int diffSymbol = in.readBit();
                int rng = in.readInt(1);
                if (rng == 0) {
                    diffVal = in.readInt(3);
                } else {
                    diffVal = in.readInt(16);
                }
                diffVal = (diffSymbol == 0 ? -1 : 1) * diffVal;
            }
            storedVal = diffVal + storedVal;
            int sig = storedVal >= 0 ? 1 : 0;
            if (storedVal == 0 && intSignal == 0) {
                sig = 0;
            }
            long ab = storedVal >= 0 ? storedVal : -storedVal;
            return new int[] {sig, (int) ab};
        }

        DecimalParts readDecimal() {
            int decimalCount = in.readInt(2) + 1;
            int isM = in.readInt(1);
            long xorString = 0L;
            if (isM == 1) {
                long xor = in.readLongBits(decimalCount);
                xorString = xor << (52 - decimalCount);
            }
            int mInt;
            if (decimalCount <= 1) {
                mInt = in.readInt(3);
            } else if (decimalCount == 2) {
                int temp = in.readInt(1);
                mInt = temp == 0 ? in.readInt(3) : in.readInt(5);
            } else if (decimalCount == 3) {
                int temp = in.readInt(2);
                if (temp == 0) {
                    mInt = in.readInt(1);
                } else if (temp == 1) {
                    mInt = in.readInt(3);
                } else if (temp == 2) {
                    mInt = in.readInt(5);
                } else {
                    mInt = in.readInt(M_VALUE_BITS[decimalCount - 1]);
                }
            } else {
                int temp = in.readInt(2);
                if (temp == 0) {
                    mInt = in.readInt(4);
                } else if (temp == 1) {
                    mInt = in.readInt(6);
                } else if (temp == 2) {
                    mInt = in.readInt(8);
                } else {
                    mInt = in.readInt(M_VALUE_BITS[decimalCount - 1]);
                }
            }

            double scale = POWERS[decimalCount];
            double decimalVal;
            if (isM == 1) {
                double m = (double) mInt / scale + 1.0;
                long mPrime = Double.doubleToRawLongBits(m);
                long decLong = xorString ^ mPrime;
                decimalVal = Double.longBitsToDouble(decLong) - 1.0;
            } else {
                decimalVal = (double) mInt / POWERS[decimalCount];
            }
            decimalVal = javaRoundDouble(decimalVal * scale) / scale;
            return new DecimalParts(decimalCount, decimalVal);
        }

        double nextValue() {
            if (in.readBit() == 1) {
                long hi = in.readInt(32) & 0xFFFFFFFFL;
                long lo = in.readInt(32) & 0xFFFFFFFFL;
                long xor = (hi << 32) | lo;
                long bits = xor ^ storedDoubleBits;
                double v = Double.longBitsToDouble(bits);
                storedDoubleBits = bits;
                storedVal = (long) v;
                return v;
            }
            int[] intVal = readIntPart();
            DecimalParts dec = readDecimal();
            int dc = dec.dc;
            double scale = POWERS[dc];
            double sign = intVal[0] == 0 ? -1.0 : 1.0;
            double cur = sign * ((intVal[1] * scale + dec.decimalVal * scale) / scale);
            storedDoubleBits = Double.doubleToRawLongBits(cur);
            return cur;
        }

        Double readPair() {
            if (eos) {
                return null;
            }
            readNum++;
            if (first) {
                first = false;
                long hi = in.readInt(32) & 0xFFFFFFFFL;
                long lo = in.readInt(32) & 0xFFFFFFFFL;
                long firstLong = (hi << 32) | lo;
                double firstVal = Double.longBitsToDouble(firstLong);
                storedVal = (long) firstVal;
                storedDoubleBits = firstLong;
                return firstVal;
            }
            if (readNum - 1 == CHUNK_POINTS) {
                eos = true;
                return null;
            }
            return nextValue();
        }
    }

    static byte[] encodeChunk(double[] values) {
        CamelCompressor c = new CamelCompressor();
        for (double v : values) {
            c.addValue(v);
        }
        c.close();
        return c.finish();
    }

    static void decodeChunk(byte[] payload, double[] out) {
        CamelDecompressor dec = new CamelDecompressor(payload);
        for (int i = 0; i < out.length; i++) {
            Double v = dec.readPair();
            if (v == null) {
                throw new IllegalStateException("camel decompress early end");
            }
            out[i] = v;
        }
    }

    public static boolean benchDoubleColumns(List<ArrayList<Double>> columns, long[] out) {
        return benchDoubleColumns(columns, out, null);
    }

    public static boolean benchDoubleColumns(
            List<ArrayList<Double>> columns, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        final boolean debug = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CAMEL_DEBUG"));
        for (List<Double> col : columns) {
            int n = col.size();
            if (n == 0) {
                continue;
            }
            orig += BenchmarkWireFormat.columnOriginalSize(n, true);
            long colComp = 0;
            long colCt = 0;
            long colDt = 0;
            try {
                for (int start = 0; start < n; start += CHUNK_POINTS) {
                    int end = Math.min(n, start + CHUNK_POINTS);
                    int chunkLen = end - start;
                    double[] chunk = new double[chunkLen];
                    for (int i = 0; i < chunkLen; i++) {
                        chunk[i] = col.get(start + i);
                    }
                    long t0 = System.nanoTime();
                    byte[] payload = encodeChunk(chunk);
                    long t1 = System.nanoTime();
                    long t2 = System.nanoTime();
                    double[] decoded = new double[chunkLen];
                    decodeChunk(payload, decoded);
                    long t3 = System.nanoTime();
                    for (int i = 0; i < chunkLen; i++) {
                        if (Double.doubleToRawLongBits(decoded[i])
                                != Double.doubleToRawLongBits(chunk[i])) {
                            throw new IllegalStateException("Camel round-trip mismatch");
                        }
                    }
                    if (lossAcc != null) {
                        BenchmarkLoss.accumulateDoubleArrayLoss(chunk, decoded, chunkLen, lossAcc);
                    }
                    colComp += payload.length;
                    colCt += t1 - t0;
                    colDt += t3 - t2;
                }
            } catch (Throwable ex) {
                if (debug) {
                    ex.printStackTrace(System.err);
                }
                continue;
            }
            comp += colComp;
            ct += colCt;
            dt += colDt;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }
}
