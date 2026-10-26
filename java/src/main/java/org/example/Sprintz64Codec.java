package org.example;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;

/** 64-bit SPRINTZ codec matching the benchmark C++ path for INT64 and raw DOUBLE bits. */
public final class Sprintz64Codec {
    private static final int BLOCK_SIZE = 8;

    private Sprintz64Codec() {}

    public static byte[] encodeLongs(List<Long> input) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        LongFire fire = new LongFire();
        ArrayList<Long> values = new ArrayList<>(BLOCK_SIZE + 1);
        boolean hasFirst = false;
        for (Long boxed : input) {
            long value = boxed == null ? 0L : boxed;
            if (!hasFirst) {
                values.add(value);
                hasFirst = true;
                continue;
            }
            values.add(value);
            if (values.size() == BLOCK_SIZE + 1) {
                long pre = values.get(0);
                long[] residuals = new long[BLOCK_SIZE];
                fire.reset();
                for (int i = 1; i <= BLOCK_SIZE; i++) {
                    long current = values.get(i);
                    residuals[i - 1] = zigzag(fireResidual(fire, current, pre));
                    pre = current;
                }
                int width = maxBitWidth(residuals);
                out.write(width);
                writeLongBe(out, values.get(0));
                byte[] packed = new byte[width];
                pack8(width, residuals, packed);
                out.write(packed, 0, packed.length);
                values.clear();
                hasFirst = false;
            }
        }
        if (!values.isEmpty()) {
            out.write(0x80 | values.size());
            for (long value : values) {
                writeLongBe(out, value);
            }
        }
        return out.toByteArray();
    }

    public static long[] decodeLongs(byte[] encoded) {
        ArrayList<Long> out = new ArrayList<>();
        int pos = 0;
        while (pos < encoded.length) {
            int width = encoded[pos++] & 0xff;
            if ((width & 0x80) != 0) {
                int size = width & 0x7f;
                for (int i = 0; i < size; i++) {
                    out.add(readLongBe(encoded, pos));
                    pos += Long.BYTES;
                }
                break;
            }
            long pre = readLongBe(encoded, pos);
            pos += Long.BYTES;
            long[] residuals = unpack8(width, encoded, pos);
            pos += width;
            out.add(pre);
            LongFire fire = new LongFire();
            long previous = pre;
            for (long residual : residuals) {
                long err = unzigzag(residual);
                long value = err + fire.predict(previous);
                out.add(value);
                fire.train(previous, value, err);
                previous = value;
            }
        }
        long[] result = new long[out.size()];
        for (int i = 0; i < out.size(); i++) result[i] = out.get(i);
        return result;
    }

    private static long fireResidual(LongFire fire, long value, long previous) {
        long pred = fire.predict(previous);
        long err = value - pred;
        fire.train(previous, value, err);
        return err;
    }

    private static long zigzag(long value) {
        return (value << 1) ^ (value < 0 ? -1L : 0L);
    }

    private static long unzigzag(long value) {
        return (value >>> 1) ^ (0L - (value & 1L));
    }

    private static int maxBitWidth(long[] values) {
        int width = 1;
        for (long value : values) {
            width = Math.max(width, Long.SIZE - Long.numberOfLeadingZeros(value));
        }
        return width;
    }

    private static void pack8(int width, long[] values, byte[] out) {
        int bitPos = 0;
        for (long value : values) {
            int remaining = width;
            while (remaining > 0) {
                int byteIndex = bitPos >>> 3;
                int bitOffset = bitPos & 7;
                int take = Math.min(remaining, 8 - bitOffset);
                int shift = remaining - take;
                int chunk = (int) ((value >>> shift) & ((1L << take) - 1L));
                out[byteIndex] |= (byte) (chunk << (8 - bitOffset - take));
                bitPos += take;
                remaining -= take;
            }
        }
    }

    private static long[] unpack8(int width, byte[] input, int offset) {
        long[] values = new long[BLOCK_SIZE];
        int bitPos = offset * 8;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            long value = 0L;
            int remaining = width;
            while (remaining > 0) {
                int byteIndex = bitPos >>> 3;
                int bitOffset = bitPos & 7;
                int take = Math.min(remaining, 8 - bitOffset);
                int chunk = input[byteIndex] & (0xff >>> bitOffset);
                chunk = (chunk & 0xff) >>> (8 - bitOffset - take);
                value = (value << take) | (chunk & ((1L << take) - 1L));
                bitPos += take;
                remaining -= take;
            }
            values[i] = value;
        }
        return values;
    }

    private static void writeLongBe(ByteArrayOutputStream out, long value) {
        for (int i = 7; i >= 0; i--) {
            out.write((int) ((value >>> (i * 8)) & 0xff));
        }
    }

    private static long readLongBe(byte[] input, int offset) {
        long value = 0L;
        for (int i = 0; i < Long.BYTES; i++) {
            value = (value << 8) | (input[offset + i] & 0xffL);
        }
        return value;
    }

    private static final class LongFire {
        private long accumulator;
        private long delta;

        void reset() {
            accumulator = 0L;
            delta = 0L;
        }

        long predict(long value) {
            long alpha = accumulator >> 2;
            long diff = (alpha * delta) >> 16;
            return value + diff;
        }

        void train(long previous, long value, long err) {
            long gradient = err > 0 ? -delta : delta;
            accumulator -= gradient;
            delta = value - previous;
        }
    }
}
