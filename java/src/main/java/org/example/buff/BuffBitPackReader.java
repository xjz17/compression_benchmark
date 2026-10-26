package org.example.buff;

/**
 * LSB-first bit reader aligned with upstream {@code BitPack<&[u8]>::read} and
 * {@link BuffBitPackWriter}.
 */
public final class BuffBitPackReader {

    private final byte[] data;
    private int bitPos;

    public BuffBitPackReader(byte[] data) {
        this.data = data != null ? data : new byte[0];
        this.bitPos = 0;
    }

    public int read(int bits) {
        if (bits <= 0 || bits > BuffBitPackWriter.MAX_BITS) {
            throw new IllegalArgumentException("bits");
        }
        int v = 0;
        for (int i = 0; i < bits; i++) {
            int bi = bitPos / BuffBitPackWriter.BYTE_BITS;
            int off = bitPos % BuffBitPackWriter.BYTE_BITS;
            if (bi >= data.length) {
                throw new IllegalArgumentException("BuffBitPackReader past end");
            }
            int b = (data[bi] >> off) & 1;
            v |= b << i;
            bitPos++;
        }
        return v;
    }

    public int bitPosition() {
        return bitPos;
    }

    /**
     * Sign-extend {@code bits} low bits of {@code v} (Rust {@code int_part} written as {@code u32}
     * with {@code ilen} bits).
     */
    public static long signExtendToLong(int v, int bits) {
        if (bits <= 0) {
            return 0;
        }
        if (bits == 32) {
            return (long) (int) (v & 0xFFFFFFFF);
        }
        long vL = v & 0xFFFFFFFFL;
        vL &= (1L << bits) - 1;
        long s = vL << (64 - bits);
        return s >> (64 - bits);
    }
}
