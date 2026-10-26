package org.example.buff;

import java.util.Arrays;

/**
 * Growable LSB-first bit writer matching upstream
 * {@code database/src/methods/bit_packing.rs} {@code BitPack::<Vec<u8>>::write}.
 */
public final class BuffBitPackWriter {

    public static final int MAX_BITS = 32;
    public static final int BYTE_BITS = 8;

    private byte[] buf;
    private int cursor;
    private int bitsInByte;

    public BuffBitPackWriter() {
        this.buf = new byte[64];
        this.cursor = 0;
        this.bitsInByte = 0;
    }

    public void write(int value, int bits) {
        int v = value;
        int mutBits = bits;
        if (mutBits < MAX_BITS) {
            v &= (1 << mutBits) - 1;
        }
        while (mutBits > 0) {
            int bitsLeft = BYTE_BITS - bitsInByte;
            if (mutBits <= bitsLeft) {
                buf[cursor] |= (byte) ((v & 0xFF) << bitsInByte);
                bitsInByte += mutBits;
                if (bitsInByte >= BYTE_BITS) {
                    cursor++;
                    bitsInByte = 0;
                }
                break;
            }
            int take = bitsLeft;
            int bb = v & ((1 << take) - 1);
            buf[cursor] |= (byte) (bb << bitsInByte);
            cursor++;
            bitsInByte = 0;
            v >>>= take;
            mutBits -= take;
            ensureCapacity();
        }
        ensureCapacity();
    }

    private void ensureCapacity() {
        if (cursor >= buf.length - 4) {
            buf = Arrays.copyOf(buf, buf.length * 2);
        }
    }

    public byte[] toByteArray() {
        int n = cursor;
        if (bitsInByte > 0) {
            n++;
        }
        return Arrays.copyOf(buf, n);
    }
}
