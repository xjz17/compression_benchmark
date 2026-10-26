package org.example;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Little-endian int64 / float64 column payloads (matches Python/C++ benchmark CSV paths). */
public final class BenchmarkWireFormat {

    public static final int INT_BYTES = 8;
    public static final int FLOAT_BYTES = 8;

    private BenchmarkWireFormat() {}

    public static void appendColumnLongLe(ByteArrayOutputStream baos, long value, ByteBuffer le) {
        le.clear();
        le.putLong(value);
        baos.write(le.array(), 0, INT_BYTES);
    }

    public static void appendColumnDoubleLe(ByteArrayOutputStream baos, double value, ByteBuffer le) {
        le.clear();
        le.putDouble(value);
        baos.write(le.array(), 0, FLOAT_BYTES);
    }

    public static long columnOriginalSize(int numValues, boolean isFloat) {
        return (long) numValues * (isFloat ? FLOAT_BYTES : INT_BYTES);
    }
}
