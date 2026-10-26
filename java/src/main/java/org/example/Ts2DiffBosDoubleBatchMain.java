package org.example;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Subprocess entry for 64-bit float columns.
 *
 * <p>Stdin: {@code int numCols}; for each column {@code int len}, {@code int maxPoint} then {@code len}
 * little-endian float64.
 *
 * <p>Stdout: four int64 totals (original bytes, compressed bytes, encode ns, decode ns).
 */
public final class Ts2DiffBosDoubleBatchMain {

    private Ts2DiffBosDoubleBatchMain() {}

    private static byte[] leInt(int v) {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array();
    }

    private static byte[] leLong(long v) {
        return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array();
    }

    private static void writePayloadRow(
            ByteArrayOutputStream out, long orig, long encodeNs, long decodeNs, byte[] payload)
            throws Exception {
        out.write(leLong(orig));
        out.write(leLong(encodeNs));
        out.write(leLong(decodeNs));
        if (payload == null || payload.length <= 0) {
            out.write(leInt(-1));
            return;
        }
        out.write(leInt(payload.length));
        out.write(payload);
    }

    public static void main(String[] args) throws Exception {
        byte[] payload =
                args.length > 0
                        ? Files.readAllBytes(Path.of(args[0]))
                        : System.in.readAllBytes();
        ByteBuffer buf = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        if (buf.remaining() < 4) {
            System.exit(2);
            return;
        }
        int numCols = buf.getInt();
        if (numCols < 0) {
            System.exit(2);
            return;
        }
        boolean payloadMode = "1".equals(System.getenv("WEB_COMPRESSION_BOS_PAYLOADS"));
        ByteArrayOutputStream payloadOut = payloadMode ? new ByteArrayOutputStream() : null;
        if (payloadMode) {
            payloadOut.write(leInt(numCols));
        }
        long totO = 0, totC = 0, te = 0, td = 0;
        for (int c = 0; c < numCols; c++) {
            if (buf.remaining() < 4) {
                System.exit(3);
                return;
            }
            int len = buf.getInt();
            if (buf.remaining() < 4) {
                System.exit(3);
                return;
            }
            int maxPoint = buf.getInt();
            if (len < 0 || buf.remaining() < len * 8) {
                System.exit(3);
                return;
            }
            ArrayList<Double> col = new ArrayList<>(len);
            for (int i = 0; i < len; i++) {
                col.add(buf.getDouble());
            }
            if (payloadMode) {
                long t0 = System.nanoTime();
                byte[] encoded =
                        Ts2DiffBosImprove.encodeDoubleColumnLevelPayloadOrNull(col, maxPoint);
                long t1 = System.nanoTime();
                long[] decoded =
                        encoded == null
                                ? null
                                : Ts2DiffBosImprove.decodeLongColumnLevelPayloadOrNull(encoded);
                long t2 = System.nanoTime();
                boolean ok = decoded != null && decoded.length == len;
                writePayloadRow(
                        payloadOut,
                        ok ? (long) len * BenchmarkWireFormat.FLOAT_BYTES : 0L,
                        ok ? t1 - t0 : 0L,
                        ok ? t2 - t1 : 0L,
                        ok ? encoded : null);
                continue;
            }
            long[] t = new long[4];
            Ts2DiffBosImprove.benchDoubleColumns(
                    Collections.singletonList(col),
                    Collections.singletonList(maxPoint),
                    t);
            totO += t[0];
            totC += t[1];
            te += t[2];
            td += t[3];
        }

        if (payloadMode) {
            System.out.write(payloadOut.toByteArray());
            System.out.flush();
            return;
        }

        ByteBuffer bb = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(totO);
        bb.putLong(totC);
        bb.putLong(te);
        bb.putLong(td);
        System.out.write(bb.array());
        System.out.flush();
        if (totO <= 0) {
            System.exit(4);
        }
    }
}
