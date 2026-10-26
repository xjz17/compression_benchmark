package org.example;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Subprocess entry: float columns scaled to int (max decimal precision per column) then
 * {@link Ts2DiffBosImprove#benchFloatColumns}.
 * <p>
 * Stdin: {@code int numCols}; for each column {@code int len} then {@code len} little-endian float32.
 * <p>
 * Stdout: four int64 totals (original bytes, compressed bytes, encode ns, decode ns).
 * <p>
 * Processes one column at a time to limit peak heap when many columns share a large patch.
 */
public final class Ts2DiffBosFloatBatchMain {

    private Ts2DiffBosFloatBatchMain() {}

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
        long totO = 0, totC = 0, te = 0, td = 0;
        for (int c = 0; c < numCols; c++) {
            if (buf.remaining() < 4) {
                System.exit(3);
                return;
            }
            int len = buf.getInt();
            if (len < 0 || buf.remaining() < len * 4) {
                System.exit(3);
                return;
            }
            ArrayList<Float> col = new ArrayList<>(len);
            for (int i = 0; i < len; i++) {
                col.add(buf.getFloat());
            }
            long[] t = new long[4];
            Ts2DiffBosImprove.benchFloatColumns(Collections.singletonList(col), t);
            totO += t[0];
            totC += t[1];
            te += t[2];
            td += t[3];
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
