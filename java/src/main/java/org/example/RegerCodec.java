package org.example;

import java.io.ByteArrayOutputStream;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** REGER-style block reorder codec for benchmark rows. */
public final class RegerCodec {
    private static final byte[] MAGIC = new byte[] {'R', 'E', 'G', 'E', 'R', '3'};
    private static final int RAW_SERIES = 0;
    private static final int CONST_SERIES = 1;
    private static final int FOR_SERIES = 2;
    private static final int DELTA_SERIES = 3;
    private static final int ORDER_PERMUTED_FLAG = 1;
    private static final int TIME_STREAM_FLAG = 2;
    private static final BigInteger U64_MAX = BigInteger.ONE.shiftLeft(64).subtract(BigInteger.ONE);

    private RegerCodec() {}

    private static boolean fastProfile() {
        String raw = System.getenv("WEB_COMPRESSION_REGER_PROFILE");
        return raw != null && raw.equalsIgnoreCase("fast");
    }

    public static void benchInt64ColumnsReger(
            List<List<Long>> columns, List<Long> times, long[] result, double[] lossAcc) {
        int rowCount = commonRowCount(columns);
        if (rowCount <= 0 || columns == null || columns.isEmpty()) {
            Arrays.fill(result, 0L);
            return;
        }
        long[][] prepared = new long[columns.size()][rowCount];
        for (int c = 0; c < columns.size(); c++) {
            List<Long> col = columns.get(c);
            for (int i = 0; i < rowCount; i++) {
                Long v = col.get(i);
                prepared[c][i] = v == null ? 0L : v.longValue();
            }
        }
        long[] preparedTimes = normalizeTimes(times, rowCount);
        long t0 = System.nanoTime();
        byte[] encoded = encodeInt64Columns(prepared, preparedTimes);
        long t1 = System.nanoTime();
        long[][] decoded = decodeInt64Columns(encoded);
        long t2 = System.nanoTime();
        if (lossAcc != null && lossAcc.length >= 2) {
            for (int c = 0; c < prepared.length; c++) {
                for (int i = 0; i < rowCount; i++) {
                    lossAcc[0] += Math.abs((double) prepared[c][i] - (double) decoded[c][i]);
                    lossAcc[1] += Math.abs((double) prepared[c][i]);
                }
            }
        }
        result[0] = (long) rowCount * prepared.length * Long.BYTES;
        result[1] = encoded.length;
        result[2] = t1 - t0;
        result[3] = t2 - t1;
    }

    public static byte[] encodeInt64Columns(long[][] columns, long[] times) {
        int rowCount = commonRowCount(columns);
        int colCount = columns == null ? 0 : columns.length;
        int blockSize = defaultBlockSize();
        int segmentSize = defaultSegmentSize();
        int blockCount = rowCount == 0 ? 0 : (rowCount + blockSize - 1) / blockSize;
        long[] rowTimes = times == null ? normalizeTimes(null, rowCount) : Arrays.copyOf(times, rowCount);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes(MAGIC);
        putU32(out, rowCount);
        putU16(out, colCount);
        putU16(out, blockSize);
        putU16(out, segmentSize);
        putU32(out, blockCount);
        for (int start = 0; start < rowCount; start += blockSize) {
            int end = Math.min(rowCount, start + blockSize);
            int n = end - start;
            long[][] blockCols = new long[colCount][n];
            for (int c = 0; c < colCount; c++) {
                System.arraycopy(columns[c], start, blockCols[c], 0, n);
            }
            long[] blockTimes = Arrays.copyOfRange(rowTimes, start, end);
            BlockPayload block = bestBlockPayload(blockCols, blockTimes, segmentSize);
            putU16(out, n);
            putU8(out, block.flags);
            putU32(out, block.payload.length);
            out.writeBytes(block.payload);
        }
        return out.toByteArray();
    }

    public static long[][] decodeInt64Columns(byte[] payload) {
        if (payload == null || payload.length < 20) {
            throw new IllegalArgumentException("REGER payload too short");
        }
        Cursor cur = new Cursor(payload);
        for (byte expected : MAGIC) {
            if (cur.readU8() != (expected & 0xff)) {
                throw new IllegalArgumentException("invalid REGER payload");
            }
        }
        int rowCount = cur.readU32();
        int colCount = cur.readU16();
        cur.readU16();
        int segmentSize = cur.readU16();
        int blockCount = cur.readU32();
        long[][] out = new long[colCount][rowCount];
        int blockStart = 0;
        for (int b = 0; b < blockCount; b++) {
            int n = cur.readU16();
            int flags = cur.readU8();
            int bodyLen = cur.readU32();
            Cursor bodyCur = cur.slice(bodyLen);
            int[] order = new int[n];
            if ((flags & ORDER_PERMUTED_FLAG) != 0) {
                int width = ceilLog2(n);
                int orderBodyLength = width > 0 ? (n * width + 7) / 8 : 0;
                long[] packedOrder = unpackValues(bodyCur.readBytes(orderBodyLength), n, width);
                boolean[] seen = new boolean[n];
                for (int i = 0; i < n; i++) {
                    int original = Math.toIntExact(packedOrder[i]);
                    if (original < 0 || original >= n || seen[original]) {
                        throw new IllegalArgumentException("invalid REGER row permutation");
                    }
                    seen[original] = true;
                    order[i] = original;
                }
            } else {
                for (int i = 0; i < n; i++) {
                    order[i] = i;
                }
            }
            long[] orderedTimes = null;
            if ((flags & TIME_STREAM_FLAG) != 0) {
                orderedTimes = readSeries(bodyCur, n, segmentSize);
            }
            long[][] decodedCols = new long[colCount][n];
            for (int c = 0; c < colCount; c++) {
                decodedCols[c] = readSeries(bodyCur, n, segmentSize);
            }
            for (int ordered = 0; ordered < n; ordered++) {
                int row = blockStart + order[ordered];
                for (int c = 0; c < colCount; c++) {
                    out[c][row] = decodedCols[c][ordered];
                }
            }
            blockStart += n;
        }
        return out;
    }

    private static BlockPayload bestBlockPayload(long[][] blockColumns, long[] blockTimes, int segmentSize) {
        List<int[]> orders = rowOrderCandidates(blockColumns, blockTimes);
        BlockPayload best = null;
        for (int[] order : orders) {
            try {
                BlockPayload candidate = encodeBlockCandidate(blockColumns, blockTimes, order, segmentSize);
                if (isIdentity(order)) {
                    BlockPayload withoutTime = encodeBlockCandidate(blockColumns, null, order, segmentSize);
                    if (withoutTime.payload.length < candidate.payload.length) {
                        candidate = withoutTime;
                    }
                }
                if (best == null || candidate.payload.length < best.payload.length) {
                    best = candidate;
                }
            } catch (RuntimeException ignored) {
                // Try the next candidate order.
            }
        }
        if (best == null) {
            int n = blockTimes.length;
            int[] identity = new int[n];
            for (int i = 0; i < n; i++) {
                identity[i] = i;
            }
            best = encodeBlockCandidate(blockColumns, blockTimes, identity, segmentSize);
        }
        return best;
    }

    private static BlockPayload encodeBlockCandidate(
            long[][] blockColumns, long[] blockTimesOrNull, int[] order, int segmentSize) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        int flags = 0;
        if (!isIdentity(order)) {
            flags |= ORDER_PERMUTED_FLAG;
            long[] orderValues = new long[order.length];
            for (int i = 0; i < order.length; i++) {
                orderValues[i] = order[i];
            }
            out.writeBytes(packValues(orderValues, ceilLog2(order.length)));
        }
        if (blockTimesOrNull != null) {
            flags |= TIME_STREAM_FLAG;
            long[] orderedTimes = orderedValues(blockTimesOrNull, order);
            writeSeries(out, bestSeries(orderedTimes, segmentSize));
        }
        for (long[] col : blockColumns) {
            writeSeries(out, bestSeries(orderedValues(col, order), segmentSize));
        }
        return new BlockPayload(flags, out.toByteArray());
    }

    private static SeriesPayload bestSeries(long[] values, int segmentSize) {
        SeriesPayload best = null;
        int bestSize = values.length * Long.BYTES;
        SeriesPayload constant = constSeries(values);
        if (constant != null && constant.payload.length < bestSize) {
            best = constant;
            bestSize = constant.payload.length;
        }
        try {
            SeriesPayload frameOfReference = forSeries(values, segmentSize);
            if (frameOfReference.payload.length < bestSize) {
                best = frameOfReference;
                bestSize = frameOfReference.payload.length;
            }
        } catch (RuntimeException ignored) {
        }
        try {
            SeriesPayload delta = deltaSeries(values, segmentSize);
            if (delta != null && delta.payload.length < bestSize) {
                best = delta;
                bestSize = delta.payload.length;
            }
        } catch (RuntimeException ignored) {
        }
        return best == null ? rawSeries(values) : best;
    }

    private static SeriesPayload rawSeries(long[] values) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        for (long v : values) {
            putI64(out, v);
        }
        return new SeriesPayload(RAW_SERIES, out.toByteArray());
    }

    private static SeriesPayload constSeries(long[] values) {
        if (values.length == 0) {
            return new SeriesPayload(CONST_SERIES, new byte[0]);
        }
        long first = values[0];
        for (long v : values) {
            if (v != first) {
                return null;
            }
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        putI64(out, first);
        return new SeriesPayload(CONST_SERIES, out.toByteArray());
    }

    private static SeriesPayload forSeries(long[] values, int segmentSize) {
        long min = Long.MAX_VALUE;
        for (long v : values) {
            if (v < min) {
                min = v;
            }
        }
        long[] diffs = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            diffs[i] = unsignedDiff(values[i], min);
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        putI64(out, values.length == 0 ? 0L : min);
        out.writeBytes(encodeSegmentedDiffs(diffs, segmentSize));
        return new SeriesPayload(FOR_SERIES, out.toByteArray());
    }

    private static SeriesPayload deltaSeries(long[] values, int segmentSize) {
        if (values.length <= 1) {
            return null;
        }
        long[] deltas = new long[values.length - 1];
        for (int i = 1; i < values.length; i++) {
            deltas[i - 1] = Math.subtractExact(values[i], values[i - 1]);
        }
        long min = Long.MAX_VALUE;
        for (long d : deltas) {
            if (d < min) {
                min = d;
            }
        }
        long[] diffs = new long[deltas.length];
        for (int i = 0; i < deltas.length; i++) {
            diffs[i] = unsignedDiff(deltas[i], min);
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        putI64(out, values[0]);
        putI64(out, min);
        out.writeBytes(encodeSegmentedDiffs(diffs, segmentSize));
        return new SeriesPayload(DELTA_SERIES, out.toByteArray());
    }

    private static void writeSeries(ByteArrayOutputStream out, SeriesPayload series) {
        putU8(out, series.mode);
        putU32(out, series.payload.length);
        out.writeBytes(series.payload);
    }

    private static long[] readSeries(Cursor cur, int count, int segmentSize) {
        int mode = cur.readU8();
        int len = cur.readU32();
        Cursor body = cur.slice(len);
        if (mode == RAW_SERIES) {
            long[] out = new long[count];
            for (int i = 0; i < count; i++) {
                out[i] = body.readI64();
            }
            return out;
        }
        if (mode == CONST_SERIES) {
            long value = count == 0 ? 0L : body.readI64();
            long[] out = new long[count];
            Arrays.fill(out, value);
            return out;
        }
        if (mode == FOR_SERIES) {
            long min = body.readI64();
            long[] diffs = decodeSegmentedDiffs(body, count, segmentSize);
            long[] out = new long[count];
            for (int i = 0; i < count; i++) {
                out[i] = min + diffs[i];
            }
            return out;
        }
        if (mode == DELTA_SERIES) {
            long first = body.readI64();
            long minDelta = body.readI64();
            long[] diffs = decodeSegmentedDiffs(body, Math.max(0, count - 1), segmentSize);
            long[] out = new long[count];
            if (count > 0) {
                out[0] = first;
            }
            for (int i = 1; i < count; i++) {
                out[i] = out[i - 1] + minDelta + diffs[i - 1];
            }
            return out;
        }
        throw new IllegalArgumentException("invalid REGER series mode");
    }

    private static byte[] encodeSegmentedDiffs(long[] diffs, int segmentSize) {
        List<Integer> widths = new ArrayList<>();
        for (int start = 0; start < diffs.length; start += segmentSize) {
            int end = Math.min(diffs.length, start + segmentSize);
            int width = 0;
            for (int i = start; i < end; i++) {
                width = Math.max(width, unsignedBitWidth(diffs[i]));
            }
            widths.add(width);
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        putRleWidths(out, widths);
        int segmentIndex = 0;
        for (int start = 0; start < diffs.length; start += segmentSize, segmentIndex++) {
            int end = Math.min(diffs.length, start + segmentSize);
            out.writeBytes(packValues(diffs, start, end, widths.get(segmentIndex)));
        }
        return out.toByteArray();
    }

    private static long[] decodeSegmentedDiffs(Cursor cur, int count, int segmentSize) {
        List<Integer> widths = readRleWidths(cur);
        long[] out = new long[count];
        int written = 0;
        for (int width : widths) {
            if (written >= count) {
                break;
            }
            int n = Math.min(segmentSize, count - written);
            int bodyLen = width > 0 ? (n * width + 7) / 8 : 0;
            byte[] body = cur.readBytes(bodyLen);
            long[] vals = unpackValues(body, n, width);
            System.arraycopy(vals, 0, out, written, n);
            written += n;
        }
        if (written != count) {
            throw new IllegalArgumentException("REGER segmented length mismatch");
        }
        return out;
    }

    private static void putRleWidths(ByteArrayOutputStream out, List<Integer> widths) {
        List<int[]> runs = new ArrayList<>();
        for (int width : widths) {
            if (runs.isEmpty() || runs.get(runs.size() - 1)[1] != width || runs.get(runs.size() - 1)[0] == 255) {
                runs.add(new int[] {1, width});
            } else {
                runs.get(runs.size() - 1)[0]++;
            }
        }
        putU16(out, runs.size());
        for (int[] run : runs) {
            putU8(out, run[0]);
            putU8(out, run[1]);
        }
    }

    private static List<Integer> readRleWidths(Cursor cur) {
        int runCount = cur.readU16();
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < runCount; i++) {
            int count = cur.readU8();
            int width = cur.readU8();
            for (int j = 0; j < count; j++) {
                out.add(width);
            }
        }
        return out;
    }

    private static List<int[]> rowOrderCandidates(long[][] blockColumns, long[] blockTimes) {
        int n = blockTimes.length;
        List<int[]> out = new ArrayList<>();
        int[] base = new int[n];
        for (int i = 0; i < n; i++) {
            base[i] = i;
        }
        appendUnique(out, base);
        int[] timeOrder = base.clone();
        sortOrder(timeOrder, (a, b) -> {
            int cmp = Long.compare(blockTimes[a], blockTimes[b]);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        appendUnique(out, timeOrder);
        if (fastProfile()) {
            return out;
        }
        for (long[] col : blockColumns) {
            int[] valueOrder = base.clone();
            sortOrder(valueOrder, (a, b) -> {
                int cmp = Long.compare(col[a], col[b]);
                if (cmp != 0) {
                    return cmp;
                }
                cmp = Long.compare(blockTimes[a], blockTimes[b]);
                return cmp != 0 ? cmp : Integer.compare(a, b);
            });
            appendUnique(out, valueOrder);
        }
        if (blockColumns.length > 1) {
            int[] lexOrder = base.clone();
            sortOrder(lexOrder, (a, b) -> {
                for (long[] col : blockColumns) {
                    int cmp = Long.compare(col[a], col[b]);
                    if (cmp != 0) {
                        return cmp;
                    }
                }
                int cmp = Long.compare(blockTimes[a], blockTimes[b]);
                return cmp != 0 ? cmp : Integer.compare(a, b);
            });
            appendUnique(out, lexOrder);
        }
        return out;
    }

    private static void sortOrder(int[] order, Comparator<Integer> comparator) {
        Integer[] boxed = new Integer[order.length];
        for (int i = 0; i < order.length; i++) {
            boxed[i] = order[i];
        }
        Arrays.sort(boxed, comparator);
        for (int i = 0; i < order.length; i++) {
            order[i] = boxed[i];
        }
    }

    private static void appendUnique(List<int[]> out, int[] candidate) {
        for (int[] existing : out) {
            if (Arrays.equals(existing, candidate)) {
                return;
            }
        }
        out.add(candidate);
    }

    private static long[] orderedValues(long[] values, int[] order) {
        long[] out = new long[order.length];
        for (int i = 0; i < order.length; i++) {
            out[i] = values[order[i]];
        }
        return out;
    }

    private static boolean isIdentity(int[] order) {
        for (int i = 0; i < order.length; i++) {
            if (order[i] != i) {
                return false;
            }
        }
        return true;
    }

    private static int ceilLog2(int n) {
        if (n <= 1) {
            return 0;
        }
        return Integer.SIZE - Integer.numberOfLeadingZeros(n - 1);
    }

    private static byte[] packValues(long[] values, int width) {
        if (values.length == 0 || width <= 0) {
            return new byte[0];
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        if (width >= 64) {
            for (long v : values) {
                putI64(out, v);
            }
            return out.toByteArray();
        }
        BitWriter writer = new BitWriter();
        for (long v : values) {
            writer.write(v, width);
        }
        return writer.finish();
    }

    private static byte[] packValues(long[] values, int start, int end, int width) {
        if (start >= end || width <= 0) {
            return new byte[0];
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        if (width >= 64) {
            for (int i = start; i < end; i++) {
                putI64(out, values[i]);
            }
            return out.toByteArray();
        }
        BitWriter writer = new BitWriter();
        for (int i = start; i < end; i++) {
            writer.write(values[i], width);
        }
        return writer.finish();
    }

    private static long[] unpackValues(byte[] data, int count, int width) {
        long[] out = new long[count];
        if (width <= 0) {
            return out;
        }
        if (width >= 64) {
            Cursor cur = new Cursor(data);
            for (int i = 0; i < count; i++) {
                out[i] = cur.readI64();
            }
            return out;
        }
        BitReader reader = new BitReader(data);
        for (int i = 0; i < count; i++) {
            out[i] = reader.read(width);
        }
        return out;
    }

    private static int commonRowCount(List<List<Long>> columns) {
        if (columns == null || columns.isEmpty()) {
            return 0;
        }
        int n = Integer.MAX_VALUE;
        for (List<Long> col : columns) {
            n = Math.min(n, col == null ? 0 : col.size());
        }
        return n == Integer.MAX_VALUE ? 0 : n;
    }

    private static int commonRowCount(long[][] columns) {
        if (columns == null || columns.length == 0) {
            return 0;
        }
        int n = Integer.MAX_VALUE;
        for (long[] col : columns) {
            n = Math.min(n, col == null ? 0 : col.length);
        }
        return n == Integer.MAX_VALUE ? 0 : n;
    }

    private static long[] normalizeTimes(List<Long> times, int rowCount) {
        long[] out = new long[rowCount];
        if (times == null || times.size() < rowCount) {
            for (int i = 0; i < rowCount; i++) {
                out[i] = i;
            }
            return out;
        }
        for (int i = 0; i < rowCount; i++) {
            Long v = times.get(i);
            out[i] = v == null ? i : v.longValue();
        }
        return out;
    }

    private static long unsignedDiff(long value, long minimum) {
        long diff = value - minimum;
        if (((value ^ minimum) & (value ^ diff)) >= 0) {
            return diff;
        }
        BigInteger wideDiff = BigInteger.valueOf(value).subtract(BigInteger.valueOf(minimum));
        if (wideDiff.signum() < 0 || wideDiff.compareTo(U64_MAX) > 0) {
            throw new ArithmeticException("REGER residual range exceeds uint64");
        }
        return wideDiff.longValue();
    }

    private static int unsignedBitWidth(long value) {
        return value == 0L ? 0 : 64 - Long.numberOfLeadingZeros(value);
    }

    private static int defaultBlockSize() {
        String raw = System.getenv("WEB_COMPRESSION_REGER_BLOCK_SIZE");
        if (raw == null || raw.isBlank()) {
            return 513;
        }
        try {
            int n = Integer.parseInt(raw.trim());
            return n > 0 ? n : 513;
        } catch (NumberFormatException ignored) {
            return 513;
        }
    }

    private static int defaultSegmentSize() {
        String raw = System.getenv("WEB_COMPRESSION_REGER_SEGMENT_SIZE");
        if (raw == null || raw.isBlank()) {
            return 16;
        }
        try {
            int n = Integer.parseInt(raw.trim());
            return n > 0 ? n : 16;
        } catch (NumberFormatException ignored) {
            return 16;
        }
    }

    private static void putU8(ByteArrayOutputStream out, int value) {
        out.write(value & 0xff);
    }

    private static void putU16(ByteArrayOutputStream out, int value) {
        out.write(value & 0xff);
        out.write((value >>> 8) & 0xff);
    }

    private static void putU32(ByteArrayOutputStream out, int value) {
        for (int i = 0; i < 4; i++) {
            out.write((value >>> (8 * i)) & 0xff);
        }
    }

    private static void putI64(ByteArrayOutputStream out, long value) {
        for (int i = 0; i < 8; i++) {
            out.write((int) ((value >>> (8 * i)) & 0xff));
        }
    }

    private static final class SeriesPayload {
        final int mode;
        final byte[] payload;

        SeriesPayload(int mode, byte[] payload) {
            this.mode = mode;
            this.payload = payload;
        }
    }

    private static final class BlockPayload {
        final int flags;
        final byte[] payload;

        BlockPayload(int flags, byte[] payload) {
            this.flags = flags;
            this.payload = payload;
        }
    }

    private static final class Cursor {
        private final byte[] data;
        private final int limit;
        private int pos;

        Cursor(byte[] data) {
            this.data = data;
            this.limit = data.length;
        }

        Cursor(byte[] data, int pos, int limit) {
            this.data = data;
            this.pos = pos;
            this.limit = limit;
        }

        int readU8() {
            if (pos >= limit) {
                throw new IllegalArgumentException("REGER truncated u8");
            }
            return data[pos++] & 0xff;
        }

        int readU16() {
            int lo = readU8();
            int hi = readU8();
            return lo | (hi << 8);
        }

        int readU32() {
            int out = 0;
            for (int i = 0; i < 4; i++) {
                out |= readU8() << (8 * i);
            }
            return out;
        }

        long readI64() {
            long out = 0L;
            for (int i = 0; i < 8; i++) {
                out |= (long) readU8() << (8 * i);
            }
            return out;
        }

        byte[] readBytes(int len) {
            if (len < 0 || pos + len > limit) {
                throw new IllegalArgumentException("REGER truncated body");
            }
            byte[] out = Arrays.copyOfRange(data, pos, pos + len);
            pos += len;
            return out;
        }

        Cursor slice(int len) {
            if (len < 0 || pos + len > limit) {
                throw new IllegalArgumentException("REGER truncated body");
            }
            Cursor out = new Cursor(data, pos, pos + len);
            pos += len;
            return out;
        }
    }

    private static final class BitWriter {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private int cur;
        private int used;

        void write(long value, int bits) {
            while (bits > 0) {
                int take = Math.min(8 - used, bits);
                int shift = bits - take;
                long mask = (1L << take) - 1L;
                cur = (cur << take) | (int) ((value >>> shift) & mask);
                used += take;
                bits -= take;
                if (used == 8) {
                    out.write(cur & 0xff);
                    cur = 0;
                    used = 0;
                }
            }
        }

        byte[] finish() {
            if (used > 0) {
                out.write((cur << (8 - used)) & 0xff);
            }
            return out.toByteArray();
        }
    }

    private static final class BitReader {
        private final byte[] data;
        private int bitPos;

        BitReader(byte[] data) {
            this.data = data;
        }

        long read(int bits) {
            long out = 0L;
            for (int i = 0; i < bits; i++) {
                int byteIdx = bitPos >>> 3;
                if (byteIdx >= data.length) {
                    throw new IllegalArgumentException("REGER bitread past end");
                }
                int shift = 7 - (bitPos & 7);
                out = (out << 1) | ((data[byteIdx] >>> shift) & 1L);
                bitPos++;
            }
            return out;
        }
    }
}
