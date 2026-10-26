package org.example;

import java.util.ArrayList;
import java.util.List;

/** Native timestamp-union Grouping layout used by the paper's Grouping method. */
public final class GroupingCodec {
    public static final String ROW_NAME = "grouping";
    private static final long TIME_BYTES = 8L;
    private static final long VALUE_BYTES = 8L;

    public static final class ColumnStream {
        final ArrayList<Long> timestamps = new ArrayList<>();
        final ArrayList<Long> values = new ArrayList<>();
    }

    private static final class GroupedLayout {
        final ArrayList<Long> timestamps = new ArrayList<>();
        final ArrayList<Long> bitmap = new ArrayList<>();
        final ArrayList<ArrayList<Long>> values;

        GroupedLayout(int columnCount) {
            values = new ArrayList<>(columnCount);
            for (int i = 0; i < columnCount; i++) {
                values.add(new ArrayList<>());
            }
        }
    }

    private GroupingCodec() {}

    public static ColumnStream intColumnStream(List<String> cells, long rowOffset) {
        ColumnStream out = new ColumnStream();
        for (int i = 0; i < cells.size(); i++) {
            String t = normalizeCell(cells.get(i));
            if (t.isEmpty()) {
                continue;
            }
            try {
                out.timestamps.add(rowOffset + i);
                out.values.add(Long.parseLong(t));
            } catch (NumberFormatException ignored) {
                // skip invalid numeric cells
            }
        }
        return out;
    }

    public static ColumnStream doubleColumnStream(List<String> cells, long rowOffset) {
        ColumnStream out = new ColumnStream();
        for (int i = 0; i < cells.size(); i++) {
            String t = normalizeCell(cells.get(i));
            if (t.isEmpty()) {
                continue;
            }
            try {
                double value = Double.parseDouble(t);
                if (!ExperimentDataPolicy.keepNumericValue(value)) {
                    continue;
                }
                out.timestamps.add(rowOffset + i);
                out.values.add(Double.doubleToRawLongBits(value));
            } catch (NumberFormatException ignored) {
                // skip invalid numeric cells
            }
        }
        return out;
    }

    public static void benchStreams(List<ColumnStream> rawColumns, long[] out) {
        List<ColumnStream> columns = new ArrayList<>();
        for (ColumnStream column : rawColumns) {
            if (column != null && !column.values.isEmpty()) {
                columns.add(column);
            }
        }
        long original = originalSize(columns);
        if (original <= 0) {
            out[0] = out[1] = out[2] = out[3] = 0L;
            return;
        }
        long t0 = System.nanoTime();
        GroupedLayout layout = groupLayout(columns);
        long t1 = System.nanoTime();
        List<ColumnStream> recovered = recoverLayout(layout);
        long t2 = System.nanoTime();
        verify(columns, recovered);
        out[0] = original;
        out[1] = groupedSize(layout);
        out[2] = t1 - t0;
        out[3] = t2 - t1;
    }

    private static GroupedLayout groupLayout(List<ColumnStream> columns) {
        GroupedLayout out = new GroupedLayout(columns.size());
        if (columns.isEmpty()) {
            return out;
        }
        if (columns.size() == 1) {
            out.timestamps.addAll(columns.get(0).timestamps);
            out.values.get(0).addAll(columns.get(0).values);
            return out;
        }
        int[] positions = new int[columns.size()];
        while (true) {
            long next = Long.MAX_VALUE;
            boolean found = false;
            for (int i = 0; i < columns.size(); i++) {
                ColumnStream col = columns.get(i);
                if (positions[i] < col.timestamps.size()) {
                    long ts = col.timestamps.get(positions[i]);
                    if (ts < next) {
                        next = ts;
                    }
                    found = true;
                }
            }
            if (!found) {
                break;
            }
            int row = out.timestamps.size();
            out.timestamps.add(next);
            int needWords = (out.timestamps.size() * columns.size() + 63) / 64;
            while (out.bitmap.size() < needWords) {
                out.bitmap.add(0L);
            }
            for (int i = 0; i < columns.size(); i++) {
                ColumnStream col = columns.get(i);
                int pos = positions[i];
                if (pos < col.timestamps.size() && col.timestamps.get(pos) == next) {
                    int bit = row * columns.size() + i;
                    out.bitmap.set(bit / 64, out.bitmap.get(bit / 64) | (1L << (bit % 64)));
                    out.values.get(i).add(col.values.get(pos));
                    positions[i]++;
                }
            }
        }
        return out;
    }

    private static List<ColumnStream> recoverLayout(GroupedLayout layout) {
        ArrayList<ColumnStream> out = new ArrayList<>(layout.values.size());
        for (int i = 0; i < layout.values.size(); i++) {
            out.add(new ColumnStream());
        }
        if (layout.values.isEmpty()) {
            return out;
        }
        if (layout.values.size() == 1) {
            out.get(0).timestamps.addAll(layout.timestamps);
            out.get(0).values.addAll(layout.values.get(0));
            return out;
        }
        int[] positions = new int[layout.values.size()];
        for (int row = 0; row < layout.timestamps.size(); row++) {
            long ts = layout.timestamps.get(row);
            for (int col = 0; col < layout.values.size(); col++) {
                int bit = row * layout.values.size() + col;
                if (((layout.bitmap.get(bit / 64) >>> (bit % 64)) & 1L) == 0L) {
                    continue;
                }
                int pos = positions[col]++;
                out.get(col).timestamps.add(ts);
                out.get(col).values.add(layout.values.get(col).get(pos));
            }
        }
        return out;
    }

    private static long originalSize(List<ColumnStream> columns) {
        long points = 0L;
        for (ColumnStream column : columns) {
            points += column.values.size();
        }
        return points * (TIME_BYTES + VALUE_BYTES);
    }

    private static long groupedSize(GroupedLayout layout) {
        long values = 0L;
        for (List<Long> column : layout.values) {
            values += column.size();
        }
        long bitmapBytes =
                layout.values.size() <= 1
                        ? 0L
                        : (layout.timestamps.size() * (long) layout.values.size() + 7L) / 8L;
        return layout.timestamps.size() * TIME_BYTES + values * VALUE_BYTES + bitmapBytes;
    }

    private static void verify(List<ColumnStream> want, List<ColumnStream> got) {
        if (want.size() != got.size()) {
            throw new IllegalStateException("grouping column count mismatch");
        }
        for (int i = 0; i < want.size(); i++) {
            if (!want.get(i).timestamps.equals(got.get(i).timestamps)
                    || !want.get(i).values.equals(got.get(i).values)) {
                throw new IllegalStateException("grouping lossless check failed at column " + i);
            }
        }
    }

    private static String normalizeCell(String raw) {
        if (raw == null) {
            return "";
        }
        String t = raw.trim();
        if (t.length() >= 2 && t.startsWith("\"") && t.endsWith("\"")) {
            t = t.substring(1, t.length() - 1).trim();
        }
        return ExperimentDataPolicy.isNaNCell(t) ? "" : t;
    }
}
