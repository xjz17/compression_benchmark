import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Reference lossless user codec: per-column delta encoding (stdlib only).
 * Entry class must be Main (Main.java).
 *
 * <p>Wire: magic WCUC1, dtype tag, column count, then per column length + delta stream.
 */
public class Main {
    private static final int MAGIC = 0x31435557; // WCUC1 little-endian

    public static void main(String[] args) throws Exception {
        String mode = "compress";
        String input = null;
        String output = null;
        String dataType = "float";
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--mode" -> mode = args[++i];
                case "--input" -> input = args[++i];
                case "--output" -> output = args[++i];
                case "--data-type" -> dataType = args[++i];
                default -> { }
            }
        }
        if (input == null || output == null) {
            System.err.println("missing --input or --output");
            System.exit(2);
        }
        if ("compress".equals(mode)) {
            compress(Paths.get(input), Paths.get(output), dataType);
        } else {
            decompress(Paths.get(input), Paths.get(output), dataType);
        }
    }

    static List<List<Double>> readColumns(Path csv, String dataType) throws IOException {
        List<String> lines = Files.readAllLines(csv);
        if (lines.isEmpty()) {
            return List.of();
        }
        String[] header = lines.get(0).split(",", -1);
        int n = header.length;
        List<List<Double>> cols = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            cols.add(new ArrayList<>());
        }
        for (int r = 1; r < lines.size(); r++) {
            String[] cells = lines.get(r).split(",", -1);
            for (int c = 0; c < Math.min(cells.length, n); c++) {
                String t = cells[c].trim();
                if (t.isEmpty()) {
                    continue;
                }
                double v = Double.parseDouble(t);
                if ("int".equals(dataType) && v != Math.rint(v)) {
                    continue;
                }
                cols.get(c).add(v);
            }
        }
        cols.removeIf(List::isEmpty);
        return cols;
    }

    static byte[] encodeColumns(List<List<Double>> cols, String dataType) {
        boolean isInt = "int".equals(dataType);
        int tag = isInt ? 1 : 2;
        int ncol = cols.size();
        int cap = 12;
        for (List<Double> col : cols) {
            cap += 4 + col.size() * 8;
        }
        ByteBuffer buf = ByteBuffer.allocate(cap).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(MAGIC);
        buf.putInt(tag);
        buf.putInt(ncol);
        for (List<Double> col : cols) {
            buf.putInt(col.size());
            if (col.isEmpty()) {
                continue;
            }
            if (isInt) {
                long prev = col.get(0).longValue();
                buf.putLong(prev);
                for (int i = 1; i < col.size(); i++) {
                    long cur = col.get(i).longValue();
                    buf.putLong(cur - prev);
                    prev = cur;
                }
            } else {
                double prev = col.get(0);
                buf.putDouble(prev);
                for (int i = 1; i < col.size(); i++) {
                    double cur = col.get(i);
                    buf.putDouble(cur - prev);
                    prev = cur;
                }
            }
        }
        byte[] out = new byte[buf.position()];
        buf.flip();
        buf.get(out);
        return out;
    }

    static List<List<Double>> decodeColumns(byte[] blob, String dataType) {
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int magic = buf.getInt();
        int tag = buf.getInt();
        int ncol = buf.getInt();
        if (magic != MAGIC) {
            throw new IllegalArgumentException("bad magic");
        }
        boolean isInt = tag == 1;
        List<List<Double>> cols = new ArrayList<>();
        for (int c = 0; c < ncol; c++) {
            int n = buf.getInt();
            List<Double> col = new ArrayList<>(n);
            if (n == 0) {
                cols.add(col);
                continue;
            }
            if (isInt) {
                long v = buf.getLong();
                col.add((double) v);
                for (int i = 1; i < n; i++) {
                    v += buf.getLong();
                    col.add((double) v);
                }
            } else {
                double v = buf.getDouble();
                col.add(v);
                for (int i = 1; i < n; i++) {
                    v += buf.getDouble();
                    col.add(v);
                }
            }
            cols.add(col);
        }
        return cols;
    }

    static void compress(Path in, Path out, String dataType) throws IOException {
        List<List<Double>> cols = readColumns(in, dataType);
        if (cols.isEmpty()) {
            System.err.println("no columns");
            System.exit(2);
        }
        Files.write(out, encodeColumns(cols, dataType));
    }

    static void decompress(Path in, Path out, String dataType) throws IOException {
        List<List<Double>> cols = decodeColumns(Files.readAllBytes(in), dataType);
        int nrows = cols.stream().mapToInt(List::size).max().orElse(0);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cols.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append("col").append(i);
        }
        sb.append('\n');
        for (int r = 0; r < nrows; r++) {
            for (int i = 0; i < cols.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                if (r < cols.get(i).size()) {
                    double v = cols.get(i).get(r);
                    sb.append("int".equals(dataType) ? String.valueOf((long) v) : String.valueOf(v));
                }
            }
            sb.append('\n');
        }
        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8);
    }
}
