package org.example;

import org.apache.commons.compress.compressors.CompressorException;
import org.apache.commons.compress.compressors.CompressorStreamFactory;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Column encodings in the spirit of Parquet / Arrow (delta+zigzag+varint, dictionary,
 * RLE, byte-stream-split) and analytics DB patterns. Not on-the-wire compatible.
 */
public final class ColumnarEncodings {

    /** Benchmark CSV / report row name (Parquet-style delta + zigzag + varint). */
    public static final String DELTA_BINARY_PACKED_ALGORITHM = "DELTA_BINARY_PACKED";

    private ColumnarEncodings() {
    }

    public static boolean isDeltaBinaryPackedAlgorithm(String name) {
        if (name == null) {
            return false;
        }
        return DELTA_BINARY_PACKED_ALGORITHM.equals(name.trim());
    }

    /** True when incremental benchmark should emit a delta-binary-packed row. */
    public static boolean shouldRunDeltaBinaryPacked(java.util.Set<String> skipOrExisting) {
        return AlgorithmAllowList.shouldRun(skipOrExisting, DELTA_BINARY_PACKED_ALGORITHM);
    }

    private static void writeVarintU32(ByteArrayOutputStream out, int v) {
        long x = v & 0xFFFFFFFFL;
        while (x >= 0x80) {
            out.write((int) ((x & 0x7F) | 0x80));
            x >>>= 7;
        }
        out.write((int) x);
    }

    private static int readVarintU32(ByteBuffer buf) {
        int result = 0;
        int shift = 0;
        while (buf.hasRemaining()) {
            int b = buf.get() & 0xFF;
            result |= (b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                return result;
            }
            shift += 7;
            if (shift > 35) {
                throw new IllegalStateException("bad varint");
            }
        }
        throw new IllegalStateException("truncated varint");
    }

    private static int zigzag32(int d) {
        return (d << 1) ^ (d >> 31);
    }

    private static int unzigzag32(int u) {
        return (u >>> 1) ^ -(u & 1);
    }

    private static void writeIntLE(ByteArrayOutputStream out, int v) {
        out.write(v & 0xFF);
        out.write((v >> 8) & 0xFF);
        out.write((v >> 16) & 0xFF);
        out.write((v >> 24) & 0xFF);
    }

    private static void writeLongLE(ByteArrayOutputStream out, long v) {
        for (int i = 0; i < 8; i++) {
            out.write((int) ((v >> (8 * i)) & 0xFF));
        }
    }

    private static long readVarintU64(ByteBuffer buf) {
        long result = 0;
        int shift = 0;
        while (buf.hasRemaining()) {
            int b = buf.get() & 0xFF;
            result |= (long) (b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                return result;
            }
            shift += 7;
            if (shift > 70) {
                throw new IllegalStateException("bad varint");
            }
        }
        throw new IllegalStateException("truncated varint");
    }

    private static long unzigzag64(long u) {
        return (u >>> 1) ^ -(u & 1L);
    }

    public static List<Long> decodeDeltaZigZagVarintLong(byte[] blob) {
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        List<Long> res = new ArrayList<>();
        if (!buf.hasRemaining()) {
            return res;
        }
        long cur = buf.getLong();
        res.add(cur);
        while (buf.hasRemaining()) {
            long u = readVarintU64(buf);
            long d = unzigzag64(u);
            cur = cur + d;
            res.add(cur);
        }
        return res;
    }

    public static List<Long> decodeRleLong(byte[] blob) {
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        List<Long> out = new ArrayList<>();
        while (buf.hasRemaining()) {
            if (buf.remaining() < 8) {
                throw new IllegalStateException("truncated RLE value");
            }
            long val = buf.getLong();
            long run = readVarintU64(buf);
            for (long i = 0; i < run; i++) {
                out.add(val);
            }
        }
        return out;
    }

    public static byte[] encodeDeltaZigZagVarintLong(List<Long> values) {
        if (values.isEmpty()) {
            return new byte[0];
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        writeLongLE(baos, values.get(0));
        long prev = values.get(0);
        for (int i = 1; i < values.size(); i++) {
            long d = values.get(i) - prev;
            prev = values.get(i);
            writeVarintU64(baos, zigzag64(d));
        }
        return baos.toByteArray();
    }

    private static void writeVarintU64(ByteArrayOutputStream out, long v) {
        long x = v;
        while ((x & ~0x7FL) != 0) {
            out.write((int) ((x & 0x7F) | 0x80));
            x >>>= 7;
        }
        out.write((int) x);
    }

    private static long zigzag64(long d) {
        return (d << 1) ^ (d >> 63);
    }

    public static byte[] encodeDeltaZigZagVarint(List<Integer> values) {
        if (values.isEmpty()) {
            return new byte[0];
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        writeIntLE(baos, values.get(0));
        int prev = values.get(0);
        for (int i = 1; i < values.size(); i++) {
            int d = values.get(i) - prev;
            prev = values.get(i);
            writeVarintU32(baos, zigzag32(d));
        }
        return baos.toByteArray();
    }

    public static List<Integer> decodeDeltaZigZagVarint(byte[] blob) {
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        List<Integer> res = new ArrayList<>();
        if (!buf.hasRemaining()) {
            return res;
        }
        int cur = buf.getInt();
        res.add(cur);
        while (buf.hasRemaining()) {
            int u = readVarintU32(buf);
            int d = unzigzag32(u);
            cur = cur + d;
            res.add(cur);
        }
        return res;
    }

    public static byte[] encodeDictionaryInt(List<Integer> values) {
        List<Integer> order = new ArrayList<>();
        Map<Integer, Integer> ix = new HashMap<>();
        for (int v : values) {
            ix.computeIfAbsent(v, k -> {
                order.add(k);
                return order.size() - 1;
            });
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        writeIntLE(out, order.size());
        for (int v : order) {
            writeIntLE(out, v);
        }
        writeIntLE(out, values.size());
        for (int v : values) {
            writeIntLE(out, ix.get(v));
        }
        return out.toByteArray();
    }

    public static byte[] encodeDictionaryFloat(List<Float> values) {
        List<Integer> order = new ArrayList<>();
        Map<Integer, Integer> ix = new HashMap<>();
        for (float f : values) {
            int b = Float.floatToIntBits(f);
            ix.computeIfAbsent(b, k -> {
                order.add(k);
                return order.size() - 1;
            });
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        writeIntLE(out, order.size());
        for (int b : order) {
            writeIntLE(out, b);
        }
        writeIntLE(out, values.size());
        for (float f : values) {
            writeIntLE(out, ix.get(Float.floatToIntBits(f)));
        }
        return out.toByteArray();
    }

    public static List<Integer> decodeDictionaryInt(byte[] blob) {
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int dictSize = buf.getInt();
        List<Integer> dict = new ArrayList<>(dictSize);
        for (int i = 0; i < dictSize; i++) {
            dict.add(buf.getInt());
        }
        int count = buf.getInt();
        List<Integer> res = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            res.add(dict.get(buf.getInt()));
        }
        return res;
    }

    public static List<Float> decodeDictionaryFloat(byte[] blob) {
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int dictSize = buf.getInt();
        int[] dictBits = new int[dictSize];
        for (int i = 0; i < dictSize; i++) {
            dictBits[i] = buf.getInt();
        }
        int count = buf.getInt();
        List<Float> res = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            res.add(Float.intBitsToFloat(dictBits[buf.getInt()]));
        }
        return res;
    }

    public static List<Long> decodeDictionaryLong(byte[] blob) {
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int dictSize = buf.getInt();
        List<Long> dict = new ArrayList<>(dictSize);
        for (int i = 0; i < dictSize; i++) {
            dict.add(buf.getLong());
        }
        int count = buf.getInt();
        List<Long> res = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            res.add(dict.get((int) buf.getLong()));
        }
        return res;
    }

    public static List<Double> decodeDictionaryDouble(byte[] blob) {
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int dictSize = buf.getInt();
        long[] dictBits = new long[dictSize];
        for (int i = 0; i < dictSize; i++) {
            dictBits[i] = buf.getLong();
        }
        int count = buf.getInt();
        List<Double> res = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            res.add(Double.longBitsToDouble(dictBits[(int) buf.getLong()]));
        }
        return res;
    }

    public static List<Integer> decodeRleInt(byte[] blob) {
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        List<Integer> out = new ArrayList<>();
        while (buf.hasRemaining()) {
            if (buf.remaining() < 4) {
                throw new IllegalStateException("truncated RLE value");
            }
            int val = buf.getInt();
            long run = readVarintU64(buf);
            for (long i = 0; i < run; i++) {
                out.add(val);
            }
        }
        return out;
    }

    public static byte[] encodeRleInt(List<Integer> values) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        int i = 0;
        while (i < values.size()) {
            int v = values.get(i);
            int j = i + 1;
            while (j < values.size() && values.get(j).equals(v)) {
                j++;
            }
            int run = j - i;
            writeIntLE(out, v);
            writeVarintU32(out, run);
            i = j;
        }
        return out.toByteArray();
    }

    public static byte[] byteStreamSplitFloat(byte[] raw) {
        int n = raw.length / 4;
        byte[] bss = new byte[n * 4];
        for (int b = 0; b < 4; b++) {
            for (int i = 0; i < n; i++) {
                bss[b * n + i] = raw[i * 4 + b];
            }
        }
        return bss;
    }

    public static byte[] inverseByteStreamSplitFloat(byte[] bss, int n) {
        byte[] raw = new byte[n * 4];
        for (int b = 0; b < 4; b++) {
            for (int i = 0; i < n; i++) {
                raw[i * 4 + b] = bss[b * n + i];
            }
        }
        return raw;
    }

    public static byte[] compressLz4Framed(byte[] data) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (OutputStream cos = new CompressorStreamFactory()
                .createCompressorOutputStream(CompressorStreamFactory.LZ4_FRAMED, out)) {
            cos.write(data);
        } catch (CompressorException e) {
            throw new IOException(e);
        }
        return out.toByteArray();
    }

    public static byte[] decompressLz4Framed(byte[] data) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (InputStream cis = new CompressorStreamFactory()
                .createCompressorInputStream(CompressorStreamFactory.LZ4_FRAMED,
                        new ByteArrayInputStream(data))) {
            cis.transferTo(out);
        } catch (CompressorException e) {
            throw new IOException(e);
        }
        return out.toByteArray();
    }

    public static byte[] encodeBssLz4Float(List<Float> values) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(values.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : values) {
            bb.putFloat(f);
        }
        byte[] bss = byteStreamSplitFloat(bb.array());
        return compressLz4Framed(bss);
    }

    public static void benchDeltaZigZagVarint(List<List<Integer>> columns, long[] total) {
        benchDeltaZigZagVarint(columns, total, null);
    }

    public static void benchDeltaZigZagVarint(List<List<Integer>> columns, long[] total, double[] lossAcc) {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Integer> col : columns) {
            long t0 = System.nanoTime();
            byte[] comp = encodeDeltaZigZagVarint(col);
            long t1 = System.nanoTime();
            List<Integer> dec = decodeDeltaZigZagVarint(comp);
            long t2 = System.nanoTime();
            if (!dec.equals(col)) {
                continue;
            }
            if (lossAcc != null) {
                List<Long> orig = new ArrayList<>(col.size());
                List<Long> decoded = new ArrayList<>(dec.size());
                for (Integer v : col) {
                    orig.add(v == null ? 0L : v.longValue());
                }
                for (Integer v : dec) {
                    decoded.add(v == null ? 0L : v.longValue());
                }
                BenchmarkLoss.accumulateLongListLoss(orig, decoded, lossAcc);
            }
            o += (long) col.size() * 4;
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    public static void benchDictionaryInt(List<List<Integer>> columns, long[] total) {
        benchDictionaryInt(columns, total, null);
    }

    public static void benchDictionaryInt(List<List<Integer>> columns, long[] total, double[] lossAcc) {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Integer> col : columns) {
            long t0 = System.nanoTime();
            byte[] comp = encodeDictionaryInt(col);
            long t1 = System.nanoTime();
            List<Integer> dec = decodeDictionaryInt(comp);
            long t2 = System.nanoTime();
            if (!dec.equals(col)) {
                continue;
            }
            if (lossAcc != null) {
                List<Long> orig = new ArrayList<>(col.size());
                List<Long> decoded = new ArrayList<>(dec.size());
                for (Integer v : col) {
                    orig.add(v == null ? 0L : v.longValue());
                }
                for (Integer v : dec) {
                    decoded.add(v == null ? 0L : v.longValue());
                }
                BenchmarkLoss.accumulateLongListLoss(orig, decoded, lossAcc);
            }
            o += (long) col.size() * 4;
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    public static void benchDictionaryFloat(List<List<Float>> columns, long[] total) {
        benchDictionaryFloat(columns, total, null);
    }

    public static void benchDictionaryFloat(List<List<Float>> columns, long[] total, double[] lossAcc) {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Float> col : columns) {
            long t0 = System.nanoTime();
            byte[] comp = encodeDictionaryFloat(col);
            long t1 = System.nanoTime();
            List<Float> dec = decodeDictionaryFloat(comp);
            long t2 = System.nanoTime();
            if (!dec.equals(col)) {
                continue;
            }
            if (lossAcc != null) {
                List<Double> orig = new ArrayList<>(col.size());
                List<Double> decoded = new ArrayList<>(dec.size());
                for (Float v : col) {
                    orig.add(v == null ? Double.NaN : v.doubleValue());
                }
                for (Float v : dec) {
                    decoded.add(v == null ? Double.NaN : v.doubleValue());
                }
                BenchmarkLoss.accumulateDoubleListLoss(orig, decoded, lossAcc);
            }
            o += (long) col.size() * 4;
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    public static void benchRleInt(List<List<Integer>> columns, long[] total) {
        benchRleInt(columns, total, null);
    }

    public static void benchRleInt(List<List<Integer>> columns, long[] total, double[] lossAcc) {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Integer> col : columns) {
            long t0 = System.nanoTime();
            byte[] comp = encodeRleInt(col);
            long t1 = System.nanoTime();
            List<Integer> dec = decodeRleInt(comp);
            long t2 = System.nanoTime();
            if (!dec.equals(col)) {
                continue;
            }
            if (lossAcc != null) {
                List<Long> orig = new ArrayList<>(col.size());
                List<Long> decoded = new ArrayList<>(dec.size());
                for (Integer v : col) {
                    orig.add(v == null ? 0L : v.longValue());
                }
                for (Integer v : dec) {
                    decoded.add(v == null ? 0L : v.longValue());
                }
                BenchmarkLoss.accumulateLongListLoss(orig, decoded, lossAcc);
            }
            o += (long) col.size() * 4;
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    public static void benchBssLz4Float(List<List<Float>> columns, long[] total) throws IOException {
        benchBssLz4Float(columns, total, null);
    }

    public static void benchBssLz4Float(List<List<Float>> columns, long[] total, double[] lossAcc)
            throws IOException {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Float> col : columns) {
            if (col.isEmpty()) {
                continue;
            }
            long t0 = System.nanoTime();
            byte[] comp = encodeBssLz4Float(col);
            long t1 = System.nanoTime();
            byte[] bss = decompressLz4Framed(comp);
            byte[] rawDec = inverseByteStreamSplitFloat(bss, col.size());
            long t2 = System.nanoTime();
            ByteBuffer bb = ByteBuffer.wrap(rawDec).order(ByteOrder.LITTLE_ENDIAN);
            List<Double> orig = new ArrayList<>(col.size());
            List<Double> decoded = new ArrayList<>(col.size());
            boolean ok = bb.remaining() >= col.size() * 4;
            for (int i = 0; i < col.size() && ok; i++) {
                Float fv = col.get(i);
                orig.add(fv == null ? Double.NaN : fv.doubleValue());
                decoded.add((double) bb.getFloat());
            }
            if (!ok) {
                continue;
            }
            if (lossAcc != null) {
                BenchmarkLoss.accumulateDoubleListLoss(orig, decoded, lossAcc);
            }
            o += (long) col.size() * 4;
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    public static void benchDeltaZigZagVarint64(List<List<Long>> columns, long[] total) {
        benchDeltaZigZagVarint64(columns, total, false);
    }

    public static void benchDeltaZigZagVarint64(
            List<List<Long>> columns, long[] total, boolean doubleWire) {
        benchDeltaZigZagVarint64(columns, total, doubleWire, null);
    }

    public static void benchDeltaZigZagVarint64(
            List<List<Long>> columns, long[] total, boolean doubleWire, double[] lossAcc) {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Long> col : columns) {
            if (col.isEmpty()) {
                continue;
            }
            o += BenchmarkWireFormat.columnOriginalSize(col.size(), doubleWire);
            long t0 = System.nanoTime();
            byte[] comp = encodeDeltaZigZagVarintLong(col);
            long t1 = System.nanoTime();
            List<Long> dec = decodeDeltaZigZagVarintLong(comp);
            long t2 = System.nanoTime();
            if (!dec.equals(col)) {
                continue;
            }
            if (lossAcc != null) {
                BenchmarkLoss.accumulateLongListLoss(col, dec, lossAcc);
            }
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    /** Benchmark complete DBF1 float64 frames, including mode/scale selection and restoration. */
    public static void benchDbpFloat64(
            List<List<Double>> columns,
            List<Integer> maxPoints,
            long[] total,
            double[] lossAcc) {
        long original = 0, compressed = 0, encodeNanos = 0, decodeNanos = 0;
        for (int columnIndex = 0; columnIndex < columns.size(); columnIndex++) {
            List<Double> column = columns.get(columnIndex);
            if (column == null || column.isEmpty()) continue;
            Integer configuredMaxPoint = maxPoints != null && columnIndex < maxPoints.size()
                    ? maxPoints.get(columnIndex)
                    : null;
            int maxPoint = configuredMaxPoint == null ? -1 : configuredMaxPoint;
            for (int offset = 0; offset < column.size(); offset += DbpFloat64Codec.MAX_BLOCK_VALUES) {
                int end = Math.min(column.size(), offset + DbpFloat64Codec.MAX_BLOCK_VALUES);
                List<Double> block = column.subList(offset, end);
                long t0 = System.nanoTime();
                byte[] frame = DbpFloat64Codec.encodeFrame(block, maxPoint);
                long t1 = System.nanoTime();
                double[] decoded = DbpFloat64Codec.decodeFrame(frame);
                long t2 = System.nanoTime();
                if (!rawDoubleEqual(block, decoded)) {
                    throw new IllegalStateException(
                            "DBF1 bitwise round trip failed at column " + columnIndex + ", offset " + offset);
                }
                BenchmarkLoss.accumulateDoubleLoss(block, decoded, decoded.length, lossAcc);
                original += (long) block.size() * Double.BYTES;
                compressed += frame.length;
                encodeNanos += t1 - t0;
                decodeNanos += t2 - t1;
            }
        }
        total[0] = original;
        total[1] = compressed;
        total[2] = encodeNanos;
        total[3] = decodeNanos;
    }

    /** Benchmark complete DBF2 float32 frames from original raw IEEE words. */
    public static void benchDbpFloat32(
            List<List<Float>> columns,
            List<Integer> maxPoints,
            long[] total,
            double[] lossAcc) {
        long original = 0, compressed = 0, encodeNanos = 0, decodeNanos = 0;
        for (int columnIndex = 0; columnIndex < columns.size(); columnIndex++) {
            List<Float> column = columns.get(columnIndex);
            if (column == null || column.isEmpty()) continue;
            Integer configuredMaxPoint = maxPoints != null && columnIndex < maxPoints.size()
                    ? maxPoints.get(columnIndex)
                    : null;
            int maxPoint = configuredMaxPoint == null ? -1 : configuredMaxPoint;
            for (int offset = 0; offset < column.size(); offset += DbpFloat64Codec.MAX_BLOCK_VALUES) {
                int end = Math.min(column.size(), offset + DbpFloat64Codec.MAX_BLOCK_VALUES);
                List<Float> block = column.subList(offset, end);
                long t0 = System.nanoTime();
                int[] words = new int[block.size()];
                for (int i = 0; i < words.length; i++) {
                    Float value = block.get(i);
                    if (value == null) {
                        throw new IllegalArgumentException(
                                "DBF2 null value at column " + columnIndex + ", offset " + (offset + i));
                    }
                    words[i] = Float.floatToRawIntBits(value);
                }
                byte[] frame = DbpFloat64Codec.encodeFrame32(words, maxPoint);
                long t1 = System.nanoTime();
                int[] decoded = DbpFloat64Codec.decodeFrame32(frame);
                long t2 = System.nanoTime();
                if (!java.util.Arrays.equals(words, decoded)) {
                    throw new IllegalStateException(
                            "DBF2 bitwise round trip failed at column " + columnIndex + ", offset " + offset);
                }
                if (lossAcc != null && lossAcc.length >= 2) lossAcc[1] += words.length;
                original += (long) words.length * Float.BYTES;
                compressed += frame.length;
                encodeNanos += t1 - t0;
                decodeNanos += t2 - t1;
            }
        }
        total[0] = original;
        total[1] = compressed;
        total[2] = encodeNanos;
        total[3] = decodeNanos;
    }

    private static boolean rawDoubleEqual(List<Double> original, double[] decoded) {
        if (decoded.length != original.size()) return false;
        for (int i = 0; i < decoded.length; i++) {
            Double value = original.get(i);
            if (value == null
                    || Double.doubleToRawLongBits(value) != Double.doubleToRawLongBits(decoded[i])) {
                return false;
            }
        }
        return true;
    }

    public static void benchDictionaryInt64(List<List<Long>> columns, long[] total) {
        benchDictionaryInt64(columns, total, null);
    }

    public static void benchDictionaryInt64(List<List<Long>> columns, long[] total, double[] lossAcc) {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Long> col : columns) {
            long t0 = System.nanoTime();
            byte[] comp = encodeDictionaryLong(col);
            long t1 = System.nanoTime();
            List<Long> dec = decodeDictionaryLong(comp);
            long t2 = System.nanoTime();
            if (!dec.equals(col)) {
                continue;
            }
            if (lossAcc != null) {
                BenchmarkLoss.accumulateLongListLoss(col, dec, lossAcc);
            }
            o += (long) col.size() * BenchmarkWireFormat.INT_BYTES;
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    public static void benchRleInt64(List<List<Long>> columns, long[] total) {
        benchRleInt64(columns, total, false);
    }

    public static void benchRleInt64(List<List<Long>> columns, long[] total, boolean doubleWire) {
        benchRleInt64(columns, total, doubleWire, null);
    }

    public static void benchRleInt64(
            List<List<Long>> columns, long[] total, boolean doubleWire, double[] lossAcc) {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Long> col : columns) {
            if (col.isEmpty()) {
                continue;
            }
            o += BenchmarkWireFormat.columnOriginalSize(col.size(), doubleWire);
            long t0 = System.nanoTime();
            byte[] comp = encodeRleLong(col);
            long t1 = System.nanoTime();
            List<Long> dec = decodeRleLong(comp);
            long t2 = System.nanoTime();
            if (!dec.equals(col)) {
                continue;
            }
            if (lossAcc != null) {
                BenchmarkLoss.accumulateLongListLoss(col, dec, lossAcc);
            }
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    public static void benchDictionaryDouble64(List<List<Double>> columns, long[] total) {
        benchDictionaryDouble64(columns, total, null);
    }

    public static void benchDictionaryDouble64(List<List<Double>> columns, long[] total, double[] lossAcc) {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Double> col : columns) {
            long t0 = System.nanoTime();
            byte[] comp = encodeDictionaryDouble(col);
            long t1 = System.nanoTime();
            List<Double> dec = decodeDictionaryDouble(comp);
            long t2 = System.nanoTime();
            if (!dec.equals(col)) {
                continue;
            }
            if (lossAcc != null) {
                BenchmarkLoss.accumulateDoubleListLoss(col, dec, lossAcc);
            }
            o += (long) col.size() * BenchmarkWireFormat.FLOAT_BYTES;
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    public static void benchBssLz4Double64(List<List<Double>> columns, long[] total) throws IOException {
        benchBssLz4Double64(columns, total, null);
    }

    public static void benchBssLz4Double64(
            List<List<Double>> columns, long[] total, double[] lossAcc) throws IOException {
        long o = 0, c = 0, te = 0, td = 0;
        for (List<Double> col : columns) {
            if (col.isEmpty()) {
                continue;
            }
            long t0 = System.nanoTime();
            byte[] comp = encodeBssLz4Double(col);
            long t1 = System.nanoTime();
            byte[] bss = decompressLz4Framed(comp);
            byte[] rawDec = inverseByteStreamSplitDouble(bss, col.size());
            long t2 = System.nanoTime();
            ByteBuffer bb = ByteBuffer.wrap(rawDec).order(ByteOrder.LITTLE_ENDIAN);
            List<Double> decoded = new ArrayList<>(col.size());
            boolean ok = bb.remaining() >= col.size() * 8;
            for (int i = 0; i < col.size() && ok; i++) {
                decoded.add(bb.getDouble());
            }
            if (!ok || !decoded.equals(col)) {
                continue;
            }
            if (lossAcc != null) {
                BenchmarkLoss.accumulateDoubleListLoss(col, decoded, lossAcc);
            }
            o += (long) col.size() * BenchmarkWireFormat.FLOAT_BYTES;
            c += comp.length;
            te += t1 - t0;
            td += t2 - t1;
        }
        total[0] = o;
        total[1] = c;
        total[2] = te;
        total[3] = td;
    }

    private static byte[] encodeDictionaryLong(List<Long> values) {
        Map<Long, Integer> idxMap = new HashMap<>();
        List<Long> order = new ArrayList<>();
        for (Long v : values) {
            if (!idxMap.containsKey(v)) {
                idxMap.put(v, order.size());
                order.add(v);
            }
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        writeIntLE(baos, order.size());
        for (Long v : order) {
            writeLongLE(baos, v);
        }
        writeIntLE(baos, values.size());
        for (Long v : values) {
            writeLongLE(baos, idxMap.get(v));
        }
        return baos.toByteArray();
    }

    private static byte[] encodeRleLong(List<Long> values) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        int i = 0;
        while (i < values.size()) {
            int j = i + 1;
            while (j < values.size() && values.get(j).equals(values.get(i))) {
                j++;
            }
            writeLongLE(baos, values.get(i));
            writeVarintU64(baos, j - i);
            i = j;
        }
        return baos.toByteArray();
    }

    private static byte[] encodeDictionaryDouble(List<Double> values) {
        Map<Long, Integer> idxMap = new HashMap<>();
        List<Long> order = new ArrayList<>();
        for (Double v : values) {
            long bits = Double.doubleToRawLongBits(v);
            if (!idxMap.containsKey(bits)) {
                idxMap.put(bits, order.size());
                order.add(bits);
            }
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        writeIntLE(baos, order.size());
        for (Long b : order) {
            writeLongLE(baos, b);
        }
        writeIntLE(baos, values.size());
        for (Double v : values) {
            writeLongLE(baos, idxMap.get(Double.doubleToRawLongBits(v)));
        }
        return baos.toByteArray();
    }

    public static byte[] encodeBssLz4Double(List<Double> values) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(values.size() * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (double f : values) {
            bb.putDouble(f);
        }
        byte[] bss = byteStreamSplitDouble(bb.array());
        return compressLz4Framed(bss);
    }

    private static byte[] byteStreamSplitDouble(byte[] raw) {
        int n = raw.length / 8;
        byte[][] lanes = new byte[8][n];
        for (int i = 0; i < n; i++) {
            for (int b = 0; b < 8; b++) {
                lanes[b][i] = raw[i * 8 + b];
            }
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream(n * 8);
        for (int b = 0; b < 8; b++) {
            out.write(lanes[b], 0, n);
        }
        return out.toByteArray();
    }

    private static byte[] inverseByteStreamSplitDouble(byte[] bss, int n) {
        byte[] raw = new byte[n * 8];
        for (int b = 0; b < 8; b++) {
            for (int i = 0; i < n; i++) {
                raw[i * 8 + b] = bss[b * n + i];
            }
        }
        return raw;
    }
}
