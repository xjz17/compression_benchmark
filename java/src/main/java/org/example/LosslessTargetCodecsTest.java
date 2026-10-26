package org.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;
import org.example.buff.BuffSplitDoubleColumnCodec;

/** Strict bit-exact regression for the seven paper algorithms. */
public final class LosslessTargetCodecsTest {
    private LosslessTargetCodecsTest() {}

    public static void main(String[] args) throws Exception {
        ArrayList<Double> values = new ArrayList<>(List.of(
                0.0,
                -0.0,
                1.0,
                -1.0,
                0.1,
                Double.MIN_VALUE,
                Double.MIN_NORMAL,
                Double.MAX_VALUE,
                Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY,
                Double.longBitsToDouble(0x7ff8000000000042L),
                Double.longBitsToDouble(0xfff8000000001234L),
                Double.longBitsToDouble(0x3fd5555555555555L),
                Double.longBitsToDouble(0x43b0000000000001L)));
        ArrayList<ArrayList<Double>> columns = new ArrayList<>();
        columns.add(values);
        ArrayList<Double> finiteValues = new ArrayList<>(List.of(
                0.0,
                -0.0,
                1.0,
                -1.0,
                0.125,
                0.25,
                -2.5,
                12.5,
                -98765.25));
        ArrayList<ArrayList<Double>> finiteColumns = new ArrayList<>();
        finiteColumns.add(finiteValues);
        List<Integer> maxPoints = List.of(4);
        List<Integer> finiteMaxPoints = List.of(4);

        for (TSEncoding encoding : List.of(TSEncoding.SPRINTZ, TSEncoding.RLBE, TSEncoding.RLE)) {
            long[] result = new long[4];
            double[] loss = new double[2];
            CompressionUtils.testEncoding(TSDataType.DOUBLE, encoding, values, 4, result, loss);
            check(encoding.toString(), result, loss, values.size());
        }
        checkSprintz64Random();

        checkBench("LECO", values.size(), (result, loss) ->
                ExternalCompressionBench.benchLeCoDoubleScaledIntegersFastPFor(columns, maxPoints, result, loss));
        checkBench("Simple8b", finiteValues.size(), (result, loss) ->
                Simple8bIntColumn.benchSimple8bDouble(finiteColumns, finiteMaxPoints, result, loss));
        checkBench("Sub-column", finiteValues.size(), (result, loss) ->
                SubcolumnCodec.benchSubcolumnDouble(finiteColumns, finiteMaxPoints, result, loss));

        double[] input = values.stream().mapToDouble(Double::doubleValue).toArray();
        byte[] encoded = BuffSplitDoubleColumnCodec.encodeDoubles(input, 10000);
        double[] decoded = BuffSplitDoubleColumnCodec.decodeDoubles(encoded).values;
        if (!rawEqual(input, decoded)) throw new AssertionError("BUFF is not bit-exact");
    }

    private interface Bench {
        void run(long[] result, double[] loss);
    }

    private static void checkBench(String name, int count, Bench bench) {
        long[] result = new long[4];
        double[] loss = new double[2];
        bench.run(result, loss);
        check(name, result, loss, count);
    }

    private static void check(String name, long[] result, double[] loss, int count) {
        if (result[0] != (long) count * Double.BYTES || loss[0] != 0.0 || !(loss[1] > 0.0)) {
            throw new AssertionError(name + " failed: result=" + Arrays.toString(result)
                    + " loss=" + Arrays.toString(loss));
        }
    }

    private static boolean rawEqual(double[] left, double[] right) {
        if (left.length != right.length) return false;
        for (int i = 0; i < left.length; i++) {
            if (Double.doubleToRawLongBits(left[i]) != Double.doubleToRawLongBits(right[i])) return false;
        }
        return true;
    }

    private static void checkSprintz64Random() {
        ArrayList<Long> values = new ArrayList<>(1_000_000);
        long x = 0x123456789abcdef0L;
        for (int i = 0; i < 1_000_000; i++) {
            x ^= x << 13;
            x ^= x >>> 7;
            x ^= x << 17;
            values.add(x);
        }
        long[] decoded = Sprintz64Codec.decodeLongs(Sprintz64Codec.encodeLongs(values));
        if (decoded.length != values.size()) {
            throw new AssertionError("SPRINTZ64 length mismatch");
        }
        for (int i = 0; i < decoded.length; i++) {
            if (decoded[i] != values.get(i)) {
                throw new AssertionError("SPRINTZ64 mismatch at " + i);
            }
        }
    }
}
