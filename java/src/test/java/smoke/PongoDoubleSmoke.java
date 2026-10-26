package smoke;

import java.lang.reflect.Method;
import java.util.HexFormat;

public final class PongoDoubleSmoke {
    public static void main(String[] args) throws Exception {
        Class<?> codec = Class.forName("org.example.PongoDouble");
        Method encode = codec.getMethod("encode", double[].class);
        Method decode = codec.getMethod("decode", byte[].class);
        double[] values = {0.0, -0.0, 1.2, 1.23, 0.00123, -42.125, 1e20,
                Double.NaN, Double.POSITIVE_INFINITY, Double.MIN_VALUE};
        byte[] compressed = (byte[]) encode.invoke(null, (Object) values);
        String expected = "0a0000000000000000000000000000007001f00ebff7982d42c82e8df018ff18bf703a8450f01d78"
                + "b58c58efb6bc75e2d6316000800000000001c1ffff80000000000008";
        if (!HexFormat.of().formatHex(compressed).equals(expected)) {
            throw new AssertionError("Java payload differs from the cross-language golden vector");
        }
        double[] restored = (double[]) decode.invoke(null, (Object) compressed);
        if (restored == null || restored.length != values.length) throw new AssertionError("decode failed");
        for (int i = 0; i < values.length; i++) {
            if (Double.doubleToRawLongBits(values[i]) != Double.doubleToRawLongBits(restored[i])) {
                throw new AssertionError("mismatch at " + i);
            }
        }
        System.out.println("Pongo OK bytes=" + compressed.length);
    }
}
