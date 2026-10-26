import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import org.example.PongoDouble;

/** Materialize a PongoDouble payload from a raw little-endian float64 column. */
public final class PongoPayloadMain {
    private PongoPayloadMain() {}

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("usage: PongoPayloadMain input.f64le output.pongo");
        }
        byte[] raw = Files.readAllBytes(Path.of(args[0]));
        if ((raw.length & 7) != 0) {
            throw new IllegalArgumentException("input length is not a multiple of 8");
        }
        ByteBuffer input = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        double[] values = new double[raw.length / 8];
        for (int i = 0; i < values.length; i++) {
            values[i] = input.getDouble();
        }
        byte[] encoded = PongoDouble.encode(values);
        double[] decoded = PongoDouble.decode(encoded);
        if (decoded.length != values.length) {
            throw new IllegalStateException("Pongo decoded length mismatch");
        }
        for (int i = 0; i < values.length; i++) {
            if (Double.doubleToRawLongBits(decoded[i]) != Double.doubleToRawLongBits(values[i])) {
                throw new IllegalStateException("Pongo roundtrip mismatch at row " + i);
            }
        }
        Files.write(Path.of(args[1]), encoded);
        System.out.printf("rows=%d raw=%d encoded=%d%n", values.length, raw.length, encoded.length);
    }
}
