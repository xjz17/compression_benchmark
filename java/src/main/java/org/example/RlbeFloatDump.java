package org.example;

import org.apache.tsfile.encoding.encoder.Encoder;
import org.apache.tsfile.encoding.encoder.TSEncodingBuilder;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;

import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

/** Optional: dump RLBE FLOAT bytes (same slice as C++ golden test) to compare with {@code tools_rlbe_dump.cpp}. */
public final class RlbeFloatDump {
  public static void main(String[] args) throws Exception {
    Encoder e = TSEncodingBuilder.getEncodingBuilder(TSEncoding.RLBE).getEncoder(TSDataType.FLOAT);
    ByteArrayOutputStream b = new ByteArrayOutputStream();
    for (int i = 0; i < 64; i++) {
      e.encode(1.0f + i * 0.03125f, b);
    }
    e.flush(b);
    byte[] x = b.toByteArray();
    System.out.println(x.length);
    Files.write(Paths.get(args.length > 0 ? args[0] : "rlbe_java.bin"), x);
  }
}
