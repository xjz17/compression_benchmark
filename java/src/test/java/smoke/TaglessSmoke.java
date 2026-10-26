import org.apache.tsfile.enums.ColumnCategory;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.TableSchema;
import org.apache.tsfile.file.metadata.enums.CompressionType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;
import org.apache.tsfile.write.record.Tablet;
import org.apache.tsfile.write.schema.MeasurementSchema;
import org.apache.tsfile.write.v4.ITsFileWriter;
import org.apache.tsfile.write.v4.TsFileWriterBuilder;
import java.io.File;
import java.util.Collections;

public class TaglessSmoke {
  public static void main(String[] args) throws Exception {
    File f = new File("target/tagless-smoke.tsfile");
    if (f.exists()) f.delete();
    TableSchema schema = new TableSchema("t1",
        Collections.singletonList(new MeasurementSchema("s1", TSDataType.INT64, TSEncoding.TS_2DIFF, CompressionType.LZ4)),
        Collections.singletonList(ColumnCategory.FIELD));
    try (ITsFileWriter w = new TsFileWriterBuilder().file(f).tableSchema(schema).build()) {
      Tablet tablet = new Tablet("t1", Collections.singletonList("s1"), Collections.singletonList(TSDataType.INT64), Collections.singletonList(ColumnCategory.FIELD), 10);
      tablet.addTimestamp(0, 0L);
      tablet.addValue(0, "s1", 42L);
      w.write(tablet);
    }
    System.out.println("OK size=" + f.length());
  }
}
