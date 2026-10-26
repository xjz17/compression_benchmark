# C++ oracle-quality recommender

The native implementation provides label-free feature extraction, development-only
ridge/kNN fitting, model serialization, and prepared model ranking. Follow the
staged execution protocol described by the source and command examples below
for training, independent verification, locking, and final confirm evaluation.
The commands below are standalone feature-builder witnesses.

`quality_features` is the label-free first stage of the current-135
recommendation-quality experiment. It reads only the frozen source-column list,
source manifests, and the first nonempty source block selected for each column.
It never accepts an oracle-label path.

Build a local witness binary:

```sh
c++ -std=c++20 -O3 -ffp-contract=off \
  -I/opt/homebrew/include -Icpp -Icpp/operator_recommend135_v1 \
  -Icpp/operator_oracle_quality135_v1 \
  cpp/operator_oracle_quality135_v1/quality_data.cpp \
  cpp/operator_oracle_quality135_v1/quality_features_main.cpp \
  cpp/operator_recommend135_v1/model.cpp \
  -L/opt/homebrew/lib -lcrypto -o /tmp/quality_features
```

Run the production invocation on the benchmark host because the
frozen `COLUMNS.json` contains source-manifest paths from that host. Use a new
output directory; the CLI refuses to replace either file:

```sh
/tmp/quality_features \
  /path/to/COLUMNS.json \
  /new/output/features.jsonl \
  /new/output/FEATURES_RECEIPT.json
```

The local unit witness uses generated source manifests and payload regions; it
does not access the benchmark host or any oracle labels:

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I/opt/homebrew/include -Icpp -Icpp/operator_recommend135_v1 \
  -Icpp/operator_oracle_quality135_v1 \
  cpp/operator_oracle_quality135_v1/quality_data.cpp \
  cpp/operator_oracle_quality135_v1/quality_data_test.cpp \
  cpp/operator_recommend135_v1/model.cpp \
  -L/opt/homebrew/lib -lcrypto \
  -o /tmp/operator_oracle_quality135_v1_test
/tmp/operator_oracle_quality135_v1_test
```

The JSONL join key is `(dataset_id, column_id)`. `input_dtype` is the canonical
model dtype, while `dtype` is retained as an equal source-provenance field.
Every row binds the frozen `COLUMNS.json`, oracle reference hash, source
manifest, selected full source regions, and the exact sampled values and null
bitmap. The online `PreparedRegistry` remains a separate 23-compression-config
runtime; this builder does not claim that it executes the historical 85
algorithm or 97 operator pools.
