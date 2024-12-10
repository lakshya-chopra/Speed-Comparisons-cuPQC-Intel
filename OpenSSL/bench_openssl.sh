#!/bin/bash
if [ -z "$1" ]; then
        echo "Usage: $0 %s <OPENSSL_BIN_PATH> [<cores> <seconds>]"
  exit 1
fi


# Define the algorithms you want to test
algorithms=("rsa3072" "X25519" "X448" "mlkem768" "mlkem512" "mlkem1024" "frodo976aes" "frodo976shake" "bikel5" "hqc192")
time="${3:-5}"
cores="${2:-$(nproc)}"

output_file="benchmark_results_cores${cores}_sec${time}.json"
echo "[]" > "$output_file"

# Note: Use OpenSSL with OQS Provider
OPENSSL_BIN_PATH=$1

for algorithm in "${algorithms[@]}"; do
  echo "Testing algorithm: $algorithm"

  output=$("$OPENSSL_BIN_PATH" speed -multi "$cores" -seconds "$time" -kem-algorithms $algorithm 2>&1)


  echo "$output"
  result_line=$(echo "$output" | awk -v alg="$algorithm" '$1 == alg {print; exit}')

  echo "Extracted line: $result_line"

  keygen_ops=$(echo "$result_line" | awk '{print $5}')
  encaps_ops=$(echo "$result_line" | awk '{print $6}')
  decaps_ops=$(echo "$result_line" | awk '{print $7}')

  echo "Keygen ops/s: $keygen_ops"
  echo "Encaps ops/s: $encaps_ops"
  echo "Decaps ops/s: $decaps_ops"

  json_entry=$(jq -n \
  --arg algorithm "$algorithm" \
  --arg keygen_ops "$keygen_ops" \
  --arg encaps_ops "$encaps_ops" \
  --arg decaps_ops "$decaps_ops" \
  '{
    algorithm: $algorithm,
    keygens_s: $keygen_ops,
    encaps_s: $encaps_ops,
    decaps_s: $decaps_ops
  }')

  jq ". += [$json_entry]" "$output_file" > tmp.json && mv tmp.json "$output_file"

done

echo "Benchmark results written to $output_file"