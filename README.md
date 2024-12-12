# Speed comparisons between Nvidia's cuPQC & Intel CPUs for Post Quantum & Classical Cryptography.

## Specifications:
**GPU**:
```
GPU used: Nvidia's a4000X
CUDA version: 12.6
OS: Ubuntu 22.04 LTS (GNU/Linux 5.15.0)
Cryptographic Library: cuPQC, release 1
```
**Intel CPU #0:**
```
Processor: 12th Gen Intel(R) Core(TM) i9-12900
Architecture: x86_64
Cores : 24
OS: Ubuntu 22.04 LTS (GNU/Linux 5.15.0)
CPU Flags (used): AVX2, SSE2, RDSEED/RDRAND, SHA_NI, AES_N1, BMI2, POPCNT
Cryptographic Library: OpenSSL (3.4.0) + libOQS
LibOQS Version: v0.12.0-rc1
ML-KEM : FIPS-203
ML-DSA: FIPS-204 
```
## Results
### For cuPQC:
- **MLKEM-512:**
```
Key Generation Throughput: ~3170219.98 ops/sec
Encapsulation Throughput: ~3268917.41 ops/sec
Decapsulation Throughput: ~3144369.38 ops/sec
```
- **MLKEM-768:**
```
Key Generation Throughput: ~2421352.50 ops/sec
Encapsulation Throughput: ~2475257.25 ops/sec
Decapsulation Throughput: ~1599282.74 ops/sec
```
- **MLDSA-44:**
```
Key Generation Throughput: 2145190.54 ops/sec
Sign Throughput: 252759.62 ops/sec
Verification Throughput: 1152434.40 ops/sec
```
- **MLDSA-65:**
```
Key Generation Throughput: 1324213.74 ops/sec
Sign Throughput: 202258.85 ops/sec
Verification Throughput: 819874.50 ops/sec
``` 
### For OpenSSL
Results can be found in the [OpenSSL](https://github.com/lakshya-chopra/Speed-Comparisons-cuPQC-Intel/tree/main/OpenSSL) dir
