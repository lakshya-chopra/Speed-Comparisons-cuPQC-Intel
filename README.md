# Speed comparisons between Nvidia cuPQC & Intel CPU's for Post Quantum & Classical Cryptography.

## Specifications:
cuPQC:
```
GPU used: Nvidia's a4000X
CUDA version: 12.6
OS: Ubuntu 22.04 LTS (GNU/Linux 5.15.0)
Cryptographic Library: cuPQC, release 1
```
Intel CPU #0:
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
