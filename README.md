# Speed comparisons between Nvidia cuPQC & Intel CPU's for Post Quantum & Classical Cryptography.

## Specifications:
cuPQC:
```
GPU used: Nvidia's a4000X
CUDA version: 12.6
OS: Ubuntu 22.04 LTS (GNU/Linux 5.15.0)
Cryptographic Library: cuPQC, release 1
```
Intel:
```
Processor: 12th Gen Intel(R) Core(TM) i9-12900
Architecture: x86_64
Cores : 24
OS: Ubuntu 22.04 LTS (GNU/Linux 5.15.0)
Cryptographic Library: OpenSSL (3.4.0) + libOQS
CPU Flags (used): AVX2, SSE2, RDSEED/RDRAND, SHA_NI, AES_N1 & SM3_NI, & Multi Buffer RSA, PCLMULQDQ, BMI2, POPCNT
```
