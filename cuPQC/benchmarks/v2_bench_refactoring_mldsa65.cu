#include <vector>
#include <cupqc.hpp>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
using namespace cupqc;

#define DEBUG_KEY_GEN true // Enable/disable debugging for key generation

using MLDSA65Key = decltype(ML_DSA_65() + Function<function::Keygen>() + Block() + BlockDim<128>()); // Optional operator with default config

using MLDSA65Sign = decltype(ML_DSA_65() + Function<function::Sign>() + Block() + BlockDim<128>()); // Optional operator with default config

using MLDSA65Verify = decltype(ML_DSA_65() + Function<function::Verify>() + Block() + BlockDim<128>()); // Optional operator with default config

__global__ void keygen_kernel(uint8_t *public_keys, uint8_t *secret_keys, uint8_t *randombytes, uint8_t *workspace)
{
    __shared__ uint8_t smem_ptr[MLDSA65Key::shared_memory_size];
    int block = blockIdx.x;
    auto public_key = public_keys + block * MLDSA65Key::public_key_size;
    auto secret_key = secret_keys + block * MLDSA65Key::secret_key_size;
    auto entropy = randombytes + block * MLDSA65Key::entropy_size;
    auto work = workspace + block * MLDSA65Key::workspace_size;

    MLDSA65Key().execute(public_key, secret_key, entropy, work, smem_ptr);
}

__global__ void sign_kernel(uint8_t *signatures, const uint8_t *messages, const size_t message_size, const uint8_t *secret_keys, uint8_t *randombytes, uint8_t *workspace)
{
    __shared__ uint8_t smem_ptr[MLDSA65Sign::shared_memory_size];
    int block = blockIdx.x;
    auto signature = signatures + block * ((MLDSA65Sign::signature_size + 7) / 8 * 8);
    auto message = messages + block * message_size;
    auto secret_key = secret_keys + block * MLDSA65Sign::secret_key_size;
    auto entropy = randombytes + block * MLDSA65Sign::entropy_size;
    auto work = workspace + block * MLDSA65Sign::workspace_size;

    MLDSA65Sign().execute(signature, message, message_size, secret_key, entropy, work, smem_ptr);
}

__global__ void verify_kernel(bool *valids, const uint8_t *signatures, const uint8_t *messages, const size_t message_size, const uint8_t *public_keys, uint8_t *workspace)
{
    __shared__ uint8_t smem_ptr[MLDSA65Verify::shared_memory_size];
    int block = blockIdx.x;
    auto signature = signatures + block * ((MLDSA65Sign::signature_size + 7) / 8 * 8);
    auto message = messages + block * message_size;
    auto public_key = public_keys + block * MLDSA65Verify::public_key_size;
    auto work = workspace + block * MLDSA65Verify::workspace_size;

    valids[block] = MLDSA65Verify().execute(message, message_size, signature, public_key, work, smem_ptr);
}

void benchmark(const std::string &operation_name, const cudaEvent_t &start, const cudaEvent_t &stop, unsigned int batch)
{
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;
    double throughput = batch / seconds;
    printf("%s Throughput: %.2f ops/sec\n", operation_name.c_str(), throughput);
}

void ml_dsa_keygen(std::vector<uint8_t> &public_keys, std::vector<uint8_t> &secret_keys, const unsigned int batch)
{
    auto length_public_key = MLDSA65Key::public_key_size;
    auto length_secret_key = MLDSA65Key::secret_key_size;

    auto workspace = make_workspace<MLDSA65Key>(batch);
    auto randombytes = get_entropy<MLDSA65Key>(batch);

    uint8_t *d_public_key = nullptr;
    uint8_t *d_secret_key = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_public_key), length_public_key * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_secret_key), length_secret_key * batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    keygen_kernel<<<batch, MLDSA65Key::BlockDim>>>(d_public_key, d_secret_key, randombytes, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(public_keys.data(), d_public_key, length_public_key * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(secret_keys.data(), d_secret_key, length_secret_key * batch, cudaMemcpyDeviceToHost);

    benchmark("Key Generation", start, stop, batch);

    cudaFree(d_public_key);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ml_dsa_sign(std::vector<uint8_t> &signatures, std::vector<uint8_t> &messages, size_t message_size,
                 const std::vector<uint8_t> &secret_keys, const unsigned int batch)
{
    auto length_secret_key = MLDSA65Sign::secret_key_size;
    auto length_signature = MLDSA65Sign::signature_size;
    // length_signature is not a multiple of 8, need to pad to ensure alignment.
    auto length_signature_padded = ((length_signature + 7) / 8) * 8;

    auto workspace = make_workspace<MLDSA65Sign>(batch);
    auto randombytes = get_entropy<MLDSA65Sign>(batch);
    uint8_t *d_signatures = nullptr;
    uint8_t *d_secret_keys = nullptr;
    uint8_t *d_messages = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_signatures), length_signature_padded * batch); // These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void **>(&d_secret_keys), length_secret_key * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_messages), message_size * batch);

    cudaMemcpy(d_secret_keys, secret_keys.data(), length_secret_key * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_messages, messages.data(), message_size * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sign_kernel<<<batch, MLDSA65Sign::BlockDim>>>(d_signatures, d_messages, message_size, d_secret_keys, randombytes, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(signatures.data(), d_signatures, length_signature_padded * batch, cudaMemcpyDeviceToHost);

    benchmark("Sign", start, stop, batch);

    cudaFree(d_secret_keys);
    cudaFree(d_signatures);
    cudaFree(d_messages);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ml_dsa_verify(bool *is_valids, const std::vector<uint8_t> &signatures, const std::vector<uint8_t> &messages, size_t message_size,
                   const std::vector<uint8_t> &public_keys, const unsigned int batch)
{
    auto workspace         = make_workspace<MLDSA65Verify>(batch);
    auto length_signature  = MLDSA65Verify::signature_size;
    auto length_signature_padded = ((length_signature + 7) / 8) * 8;
    auto length_public_key = MLDSA65Verify::public_key_size;

    uint8_t* d_signatures  = nullptr;
    uint8_t* d_messages    = nullptr;
    uint8_t* d_public_keys = nullptr;
    bool*    d_valids      = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_signatures), length_signature_padded * batch); // These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void **>(&d_public_keys), length_public_key * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_messages), message_size * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_valids), batch);

    cudaMemcpy(d_public_keys, public_keys.data(), length_public_key * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatures, signatures.data(), length_signature_padded * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_messages, messages.data(), message_size * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    verify_kernel<<<batch, MLDSA65Verify::BlockDim>>>(d_valids, d_signatures, d_messages, message_size, d_public_keys, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(is_valids, d_valids, batch, cudaMemcpyDeviceToHost);

    benchmark("Verification", start, stop, batch);

    cudaFree(d_public_keys);
    cudaFree(d_signatures);
    cudaFree(d_messages);
    cudaFree(d_valids);
    destroy_workspace(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
    unsigned int batch = 100000; // Adjust the batch size for benchmarking

    constexpr size_t message_size = 1024;

    std::vector<uint8_t> public_keys(MLDSA65Key::public_key_size * batch);
    std::vector<uint8_t> secret_keys(MLDSA65Key::secret_key_size * batch);

    ml_dsa_keygen(public_keys, secret_keys, batch);

    std::vector<uint8_t> signatures((MLDSA65Sign::signature_size + 7) / 8 * 8 * batch);
    std::vector<uint8_t> messages(message_size * batch);
    ml_dsa_sign(signatures, messages, message_size, secret_keys, batch);

    bool is_valids[batch];
    ml_dsa_verify(is_valids, signatures, messages, message_size, public_keys, batch);

    printf("Key generation, Signing, and Verification completed successfully.\n");
}
