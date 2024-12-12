#include <vector>
#include <cupqc.hpp>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
using namespace cupqc;

#define DEBUG_KEY_GEN true // Enable/disable debugging for key generation

using MLKEM512Key = decltype(ML_KEM_512()
                            + Function<function::Keygen>()
                            + Block()
                            + BlockDim<128>());  // Optional operator with default config

using MLKEM512Encaps = decltype(ML_KEM_512()
                               + Function<function::Encaps>()
                               + Block()
                               + BlockDim<128>());  // Optional operator with default config

using MLKEM512Decaps = decltype(ML_KEM_512()
                               + Function<function::Decaps>()
                               + Block()
                               + BlockDim<128>());  // Optional operator with default config

__global__ void keygen_kernel(uint8_t* public_keys, uint8_t* secret_keys, uint8_t* workspace, uint8_t* randombytes)
{
    __shared__ uint8_t smem_ptr[MLKEM512Key::shared_memory_size];
    int block = blockIdx.x;
    auto public_key = public_keys + block * MLKEM512Key::public_key_size;
    auto secret_key = secret_keys + block * MLKEM512Key::secret_key_size;
    auto entropy    = randombytes + block * MLKEM512Key::entropy_size;
    auto work       = workspace   + block * MLKEM512Key::workspace_size;

    MLKEM512Key().execute(public_key, secret_key, entropy, work, smem_ptr);
}

__global__ void encaps_kernel(uint8_t* ciphertexts, uint8_t* shared_secrets, const uint8_t* public_keys, uint8_t* workspace, uint8_t* randombytes)
{
    __shared__ uint8_t smem_ptr[MLKEM512Encaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM512Encaps::shared_secret_size;
    auto ciphertext    = ciphertexts + block * MLKEM512Encaps::ciphertext_size;
    auto public_key    = public_keys + block * MLKEM512Encaps::public_key_size;
    auto entropy       = randombytes + block * MLKEM512Encaps::entropy_size;
    auto work          = workspace   + block * MLKEM512Encaps::workspace_size;

    MLKEM512Encaps().execute(ciphertext, shared_secret, public_key, entropy, work, smem_ptr);
}

__global__ void decaps_kernel(uint8_t* shared_secrets, const uint8_t* ciphertexts, const uint8_t* secret_keys, uint8_t* workspace)
{
    __shared__ uint8_t smem_ptr[MLKEM512Decaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM512Decaps::shared_secret_size;
    auto ciphertext    = ciphertexts + block * MLKEM512Decaps::ciphertext_size;
    auto secret_key    = secret_keys + block * MLKEM512Decaps::secret_key_size;
    auto work          = workspace   + block * MLKEM512Decaps::workspace_size;

    MLKEM512Decaps().execute(shared_secret, ciphertext, secret_key, work, smem_ptr);
}

void verify_key_pair(const std::vector<uint8_t>& public_keys, const std::vector<uint8_t>& secret_keys, unsigned int batch) {
    for (unsigned int i = 0; i < batch; ++i) {
        const auto pub_key = &public_keys[i * MLKEM512Key::public_key_size];
        const auto sec_key = &secret_keys[i * MLKEM512Key::secret_key_size];

        bool public_key_valid = false;
        for (size_t j = 0; j < MLKEM512Key::public_key_size; ++j) {
            if (pub_key[j] != 0) {
                public_key_valid = true;
                break;
            }
        }

        bool secret_key_valid = false;
        for (size_t j = 0; j < MLKEM512Key::secret_key_size; ++j) {
            if (sec_key[j] != 0) {
                secret_key_valid = true;
                break;
            }
        }

        assert(public_key_valid && "Generated public key is invalid (all zeros).\n");
        assert(secret_key_valid && "Generated secret key is invalid (all zeros).\n");

        if (DEBUG_KEY_GEN) {
            printf("Verified Key Pair %d:\n", i);
            printf("  Public Key (First 16 bytes): ");
            for (size_t j = 0; j < 16 && j < MLKEM512Key::public_key_size; ++j) {
                printf("%02x", pub_key[j]);
            }
            printf("\n");

            printf("  Secret Key (First 16 bytes): ");
            for (size_t j = 0; j < 16 && j < MLKEM512Key::secret_key_size; ++j) {
                printf("%02x", sec_key[j]);
            }
            printf("\n");
        }
    }
}

void verify_shared_secrets(const std::vector<uint8_t>& encaps_shared_secrets, const std::vector<uint8_t>& decaps_shared_secrets, unsigned int batch) {
    for (unsigned int i = 0; i < batch; ++i) {
        const auto enc_secret = &encaps_shared_secrets[i * MLKEM512Encaps::shared_secret_size];
        const auto dec_secret = &decaps_shared_secrets[i * MLKEM512Encaps::shared_secret_size];

        bool secrets_match = true;
        for (size_t j = 0; j < MLKEM512Encaps::shared_secret_size; ++j) {
            if (enc_secret[j] != dec_secret[j]) {
                secrets_match = false;
                break;
            }
        }

        assert(secrets_match && "Encapsulation and Decapsulation shared secrets do not match.\n");

        printf("Shared Secret Match %d: %s\n", i, secrets_match ? "PASS" : "FAIL");

        if (secrets_match) {
            printf("  Shared Secret (Encapsulation & Decapsulation): ");
            for (size_t j = 0; j < MLKEM512Encaps::shared_secret_size; ++j) {
                printf("%02x", enc_secret[j]);
            }
            printf("\n");
        }
    }
}

void benchmark(const std::string& operation_name, const cudaEvent_t& start, const cudaEvent_t& stop, unsigned int batch) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;
    double throughput = batch / seconds;
    printf("%s Throughput: %.2f ops/sec\n", operation_name.c_str(), throughput);
}

void ml_kem_keygen(std::vector<uint8_t>& public_keys, std::vector<uint8_t>& secret_keys, const unsigned int batch)
{
    auto length_public_key = MLKEM512Key::public_key_size;
    auto length_secret_key = MLKEM512Key::secret_key_size;

    auto workspace   = make_workspace<MLKEM512Key>(batch);
    auto randombytes = get_entropy<MLKEM512Key>(batch);

    uint8_t* d_public_key = nullptr;
    uint8_t* d_secret_key = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    keygen_kernel<<<batch, MLKEM512Key::BlockDim>>>(d_public_key, d_secret_key, workspace, randombytes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(public_keys.data(), d_public_key, length_public_key * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(secret_keys.data(), d_secret_key, length_secret_key * batch, cudaMemcpyDeviceToHost);

    benchmark("Key Generation", start, stop, batch);

    verify_key_pair(public_keys, secret_keys, batch);

    cudaFree(d_public_key);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ml_kem_encaps(std::vector<uint8_t>& ciphertexts, std::vector<uint8_t>& shared_secrets,
                   const std::vector<uint8_t>& public_keys, const unsigned int batch)
{
    auto length_ciphertext   = MLKEM512Encaps::ciphertext_size;
    auto length_sharedsecret = MLKEM512Encaps::shared_secret_size;
    auto length_public_key   = MLKEM512Encaps::public_key_size;

    auto workspace   = make_workspace<MLKEM512Encaps>(batch);
    auto randombytes = get_entropy<MLKEM512Encaps>(batch);

    uint8_t* d_ciphertext   = nullptr;
    uint8_t* d_sharedsecret = nullptr;
    uint8_t* d_public_key   = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_ciphertext), length_ciphertext * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_sharedsecret), length_sharedsecret * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch);

    cudaMemcpy(d_public_key, public_keys.data(), length_public_key * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    encaps_kernel<<<batch, MLKEM512Encaps::BlockDim>>>(d_ciphertext, d_sharedsecret, d_public_key, workspace, randombytes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(ciphertexts.data(), d_ciphertext, length_ciphertext * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(shared_secrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost);

    benchmark("Encapsulation", start, stop, batch);

    cudaFree(d_ciphertext);
    cudaFree(d_sharedsecret);
    cudaFree(d_public_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ml_kem_decaps(std::vector<uint8_t>& shared_secrets, const std::vector<uint8_t>& ciphertexts,
                   const std::vector<uint8_t>& secret_keys, const unsigned int batch)
{
    auto length_ciphertext   = MLKEM512Decaps::ciphertext_size;
    auto length_sharedsecret = MLKEM512Decaps::shared_secret_size;
    auto length_secret_key   = MLKEM512Decaps::secret_key_size;

    auto workspace   = make_workspace<MLKEM512Decaps>(batch);

    uint8_t* d_ciphertext   = nullptr;
    uint8_t* d_sharedsecret = nullptr;
    uint8_t* d_secret_key   = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_ciphertext), length_ciphertext * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_sharedsecret), length_sharedsecret * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);

    cudaMemcpy(d_ciphertext, ciphertexts.data(), length_ciphertext * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_secret_key, secret_keys.data(), length_secret_key * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    decaps_kernel<<<batch, MLKEM512Decaps::BlockDim>>>(d_sharedsecret, d_ciphertext, d_secret_key, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(shared_secrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost);

    benchmark("Decapsulation", start, stop, batch);

    cudaFree(d_ciphertext);
    cudaFree(d_sharedsecret);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    unsigned int batch = 100000; // Adjust the batch size for benchmarking

    std::vector<uint8_t> public_keys(MLKEM512Key::public_key_size * batch);
    std::vector<uint8_t> secret_keys(MLKEM512Key::secret_key_size * batch);
    std::vector<uint8_t> ciphertexts(MLKEM512Encaps::ciphertext_size * batch);
    std::vector<uint8_t> encaps_shared_secrets(MLKEM512Encaps::shared_secret_size * batch);
    std::vector<uint8_t> decaps_shared_secrets(MLKEM512Decaps::shared_secret_size * batch);

    ml_kem_keygen(public_keys, secret_keys, batch);

    ml_kem_encaps(ciphertexts, encaps_shared_secrets, public_keys, batch);

    ml_kem_decaps(decaps_shared_secrets, ciphertexts, secret_keys, batch);

    verify_shared_secrets(encaps_shared_secrets, decaps_shared_secrets, batch);

    printf("Key generation, encapsulation, and decapsulation completed successfully.\n");
}

