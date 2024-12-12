#include <vector>
#include <cupqc.hpp>
#include <cassert>
#include <cstdio>
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

    keygen_kernel<<<batch, MLKEM512Key::BlockDim>>>(d_public_key, d_secret_key, workspace, randombytes);

    cudaMemcpy(public_keys.data(), d_public_key, length_public_key * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(secret_keys.data(), d_secret_key, length_secret_key * batch, cudaMemcpyDeviceToHost);

    verify_key_pair(public_keys, secret_keys, batch); // Verify keys after generation

    cudaFree(d_public_key);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    unsigned int batch = 10;

    std::vector<uint8_t> public_keys(MLKEM512Key::public_key_size * batch);
    std::vector<uint8_t> secret_keys(MLKEM512Key::secret_key_size * batch);

    ml_kem_keygen(public_keys, secret_keys, batch);

    printf("Key generation and verification completed successfully.\n");
}

