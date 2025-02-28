# cuPQC Setup

## Pre-requisites:
```
- Nvidia's GPU of the given architectures: 70, 75, 80, 86, 89, 90 
- x86_64 CPU
- Cuda (>=12.4)
```
Check your GPU's compute capability [here](https://developer.nvidia.com/cuda-gpus)

## Cuda installation:
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```
```sh
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Check NVCC
```sh
nvcc --version
```

## cuPQC installation (v 0.2.0)
```sh
wget https://developer.download.nvidia.com/compute/cupqc/redist/cupqc/cupqc-pkg-0.2.0.tar.gz
tar -xvzf cupqc-pkg-0.2.0-tar.gz
```

## Run cuPQC's examples:
```sh
cd cupqc-pkg-0.2.0/example
make
```
Make sure to adjust the arch as per your GPU's compute capability.

> Note: if programs dont run then delete the binary and build again using any of the given above two commands

For further testing, copy the examples in this directory (`benchmarks` & `examples`) and run `make` again, or run it manually:
```sh
nvcc -dlto -arch=native -std=c++17 -O3  -L../lib/ -lcupqc  -o <binary_name> <file_name.cu>  -I../include/ -I../include/cupqc
```
for example:
```shell
nvcc -dlto -arch=native -std=c++17 -O3  -L../lib/ -lcupqc  -o v2_bench_refactoring v2_bench_refactoring.cu  -I../include/ -I../include/cupqc
```
Run:
```sh
./v2_bench_refactoring
```

If using a realtime kernel, you may want to setup the nvidia-driver like [this](https://github.com/lakshya-chopra/nvidia-driver-rtk22)

**Check the GPU status**:
```sh
watch gpustat -cp
```
```sh
watch -n0.1 nvidia-smi
```

## liboqs integration:
cuPQC can now be used as a backend for ML-KEM in liboqs as well. Build liboqs using the following cmd for this:
```sh
cmake -DBUILD_SHARED_LIBS=ON  -DOQS_USE_OPENSSL=OFF  -DCMAKE_BUILD_TYPE=Release -DOQS_DIST_BUILD=ON  \
-DOQS_USE_CUPQC=ON  -DCMAKE_PREFIX_PATH=/home/master/cupqc/cupqc-pkg-0.2.0/cmake   \    
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc  -DCMAKE_CUDA_ARCHITECTURES=86    \    
-DOQS_ENABLE_KEM_ml_kem_768_cuda=ON ..
```


