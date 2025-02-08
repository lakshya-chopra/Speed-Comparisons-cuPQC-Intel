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

