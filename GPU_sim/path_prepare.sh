nvcc -V
export CUDA_HOME=/home/sr/CUDA/cuda-4.0
export CUDA_INSTALL_PATH=${CUDA_HOME}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=PATH=PATH=${CUDA_HOME}/bin:$PATH


echo $CUDA_HOME
echo $CUDA_INSTALL_PATH
echo $LD_LIBRARY_PATH
echo $PATH


nvcc -V