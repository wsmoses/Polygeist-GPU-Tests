
CC=$${HOME}/git/Polygeist/build/mlir-clang/mlir-clang
CFLAGS=-L ~/llvm-project/build/lib \
       -resource-dir=$${HOME}/git/Polygeist/mlir-build/lib/clang/14.0.0/ \
       -DLARGE_DATASET \
	   --function=* \
       -lm

CUCC=$${HOME}/git/Polygeist/build/mlir-clang/mlir-clang
CUCFLAGS=-L ~/llvm-project/build/lib \
         -resource-dir=$${HOME}/git/Polygeist/mlir-build/lib/clang/14.0.0/ \
         --cuda-gpu-arch=sm_60 \
         --cuda-lower \
         --cpuify="distribute" \
         -DLARGE_DATASET \
	     --function=* \
		 -L /usr/local/cuda-11.2/lib64 \
		 -L ${HOME}/git/Polygeist/mlir-build/lib \
         -lcudart_static -ldl -lrt -lpthread -lm

         #--cpuify="continuation" \
