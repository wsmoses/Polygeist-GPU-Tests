
CC=$${HOME}/Polygeist/build/mlir-clang/mlir-clang
CFLAGS=-L ~/llvm-project/build/lib \
       -resource-dir=$${HOME}/llvm-project/build/lib/clang/14.0.0/ \
       -DLARGE_DATASET \
       -lm

CUCC=$${HOME}/Polygeist/build/mlir-clang/mlir-clang
CUCFLAGS=-L ~/llvm-project/build/lib \
         -resource-dir=$${HOME}/llvm-project/build/lib/clang/14.0.0/ \
         --cuda-gpu-arch=sm_60 \
         --cuda-lower \
         --cpuify="continuation" \
         -DLARGE_DATASET \
         -lcudart_static -ldl -lrt -lpthread -lm

