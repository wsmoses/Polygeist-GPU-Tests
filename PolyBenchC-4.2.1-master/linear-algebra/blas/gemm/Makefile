include ../../../config.mk

EXTRA_FLAGS=

all: gemm gemm_cu

gemm: gemm.c gemm.h
	${VERBOSE} ${CC} -o gemm gemm.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

gemm_cu: gemm_host.cu gemm_kernel.cu gemm_kernel.hu
	${VERBOSE} ${CUCC} -o gemm_cu gemm_host.cu gemm_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f gemm gemm_cu

