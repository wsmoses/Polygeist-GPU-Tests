include ../../config.mk

EXTRA_FLAGS=

all: jacobi-1d jacobi-1d_cu

jacobi-1d: jacobi-1d.c jacobi-1d.h
	${VERBOSE} ${CC} -o jacobi-1d jacobi-1d.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

jacobi-1d_cu: jacobi-1d_host.cu jacobi-1d_kernel.cu jacobi-1d_kernel.hu
	${VERBOSE} ${CUCC} -o jacobi-1d_cu jacobi-1d_host.cu jacobi-1d_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f jacobi-1d jacobi-1d_cu

