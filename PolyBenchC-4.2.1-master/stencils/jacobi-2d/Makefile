include ../../config.mk

EXTRA_FLAGS=

all: jacobi-2d jacobi-2d_cu

jacobi-2d: jacobi-2d.c jacobi-2d.h
	${VERBOSE} ${CC} -o jacobi-2d jacobi-2d.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

jacobi-2d_cu: jacobi-2d_host.cu jacobi-2d_kernel.cu jacobi-2d_kernel.hu
	${VERBOSE} ${CUCC} -o jacobi-2d_cu jacobi-2d_host.cu jacobi-2d_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f jacobi-2d jacobi-2d_cu

