include ../../config.mk

EXTRA_FLAGS=

all: seidel-2d seidel-2d_cu

seidel-2d: seidel-2d.c seidel-2d.h
	${VERBOSE} ${CC} -o seidel-2d seidel-2d.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

seidel-2d_cu: seidel-2d_host.cu seidel-2d_kernel.cu seidel-2d_kernel.hu
	${VERBOSE} ${CUCC} -o seidel-2d_cu seidel-2d_host.cu seidel-2d_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f seidel-2d seidel-2d_cu

