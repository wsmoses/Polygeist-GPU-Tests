include ../../config.mk

EXTRA_FLAGS=

all: fdtd-2d fdtd-2d_cu

fdtd-2d: fdtd-2d.c fdtd-2d.h
	${VERBOSE} ${CC} -o fdtd-2d fdtd-2d.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

fdtd-2d_cu: fdtd-2d_host.cu fdtd-2d_kernel.cu fdtd-2d_kernel.hu
	${VERBOSE} ${CUCC} -o fdtd-2d_cu fdtd-2d_host.cu fdtd-2d_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f fdtd-2d fdtd-2d_cu

