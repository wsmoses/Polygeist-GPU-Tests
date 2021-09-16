include ../../config.mk

EXTRA_FLAGS=

all: nussinov nussinov_cu

nussinov: nussinov.c nussinov.h
	${VERBOSE} ${CC} -o nussinov nussinov.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

nussinov_cu: nussinov_host.cu nussinov_kernel.cu nussinov_kernel.hu
	${VERBOSE} ${CUCC} -o nussinov_cu nussinov_host.cu nussinov_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f nussinov nussinov_cu

