include ../../../config.mk

EXTRA_FLAGS=-lm

all: cholesky cholesky_cu

cholesky: cholesky.c cholesky.h
	${VERBOSE} ${CC} -o cholesky cholesky.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

cholesky_cu: cholesky_host.cu cholesky_kernel.cu cholesky_kernel.hu
	${VERBOSE} ${CUCC} -o cholesky_cu cholesky_host.cu cholesky_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f cholesky cholesky_cu

