include ../../../config.mk

EXTRA_FLAGS=

all: symm symm_cu

symm: symm.c symm.h
	${VERBOSE} ${CC} -o symm symm.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

symm_cu: symm_host.cu symm_kernel.cu symm_kernel.hu
	${VERBOSE} ${CUCC} -o symm_cu symm_host.cu symm_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f symm symm_cu

