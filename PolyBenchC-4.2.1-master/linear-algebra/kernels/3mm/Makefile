include ../../../config.mk

EXTRA_FLAGS=

all: 3mm 3mm_cu

3mm: 3mm.c 3mm.h
	${VERBOSE} ${CC} -o 3mm 3mm.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

3mm_cu: 3mm_host.cu 3mm_kernel.cu 3mm_kernel.hu
	${VERBOSE} ${CUCC} -o 3mm_cu 3mm_host.cu 3mm_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f 3mm 3mm_cu

