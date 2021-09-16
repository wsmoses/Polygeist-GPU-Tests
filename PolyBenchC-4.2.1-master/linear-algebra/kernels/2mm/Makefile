include ../../../config.mk

EXTRA_FLAGS=

all: 2mm 2mm_cu

2mm: 2mm.c 2mm.h
	${VERBOSE} ${CC} -o 2mm 2mm.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

2mm_cu: 2mm_host.cu 2mm_kernel.cu 2mm_kernel.hu
	${VERBOSE} ${CUCC} -o 2mm_cu 2mm_host.cu 2mm_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f 2mm 2mm_cu

