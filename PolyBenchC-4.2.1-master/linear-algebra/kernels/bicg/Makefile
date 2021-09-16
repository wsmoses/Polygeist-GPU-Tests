include ../../../config.mk

EXTRA_FLAGS=

all: bicg bicg_cu

bicg: bicg.c bicg.h
	${VERBOSE} ${CC} -o bicg bicg.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

bicg_cu: bicg_host.cu bicg_kernel.cu bicg_kernel.hu
	${VERBOSE} ${CUCC} -o bicg_cu bicg_host.cu bicg_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f bicg bicg_cu

