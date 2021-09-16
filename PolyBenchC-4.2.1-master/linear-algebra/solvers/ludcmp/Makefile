include ../../../config.mk

EXTRA_FLAGS=

all: ludcmp ludcmp_cu

ludcmp: ludcmp.c ludcmp.h
	${VERBOSE} ${CC} -o ludcmp ludcmp.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

ludcmp_cu: ludcmp_host.cu ludcmp_kernel.cu ludcmp_kernel.hu
	${VERBOSE} ${CUCC} -o ludcmp_cu ludcmp_host.cu ludcmp_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f ludcmp ludcmp_cu

