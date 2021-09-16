include ../../../config.mk

EXTRA_FLAGS=

all: lu lu_cu

lu: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

lu_cu: lu_host.cu lu_kernel.cu lu_kernel.hu
	${VERBOSE} ${CUCC} -o lu_cu lu_host.cu lu_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f lu lu_cu

