include ../../../config.mk

EXTRA_FLAGS=

all: mvt mvt_cu

mvt: mvt.c mvt.h
	${VERBOSE} ${CC} -o mvt mvt.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

mvt_cu: mvt_host.cu mvt_kernel.cu mvt_kernel.hu
	${VERBOSE} ${CUCC} -o mvt_cu mvt_host.cu mvt_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f mvt mvt_cu

