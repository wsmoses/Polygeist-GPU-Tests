include ../../../config.mk

EXTRA_FLAGS=

all: trmm trmm_cu

trmm: trmm.c trmm.h
	${VERBOSE} ${CC} -o trmm trmm.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

trmm_cu: trmm_host.cu trmm_kernel.cu trmm_kernel.hu
	${VERBOSE} ${CUCC} -o trmm_cu trmm_host.cu trmm_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f trmm trmm_cu

