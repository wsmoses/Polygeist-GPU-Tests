include ../../../config.mk

EXTRA_FLAGS=

all: gemver gemver_cu

gemver: gemver.c gemver.h
	${VERBOSE} ${CC} -o gemver gemver.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

gemver_cu: gemver_host.cu gemver_kernel.cu gemver_kernel.hu
	${VERBOSE} ${CUCC} -o gemver_cu gemver_host.cu gemver_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f gemver gemver_cu

