include ../../../config.mk

EXTRA_FLAGS=

all: atax atax_cu

atax: atax.c atax.h
	${VERBOSE} ${CC} -o atax atax.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

atax_cu: atax_host.cu atax_kernel.cu atax_kernel.hu
	${VERBOSE} ${CUCC} -o atax_cu atax_host.cu atax_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f atax atax_cu

