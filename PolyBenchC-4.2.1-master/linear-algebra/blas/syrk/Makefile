include ../../../config.mk

EXTRA_FLAGS=

all: syrk syrk_cu

syrk: syrk.c syrk.h
	${VERBOSE} ${CC} -o syrk syrk.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

syrk_cu: syrk_host.cu syrk_kernel.cu syrk_kernel.hu
	${VERBOSE} ${CUCC} -o syrk_cu syrk_host.cu syrk_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f syrk syrk_cu

