include ../../../config.mk

EXTRA_FLAGS=

all: syr2k syr2k_cu

syr2k: syr2k.c syr2k.h
	${VERBOSE} ${CC} -o syr2k syr2k.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

syr2k_cu: syr2k_host.cu syr2k_kernel.cu syr2k_kernel.hu
	${VERBOSE} ${CUCC} -o syr2k_cu syr2k_host.cu syr2k_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f syr2k syr2k_cu

