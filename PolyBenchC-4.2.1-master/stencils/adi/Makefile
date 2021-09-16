include ../../config.mk

EXTRA_FLAGS=

all: adi adi_cu

adi: adi.c adi.h
	${VERBOSE} ${CC} -o adi adi.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

adi_cu: adi_host.cu adi_kernel.cu adi_kernel.hu
	${VERBOSE} ${CUCC} -o adi_cu adi_host.cu adi_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f adi adi_cu

