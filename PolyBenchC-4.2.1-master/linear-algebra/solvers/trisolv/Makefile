include ../../../config.mk

EXTRA_FLAGS=

all: trisolv trisolv_cu

trisolv: trisolv.c trisolv.h
	${VERBOSE} ${CC} -o trisolv trisolv.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

trisolv_cu: trisolv_host.cu trisolv_kernel.cu trisolv_kernel.hu
	${VERBOSE} ${CUCC} -o trisolv_cu trisolv_host.cu trisolv_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f trisolv trisolv_cu

