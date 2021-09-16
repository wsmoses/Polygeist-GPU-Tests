include ../../../config.mk

EXTRA_FLAGS=

all: gesummv gesummv_cu

gesummv: gesummv.c gesummv.h
	${VERBOSE} ${CC} -o gesummv gesummv.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

gesummv_cu: gesummv_host.cu gesummv_kernel.cu gesummv_kernel.hu
	${VERBOSE} ${CUCC} -o gesummv_cu gesummv_host.cu gesummv_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f gesummv gesummv_cu

