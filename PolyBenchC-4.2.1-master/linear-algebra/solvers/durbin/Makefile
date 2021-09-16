include ../../../config.mk

EXTRA_FLAGS=

all: durbin durbin_cu

durbin: durbin.c durbin.h
	${VERBOSE} ${CC} -o durbin durbin.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

durbin_cu: durbin_host.cu durbin_kernel.cu durbin_kernel.hu
	${VERBOSE} ${CUCC} -o durbin_cu durbin_host.cu durbin_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f durbin durbin_cu

