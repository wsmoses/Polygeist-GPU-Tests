include ../../../config.mk

EXTRA_FLAGS=

all: doitgen doitgen_cu

doitgen: doitgen.c doitgen.h
	${VERBOSE} ${CC} -o doitgen doitgen.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

doitgen_cu: doitgen_host.cu doitgen_kernel.cu doitgen_kernel.hu
	${VERBOSE} ${CUCC} -o doitgen_cu doitgen_host.cu doitgen_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f doitgen doitgen_cu

