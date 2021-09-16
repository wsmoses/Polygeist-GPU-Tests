include ../../config.mk

EXTRA_FLAGS=-lm

all: correlation correlation_cu

correlation: correlation.c correlation.h
	${VERBOSE} ${CC} -o correlation correlation.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

correlation_cu: correlation_host.cu correlation_kernel.cu correlation_kernel.hu
	${VERBOSE} ${CUCC} -o correlation_cu correlation_host.cu correlation_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f correlation correlation_cu

