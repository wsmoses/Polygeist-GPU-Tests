include ../../config.mk

EXTRA_FLAGS=

all: floyd-warshall floyd-warshall_cu

floyd-warshall: floyd-warshall.c floyd-warshall.h
	${VERBOSE} ${CC} -o floyd-warshall floyd-warshall.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

floyd-warshall_cu: floyd-warshall_host.cu floyd-warshall_kernel.cu floyd-warshall_kernel.hu
	${VERBOSE} ${CUCC} -o floyd-warshall_cu floyd-warshall_host.cu floyd-warshall_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f floyd-warshall floyd-warshall_cu

