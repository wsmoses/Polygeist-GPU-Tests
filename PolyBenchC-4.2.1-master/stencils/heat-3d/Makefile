include ../../config.mk

EXTRA_FLAGS=

all: heat-3d heat-3d_cu

heat-3d: heat-3d.c heat-3d.h
	${VERBOSE} ${CC} -o heat-3d heat-3d.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

heat-3d_cu: heat-3d_host.cu heat-3d_kernel.cu heat-3d_kernel.hu
	${VERBOSE} ${CUCC} -o heat-3d_cu heat-3d_host.cu heat-3d_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f heat-3d heat-3d_cu

