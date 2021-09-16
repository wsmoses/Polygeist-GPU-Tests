include ../../config.mk

EXTRA_FLAGS=

all: deriche deriche_cu

deriche: deriche.c deriche.h
	${VERBOSE} ${CC} -o deriche deriche.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

deriche_cu: deriche_host.cu deriche_kernel.cu deriche_kernel.hu
	${VERBOSE} ${CUCC} -o deriche_cu deriche_host.cu deriche_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f deriche deriche_cu

