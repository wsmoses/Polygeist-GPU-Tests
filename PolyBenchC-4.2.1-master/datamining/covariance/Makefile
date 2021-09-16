include ../../config.mk

EXTRA_FLAGS=

all: covariance covariance_cu

covariance: covariance.c covariance.h
	${VERBOSE} ${CC} -o covariance covariance.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

covariance_cu: covariance_host.cu covariance_kernel.cu covariance_kernel.hu
	${VERBOSE} ${CUCC} -o covariance_cu covariance_host.cu covariance_kernel.cu ${CUCFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f covariance covariance_cu

