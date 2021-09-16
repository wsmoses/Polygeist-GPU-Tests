include ../../../config.mk

EXTRA_FLAGS=-lm

all: gramschmidt gramschmidt_cu

gramschmidt: gramschmidt.c gramschmidt.h
	${VERBOSE} ${CC} -o gramschmidt gramschmidt.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

gramschmidt_cu: gramschmidt_host.cu gramschmidt_kernel.cu gramschmidt_kernel.hu
	${VERBOSE} ${CUCC} -o gramschmidt_cu gramschmidt_host.cu gramschmidt_kernel.cu ${CUCFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f gramschmidt gramschmidt_cu

