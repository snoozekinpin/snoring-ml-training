CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -I. -Igenerated
GOLDEN ?= ../../output/v5/deliverables/golden

build:
	mkdir -p build

HEADERS = $(wildcard *.h) $(wildcard generated/*.h)

build/test_features: host_test/test_features.c fft512.c snore_features.c $(HEADERS) | build
	$(CC) $(CFLAGS) -o $@ host_test/test_features.c fft512.c snore_features.c -lm

build/test_fsm: host_test/test_fsm.c snore_episode_fsm.c $(HEADERS) | build
	$(CC) $(CFLAGS) -o $@ host_test/test_fsm.c snore_episode_fsm.c -lm

build/test_doa: host_test/test_doa.c fft512.c doa_gccphat.c $(HEADERS) | build
	$(CC) $(CFLAGS) -o $@ host_test/test_doa.c fft512.c doa_gccphat.c -lm

test-features: build/test_features
	./build/test_features $(GOLDEN)/features.bin

test-fsm: build/test_fsm
	./build/test_fsm $(GOLDEN)/fsm_trace.txt
	./build/test_fsm $(GOLDEN)/fsm_trace_boundary.txt

test-doa: build/test_doa
	./build/test_doa $(GOLDEN)/doa_cases.bin $(GOLDEN)/doa_tracker.bin

test: test-features test-fsm test-doa

clean:
	rm -rf build

.PHONY: test test-features test-fsm test-doa clean
