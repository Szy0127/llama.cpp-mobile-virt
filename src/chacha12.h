#ifndef __CHACHA12_H
#define __CHACHA12_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void ChaCha12XOR(uint8_t key[32], uint32_t counter, uint8_t nonce[12], uint8_t *input, uint8_t *output, int inputlen);

#ifdef __cplusplus
}
#endif

#endif
