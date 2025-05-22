#include <stdint.h>
#include <string.h>

#ifdef __ARM_NEON
#define HAS_NEON 1
#include <arm_neon.h>
#else
#define HAS_NEON 0
#endif

static inline void u32t8le(uint32_t v, uint8_t p[4]) {
    p[0] = v & 0xff;
    p[1] = (v >> 8) & 0xff;
    p[2] = (v >> 16) & 0xff;
    p[3] = (v >> 24) & 0xff;
}

static inline uint32_t u8t32le(uint8_t p[4]) {
    uint32_t value = p[3];
    value = (value << 8) | p[2];
    value = (value << 8) | p[1];
    value = (value << 8) | p[0];
    return value;
}

static inline uint32_t rotl32(uint32_t x, int n) {
    return x << n | (x >> (-n & 31));
}

static void chacha20_init_state(uint32_t s[16], uint8_t key[32], uint32_t counter, uint8_t nonce[12]) {
    s[0] = 0x61707865;
    s[1] = 0x3320646e;
    s[2] = 0x79622d32;
    s[3] = 0x6b206574;

    for (int i = 0; i < 8; i++) {
        s[4 + i] = u8t32le(key + i * 4);
    }

    s[12] = counter;

    for (int i = 0; i < 3; i++) {
        s[13 + i] = u8t32le(nonce + i * 4);
    }
}




static void chacha_quarterround(uint32_t *x, int a, int b, int c, int d) {
    x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16);
    x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12);
    x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8);
    x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);
}

static void chacha_serialize(uint32_t in[16], uint8_t output[64]) {
    for (int i = 0; i < 16; i++) {
        u32t8le(in[i], output + (i << 2));
    }
}

static void chacha12_block_std(uint32_t in[16], uint8_t out[64]) {
    uint32_t x[16];
    memcpy(x, in, sizeof(uint32_t) * 16);

    for (int i = 6; i > 0; i--) {
        chacha_quarterround(x, 0, 4, 8, 12);
        chacha_quarterround(x, 1, 5, 9, 13);
        chacha_quarterround(x, 2, 6, 10, 14);
        chacha_quarterround(x, 3, 7, 11, 15);
        chacha_quarterround(x, 0, 5, 10, 15);
        chacha_quarterround(x, 1, 6, 11, 12);
        chacha_quarterround(x, 2, 7, 8, 13);
        chacha_quarterround(x, 3, 4, 9, 14);
    }

    for (int i = 0; i < 16; i++) {
        x[i] += in[i];
    }

    chacha_serialize(x, out);
}

void ChaCha12XOR_std(uint8_t key[32], uint32_t counter, uint8_t nonce[12], 
                    uint8_t *in, uint8_t *out, int inlen) {
    uint32_t s[16];
    uint8_t block[64];

    chacha20_init_state(s, key, counter, nonce);

    for (int i = 0; i < inlen; i += 64) {
        chacha12_block_std(s, block);
        s[12]++;

        for (int j = i; j < i + 64 && j < inlen; j++) {
            out[j] = in[j] ^ block[j - i];
        }
    }
}

#ifdef __ARM_NEON

static void chacha_quarterround_neon(uint32x4_t *a, uint32x4_t *b, 
                                   uint32x4_t *c, uint32x4_t *d) {
    *a = vaddq_u32(*a, *b);
    *d = veorq_u32(*d, *a);
    *d = vorrq_u32(vshlq_n_u32(*d, 16), vshrq_n_u32(*d, 16));
    
    *c = vaddq_u32(*c, *d);
    *b = veorq_u32(*b, *c);
    *b = vorrq_u32(vshlq_n_u32(*b, 12), vshrq_n_u32(*b, 20));
    
    *a = vaddq_u32(*a, *b);
    *d = veorq_u32(*d, *a);
    *d = vorrq_u32(vshlq_n_u32(*d, 8), vshrq_n_u32(*d, 24));
    
    *c = vaddq_u32(*c, *d);
    *b = veorq_u32(*b, *c);
    *b = vorrq_u32(vshlq_n_u32(*b, 7), vshrq_n_u32(*b, 25));
}

static void chacha12_block_neon(uint32_t in[16], uint8_t out[64]) {
    uint32x4_t x0, x1, x2, x3;
    uint32x4_t orig0, orig1, orig2, orig3;
    
    orig0 = x0 = vld1q_u32(in);
    orig1 = x1 = vld1q_u32(in + 4);
    orig2 = x2 = vld1q_u32(in + 8);
    orig3 = x3 = vld1q_u32(in + 12);
    
    for (int i = 0; i < 6; i++) {
        chacha_quarterround_neon(&x0, &x1, &x2, &x3);
        chacha_quarterround_neon(&x0, &x1, &x2, &x3);
    }
    
    x0 = vaddq_u32(x0, orig0);
    x1 = vaddq_u32(x1, orig1);
    x2 = vaddq_u32(x2, orig2);
    x3 = vaddq_u32(x3, orig3);
    
    vst1q_u8(out, vreinterpretq_u8_u32(x0));
    vst1q_u8(out + 16, vreinterpretq_u8_u32(x1));
    vst1q_u8(out + 32, vreinterpretq_u8_u32(x2));
    vst1q_u8(out + 48, vreinterpretq_u8_u32(x3));
}

void ChaCha12XOR_neon(uint8_t key[32], uint32_t counter, uint8_t nonce[12], 
                     uint8_t *in, uint8_t *out, int inlen) {
    uint32_t s[16];
    uint8_t block[64] __attribute__((aligned(16)));

    chacha20_init_state(s, key, counter, nonce);

    for (int i = 0; i < inlen; i += 64) {
        chacha12_block_neon(s, block);
        s[12]++;

        for (int j = i; j < i + 64 && j < inlen; j++) {
            out[j] = in[j] ^ block[j - i];
        }
    }
}
#endif



void ChaCha12XOR(uint8_t key[32], uint32_t counter, uint8_t nonce[12], 
                uint8_t *in, uint8_t *out, int inlen) {
#ifdef __ARM_NEON
        ChaCha12XOR_neon(key, counter, nonce, in, out, inlen);
#else
        ChaCha12XOR_std(key, counter, nonce, in, out, inlen);
#endif
}
