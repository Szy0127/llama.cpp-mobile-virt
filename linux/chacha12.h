
//#define __ARM_NEON
#ifdef __ARM_NEON
#include <asm/neon.h>
typedef uint32_t uint32x4_t __attribute__((vector_size(16)));
#else
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
    /*
    printk("state:\n");
    for(int i = 0 ; i < 16 ; i++){
        printk("%x,", s[i]);
    }
    printk("\n");
    */

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

const uint32_t k[64] = {
0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee ,
0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501 ,
0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be ,
0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821 ,
0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa ,
0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8 ,
0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed ,
0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a ,
0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c ,
0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70 ,
0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05 ,
0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665 ,
0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039 ,
0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1 ,
0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1 ,
0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 };
 
// r specifies the per-round shift amounts
const uint32_t r[] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                      5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
                      4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                      6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
 
// leftrotate function definition
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))
 
void to_bytes(uint32_t val, uint8_t *bytes)
{
    bytes[0] = (uint8_t) val;
    bytes[1] = (uint8_t) (val >> 8);
    bytes[2] = (uint8_t) (val >> 16);
    bytes[3] = (uint8_t) (val >> 24);
}
 
uint32_t to_int32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0]
        | ((uint32_t) bytes[1] << 8)
        | ((uint32_t) bytes[2] << 16)
        | ((uint32_t) bytes[3] << 24);
}
 
uint8_t buffer[180*1024*1024];
void md5(const uint8_t *initial_msg, size_t initial_len, uint8_t *digest) {
 
    // These vars will contain the hash
    uint32_t h0, h1, h2, h3;
 
    // Message (to prepare)
    uint8_t *msg = NULL;
 
    size_t new_len, offset;
    uint32_t w[16];
    uint32_t a, b, c, d, i, f, g, temp;
 
    // Initialize variables - simple count in nibbles:
    h0 = 0x67452301;
    h1 = 0xefcdab89;
    h2 = 0x98badcfe;
    h3 = 0x10325476;
 
    //Pre-processing:
    //append "1" bit to message    
    //append "0" bits until message length in bits ≡ 448 (mod 512)
    //append length mod (2^64) to message
 
    for (new_len = initial_len + 1; new_len % (512/8) != 448/8; new_len++)
        ;
 
    msg = buffer;//(uint8_t*)malloc(new_len + 8);
    memcpy(msg, initial_msg, initial_len);
    msg[initial_len] = 0x80; // append the "1" bit; most significant bit is "first"
    for (offset = initial_len + 1; offset < new_len; offset++)
        msg[offset] = 0; // append "0" bits
 
    // append the len in bits at the end of the buffer.
    to_bytes(initial_len*8, msg + new_len);
    // initial_len>>29 == initial_len*8>>32, but avoids overflow.
    to_bytes(initial_len>>29, msg + new_len + 4);
 
    // Process the message in successive 512-bit chunks:
    //for each 512-bit chunk of message:
    for(offset=0; offset<new_len; offset += (512/8)) {
 
        // break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
        for (i = 0; i < 16; i++)
            w[i] = to_int32(msg + offset + i*4);
 
        // Initialize hash value for this chunk:
        a = h0;
        b = h1;
        c = h2;
        d = h3;
        // Main loop:
        for(i = 0; i<64; i++) {
 
            if (i < 16) {
                f = (b & c) | ((~b) & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | ((~d) & c);
                g = (5*i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3*i + 5) % 16;          
            } else {
                f = c ^ (b | (~d));
                g = (7*i) % 16;
            }
 
            temp = d;
            d = c;
            c = b;
            b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
            a = temp;
 
        }
 
        // Add this chunk's hash to result so far:
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
 
    }
 
    // cleanup
    //free(msg);
 
    //var char digest[16] := h0 append h1 append h2 append h3 //(Output is in little-endian)
    to_bytes(h0, digest);
    to_bytes(h1, digest + 4);
    to_bytes(h2, digest + 8);
    to_bytes(h3, digest + 12);
}
 
