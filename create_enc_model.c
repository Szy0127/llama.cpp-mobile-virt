/*
 * gcc create_enc_model.c -lcrypto
 * ./a.out tensor_info model.gguf model_enc.gguf
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <openssl/evp.h>

#define MAX_LINE_LENGTH 256
#define AES_BLOCK_SIZE 16

static uint8_t key[] = {
    0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f
};

static int aes128_ecb_encrypt_inplace(unsigned char *buf, int len, const unsigned char key[16]) {
    EVP_CIPHER_CTX *ctx = NULL;
    int outlen = 0;
    int tmplen = 0;
    int ret = -1;

    if (len % AES_BLOCK_SIZE != 0) {
        fprintf(stderr, "Error: input length must be multiple of 16 bytes\n");
        return -1;
    }

    printf("encrypt size 0x%lx\n", len);
    ctx = EVP_CIPHER_CTX_new();
    if (ctx == NULL) {
        return -1;
    }

    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), NULL, key, NULL) != 1) {
        goto cleanup;
    }

    EVP_CIPHER_CTX_set_padding(ctx, 0);

    if (EVP_EncryptUpdate(ctx, buf, &outlen, buf, len) != 1) {
        goto cleanup;
    }

    if (EVP_EncryptFinal_ex(ctx, buf + outlen, &tmplen) != 1) {
        goto cleanup;
    }

    ret = outlen + tmplen;

cleanup:
    EVP_CIPHER_CTX_free(ctx);
    return ret;
}

static int ends_with(const char *s, const char *suffix) {
    size_t s_len = strlen(s);
    size_t suffix_len = strlen(suffix);
    if (s_len < suffix_len) {
        return 0;
    }
    return strcmp(s + s_len - suffix_len, suffix) == 0;
}

static void trim_newline(char *s) {
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[--len] = '\0';
    }
}

static int copy_file(FILE *src, FILE *dst) {
    unsigned char buf[1024 * 1024];
    size_t n;

    if (fseek(src, 0, SEEK_SET) != 0) {
        return -1;
    }
    if (fseek(dst, 0, SEEK_SET) != 0) {
        return -1;
    }

    while ((n = fread(buf, 1, sizeof(buf), src)) > 0) {
        if (fwrite(buf, 1, n, dst) != n) {
            return -1;
        }
    }
    if (ferror(src)) {
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <offset_file> <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char *offset_file = argv[1];  
    const char *input_file = argv[2];   
    const char *output_file = argv[3];  

    FILE *fp_offset = fopen(offset_file, "r");
    if (!fp_offset) {
        perror("Failed to open offset file");
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    if (!fgets(line, sizeof(line), fp_offset)) {
        fprintf(stderr, "Failed to read tensor count line\n");
        fclose(fp_offset);
        return 1;
    }
    errno = 0;
    char *endptr = NULL;
    unsigned long total_tensors = strtoul(line, &endptr, 10);
    if (errno != 0 || endptr == line) {
        fprintf(stderr, "Invalid tensor count line: %s\n", line);
        fclose(fp_offset);
        return 1;
    }

    FILE *fp_input = fopen(input_file, "rb");
    if (!fp_input) {
        perror("Failed to open input file");
        fclose(fp_offset);
        return 1;
    }

    FILE *fp_output = fopen(output_file, "wb+");
    if (!fp_output) {
        perror("Failed to open output file");
        fclose(fp_offset);
        fclose(fp_input);
        return 1;
    }

    if (copy_file(fp_input, fp_output) != 0) {
        fprintf(stderr, "Failed to copy input file to output file\n");
        fclose(fp_offset);
        fclose(fp_input);
        fclose(fp_output);
        return 1;
    }

    unsigned long encrypted_count = 0;
    unsigned long skipped_unaligned_count = 0;
    unsigned long skipped_meta_count = 0;
    for (unsigned long i = 0; i < total_tensors; i++) {
        char name_line[MAX_LINE_LENGTH];
        char offset_line[MAX_LINE_LENGTH];
        char tensor_name[MAX_LINE_LENGTH];
        unsigned long long offset = 0;
        unsigned long long size = 0;

        if (!fgets(name_line, sizeof(name_line), fp_offset)) {
            fprintf(stderr, "Unexpected EOF while reading tensor name line (index %lu)\n", i + 1);
            break;
        }
        if (!fgets(offset_line, sizeof(offset_line), fp_offset)) {
            fprintf(stderr, "Unexpected EOF while reading tensor offset line (index %lu)\n", i + 1);
            break;
        }

        trim_newline(name_line);
        if (sscanf(name_line, "Tensor %*u: %255s", tensor_name) != 1) {
            fprintf(stderr, "Invalid tensor name line: %s\n", name_line);
            continue;
        }

        if (sscanf(offset_line, "offset:%llu size:%llu", &offset, &size) != 2) {
            fprintf(stderr, "Invalid tensor offset line: %s\n", offset_line);
            continue;
        }

        if (strstr(tensor_name, ".__rknpu") != NULL && ends_with(tensor_name, "__rknpu_meta")) {
            skipped_meta_count++;
            continue;
        }

        if (fseek(fp_output, (long)offset, SEEK_SET) != 0) {
            perror("Failed to seek output file");
            continue;
        }

        unsigned char *buffer = (unsigned char *)malloc((size_t)size);
        if (!buffer) {
            perror("Failed to allocate memory");
            continue;
        }

        size_t bytes_read = fread(buffer, 1, (size_t)size, fp_output);
        if (bytes_read != size) {
            fprintf(stderr, "Read error at offset %llu (expected %llu, got %zu)\n", offset, size, bytes_read);
            free(buffer);
            continue;
        }

        if (size % AES_BLOCK_SIZE != 0) {
            fprintf(stderr, "Skip tensor %s: size %llu is not aligned to %d bytes\n",
                    tensor_name, size, AES_BLOCK_SIZE);
            free(buffer);
            skipped_unaligned_count++;
            continue;
        }

        if (aes128_ecb_encrypt_inplace(buffer, (int)size, key) < 0) {
            fprintf(stderr, "AES-128-ECB encryption failed for tensor %s\n", tensor_name);
            free(buffer);
            continue;
        }


        if (fseek(fp_output, (long)offset, SEEK_SET) != 0) {
            perror("Failed to seek output file");
            free(buffer);
            continue;
        }

        size_t bytes_written = fwrite(buffer, 1, (size_t)size, fp_output);
        if (bytes_written != size) {
            fprintf(stderr, "Write error at offset %llu (expected %llu, got %zu)\n", offset, size, bytes_written);
            free(buffer);
            continue;
        }

        free(buffer);
        encrypted_count++;
    }

    fclose(fp_offset);
    fclose(fp_input);
    fclose(fp_output);
    printf("Processed tensors: total=%lu encrypted=%lu skipped_meta=%lu skipped_unaligned=%lu, output=%s\n",
           total_tensors, encrypted_count, skipped_meta_count, skipped_unaligned_count, output_file);
    return 0;
}
