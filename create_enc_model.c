/*
 * gcc create_enc_model.c src/chacha20.c
 * ./a.out tensor_info model.gguf model_enc.gguf
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "src/chacha20.h"

#define MAX_LINE_LENGTH 256

typedef struct {
    unsigned long offset;
    unsigned long size;
} MemoryBlock;

    uint8_t key[] = {
        0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b,
        0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13,
        0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b,
        0x1c, 0x1d, 0x1e, 0x1f
    };

    uint8_t nonce[] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00
    };
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

    MemoryBlock blocks[500];
    int count = 0;
    char line[MAX_LINE_LENGTH];

    while (fgets(line, sizeof(line), fp_offset)&& count < 291) {
        char *offset_str = strstr(line, "offset:");
        if (!offset_str) offset_str = strstr(line, ".offset:");
        if (offset_str) {
            unsigned long offset, size;
            if (sscanf(offset_str, "offset:%lu size:%lu", &offset, &size) == 2) {
                blocks[count].offset = offset;
                blocks[count].size = size;
                count++;
            }
        }
    }
    fclose(fp_offset);
    FILE *fp_input = fopen(input_file, "rb");
    if (!fp_input) {
        perror("Failed to open input file");
        return 1;
    }

    FILE *fp_output = fopen(output_file, "rb+");  
    if (!fp_output) {
        perror("Failed to open output file");
        fclose(fp_input);
        return 1;
    }

    for (int i = 0; i < count; i++) {
        unsigned long offset = blocks[i].offset;
        unsigned long size = blocks[i].size;
        printf("%ld %ld\n", offset,size);

        if (fseek(fp_input, offset, SEEK_SET) != 0) {
            perror("Failed to seek input file");
            continue;
        }

        unsigned char *buffer = (unsigned char *)malloc(size);
        if (!buffer) {
            perror("Failed to allocate memory");
            continue;
        }

        size_t bytes_read = fread(buffer, 1, size, fp_input);
        if (bytes_read != size) {
            fprintf(stderr, "Read error at offset %lu (expected %lu, got %zu)\n", offset, size, bytes_read);
            free(buffer);
            continue;
        }
        ChaCha20XOR(key, 1, nonce, buffer,buffer, size);

        if (fseek(fp_output, offset, SEEK_SET) != 0) {
            perror("Failed to seek output file");
            free(buffer);
            continue;
        }

        size_t bytes_written = fwrite(buffer, 1, size, fp_output);
        if (bytes_written != size) {
            fprintf(stderr, "Write error at offset %lu (expected %lu, got %zu)\n", offset, size, bytes_written);
        }

        free(buffer);
    }

    fclose(fp_input);
    fclose(fp_output);
    printf("Processed %d blocks. Output written to %s\n", count, output_file);
    return 0;
}
