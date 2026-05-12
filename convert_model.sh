python3 read_tensor.py ~/models/llama-3-8b-instruct-q8_0-test.gguf > llama3.info
gcc create_enc_model.c src/chacha12.c
./a.out llama3.info ~/models/llama-3-8b-instruct-q8_0-test.gguf ~/models/llama3-8b-enc.gguf
