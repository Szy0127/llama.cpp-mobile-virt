python3 read_tensor.py ~/models/llama-3-8b-instruct-q8_0-test.gguf > llama3.info
gcc create_enc_model.c -o create_enc_model -lcrypto
./create_enc_model llama3.info ~/models/llama-3-8b-instruct-q8_0-test.gguf ~/models/llama3-8b-enc.gguf
