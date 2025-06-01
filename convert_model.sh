python3 read_tensor.py ~/qwen2.5-0.5b-instruct-q4_0.gguf > tensor_info
gcc create_enc_model.c src/chacha12.c
./a.out tensor_info ~/qwen2.5-0.5b-instruct-q4_0.gguf ~/qwen2.5-0.5B-enc.gguf
