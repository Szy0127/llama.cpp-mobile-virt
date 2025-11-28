python3 read_tensor.py ~/models/qwen2.5-3b-q4.gguf > tensor_info
gcc create_enc_model.c src/chacha12.c
./a.out tensor_info ~/models/qwen2.5-3b-q4.gguf ~/models/qwen2.5-3b-q4-enc.gguf
