import gguf
import os
import sys

'''
gguf qwen2.5-0.5B 3-1.7B 3-0.5B

1. tensor.size = 512n
2. tensor data 存储连续 中间不存在其他metadata 即只要第一个512对齐 后续都512对齐
3. 整个tensor data区域延续到gguf文件末尾 即不影响其他data读取
'''



# 加载 .gguf 文件
reader = gguf.GGUFReader(sys.argv[1], "r")
file_size = os.path.getsize(sys.argv[1])
#print(file_size)


# 输出模型信息
#print(f"Architecture: {reader.architecture}")
#print(f"Tensor count: {len(reader.tensors)}")
#print("-" * 60)

# 遍历每个 tensor 并输出大小、offset 等信息

offsets = []
size = 0
last_size = 0
print(len(reader.tensors))
for i, tensor in enumerate(reader.tensors):
    print(f"Tensor {i+1}: {tensor.name}")
    #print(f"  Shape: {tensor.shape}")
    #print(f"  Data offset: {tensor.data.offset}")
    #print(f"  Data offset: {tensor.data_offset}")
    #print(f"  Data size (bytes): {tensor.data.nbytes}")
    #print(f"offset:{tensor.data_offset} size:{tensor.data.nbytes} {tensor.name}")
    print(f"offset:{tensor.data_offset} size:{tensor.data.nbytes}")
    size += tensor.data.nbytes
    last_size = tensor.data.nbytes
    offsets.append(tensor.data_offset)

    #if tensor.data.nbytes % 512 !=0:
        #print("not 512!!!")
    #print("-" * 40)

print(size)
#print(offsets[-1], offsets[-1]+last_size)
#print(offsets[-1]+last_size-offsets[0],size)

