import paddle
paddle.enable_compat(scope={"deep_gemm"})
import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8
import numpy as np

# 1. 随机种子
np.random.seed(0)

# 2. Q, K, V维度
# 2.1 Q有2048个，K, V有4096个模拟长上下文
# 2.2 Q为128 * 64 = 8192维
# 2.3 K为128维
# 2.3 V为128维
seq_len = 2048
seq_len_kv = 4096
num_heads = 64
head_dim = 128

q_np = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
kv_np = np.random.randn(seq_len_kv, head_dim).astype(np.float32)
weights_np = np.random.randn(seq_len, num_heads).astype(np.float32)
ks_np = np.zeros(seq_len, dtype=np.int32)
ke_np = np.arange(seq_len, dtype=np.int32) + (seq_len_kv - seq_len)

# 3. Q, K, V张量
# 3.1 q.shape = [2048, 64, 128]，转fp8
q = paddle.to_tensor(q_np, dtype='bfloat16')
q_fp8 = paddle.cast(q, 'float8_e4m3fn')

# 3.2 k.shape = [4096, 128]，转fp8
kv = paddle.to_tensor(kv_np, dtype='bfloat16')
kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)

# 3.3 weights.shape = [2048, 64]
weights = paddle.to_tensor(weights_np, dtype='float32')

# 3.4 ks.shape = [2048]，元素值为0
# 3.4 ks.shape = [2048]，元素值为2048, 2049, ... 4095
ks = paddle.to_tensor(ks_np, dtype='int32')
ke = paddle.to_tensor(ke_np, dtype='int32')

# 4. 计算
logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)

# 5. 打印张量
output_file = "/root/paddlejob/share-storage/gpfs/system-public/changwenbin/yangrongjin/Log/2.28/paddle.txt"
logits_cpu = logits.cpu().astype('float32')
with open(output_file, "w") as f:
    for row in logits_cpu.numpy():
        f.write(" ".join([f"{v:.6f}" for v in row.tolist()]))
        f.write("\n")
print(f"Logits 已保存到 {output_file}")