import torch
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
q = torch.tensor(q_np, device='cuda', dtype=torch.bfloat16)
q_fp8 = q.to(torch.float8_e4m3fn)

# 3.2 k.shape = [4096, 128]，转fp8
kv = torch.tensor(kv_np, device='cuda', dtype=torch.bfloat16)
kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)

# 3.3 weights.shape = [2048, 64]
weights = torch.tensor(weights_np, device='cuda', dtype=torch.float32)

# 3.4 ks.shape = [2048]，元素值为0
# 3.4 ks.shape = [2048]，元素值为2048, 2049, ... 4095
ks = torch.tensor(ks_np, device='cuda', dtype=torch.int)
ke = torch.tensor(ke_np, device='cuda', dtype=torch.int)

# 4. 计算
logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)

# 5. 打印张量
output_file = "/root/paddlejob/share-storage/gpfs/system-public/changwenbin/yangrongjin/Log/2.28/tensor.txt"
logits_cpu = logits.detach().cpu().float()
with open(output_file, "w") as f:
    for row in logits_cpu:
        f.write(" ".join([f"{v:.6f}" for v in row.tolist()]))
        f.write("\n")
print(f"Logits 已保存到 {output_file}")