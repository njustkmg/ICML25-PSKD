import numpy as np
import ot  # Python Optimal Transport (POT)

# 生成两个 (1000, 1) 的向量
np.random.seed(42)
src = np.array([[0.0],[0.0]])  # 源分布 (1000, 1)
tgt = np.array([[1.0],[1.0]])  # 目标分布 (1000, 1)

# 源分布和目标分布的权重 (均匀分布)
src_weights = np.ones(len(src)) / len(src)
tgt_weights = np.ones(len(tgt)) / len(tgt)

# 构造传输成本矩阵 C(a, b) = a - b
cost_matrix = src - tgt.T  # (1000, 1000) 矩阵

# 直接调用 POT 库的 ot.emd 函数来求解最优传输问题
optimal_transport = ot.emd(src_weights, tgt_weights, normalized_cost_matrix)

# 计算最小传输成本
minimal_cost = np.sum(optimal_transport * cost_matrix)

# 输出结果
print("Optimal transport matrix:")
print(optimal_transport)
print("Minimal transport cost:", minimal_cost)
