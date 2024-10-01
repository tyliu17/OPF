from pypower.api import runpf as prunpf
from scipy.optimize import minimize
import numpy as np

# 目标函数：最小化发电成本
def cost_function(Pg, gencost):
    return sum([gencost[i, 0] * Pg[i]**2 + gencost[i, 1] * Pg[i] + gencost[i, 2] for i in range(len(Pg))])

# 负荷平衡约束 (等式约束)
def balance_constraint(Pg, P_load):
    return np.sum(Pg) - P_load  # 总发电量和总负荷平衡

# 发电机最小/最大出力约束 (不等式约束)
def gen_min_max_constraints(Pg, Pg_min, Pg_max):
    return np.concatenate([Pg - Pg_min, Pg_max - Pg])

def runpf(mpc, ppopt, fname):
    # 提取 baseMVA 和 bus 数据
    baseMVA = mpc['baseMVA']
    bus = mpc['bus']

    # 假设的发电成本系数 (a, b, c)
    gencost = np.array([
        [0.11, 5, 150],    # 生成器 1
        [0.085, 1.2, 600], # 生成器 2
        [0.1225, 1, 335]   # 生成器 3
    ])

    # 发电机出力约束
    Pg_min = np.array([0, 0, 0])   # 最小出力
    Pg_max = np.array([300, 300, 300])  # 最大出力
    P_load = bus[bus[:, 1] == 2, 2].sum()  # 计算负荷（负荷总和）

    # 初始发电机出力
    Pg_initial = np.array([100, 100, 100])

    # 定义约束：一个等式约束（负荷平衡），和一个不等式约束（最小/最大出力）
    cons = [{'type': 'eq', 'fun': balance_constraint, 'args': (P_load,)},  # 负荷平衡约束
            {'type': 'ineq', 'fun': gen_min_max_constraints, 'args': (Pg_min, Pg_max)}]  # 最小/最大出力约束

    # 优化
    result = minimize(cost_function, Pg_initial, args=(gencost,), constraints=cons, 
                      bounds=[(Pg_min[i], Pg_max[i]) for i in range(len(Pg_initial))])

    print("result\n",result)
    # 输出结果
    if result.success:
        print("Optimal Generator Outputs (Pg):", result.x)
        print("Total Cost:", cost_function(result.x, gencost))
    else:
        print("Optimization failed:", result.message)

    
    
    return prunpf(mpc, ppopt, fname=fname)
    
