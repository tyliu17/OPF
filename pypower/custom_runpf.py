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

# 线路潮流约束 (不等式约束)
def branch_flow_constraints(Pg, bus, branch, baseMVA):
    # 提取线路参数
    fbus = branch[:, 0].astype(int) - 1  # 起始母线
    tbus = branch[:, 1].astype(int) - 1  # 终止母线
    x = branch[:, 3]  # 线路电抗
    rateA = branch[:, 5]  # 线路的最大传输容量

    # 假设各母线电压相角为0（简化），实际可以结合潮流计算
    # Pg 代表发电量，bus[:, 2] 是负荷功率需求
    V_angle = np.zeros(len(bus))  # 初始化母线电压相角

    # 计算各线路的潮流（使用潮流公式 P_ij = (V_i^2 - V_i * V_j * cos(θ_i - θ_j)) / x_ij）
    # 由于简化，这里只使用线性近似潮流公式 P_ij ≈ (θ_i - θ_j) / x_ij
    P_branch_flow = (V_angle[fbus] - V_angle[tbus]) / x

    # 添加约束：潮流不能超过线路的传输容量（rateA）
    return np.concatenate([P_branch_flow - rateA, -P_branch_flow - rateA])

# 输出潮流信息
def print_flow_information(branch, P_branch_flow):
    print("Line Flows (MW):")
    for i, flow in enumerate(P_branch_flow):
        print(f"Line {i+1} Flow: {flow:.4f} MW (RateA: {branch[i, 5]} MW)")

def runpf(mpc, ppopt, fname):
    # 提取 baseMVA 和 bus 数据
    version = mpc['version']
    baseMVA = mpc['baseMVA']
    bus = mpc['bus']
    gen = mpc['gen']
    branch = mpc['branch']
    gencost = mpc['gencost']
    
    """
    # 假设的发电成本系数 (a, b, c)
    gencost = np.array([
        [0.11, 5, 150],    # 生成器 1
        [0.085, 1.2, 600], # 生成器 2
        [0.1225, 1, 335]   # 生成器 3
    ])
    """

    # 发电机出力约束
    Pg_min = np.array([0, 0, 0])   # 最小出力
    Pg_max = np.array([300, 300, 300])  # 最大出力
    P_load = bus[bus[:, 1] == 2, 2].sum()  # 计算负荷（负荷总和）

    # 初始发电机出力
    Pg_initial = np.array([100, 100, 100])

     # 定义约束：负荷平衡约束，发电机出力上下限，线路潮流约束
    cons = [{'type': 'eq', 'fun': balance_constraint, 'args': (P_load,)},  # 负荷平衡约束
            {'type': 'ineq', 'fun': gen_min_max_constraints, 'args': (Pg_min, Pg_max)},  # 发电机最小/最大出力约束
            {'type': 'ineq', 'fun': branch_flow_constraints, 'args': (bus, branch, baseMVA)}]  # 线路潮流约束

    # 优化
    result = minimize(cost_function, Pg_initial, args=(gencost,), constraints=cons, 
                      bounds=[(Pg_min[i], Pg_max[i]) for i in range(len(Pg_initial))])
    print("result : " , result)
    """
    print("result\n",result)

    # 输出结果
    if result.success:
        print("Optimal Generator Outputs (Pg):", result.x)
        print("Total Cost:", cost_function(result.x, gencost))
    else:
        print("Optimization failed:", result.message)
    """
    # 输出结果
    if result.success:
        Pg_optimal = result.x
        total_cost = cost_function(Pg_optimal, gencost)
        print("Optimal Generator Outputs (Pg):", Pg_optimal)
        print("Total Cost:", total_cost)
        
        # 计算潮流
        fbus = branch[:, 0].astype(int) - 1
        tbus = branch[:, 1].astype(int) - 1
        x = branch[:, 3]
        V_angle = np.zeros(len(bus))  # 简化相角
        P_branch_flow = (V_angle[fbus] - V_angle[tbus]) / x

        # 输出潮流信息
        print_flow_information(branch, P_branch_flow)

        # 输出发电机约束情况
        gen_constraints = (Pg_optimal <= Pg_min) | (Pg_optimal >= Pg_max)
        print("Generator Constraints Reached:", gen_constraints)

        # 输出线路潮流约束情况
        line_constraints = (P_branch_flow >= branch[:, 5]) | (-P_branch_flow >= branch[:, 5])
        print("Line Flow Constraints Reached:", line_constraints)

    else:
        print("Optimization failed:", result.message)
    print("===================================")
    return prunpf(mpc, ppopt, fname=fname)
    
