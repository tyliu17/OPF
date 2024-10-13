import numpy as np
from pypower.api import case9, ppoption, runopf, runpf, loadcase
from pypower.idx_bus import PD, QD
from pypower.idx_gen import PG
from pypower.idx_brch import RATE_A, PF
from custom_functions import runpf as crunpf

# 載入案例
ppc = case9()

# 設置 PYPOWER 選項
ppopt = ppoption(
    OUT_ALL=1,
    # OUT_ALL 輸出所有結果：
    PF_ALG=2,
    # PF_ALG 電力流計算的算法：
        # 1 - Newton's method,
        # 2 - Fast-Decoupled (XB version),
        # 3 - Fast-Decoupled (BX version),
        # 4 - Gauss Seidel,
)

n = 1  # 數據集數量

N = len(ppc['bus'])  # 匯流排數量
L = len(ppc['branch'])  # 線路數量

# 非零負載的索引
index = np.where(ppc['bus'][:, PD] != 0)[0]
n_load = len(index)

load_file0_real = np.zeros((N, n))
load_file0_reac = np.zeros((N, n))

# 設置線路限制
ppc['branch'][:, RATE_A] *= 1.1  # 增加 10% 的流量限制
p_max = ppc['branch'][:, RATE_A]

# 調整負載
ppc['bus'][:, PD] *= 0.95  # 減少 5% 的負載

ppc2 = ppc.copy()  # 儲存參考系統

# 生成隨機負載並運行最優潮流
result = []
for k in range(n):
    r0 = np.random.uniform(-0.05, 0.05, N)  # 負載 p
    r1 = np.random.uniform(-0.05, 0.05, N)  # 負載 q
    
    # 有功功率
    ppc['bus'][:, PD] = ppc2['bus'][:, PD] * (1 + r0)
    # 無功功率
    ppc['bus'][:, QD] = ppc2['bus'][:, QD] * (1 + r1 * 0.2)
    
    load_file0_real[:, k] = ppc['bus'][:, PD]
    load_file0_reac[:, k] = ppc['bus'][:, QD]
    
    if np.min(load_file0_real[:, k]) < -300:
        result.append({'success': 0})
        print('負載過小')
    else:
        try:
            # res = runopf(ppc, ppopt, fname='result.txt')
            # res = runpf(ppc, ppopt, fname='result.txt')[0]
            cres = crunpf.runpf(ppc, ppopt, fname='result.txt')[0]

            result.append(cres)
            if cres['success'] == 0:
                print(f"OPF 失敗: {res.get('et', 'Unknown error')}")
        except Exception as e:
            print(f"OPF 執行錯誤: {str(e)}")
            result.append({'success': 0})
    
    ppc = ppc2.copy()  # 在修改後重置 ppc

# 計算有效數據集數量
index = [i for i, res in enumerate(result) if res['success']]
idx_count = len(index)
index0 = np.zeros(n)
index0[index] = 1

if idx_count == 0:
    print("所有模擬都失敗了。")
else:
    # 初始化電壓、匯流排角度和線路潮流
    v = np.zeros((N, idx_count))
    theta = np.zeros((N, idx_count))
    f = np.zeros((L, idx_count))
    load_file_real = np.zeros((N, idx_count))
    load_file_reac = np.zeros((N, idx_count))

    for i, idx in enumerate(index):
        v[:, i] = result[idx]['bus'][:, 7]
        theta[:, i] = result[idx]['bus'][:, 8]
        f[:, i] = result[idx]['branch'][:, PF]
        load_file_real[:, i] = load_file0_real[:, idx]
        load_file_reac[:, i] = load_file0_reac[:, idx]

    # 線路約束
    line_index_lo = np.zeros((L, idx_count))
    line_index_up = np.zeros((L, idx_count))
    for i in range(idx_count):
        line_index_up[:, i] = (p_max - f[:, i] < 1e-3) & (f[:, i] > 0)
        line_index_lo[:, i] = (p_max + f[:, i] < 1e-3) & (f[:, i] < 0)

    # 匯流排約束
    n_gen = len(ppc2['gen'])
    n_bus = len(ppc2['bus'])
    gen_idx0 = ppc2['gen'][:, 0].astype(int)
    gen_idx = [np.where(ppc['bus'][:, 0] == gen_bus)[0][0] for gen_bus in gen_idx0]
    gen_lim = ppc2['gen'][:, [8, 9]]

    # 從結果中獲取數據
    gen_data0 = np.zeros((n_gen, idx_count))
    gen_data = np.zeros((n_bus, idx_count))
    for i, idx in enumerate(index):
        gen_data0[:, i] = result[idx]['gen'][:, PG]

    for i, bus_idx in enumerate(gen_idx):
        gen_data[bus_idx, :] += gen_data0[i, :]

    print(f'有效樣本: {idx_count}')
    line = line_index_lo + line_index_up
    lsum = np.sum(line, axis=0)
    print('最大活躍線路數:')
    print(np.max(lsum))
    print('活躍比率:')
    print(np.sum(line) / L / idx_count)