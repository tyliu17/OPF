import numpy as np
from pypower.api import case9, ppoption, runopf, runpf, loadcase
from pypower.idx_bus import PD, QD
from pypower.idx_gen import PG
from pypower.idx_brch import RATE_A, PF
import json
import os
import re

# 載入案例
ppc = case9()

# 設置 PYPOWER 選項
ppopt = ppoption(
    OUT_ALL=0,
    # OUT_ALL 輸出所有結果：
    PF_ALG=2,
    # PF_ALG 電力流計算的算法：
        # 1 - Newton's method,
        # 2 - Fast-Decoupled (XB version),
        # 3 - Fast-Decoupled (BX version),
        # 4 - Gauss Seidel,
)

n = 4800  # 數據集數量（根據需要調整）

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

# 定義一個函數來轉換 ppc 和 res 為可序列化的格式
def serialize_ppc_res(ppc, res):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return str(obj)  # 其他類型轉換為字串

    serialized_ppc = convert(ppc)
    serialized_res = convert(res)
    return {
        'ppc': serialized_ppc,
        'res': serialized_res
    }

# 創建儲存 JSON 文件的目錄
output_dir = '../model/dataset/case9_data'
os.makedirs(output_dir, exist_ok=True)

# 取數字
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
max_number = 0
if files:
    max_number = max(extract_number(f) for f in files if extract_number(f) is not None)

# 生成隨機負載並運行最優潮流
for k in range(max_number, max_number + n):
    r0 = np.random.uniform(-0.05, 0.05, N)  # 負載 p
    r1 = np.random.uniform(-0.05, 0.05, N)  # 負載 q

    # 有功功率
    ppc['bus'][:, PD] = ppc2['bus'][:, PD] * (1 + r0)
    # 無功功率
    ppc['bus'][:, QD] = ppc2['bus'][:, QD] * (1 + r1 * 0.2)

    # 儲存負載
    load_number = k - max_number
    load_file0_real[:, load_number] = ppc['bus'][:, PD]
    load_file0_reac[:, load_number] = ppc['bus'][:, QD]

    if np.min(load_file0_real[:, load_number]) < -300:
        result_entry = {'success': 0, 'message': '負載過小'}
        serialized_data = serialize_ppc_res(ppc, result_entry)
        filename = f'power_flow_case_{k+1}.json'
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, ensure_ascii=False, indent=4)
        print(f'案例 {k+1}: 負載過小，已儲存為 {filename}')
    else:
        try:
            # res = runopf(ppc, ppopt, fname='result.txt')
            res, success = runpf(ppc, ppopt)
            # runpf 返回一個元組 (res, success)，確保正確獲取結果
            if isinstance(res, tuple) and len(res) > 1:
                res = res[0]
            
            serialized_data = serialize_ppc_res(ppc, res)
            filename = f'power_flow_case_{k+1}.json'
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized_data, f, ensure_ascii=False, indent=4)
            print(f'案例 {k+1}: 已成功儲存為 {filename}')

            if res.get('success', 0) == 0:
                print(f"案例 {k+1} OPF 失敗: {res.get('et', 'Unknown error')}")
        except Exception as e:
            print(f"案例 {k+1} OPF 執行錯誤: {str(e)}")
            result_entry = {'success': 0, 'message': str(e)}
            serialized_data = serialize_ppc_res(ppc, result_entry)
            filename = f'power_flow_case_{k+1}.json'
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized_data, f, ensure_ascii=False, indent=4)
            print(f'案例 {k+1}: 錯誤已儲存為 {filename}')

    ppc = ppc2.copy()  # 在修改後重置 ppc

