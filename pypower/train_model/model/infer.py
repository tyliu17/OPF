import torch
import json
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
import torch.nn.functional as F
import model as M
import os
import pypower.idx_brch as pybrch
import pypower.idx_bus as pybus

json_file_path = 'dataset/case9_data/power_flow_case_200.json'  # 替換為您的 JSON 檔案路徑
model_path = 'runs/opf_experiment_20241014-170825/best_model.pth'  # 替換為您的模型檔案路徑

# 1. 定義異構圖的元數據（Metadata）
# 這應該與訓練時的 metadata 完全一致
metadata = (
    ['bus'],
    [
        ('bus', 'connected_to', 'bus'),
    ]
)

# 2. 讀取 JSON 測試資料
with open(json_file_path, 'r') as f:
    test_data = json.load(f)

# 3. 提取數據並轉換為 PyTorch 張量
bus_data = torch.tensor(test_data['ppc']['bus'], dtype=torch.float32)     # bus 數據
gen_data = torch.tensor(test_data['ppc']['gen'], dtype=torch.float32)     # gen 數據
branch_data = torch.tensor(test_data['ppc']['branch'], dtype=torch.float32)  # branch 數據

# 4. 構建異構圖的特徵字典（x_dict）和邊索引字典（edge_index_dict）
x_dict = {
    'bus': bus_data,
    'gen': gen_data,
    'branch': branch_data
}

# 邊連接
from_bus = []
to_bus = []
for branch in branch_data:
    from_bus.append(int(branch[pybrch.F_BUS]) - 1)
    to_bus.append(int(branch[pybrch.T_BUS]) - 1)
edge_index_connected_to = torch.tensor([from_bus, to_bus], dtype=torch.long)

# 5. 創建 HeteroData 物件
data = HeteroData()
data['bus'].x = bus_data
data['gen'].x = gen_data
data['branch'].x = branch_data
data[('bus', 'connected_to', 'bus')].edge_index = edge_index_connected_to

# 6. 設置設備 (GPU 或 CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 7. 初始化並包裝模型
base_model = M.Model()
model = to_hetero(base_model, metadata)
model = model.to(device)
model.eval()

# 8. 載入最佳模型權重
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 9. 將數據移動到相同設備
data = data.to(device)

# 10. 進行推斷
with torch.no_grad():
    predictions = model(data.x_dict, data.edge_index_dict)

# 11. 打印推斷結果
total_loss = 0
for idx, branch in enumerate(predictions['bus'].cpu().numpy()):
    # print("推斷結果（'branch' 的 PF值",pybrch.PF ,"）:\n", predictions['bus'].cpu().numpy())
    print("實際值：", test_data['res']['branch'][idx][pybrch.PF], "\t推斷值：", branch[0], end="\t")

    actual_tensor = torch.tensor(int(test_data['res']['branch'][idx][pybrch.PF]), dtype=torch.float32)
    predicted_tensor = torch.tensor(int(branch[0]), dtype=torch.float32)
    
    loss = F.l1_loss(predicted_tensor, actual_tensor)
    print(int(loss))
    total_loss += loss.item()

print(total_loss/len(test_data['res']['branch']))
