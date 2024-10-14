import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.datasets import opf
from torch_geometric.loader import DataLoader
import model as M
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
import os
import argparse
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.custom_opf_dataset import CustomOPFDataset

# 讀取 config.json 文件
with open('config.json', 'r') as f:
    config = json.load(f)

# 取得超參數
epochs = config.get("epochs", 10)
batch_size = config.get("batch_size", 64)
lr = config.get("learning_rate", 0.001)

# 設置隨機種子
torch.manual_seed(42)

# 加載數據集
dataset = CustomOPFDataset(root='./dataset') 

total_length = len(dataset)
train_length = int(total_length * 0.8)
val_length = total_length - train_length
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_length, val_length])

# 創建數據加載器
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# 初始化模型
data = train_ds[0]
model = to_hetero(M.Model(), data.metadata())

# 檢查是否有可用的GPU (apple silicon使用MPS)
# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 初始化優化器和學習率調度器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

# 設置TensorBoard
log_dir = "runs/opf_experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

# 驗證函數
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            # 使用 mse_Loss 計算損失
            loss = F.mse_loss(out['bus'], batch['bus', 'connected_to', 'bus'].y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# 訓練循環
best_val_loss = float('inf')
for epoch in tqdm(range(epochs), desc="Epochs"):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.mse_loss(out['bus'], batch['bus', 'connected_to', 'bus'].y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_train_loss = epoch_loss / len(train_loader)
    val_loss = validate(model, val_loader)
    
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 保存最佳模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }, os.path.join(log_dir, 'best_model.pth'))

# 保存最終模型和配置
final_model_path = os.path.join(log_dir, 'final_model.pth')
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'val_loss': val_loss,
}, final_model_path)

# 儲存超參數設定
config = {
    'epochs': epochs,
    'batch_size': batch_size,
    'learning_rate': lr,
    'model_architecture': str(model),
    'optimizer': str(optimizer),
    'scheduler': str(scheduler),
}

with open(os.path.join(log_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

print("訓練結束.")
print(f"模型儲存位置 {final_model_path}")
print(f"TensorBoard logs 位置 {log_dir}")
