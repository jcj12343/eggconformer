import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ===================== 全局根目录配置 =====================
SCRIPT_PATH = os.path.abspath(__file__)
CODE_DIR = os.path.dirname(SCRIPT_PATH)
ROOT = os.path.dirname(CODE_DIR)
# 数据集路径
DATA_FOLDER = os.path.join(ROOT, "data", "processed", "norm_data")
# 输出目录
CHECKPOINT_ROOT = os.path.join(ROOT, "checkpoint")
LOG_ROOT = os.path.join(ROOT, "logs")
RESULT_ROOT = os.path.join(ROOT, "results")
REPORT_ROOT = os.path.join(ROOT, "report")
SUMMARY_DIR = os.path.join(RESULT_ROOT, "summary")
# 自动创建文件夹
for dir_path in [CHECKPOINT_ROOT, LOG_ROOT, RESULT_ROOT, REPORT_ROOT, SUMMARY_DIR, DATA_FOLDER]:
    os.makedirs(dir_path, exist_ok=True)

SUBJECT_LIST = [f"A{i:02d}" for i in range(1, 10)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_MAX = 100
BATCH_SIZE = 32
LR_BACKBONE = 3e-5
LR_HEAD = 8e-5
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 10
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

CLASS_NAMES = ["Left", "Right", "Feet", "Tongue"]
N_CH = 22
N_TIME = 626
N_CLS = 4

# CNN+Conformer超参
F1 = 8
D = 2
TEMP_KERNEL = 32
POOL_SIZE = 4
DROP_RATE = 0.1
D_MODEL = 128
N_HEAD = 8
TRANS_LAYERS = 4
FFN_DIM = 256
CONF_KERNEL = 31
GRAD_CLIP = 1.0

all_test_label = []
all_test_pred = []
metric_recorder = []

# 日志配置
logging.basicConfig(
    filename=os.path.join(LOG_ROOT, "train_fixed.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 加权FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return torch.mean(loss)

# 数据集+多维度数据增强
class EEGDataset(Dataset):
    def __init__(self, data, label, aug=False):
        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(label)
        self.aug = aug
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx].unsqueeze(0)  # [1, C, T]
        y = self.label[idx]
        if self.aug:
            # 高斯噪声
            x += torch.randn_like(x) * 0.002
            # 时域随机偏移
            shift = np.random.randint(-3, 3)
            if shift > 0:
                x = torch.cat([torch.zeros_like(x[:, :, :shift]), x[:, :, :-shift]], dim=-1)
            elif shift < 0:
                shift_abs = abs(shift)
                x = torch.cat([x[:, :, shift_abs:], torch.zeros_like(x[:, :, :shift_abs])], dim=-1)
            # 幅值缩放
            scale = np.random.uniform(0.85, 1.15)
            x = x * scale
            # 随机通道掩码增强
            mask_channel = np.random.choice(N_CH, size=3, replace=False)
            x[:, mask_channel, :] *= 0.0
        return x, y

# 随机打乱划分数据集，消除时序漂移
def load_subject_data(sub_id):
    data = np.load(os.path.join(DATA_FOLDER, f"{sub_id}T_data.npy"))
    label = np.load(os.path.join(DATA_FOLDER, f"{sub_id}T_label.npy"))
    perm = np.random.permutation(len(data))
    data = data[perm]
    label = label[perm]
    total = len(data)
    tr_end = int(total * 0.70)
    val_end = int(total * 0.85)
    x_tr, y_tr = data[:tr_end], label[:tr_end]
    x_val, y_val = data[tr_end:val_end], label[tr_end:val_end]
    x_ts, y_ts = data[val_end:], label[val_end:]
    info = f"{sub_id} | Train:{len(x_tr)} Val:{len(x_val)} Test:{len(x_ts)}"
    print(info)
    logger.info(info)
    return x_tr, y_tr, x_val, y_val, x_ts, y_ts

# 一轮训练/验证通用函数
def run_epoch(model, criterion, opt, loader, train_mode):
    model.train() if train_mode else model.eval()
    loss_sum = 0.0
    pred_list = []
    label_list = []
    with torch.set_grad_enabled(train_mode):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            logit = model(batch_x)
            loss = criterion(logit, batch_y)
            if train_mode:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                opt.step()
            loss_sum += loss.item()
            pred = torch.argmax(logit, dim=-1)
            pred_list.extend(pred.cpu().numpy())
            label_list.extend(batch_y.cpu().numpy())
    avg_loss = loss_sum / len(loader)
    acc = accuracy_score(label_list, pred_list)
    kappa = cohen_kappa_score(label_list, pred_list)
    return avg_loss, acc, kappa, pred_list, label_list

# 绘制loss/acc曲线
def plot_curve(save_path, tr_loss, val_loss, tr_acc, val_acc):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    ax1.plot(tr_loss, label="Train Loss")
    ax1.plot(val_loss, label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(tr_acc, label="Train Acc")
    ax2.plot(val_acc, label="Val Acc")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# 绘制混淆矩阵
def plot_confusion_matrix(save_path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=13)
    ax.set_xticks(np.arange(N_CLS))
    ax.set_yticks(np.arange(N_CLS))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    plt.colorbar(im, ax=ax)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# 导入自研EEGConformer
from EEGTransformer import EEGCNNTransformer

# 单被试完整训练流程
def train_single_subject(sub_id):
    sub_ckpt = os.path.join(CHECKPOINT_ROOT, sub_id)
    sub_res = os.path.join(RESULT_ROOT, sub_id)
    os.makedirs(sub_ckpt, exist_ok=True)
    os.makedirs(sub_res, exist_ok=True)
    best_ckpt_path = os.path.join(sub_ckpt, "best_fixed.pth")

    x_tr, y_tr, x_val, y_val, x_ts, y_ts = load_subject_data(sub_id)
    ds_tr = EEGDataset(x_tr, y_tr, aug=True)
    ds_val = EEGDataset(x_val, y_val, aug=False)
    ds_ts = EEGDataset(x_ts, y_ts, aug=False)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    dl_ts = DataLoader(ds_ts, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化EEGConformer
    model = EEGCNNTransformer(
        n_channels=N_CH,
        n_timepoints=N_TIME,
        n_classes=N_CLS,
        F1=F1,
        D=D,
        temporal_kernel=TEMP_KERNEL,
        pool_size=POOL_SIZE,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=TRANS_LAYERS,
        dim_feedforward=FFN_DIM,
        conf_kernel=CONF_KERNEL,
        dropout=DROP_RATE
    ).to(DEVICE)

    # 四类权重全部均等 [1.0,1.0,1.0,1.0]
    cls_weight = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(DEVICE)
    criterion = FocalLoss(alpha=cls_weight, gamma=0.6)

    # 分层学习率
    backbone_params = list(model.temp_conv.parameters()) + list(model.spatial_conv.parameters()) + list(model.bn_cnn.parameters()) + list(model.channel_att.parameters())
    head_params = list(model.proj.parameters()) + list(model.conformer_encoder.parameters()) + list(model.head.parameters()) + list(model.pos_enc.parameters())
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    best_val_acc = 0.0
    train_loss_rec, val_loss_rec = [], []
    train_acc_rec, val_acc_rec = [], []

    start_log = f"\n==== Start Train Fixed {sub_id} ===="
    print(start_log)
    logger.info(start_log)

    for epoch in range(1, EPOCHS_MAX + 1):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, criterion, optimizer, dl_tr, train_mode=True)
        val_loss, val_acc, _, _, _ = run_epoch(model, criterion, optimizer, dl_val, train_mode=False)
        scheduler.step()

        train_loss_rec.append(tr_loss)
        val_loss_rec.append(val_loss)
        train_acc_rec.append(tr_acc)
        val_acc_rec.append(val_acc)

        epoch_log = f"Epoch {epoch:02d} | Train Acc:{tr_acc:.4f} Val Acc:{val_acc:.4f}"
        print(epoch_log)
        logger.info(f"{sub_id} {epoch_log}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)

    # 保存训练曲线
    plot_curve(os.path.join(sub_res, "curve_fixed.png"), train_loss_rec, val_loss_rec, train_acc_rec, val_acc_rec)
    # 测试集推理
    _, test_acc, test_kappa, test_pred, test_true = run_epoch(model, criterion, optimizer, dl_ts, train_mode=False)
    # 混淆矩阵
    plot_confusion_matrix(os.path.join(sub_res, "cm_fixed.png"), test_true, test_pred)
    # 输出分类报告
    report = classification_report(test_true, test_pred, target_names=CLASS_NAMES, digits=4)
    with open(os.path.join(sub_res, "result_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best Val Acc: {best_val_acc:.4f}\nTest Acc: {test_acc:.4f}\nTest Kappa: {test_kappa:.4f}\n\n")
        f.write(report)
    all_test_label.extend(test_true)
    all_test_pred.extend(test_pred)
    metric_recorder.append([best_val_acc, test_acc, test_kappa])
    finish_log = f"{sub_id} Fixed Train Finished | Test Acc = {test_acc:.4f}"
    print(finish_log)
    logger.info(finish_log)

# 所有被试结果汇总
def summary_global():
    print("\n======== Global Fixed Result ========")
    logger.info("======== Global Fixed Result ========")
    metric_arr = np.array(metric_recorder)
    mean_val = metric_arr[:,0].mean()
    mean_test = metric_arr[:,1].mean()
    mean_kappa = metric_arr[:,2].mean()
    total_acc = accuracy_score(all_test_label, all_test_pred)
    total_kappa = cohen_kappa_score(all_test_label, all_test_pred)
    report = classification_report(all_test_label, all_test_pred, target_names=CLASS_NAMES, digits=4)
    total_txt = os.path.join(SUMMARY_DIR, "total_result_fixed.txt")
    with open(total_txt, "w", encoding="utf-8") as f:
        f.write(f"Global Test Acc: {total_acc:.4f} | Global Kappa:{total_kappa:.4f}\n")
        f.write(f"Avg Val Acc:{mean_val:.4f} | Avg Test Acc:{mean_test:.4f} | Avg Kappa:{mean_kappa:.4f}\n\n")
        f.write(report)
    plot_confusion_matrix(os.path.join(SUMMARY_DIR, "global_cm_fixed.png"), all_test_label, all_test_pred)
    final_log = f"Global Acc:{total_acc:.4f}  Mean Single Subject Test Acc:{mean_test:.4f}"
    print(final_log)
    logger.info(final_log)
    import shutil
    shutil.copy(total_txt, os.path.join(REPORT_ROOT, "global_result_fixed_backup.txt"))
    shutil.copy(os.path.join(SUMMARY_DIR, "global_cm_fixed.png"), os.path.join(REPORT_ROOT, "global_cm_fixed_backup.png"))

# 程序入口
if __name__ == "__main__":
    for sub in SUBJECT_LIST:
        train_single_subject(sub)
    summary_global()
    print("All Subject Fixed Train Complete!")