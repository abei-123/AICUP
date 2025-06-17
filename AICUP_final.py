# -*- coding: utf-8 -*-
"""
Swing-Action Attribute Prediction ‖ Enhanced 1-D CNN + Transformer
改进：预训练二元分类头，BatchNorm+Dropout，FocalLoss，动态权重，早停
"""
import argparse, random, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from focal_loss import FocalLoss

# 数值打印精度
np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)
# 固定随机种子
torch.manual_seed(42); np.random.seed(42); random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Dataset Helpers ----
def _read_txt(fp):
    lines = open(fp, 'r', encoding='utf-8').read().splitlines()[1:]
    arr = np.array([[int(x) for x in l.split()[:6]] for l in lines if l.strip()], dtype=float)
    m, s = arr.mean(0, keepdims=True), arr.std(0, keepdims=True) + 1e-8
    return ((arr - m) / s).astype(np.float32)

def _split(arr, train, n_seg=27, seg_len=128):
    idx = np.linspace(0, arr.shape[0], n_seg + 1, dtype=int)
    segs = []
    for i in range(n_seg):
        seg = arr[idx[i]:idx[i+1]]
        if seg.shape[0] == 0:
            seg = np.zeros((seg_len, 6), dtype=np.float32)
        elif seg.shape[0] < seg_len:
            seg = np.pad(seg, ((0, seg_len - seg.shape[0]), (0,0)))
        else:
            start = random.randint(0, seg.shape[0] - seg_len) if train else 0
            seg = seg[start:start+seg_len]
        segs.append(seg.astype(np.float32))
    return np.stack(segs)

class SwingDS(Dataset):
    def __init__(self, data_dir, info_csv, players=None, cols=None, train=True):
        info = pd.read_csv(info_csv)
        self.train, self.cols = train, cols
        X, y, uid = [], [], []
        for fp in Path(data_dir).glob('*.txt'):
            u = int(fp.stem)
            row = info[info['unique_id'] == u]
            if row.empty: continue
            pid = row['player_id'].iloc[0] if 'player_id' in row else None
            if players is not None and pid not in players: continue
            arr = _read_txt(fp)
            if arr.shape[0] < 28:
                if train:
                    continue
                else:
                    arr = np.pad(arr, ((0, 28-arr.shape[0]), (0,0)))
            X.append(_split(arr, train)); uid.append(u)
            if cols:
                vals = row[cols].iloc[0].to_numpy().astype(int)
                vals[0:2] -= 1; vals[3] -= 2
                mapping = {v:i for i,v in enumerate(sorted(info['play years'].unique()))}
                vals[2] = mapping[row['play years'].iloc[0]]
                y.append(vals)
        self.X = np.stack(X)
        self.y = np.stack(y) if cols else None
        self.uid = uid
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.train:
            x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
        if self.y is not None:
            return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.int64)
        return torch.from_numpy(x), self.uid[idx]

# ---- Model ----
class Net(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        # CNN blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(6,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128,d_model,3,padding=1), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.short3 = nn.Conv1d(128,d_model,1)
        self.pool   = nn.MaxPool1d(2)
        # Positional embedding and Transformer encoder
        self.pos    = nn.Parameter(torch.randn(1,64,d_model))
        enc_layer   = nn.TransformerEncoderLayer(d_model, nhead, 2*d_model, 0.1, batch_first=True)
        self.enc    = nn.TransformerEncoder(enc_layer, num_layers)
        # Feature projection: mean, var, RMS for each of 6 channels => 18 dims -> d_model
        self.feat_proj = nn.Sequential(
            nn.Linear(6*3, d_model),
            nn.ReLU()
        )
        # Prediction heads
        self.head   = nn.ModuleDict({
            'gender': nn.Linear(d_model,1), 'handed': nn.Linear(d_model,1),
            'years' : nn.Linear(d_model,3), 'level' : nn.Linear(d_model,4)
        })

    def forward(self, x):
        B = x.size(0)
        # === Compute classical statistics features ===
        # x: (B,27,128,6)
        raw_flat = x.view(B, -1, 6)  # (B, 27*128, 6)
        mean = raw_flat.mean(dim=1)  # (B,6)
        var  = raw_flat.var(dim=1)   # (B,6)
        rms  = torch.sqrt((raw_flat**2).mean(dim=1))  # (B,6)
        stats = torch.cat([mean, var, rms], dim=1)    # (B,18)
        feat_emb = self.feat_proj(stats)             # (B,d_model)

        # === CNN + Transformer pipeline ===
        y = x.view(-1,128,6).transpose(1,2)          # (B*27,6,128)
        y = self.conv1(y)
        y = self.conv2(y)
        res = self.short3(y)
        y = self.conv3(y) + res
        y = self.pool(y)                             # (B*27, d_model, 64)
        y = y.transpose(1,2) + self.pos              # (B*27,64,d_model)
        y = self.enc(y)                              # (B*27,64,d_model)
        y = y.mean(1).view(B,27,-1).mean(1)           # (B,d_model)

        # === Fuse end-to-end features with classical stats ===
        x_fused = y + feat_emb                      # (B,d_model)

        return {k: self.head[k](x_fused) for k in self.head}

# ---- Training with pretrain binary ----
def run_train(epochs=50, batch_size=64, val_every=5, patience=5, pretrain=False):  # pretrain flag controls binary head pretraining (default off)  # pretrain flag controls binary head pretraining
    cols = ['gender','hold racket handed','play years','level']
    info_csv = '39_Training_Dataset/train_info.csv'
    players  = pd.read_csv(info_csv)['player_id'].unique()
    tr, va   = train_test_split(players, test_size=0.2, random_state=42)
    ds_tr    = SwingDS('39_Training_Dataset/train_data', info_csv, tr, cols, True)
    ds_va    = SwingDS('39_Training_Dataset/train_data', info_csv, va, cols, False)
    # DataLoaders
    g = ds_tr.y[:,0]
    sampler = WeightedRandomSampler(np.where(g==1, (g==1).mean(), 1.0), len(g), True)
    dl_tr   = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, num_workers=1)
    dl_va   = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=1)
    fl_g = FocalLoss(alpha=(g==1).mean(), gamma=2.0, multiclass=False)
    fl_h = FocalLoss(alpha=(ds_tr.y[:,1]==1).mean(), gamma=2.0, multiclass=False)
    # Instantiate model
    model = Net().to(DEVICE)
    if pretrain:
        # Pretrain binary heads (optional)
        # Freeze all except gender/handed heads
        for name, param in model.named_parameters():
            param.requires_grad = ('head.gender' in name) or ('head.handed' in name)
        opt_pre = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for pre_ep in range(1, 6):
            model.train()
            pre_loss = 0.0
            for x, y in dl_tr:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt_pre.zero_grad()
                out = model(x)
                loss_b = fl_g(out['gender'].squeeze(), y[:, 0].float()) + \
                         fl_h(out['handed'].squeeze(), y[:, 1].float())
                loss_b.backward()
                opt_pre.step()
                pre_loss += loss_b.item() * x.size(0)
            print(f'Pre-Epoch {pre_ep}/5 loss={pre_loss/len(ds_tr):.4f}')
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    # Multi-task training setup
    for p in model.parameters(): p.requires_grad = True
    w_y = torch.tensor(1.0/(np.bincount(ds_tr.y[:,2], minlength=3)+1e-8), device=DEVICE, dtype=torch.float32)
    w_l = torch.tensor(1.0/(np.bincount(ds_tr.y[:,3], minlength=4)+1e-8), device=DEVICE, dtype=torch.float32)
    loss_y = nn.CrossEntropyLoss(weight=w_y)
    loss_l = nn.CrossEntropyLoss(weight=w_l)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3)
    best_auc_sum, no_imp = 0, 0
    # Validate helper
    def validate():
        model.eval(); pg, ph, yg, yh = [], [], [], []
        py_pred, py_true = [], []
        lvl_pred, lvl_true = [], []
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                o = model(x)
                pg.append(torch.sigmoid(o['gender']).cpu().numpy().ravel())
                ph.append(torch.sigmoid(o['handed']).cpu().numpy().ravel())
                yg.append(y[:,0].cpu().numpy()); yh.append(y[:,1].cpu().numpy())
                yy = torch.softmax(o['years'], dim=1).cpu().numpy()
                lv = torch.softmax(o['level'], dim=1).cpu().numpy()
                py_pred.append(yy); py_true.append(y[:,2].cpu().numpy())
                lvl_pred.append(lv); lvl_true.append(y[:,3].cpu().numpy())
        pg = np.concatenate(pg); ph = np.concatenate(ph)
        yg = np.concatenate(yg); yh = np.concatenate(yh)
        py_pred = np.vstack(py_pred); py_true = np.concatenate(py_true)
        lvl_pred = np.vstack(lvl_pred); lvl_true = np.concatenate(lvl_true)
        auc_g   = roc_auc_score(yg, pg)
        auc_h   = roc_auc_score(yh, ph)
        auc_py  = roc_auc_score(py_true, py_pred, multi_class='ovr', average='macro')
        auc_lvl = roc_auc_score(lvl_true, lvl_pred, multi_class='ovr', average='macro')

        print(f'gender: {auc_g}')
        print(f'hand: {auc_h}')
        print(f'play age AUC: {auc_py}')
        print(f'level AUC: {auc_lvl}')
        return auc_g + auc_h
    # Training loop
    for ep in range(1, epochs+1):
        model.train(); total_loss=0
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            o = model(x)
            loss = 2*fl_g(o['gender'].squeeze(), y[:,0].float()) + 2*fl_h(o['handed'].squeeze(), y[:,1].float())
            loss += loss_y(o['years'], y[:,2]) + loss_l(o['level'], y[:,3])
            loss.backward(); opt.step(); total_loss += loss.item()*x.size(0)
        print(f'Epoch {ep}/{epochs} loss={total_loss/len(ds_tr):.4f}')
        if ep % val_every == 0:
            auc_sum = validate(); sched.step(1-auc_sum)
            if auc_sum > best_auc_sum:
                best_auc_sum, no_imp = auc_sum, 0
                torch.save(model.state_dict(), 'swing_best.pth')
            else:
                no_imp +=1
            if no_imp >= patience:
                print('Early stopping'); break
# ---- Predict ----
def run_predict(batch_size=64):
    model = Net().to(DEVICE)
    model.load_state_dict(torch.load('swing_best.pth', map_location=DEVICE, weights_only=True))
    model.eval()
    ds = SwingDS('39_Test_Dataset/test_data','39_Test_Dataset/test_info.csv', None, None, False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)
    results = []
    with torch.no_grad():
        for x, uid in dl:
            x = x.to(DEVICE)
            o = model(x)
            gp = torch.sigmoid(o['gender']).cpu().numpy().squeeze(-1)
            hp = torch.sigmoid(o['handed']).cpu().numpy().squeeze(-1)
            yp = torch.softmax(o['years'], dim=1).cpu().numpy()
            lp = torch.softmax(o['level'], dim=1).cpu().numpy()
            for i, u in enumerate(uid):
                results.append({
                    'unique_id': int(u),
                    'gender': f"{1-gp[i]:.4f}", 'hold racket handed': f"{1-hp[i]:.4f}",
                    **{f'play years_{j}': f"{yp[i,j]:.4f}" for j in range(3)},
                    **{f'level_{j+2}': f"{lp[i,j]:.4f}" for j in range(4)}
                })
    pd.DataFrame(results).sort_values('unique_id').to_csv('test_predictions.csv', index=False, float_format='%.4f')
    print('Predictions saved.')

# ---- CLI ----
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', help='Enable binary head pretraining')
    parser.add_argument('--mode', choices=['train','predict'], required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--val_every', type=int, default=10)
    args = parser.parse_args()
    if args.mode == 'train':
        run_train(args.epochs, args.batch, args.val_every, pretrain=args.pretrain)
    else:
        run_predict(args.batch)
