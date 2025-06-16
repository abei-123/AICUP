import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    * binary:   logits + targets (B,)  →   `multiclass=False`
    * multi-cls logits + targets (B,)  →   `multiclass=True`  (softmax 版本)
    alpha:     None | float | list(ndim=C)   — 正樣本權重
    gamma:     調節難例 (常用 1~3)
    """
    def __init__(self, gamma=2.0, alpha=None, multiclass=False, reduction='mean'):
        super().__init__()
        self.gamma, self.reduction, self.mc = gamma, reduction, multiclass
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:  # scalar
            self.alpha = torch.tensor([alpha], dtype=torch.float32)

    def forward(self, logits, target):
        """
        logits:  (B,)   or  (B,C)
        target:  (B,)   —  int64 0…C-1   (binary 0/1)
        """
        if self.mc:                         # ─── multi-class focal ───
            # CE = -log(p_t)  ;  p = softmax
            logp  = F.log_softmax(logits, dim=1)          # (B,C)
            pt    = torch.exp(logp)                       # (B,C)
            logp  = logp.gather(1, target.unsqueeze(1))   # (B,1)
            pt    = pt.gather(1, target.unsqueeze(1))     # (B,1)
        else:                              # ─── binary focal ───
            # BCE = -[ y·logσ + (1-y)·log(1-σ) ]
            pt    = torch.sigmoid(logits)                 # (B,)
            # 取 y==1 時的 p ，y==0 時的 (1-p)
            pt    = torch.where(target == 1, pt, 1 - pt).unsqueeze(1)  # (B,1)
            logp  = torch.log(pt + 1e-12)                 # 避免 log(0)

        # focal = α·(1-pt)^γ · CE
        if self.alpha is not None:
            α = self.alpha.to(logits.device)
            if self.mc:
                α = α.gather(0, target)                   # (B,)
            else:
                α = torch.where(target == 1, α[0], 1-α[0])
            loss = -α.unsqueeze(1) * (1-pt)**self.gamma * logp
        else:
            loss = -(1-pt)**self.gamma * logp

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss.squeeze()
