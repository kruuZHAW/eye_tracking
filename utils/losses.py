import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- FOCAL LOSS -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Focal Loss for classification.

        Args:
            alpha (float): Weighting factor for the class imbalance.
            gamma (float): Focusing parameter to reduce relative loss for well-classified examples.
            reduction (str): 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=0.1)  # [B] / Soft targets help reduce overconfidence:
        pt = torch.exp(-ce_loss)  # [B]
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # [B]
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss