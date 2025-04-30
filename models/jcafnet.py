import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy


# -------------------- CROSS ATTENTION BLOCK --------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, query, key):
        Q = self.query_proj(query)  # [B, T, D]
        K = self.key_proj(key)      # [B, T, D]
        V = self.value_proj(key)    # [B, T, D]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)                # [B, T, T]
        attended = torch.matmul(attn_weights, V)                         # [B, T, D]
        return attended


# -------------------- MODALITY RESNET BACKBONE --------------------
def build_resnet_backbone(input_channels):
    base_model = resnet34(pretrained=False)
    base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base_model.fc = nn.Identity()  # remove classification head
    return base_model


# -------------------- JOINT FEATURE CNN --------------------
class JointFeatureCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.encoder(x)  # [B, 128, 1, 1]
        return x.view(x.size(0), -1)  # [B, 128]


# -------------------- JOINT CROSS ATTENTION FUSION NET --------------------
class JCAFNet(pl.LightningModule):
    def __init__(self, num_classes, gaze_dim, mouse_dim, joint_dim, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.gaze_backbone = build_resnet_backbone(gaze_dim)
        self.mouse_backbone = build_resnet_backbone(mouse_dim)
        self.joint_backbone = JointFeatureCNN(joint_dim)

        self.cross_attention = CrossAttentionBlock(d_model=512)

        self.fc_fusion = nn.Sequential(
            nn.Linear(512 * 2 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.learning_rate = learning_rate

    def forward(self, gaze_input, mouse_input, joint_input):
        gaze_feat = self.gaze_backbone(gaze_input)  # [B, 512]
        mouse_feat = self.mouse_backbone(mouse_input)  # [B, 512]

        # reshape for attention: [B, D] -> [B, 1, D]
        gaze_feat_exp = gaze_feat.unsqueeze(1)
        mouse_feat_exp = mouse_feat.unsqueeze(1)

        attended_gaze = self.cross_attention(gaze_feat_exp, mouse_feat_exp).squeeze(1)  # [B, 512]
        attended_mouse = self.cross_attention(mouse_feat_exp, gaze_feat_exp).squeeze(1)  # [B, 512]

        joint_feat = self.joint_backbone(joint_input)  # [B, 128]

        fused = torch.cat([attended_gaze, attended_mouse, joint_feat], dim=1)  # [B, 1152]
        logits = self.fc_fusion(fused)  # [B, num_classes]
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['gaze'], batch['mouse'], batch['joint'])
        loss = self.criterion(logits, batch['label'])
        acc = self.accuracy(logits.argmax(dim=1), batch['label'])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['gaze'], batch['mouse'], batch['joint'])
        loss = self.criterion(logits, batch['label'])
        acc = self.accuracy(logits.argmax(dim=1), batch['label'])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc"}}
