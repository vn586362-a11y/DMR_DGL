#
# file: face-antispoofing/models/surf_baseline.py
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_se import resnet18_se


class Net(nn.Module):
    def __init__(self, args, embedding_size=256, num_classes=2):
        super(Net, self).__init__()

        self.net_rgb = resnet18_se(args, pretrained=False, num_classes=num_classes)
        self.net_depth = resnet18_se(args, pretrained=False, num_classes=num_classes)
        self.net_ir = resnet18_se(args, pretrained=False, num_classes=num_classes)

        self.net_rgb.fc = nn.Linear(512, embedding_size)
        self.net_depth.fc = nn.Linear(512, embedding_size)
        self.net_ir.fc = nn.Linear(512, embedding_size)

        # === 核心修改 1：在融合层后加入 Dropout ===
        self.fusion_dropout = nn.Dropout(p=0.5)  # 50% 的 Dropout 比例
        # ========================================

        self.fc_fus = nn.Linear(embedding_size * 3, embedding_size)
        self.fc_final = nn.Linear(embedding_size, num_classes)

        self.unimodal_classifiers = nn.ModuleDict({
            'rgb': nn.Linear(embedding_size, num_classes),
            'depth': nn.Linear(embedding_size, num_classes),
            'ir': nn.Linear(embedding_size, num_classes)
        })

    def forward(self, x_rgb, x_depth, x_ir):
        feat_rgb = self.net_rgb.feature(x_rgb)
        feat_depth = self.net_depth.feature(x_depth)
        feat_ir = self.net_ir.feature(x_ir)

        emb_rgb = self.net_rgb.fc(feat_rgb)
        emb_depth = self.net_depth.fc(feat_depth)
        emb_ir = self.net_ir.fc(feat_ir)

        out_rgb = self.unimodal_classifiers['rgb'](emb_rgb)
        out_depth = self.unimodal_classifiers['depth'](emb_depth)
        out_ir = self.unimodal_classifiers['ir'](emb_ir)

        feat_fus = torch.cat((emb_rgb, emb_depth, emb_ir), dim=1)
        feat_fus = self.fc_fus(feat_fus)
        feat_fus = F.relu(feat_fus)

        # === 核心修改 2：应用 Dropout ===
        # 注意：Dropout 只在训练时生效，在 model.eval() 模式下会自动关闭
        feat_fus = self.fusion_dropout(feat_fus)
        # =============================

        output_fusion = self.fc_final(feat_fus)

        return output_fusion, out_rgb, out_depth, out_ir