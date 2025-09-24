#
# file: face-antispoofing/configuration/config_baseline_multi.py
#
from easydict import EasyDict as edict

config = edict()
config.margin = 0.5
config.s = 32
config.image_size = (128, 128)
config.batch_size = 96
config.epochs = 150

config.data_path = '/mnt/d/jcy/CASIA-SURF'
config.save_path = './checkpoints/baseline_multi_dgl'
config.device = 'cuda'

# === 核心修改 1：调高学习率 ===
config.lr = 0.01
# ==============================

config.milestones = [60, 100, 120]
config.gamma = 0.2

# === 核心修改 2：调高 DGL alpha 超参数 ===
config.alpha = 1.0
# ========================================

# Model settings
config.embedding_size = 512
config.num_classes = 2
config.class_num = 2  # resnet18_se 模型需要这个名称的参数

config.model_name = 'baseline_multi_dgl'

# Dataloader settings
config.pin_memory = True
config.num_workers = 4