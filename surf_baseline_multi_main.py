#
# file: face-antispoofing/src/surf_baseline_multi_main.py
#
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config_baseline_multi import config
from models.surf_baseline import Net
from lib.model_develop import train_epoch, valid_epoch
from src.surf_baseline_multi_dataloader import SURF_multi_train_loader, SURF_multi_val_loader


def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # === 核心修改：为训练集加入更强的数据增强 ===
    transform_train = transforms.Compose([
        transforms.Resize(config.image_size),
        # 增加色彩抖动、仿射变换和随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # ============================================

    # 验证集的预处理保持不变，不应包含随机增强
    transform_val = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SURF_multi_train_loader(root_path=config.data_path, transform=transform_train)
    val_dataset = SURF_multi_val_loader(root_path=config.data_path, transform=transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )

    valid_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )

    model = Net(args=config, embedding_size=config.embedding_size, num_classes=config.num_classes)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个 GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4
    )

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestones, gamma=config.gamma
    )

    best_valid_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 10

    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(train_loader, model, optimizer, epoch, lr_scheduler, criterion, config)
        valid_loss, valid_acc, valid_auc = valid_epoch(valid_loader, model, criterion, config)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            print(f"*** 新的最佳验证损失: {best_valid_loss:.4f} ***")

            if not os.path.exists(config.save_path):
                os.makedirs(config.save_path)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': best_valid_loss,
            }, os.path.join(config.save_path, f'{config.model_name}_best.pth'))
        else:
            epochs_no_improve += 1
            print(f"验证损失没有改善，已连续 {epochs_no_improve} 个 epochs。")

        if epochs_no_improve >= early_stop_patience:
            print(f"验证损失已连续 {early_stop_patience} 个 epochs 没有改善，触发早停！")
            break

    print("训练完成!")
    print(f"最好的验证集损失是: {best_valid_loss:.4f}")


if __name__ == '__main__':
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()