#
# file: face-antispoofing/lib/model_develop.py
#
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
import sys  # <-- 新增导入

from lib.model_develop_utils import *


def train_epoch(train_loader, model, optimizer, epoch, lr_scheduler, criterion, config):
    model.train()
    alpha = getattr(config, 'alpha', 0.5)
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # === 进度条修改：统一输出流 ===
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=True, file=sys.stdout)

    for (image_rgb, image_depth, image_ir), label in pbar:
        image_rgb, image_depth, image_ir = image_rgb.cuda(), image_depth.cuda(), image_ir.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        out_f, out_rgb, out_depth, out_ir = model(image_rgb, image_depth, image_ir)
        loss_f = criterion(out_f, label)
        loss_rgb = criterion(out_rgb, label)
        loss_depth = criterion(out_depth, label)
        loss_ir = criterion(out_ir, label)
        loss_f.backward(retain_graph=True)
        if isinstance(model, nn.DataParallel):
            for param in model.module.net_rgb.parameters(): param.grad = None
            for param in model.module.net_depth.parameters(): param.grad = None
            for param in model.module.net_ir.parameters(): param.grad = None
        else:
            for param in model.net_rgb.parameters(): param.grad = None
            for param in model.net_depth.parameters(): param.grad = None
            for param in model.net_ir.parameters(): param.grad = None
        loss_unimodal = (loss_rgb + loss_depth + loss_ir) * alpha
        loss_unimodal.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        _, preds = torch.max(out_f, 1)
        loss = loss_f
        if torch.isnan(loss):
            print("\n!!! 损失变为 NaN，请尝试进一步降低学习率。 !!!")
            return float('nan'), 0.0
        running_loss += loss.item() * image_rgb.size(0)
        running_corrects += torch.sum(preds == label.data)
        total_samples += image_rgb.size(0)
        pbar.set_postfix({
            'Loss': f'{running_loss / total_samples:.4f}',
            'Acc': f'{running_corrects.double().item() / total_samples:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
        })
    if lr_scheduler is not None:
        lr_scheduler.step()
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc


def valid_epoch(valid_loader, model, criterion, config):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    pred_list = []
    label_list = []

    with torch.no_grad():
        # === 进度条修改：统一输出流 ===
        pbar = tqdm(valid_loader, desc="Validating", leave=False, file=sys.stdout)
        for (image_rgb, image_depth, image_ir), label in pbar:
            # =============================
            image_rgb, image_depth, image_ir = image_rgb.cuda(), image_depth.cuda(), image_ir.cuda()
            label = label.cuda()

            output, _, _, _ = model(image_rgb, image_depth, image_ir)

            if torch.isnan(output).any():
                return 0.0, 0.0, 0.0

            prob = F.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            loss = criterion(output, label)

            running_loss += loss.item() * image_rgb.size(0)
            running_corrects += torch.sum(preds == label.data)
            total_samples += image_rgb.size(0)

            label_list.extend(label.cpu().numpy())
            pred_list.extend(prob.cpu().numpy()[:, 1])

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    auc_score = 0.0
    if len(np.unique(label_list)) > 1:
        if not np.isnan(pred_list).any():
            auc_score = metrics.roc_auc_score(label_list, pred_list)
    else:
        print("\n[Warning] 验证集中只存在一种类别，AUC分数未定义，记为0。")

    print(f'Valid Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {auc_score:.4f}')

    return epoch_loss, epoch_acc, auc_score