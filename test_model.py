import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
from tqdm import tqdm

# 将项目根目录添加到系统路径，以便导入我们自定义的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入与我们训练脚本完全一致的模块
from configuration.config_baseline_multi import config
from models.surf_baseline import Net
from src.surf_baseline_multi_dataloader import SURF_multi_val_loader


def evaluate_final_model():
    """
    加载由主训练脚本保存的最佳模型，
    并在独立的测试集上进行最终的性能评估，计算真实的AUC分数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 1. 加载我们训练好的最佳模型
    # 路径根据我们的配置文件自动生成
    model_path = os.path.join(config.save_path, f'{config.model_name}_best.pth')
    if not os.path.exists(model_path):
        print(f"!!! 错误: 找不到训练好的模型: {model_path}")
        print("!!! 请先确保您已经成功运行了主训练脚本 `surf_baseline_multi_main.py` 并生成了最佳模型。")
        return

    # 初始化与训练时完全相同的模型结构
    model = Net(args=config, embedding_size=config.embedding_size, num_classes=config.num_classes)

    # 加载模型权重
    # 这段代码会自动处理模型在保存时是否被 nn.DataParallel 包裹的情况
    saved_state_dict = torch.load(model_path)['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()  # 切换到评估模式
    print(f"--> 成功加载最佳模型: {model_path}")

    # 2. 准备测试集的数据加载器
    transform_test = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 明确指定加载 test_list.txt 文件
    test_dataset = SURF_multi_val_loader(root_path=config.data_path, transform=transform_test,
                                         list_file='test_list.txt')

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )

    # 3. 执行预测并计算最终性能指标
    pred_scores = []
    true_labels = []
    with torch.no_grad():
        for (image_rgb, image_depth, image_ir), label in tqdm(test_loader, desc="正在对测试集进行最终评估"):
            image_rgb, image_depth, image_ir = image_rgb.to(device), image_depth.to(device), image_ir.to(device)

            output, _, _, _ = model(image_rgb, image_depth, image_ir)
            # 使用 softmax 得到概率，我们关心的是“攻击”类别的概率
            probabilities = torch.nn.functional.softmax(output, dim=1)

            true_labels.extend(label.cpu().numpy())
            pred_scores.extend(probabilities.cpu().numpy()[:, 1])  # 取出判断为 "fake" (标签1) 的概率

    # 计算并打印最终的评估结果
    if len(np.unique(true_labels)) > 1:
        auc_score = metrics.roc_auc_score(true_labels, pred_scores)
        # 您也可以在这里计算其他指标，例如 EER, HTER 等

        print("\n" + "=" * 50)
        print(f"  最终模型性能评估结果:")
        print(f"  测试集 ROC AUC 分数: {auc_score:.4f}")
        print("=" * 50)
    else:
        print("\n[错误] 测试集标签文件中仍然只包含一种类别，无法计算 AUC。请检查您的 `test_list.txt` 文件。")


if __name__ == '__main__':
    evaluate_final_model()