import os

# --- 请在这里配置您的路径 ---
# 数据集根目录
dataset_root = '/mnt/d/jcy/CASIA-SURF'
# 您上传的官方列表文件名
# 请确保这三个文件与 generate_lists.py 放在同一个目录下
official_train_list_filename = 'train_list_complete.txt'
official_val_list_filename = 'val_list_complete.txt'
official_test_list_filename = 'test_list_complete.txt'


# -----------------------------

def process_official_list(split_name, official_filename, output_path):
    """
    专门处理官方列表文件，智能转换路径并生成模型可用的新列表。
    """
    print(f"\n{'=' * 20} 正在处理 '{split_name}' 数据 {'=' * 20}")

    if not os.path.exists(official_filename):
        print(f"    [致命错误] 官方列表文件 '{official_filename}' 未找到！")
        print(f"    请将您上传的 '{official_filename}' 文件与本脚本放在同一个目录下。")
        return 0

    output_data = []
    processed_count = 0
    with open(official_filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                old_rgb_path, old_depth_path, old_ir_path, label = parts

                try:
                    # 使用 split 方法，以 'CASIA-SURF/' 为分隔符，并取后半部分
                    relative_rgb = old_rgb_path.split('CASIA-SURF/')[1]
                    relative_depth = old_depth_path.split('CASIA-SURF/')[1]
                    relative_ir = old_ir_path.split('CASIA-SURF/')[1]

                    # 检查转换后的文件在您的本地路径下是否真的存在
                    if os.path.exists(os.path.join(dataset_root, relative_rgb)):
                        output_data.append(f"{relative_rgb} {relative_depth} {relative_ir} {label}\n")
                        processed_count += 1
                except IndexError:
                    continue

    with open(output_path, 'w') as f:
        f.writelines(output_data)

    print(f"--> 成功生成 '{output_path}'，包含 {processed_count} 条带有真实标签的记录。")
    return processed_count


if __name__ == '__main__':
    print("开始生成带真实标签的数据列表文件 (最终版)...")

    # 定义最终生成的文件路径
    final_train_list_path = os.path.join(dataset_root, 'train_list.txt')
    final_val_list_path = os.path.join(dataset_root, 'val_list.txt')
    final_test_list_path = os.path.join(dataset_root, 'test_list.txt')

    # 处理训练集
    process_official_list('Training', official_train_list_filename, final_train_list_path)

    # 处理验证集
    process_official_list('Validation', official_val_list_filename, final_val_list_path)

    # 处理测试集
    process_official_list('Testing', official_test_list_filename, final_test_list_path)

    print("\n脚本执行完毕。请检查记录数。")