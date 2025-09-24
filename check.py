import os

def get_subject_ids_from_list(file_path):
    """从列表文件中提取所有唯一的采集对象ID"""
    subject_ids = set()
    with open(file_path, 'r') as f:
        for line in f:
            # 假设路径格式是 'modal/subject_id/image.jpg'
            # 你需要根据你的实际路径格式修改这里的切分逻辑
            try:
                # parts[0] 是第一个模态的路径
                parts = line.strip().split()
                subject_id = parts[0].split('/')[-2] # 取倒数第二个部分作为ID
                subject_ids.add(subject_id)
            except IndexError:
                continue # 跳过空行或格式不正确的行
    return subject_ids

# --- 配置你的文件路径 ---
dataset_root = '/mnt/d/jcy/CASIA-SURF' # 和你的截图路径一致
train_list_path = os.path.join(dataset_root, 'train_list.txt')
val_list_path = os.path.join(dataset_root, 'val_list.txt')
test_list_path = os.path.join(dataset_root, 'test_list.txt')
# -------------------------

# 获取每个数据集的采集对象ID集合
train_subjects = get_subject_ids_from_list(train_list_path)
val_subjects = get_subject_ids_from_list(val_list_path)
test_subjects = get_subject_ids_from_list(test_list_path)

print(f"训练集中有 {len(train_subjects)} 个独立采集对象。")
print(f"验证集中有 {len(val_subjects)} 个独立采集对象。")
print(f"测试集中有 {len(test_subjects)} 个独立采集对象。")
print("-" * 30)

# 检查交集
train_val_overlap = train_subjects.intersection(val_subjects)
train_test_overlap = train_subjects.intersection(test_subjects)
val_test_overlap = val_subjects.intersection(test_subjects)

if not train_test_overlap:
    print("✅ 很好！训练集和测试集之间没有重叠的采集对象。")
else:
    print(f"❌ 警告：发现数据泄露！训练集和测试集之间有 {len(train_test_overlap)} 个重叠的采集对象。")
    print(f"   重叠的ID: {list(train_test_overlap)[:10]}...") # 打印前10个

# ...可以添加对其他交集的检查...