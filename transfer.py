import os
import shutil

# ========== 配置区 ==========
SOURCE_DIR = r"F:\data_generate\spectra_mimic\augmented_results"  # 源文件夹
TARGET_DIR = r"F:\data_generate\spectra_mimic\results"  # 目标文件夹

# 如果目标文件夹不存在，就创建
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# 遍历源文件夹及其所有子文件夹
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.lower().endswith(".csv"):
            source_path = os.path.join(root, file)
            target_path = os.path.join(TARGET_DIR, file)
            
            # 如果目标文件夹中有同名文件，可以改名避免覆盖
            base_name, ext = os.path.splitext(file)
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(TARGET_DIR, f"{base_name}_{counter}{ext}")
                counter += 1
            
            shutil.copy2(source_path, target_path)  # 复制文件（包含元数据）
            print(f"已复制: {source_path} -> {target_path}")

print("全部 CSV 文件已复制完成！")
