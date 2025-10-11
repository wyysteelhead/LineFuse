#!/usr/bin/env python3
"""
LineFuse 数据格式转换脚本
将augmented_results中的多光谱CSV文件转换为单光谱CSV文件

输入格式: 每个CSV包含wavelength列 + 50个spectrum列
输出格式: 每个光谱单独保存为一个CSV文件 (wavelength, intensity)

更新版本: 支持新的扁平化目录结构（所有CSV直接在augmented_results下）
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

def convert_augmented_data():
    """转换augmented_results数据为LineFuse格式"""

    # 配置路径
    input_dir = Path(".")  # augmented_results目录
    output_dir = Path("../dataset/csv_data")  # 输出到现有的csv_data目录

    print("=== LineFuse 数据格式转换 (v2.0) ===")
    print(f"输入目录: {input_dir.absolute()}")
    print(f"输出目录: {output_dir.absolute()}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有CSV文件（直接在当前目录下）
    csv_files = sorted(list(input_dir.glob("*_spectra.csv")))
    print(f"发现 {len(csv_files)} 个CSV文件")

    if len(csv_files) == 0:
        print("❌ 没有找到CSV文件，请检查目录结构")
        print("预期格式: XXX_augmented_spectra.csv")
        return

    # 显示数据概览
    print(f"文件名示例: {csv_files[0].name}")
    try:
        sample_df = pd.read_csv(csv_files[0])
        print(f"每个CSV数据点数: {len(sample_df)}")
        print(f"每个CSV光谱数量: {len([col for col in sample_df.columns if col.startswith('spectrum_')])}")
    except Exception as e:
        print(f"⚠️  读取示例文件失败: {e}")

    # 统计信息
    total_spectra = 0
    spectrum_counter = 0
    error_files = []

    # 处理每个CSV文件
    for csv_file in tqdm(csv_files, desc="处理CSV文件"):
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)

            # 检查数据格式
            if 'wavelength' not in df.columns:
                print(f"⚠️  跳过 {csv_file.name}: 缺少wavelength列")
                error_files.append((csv_file.name, "缺少wavelength列"))
                continue

            # 获取波长列
            wavelength = df['wavelength'].values

            # 获取所有光谱列 (spectrum_0, spectrum_1, ...)
            spectrum_columns = [col for col in df.columns if col.startswith('spectrum_')]

            if len(spectrum_columns) == 0:
                print(f"⚠️  跳过 {csv_file.name}: 没有找到spectrum列")
                error_files.append((csv_file.name, "没有找到spectrum列"))
                continue

            # 为每个光谱创建单独的CSV文件
            for i, spectrum_col in enumerate(spectrum_columns):
                intensity = df[spectrum_col].values

                # 检查数据有效性
                if len(intensity) != len(wavelength):
                    print(f"⚠️  {csv_file.name} - {spectrum_col}: 数据长度不匹配")
                    continue

                # 检查是否有有效数据（非全零）
                if np.all(intensity == 0):
                    continue  # 跳过全零光谱

                # 创建新的DataFrame (只包含两列: wavelength, intensity)
                new_df = pd.DataFrame({
                    'wavelength': wavelength,
                    'intensity': intensity
                })

                # 生成输出文件名: spectrum_XXXXX.csv
                output_filename = f"spectrum_{spectrum_counter:05d}.csv"
                output_path = output_dir / output_filename

                # 保存CSV文件
                new_df.to_csv(output_path, index=False)

                spectrum_counter += 1
                total_spectra += 1

        except Exception as e:
            print(f"❌ 处理 {csv_file.name} 时出错: {e}")
            error_files.append((csv_file.name, str(e)))
            continue

    print(f"\n✅ 转换完成!")
    print(f"总共处理: {len(csv_files)} 个输入文件")
    print(f"成功文件: {len(csv_files) - len(error_files)} 个")
    print(f"错误文件: {len(error_files)} 个")
    print(f"生成光谱: {total_spectra} 条")
    print(f"输出目录: {output_dir}")

    # 显示错误信息
    if error_files:
        print(f"\n⚠️  错误文件列表:")
        for filename, error in error_files[:10]:  # 只显示前10个错误
            print(f"  {filename}: {error}")
        if len(error_files) > 10:
            print(f"  ... 还有 {len(error_files) - 10} 个错误文件")

    # 验证输出文件
    output_files = list(output_dir.glob("spectrum_*.csv"))
    print(f"✅ 验证: 输出了 {len(output_files)} 个文件")

    # 显示示例文件
    if output_files:
        sample_file = output_files[0]
        print(f"\n📋 示例文件格式 ({sample_file.name}):")
        try:
            sample_df = pd.read_csv(sample_file)
            print(sample_df.head())
            print(f"数据点数: {len(sample_df)}")
            print(f"波长范围: {sample_df['wavelength'].min():.2f} - {sample_df['wavelength'].max():.2f}")
            print(f"强度范围: {sample_df['intensity'].min():.2f} - {sample_df['intensity'].max():.2f}")
        except Exception as e:
            print(f"读取示例文件失败: {e}")

    # 计算预期的训练数据规模
    print(f"\n🚀 训练数据规模预估:")
    print(f"可用光谱数量: {total_spectra}")
    print(f"推荐训练样本数: {min(total_spectra, 10000)} (使用 --samples {min(total_spectra, 10000)})")
    print(f"最大可能训练集: {total_spectra * 12} 张图像 (12种模糊效果)")

def cleanup_existing_data():
    """清理现有的测试数据，为新数据让路"""

    csv_data_dir = Path("../dataset/csv_data")

    if csv_data_dir.exists():
        existing_files = list(csv_data_dir.glob("spectrum_*.csv"))
        if existing_files:
            print(f"🧹 发现现有数据: {len(existing_files)} 个文件")
            response = input("是否清理现有数据? (y/N): ").lower()
            if response == 'y':
                for file in existing_files:
                    file.unlink()
                print("✅ 清理完成")
                return True
            else:
                print("保留现有数据，新数据将从现有编号继续")
                return False
    return True

def get_next_spectrum_number():
    """获取下一个可用的光谱编号"""
    csv_data_dir = Path("../dataset/csv_data")
    if not csv_data_dir.exists():
        return 0

    existing_files = list(csv_data_dir.glob("spectrum_*.csv"))
    if not existing_files:
        return 0

    # 提取现有文件的最大编号
    max_num = -1
    for file in existing_files:
        try:
            num_str = file.stem.split('_')[1]
            num = int(num_str)
            max_num = max(max_num, num)
        except (ValueError, IndexError):
            continue

    return max_num + 1

def main():
    """主函数"""
    print("LineFuse 数据转换工具 v2.0")
    print("支持新的扁平化目录结构")
    print("=" * 50)

    # 检查当前目录
    current_files = list(Path(".").glob("*_spectra.csv"))
    if not current_files:
        print("❌ 当前目录下没有找到光谱数据文件")
        print("请确保在 augmented_results 目录下运行此脚本")
        print("预期文件格式: XXX_augmented_spectra.csv")
        return

    print(f"✅ 发现 {len(current_files)} 个光谱数据文件")

    # 处理现有数据
    cleanup_existing_data()

    # 执行转换
    convert_augmented_data()

    print("\n🚀 转换完成! 现在可以使用以下命令生成训练数据:")
    print("cd ..")
    print("# 中等规模测试")
    print("python main.py generate --samples 2000 --output real_data_test")
    print("# 大规模训练数据")
    print("python main.py generate --samples 10000 --output real_data_training")
    print("# 或使用自动化脚本")
    print("./auto_dataset_generator.sh")

if __name__ == "__main__":
    main()