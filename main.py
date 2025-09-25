#!/usr/bin/env python3
"""
LineFuse 主程序
光谱折线图模糊数据生成和模型训练的统一入口
"""

import sys
import argparse
import random
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
from data.clean_chart_generator import CleanChartGenerator
from data.dataset_builder import DatasetBuilder

def generate_dataset(num_samples: int = 10, output_dir: str = "linefuse_dataset",
                    difficulty_levels: list = ["easy", "medium", "hard"],
                    enable_style_diversity: bool = True,
                    style_diversity_level: float = 0.8,
                    image_size: int = 1024,
                    line_width: float = 0.8,
                    pixel_perfect: bool = True,
                    pure_line_only: bool = False,
                    target_style: str = None):
    """生成完整的训练数据集"""
    print(f"=== LineFuse 数据集生成 ===")
    print(f"使用样本数量: {num_samples}")
    print(f"输出目录: {output_dir}")
    print(f"难度级别: {', '.join(difficulty_levels)}")
    print(f"样式多样化: {'启用' if enable_style_diversity else '禁用'}")
    if enable_style_diversity:
        print(f"多样化程度: {style_diversity_level:.1f} (0.0=最低, 1.0=最高)")
        if target_style:
            print(f"目标样式: {target_style}")
    print(f"像素完美对齐: {'启用' if pixel_perfect else '禁用'}")
    print(f"纯线条模式: {'启用' if pure_line_only else '禁用'}")

    # 定义难度级别配置
    difficulty_config = {
        "easy": {
            "line_width": 0.6,
            "blur_strength": 1.5,
            "contrast_reduction": 0.8,
            "description": "轻度模糊，线条适中"
        },
        "medium": {
            "line_width": 0.3,
            "blur_strength": 2.5,
            "contrast_reduction": 0.7,
            "description": "中度模糊，线条很细"
        },
        "hard": {
            "line_width": 0.15,
            "blur_strength": 3.5,
            "contrast_reduction": 0.6,
            "description": "重度模糊，线条极细"
        },
        "extreme": {
            "line_width": 0.1,
            "blur_strength": 4.5,
            "contrast_reduction": 0.5,
            "description": "极度模糊，几乎不可见线条"
        }
    }

    # 检查已有的CSV数据
    existing_csv_dir = Path('dataset/csv_data')
    if not existing_csv_dir.exists():
        print(f"✗ 未找到CSV数据目录: {existing_csv_dir}")
        print("请先运行数据生成创建CSV文件")
        return False

    # 获取可用的CSV文件
    csv_files = list(existing_csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"✗ CSV目录中没有找到任何CSV文件")
        return False

    # 限制使用的样本数量
    csv_files = csv_files[:num_samples]
    actual_samples = len(csv_files)
    print(f"✓ 找到 {len(list(existing_csv_dir.glob('*.csv')))} 个CSV文件，使用 {actual_samples} 个")

    # 直接创建最终数据集目录
    final_dir = Path(output_dir) / 'final_dataset'
    final_dir.mkdir(parents=True, exist_ok=True)

    # 临时目录用于中间处理
    temp_clean_dir = Path(output_dir) / '.temp_clean'
    temp_blur_dir = Path(output_dir) / '.temp_blur'
    temp_clean_dir.mkdir(exist_ok=True)
    temp_blur_dir.mkdir(exist_ok=True)

    # 步骤1: 从已有CSV生成清晰图表
    print(f"\n1. 从已有CSV生成清晰光谱图表...")

    # 注意：这里不再直接生成最终的clean图表
    # 而是为每个难度级别生成对应的clean基础图，确保clean/blur背景一致
    print("  清晰图表将按难度级别生成以确保与模糊图表背景一致...")

    # 步骤2: 按难度级别生成统一基础图和配对的clean/blur图
    print(f"\n2. 按难度级别生成统一基础图和配对的clean/blur图...")

    # 检查是否可以生成模糊效果
    can_generate_blur = True
    try:
        from data.blur_generator import BlurGenerator
    except ImportError as e:
        print(f"⚠️  模糊效果生成需要依赖库: {e}")
        print("请运行: pip install opencv-python albumentations")
        print("将仅生成清晰图...")
        can_generate_blur = False

    total_blur_count = 0
    total_clean_count = 0

    # 初始化模糊效果日志文件
    log_file = Path(output_dir) / 'blur_effects_log.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("LineFuse 模糊效果详细日志\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now()}\n")
        f.write(f"样本数量: {num_samples}\n")
        f.write(f"难度级别: {', '.join(difficulty_levels)}\n")
        f.write(f"样式多样化: {'启用' if enable_style_diversity else '禁用'}\n")
        f.write("=" * 50 + "\n")

    # 为每个难度级别生成配对的清晰图和模糊图
    for difficulty in difficulty_levels:
        config = difficulty_config[difficulty]
        print(f"\n  生成 {difficulty} 难度 ({config['description']})...")

        difficulty_blur_count = 0
        difficulty_clean_count = 0

        # 如果可以生成模糊效果，初始化模糊生成器
        if can_generate_blur:
            blur_generator = BlurGenerator(difficulty=difficulty)

        # 为每个CSV文件生成统一基础图，然后配对生成clean/blur
        for csv_file in csv_files:
            # 创建该难度的基础图生成器，包含样式多样化
            difficulty_generator = CleanChartGenerator(
                figure_size=(image_size, image_size),
                line_width=config['line_width'],
                enable_style_diversity=enable_style_diversity,
                style_diversity_level=style_diversity_level,
                target_style=target_style
            )

            # 统一基础图文件路径 - 这将作为该难度下该CSV的标准基础图
            base_chart_file = temp_clean_dir / f"{csv_file.stem}_{difficulty}_base.png"

            try:
                # 根据需求分别生成clean和blur的基础图
                if pure_line_only:
                    # pure_line_only模式：clean图为纯线条，blur图有坐标

                    # 1. 生成纯线条的clean图
                    clean_output_name = f"{csv_file.stem}_{difficulty}_clean.png"
                    clean_output_path = temp_clean_dir / clean_output_name
                    difficulty_generator.process_csv_to_chart(csv_file, clean_output_path,
                                                            pure_line_only=True,  # clean图纯线条
                                                            pixel_perfect=pixel_perfect)
                    difficulty_clean_count += 1

                    # 2. 如果需要模糊效果，生成有坐标的基础图用于模糊
                    if can_generate_blur:
                        # 生成有坐标的基础图（仅用于模糊图生成）
                        blur_base_generator = CleanChartGenerator(
                            figure_size=(image_size, image_size),
                            line_width=config['line_width'],
                            enable_style_diversity=enable_style_diversity,
                            style_diversity_level=style_diversity_level,
                            target_style=target_style
                        )

                        blur_base_file = temp_clean_dir / f"{csv_file.stem}_{difficulty}_blur_base.png"
                        blur_base_generator.process_csv_to_chart(csv_file, blur_base_file,
                                                               pure_line_only=False,  # blur基础图有坐标
                                                               pixel_perfect=pixel_perfect)

                        # 生成3张不同的模糊图变体
                        for variant in range(3):
                            blur_output_name = f"{csv_file.stem}_{difficulty}_variant_{variant}.png"
                            blur_output_path = temp_blur_dir / blur_output_name

                            try:
                                # 加载有坐标的基础图
                                base_image = blur_generator.load_image(blur_base_file)

                                # 1. 应用基础退化效果（每张都有）
                                base_degraded = blur_generator.apply_base_degradation(base_image)

                                # 2. 随机添加额外效果
                                final_result = blur_generator.apply_random_additional_blur(base_degraded)

                                # 保存结果
                                import cv2
                                cv2.imwrite(str(blur_output_path), final_result['image'])
                                difficulty_blur_count += 1

                            except Exception as e:
                                print(f"    🚨 BLUR GENERATION FAILED for {blur_output_name}:")
                                print(f"       Error: {str(e)}")

                        # 删除模糊基础图文件
                        if blur_base_file.exists():
                            blur_base_file.unlink()

                else:
                    # 标准模式：clean和blur使用统一基础图

                    # 生成统一基础图（有坐标）
                    difficulty_generator.process_csv_to_chart(csv_file, base_chart_file,
                                                            pure_line_only=False,  # 标准模式有坐标
                                                            pixel_perfect=pixel_perfect)

                    # 1. 将基础图作为该难度的清晰图 (直接复制，保持完全一致)
                    clean_output_name = f"{csv_file.stem}_{difficulty}_clean.png"
                    clean_output_path = temp_clean_dir / clean_output_name

                    import shutil
                    shutil.copy2(base_chart_file, clean_output_path)
                    difficulty_clean_count += 1

                    # 2. 如果可以生成模糊效果，基于同样的基础图生成模糊图
                    if can_generate_blur:
                        # 生成3张不同的模糊图变体，每张都基于相同的基础图
                        for variant in range(3):
                            blur_output_name = f"{csv_file.stem}_{difficulty}_variant_{variant}.png"
                            blur_output_path = temp_blur_dir / blur_output_name

                            try:
                                # 加载统一基础图
                                base_image = blur_generator.load_image(base_chart_file)

                                # 1. 应用基础退化效果（每张都有）
                                base_degraded, base_effects_log = blur_generator.apply_base_degradation(base_image)

                                # 2. 随机添加额外效果
                                final_result = blur_generator.apply_random_additional_blur(base_degraded)

                                # 3. 记录详细的模糊效果日志
                                blur_log = {
                                    'file': blur_output_name,
                                    'difficulty': difficulty,
                                    'variant': variant,
                                    'csv_source': csv_file.name,
                                    'base_effects': base_effects_log,
                                    'additional_effects': final_result.get('additional_effects_details', []),
                                    'total_effects': len(base_effects_log) + final_result.get('num_additional', 0)
                                }

                                # 打印简化日志
                                print(f"       📝 {blur_output_name}:")
                                print(f"          Base: {', '.join(base_effects_log)}")
                                if final_result.get('additional_effects_details'):
                                    print(f"          Extra: {', '.join(final_result['additional_effects_details'])}")
                                else:
                                    print(f"          Extra: None")

                                # 保存结果
                                import cv2
                                cv2.imwrite(str(blur_output_path), final_result['image'])
                                difficulty_blur_count += 1

                                # 保存详细日志到文件
                                log_file = Path(output_dir) / 'blur_effects_log.txt'
                                with open(log_file, 'a', encoding='utf-8') as f:
                                    f.write(f"\n=== {blur_output_name} ===\n")
                                    f.write(f"Difficulty: {difficulty}\n")
                                    f.write(f"CSV Source: {csv_file.name}\n")
                                    f.write(f"Variant: {variant}\n")
                                    f.write(f"\nBase Effects:\n")
                                    for effect in base_effects_log:
                                        f.write(f"  - {effect}\n")
                                    f.write(f"\nAdditional Effects:\n")
                                    if final_result.get('additional_effects_details'):
                                        for effect in final_result['additional_effects_details']:
                                            f.write(f"  - {effect}\n")
                                    else:
                                        f.write(f"  - None\n")
                                    f.write(f"\nTotal Effects: {blur_log['total_effects']}\n")
                                    f.write("-" * 50 + "\n")

                            except Exception as e:
                                print(f"    🚨 BLUR GENERATION FAILED for {blur_output_name}:")
                                print(f"       Error: {str(e)}")

                    # 删除临时基础图文件 (已经复制给clean，不再需要)
                    if base_chart_file.exists():
                        base_chart_file.unlink()

            except Exception as e:
                print(f"    ✗ 生成{difficulty}基础图失败 {csv_file.name}: {e}")

        print(f"    ✓ 生成 {difficulty_clean_count} 个清晰图 和 {difficulty_blur_count} 个 {difficulty} 模糊图")
        total_blur_count += difficulty_blur_count
        total_clean_count += difficulty_clean_count

    print(f"✓ 总共生成 {total_clean_count} 个清晰图 和 {total_blur_count} 个模糊图")

    # 步骤4: 构建分层训练数据集
    print(f"\n4. 构建分层训练数据集...")
    builder = DatasetBuilder()
    builder.split_paired_data_by_difficulty(temp_clean_dir, temp_blur_dir, final_dir,
                                          difficulties=difficulty_levels,
                                          split_ratios=(0.7, 0.15, 0.15))

    # 清理临时目录
    import shutil
    shutil.rmtree(temp_clean_dir, ignore_errors=True)
    shutil.rmtree(temp_blur_dir, ignore_errors=True)
    print(f"✓ 清理临时文件")

    # 统计分层数据集结果
    print(f"\n=== 分层数据集生成完成 ===")
    total_train_clean = total_train_blur = 0
    total_val_clean = total_val_blur = 0
    total_test_clean = total_test_blur = 0

    for difficulty in difficulty_levels:
        difficulty_dir = final_dir / difficulty
        if difficulty_dir.exists():
            train_clean = len(list((difficulty_dir / 'train' / 'clean').glob("*.png")))
            train_blur = len(list((difficulty_dir / 'train' / 'blur').glob("*.png")))
            val_clean = len(list((difficulty_dir / 'val' / 'clean').glob("*.png")))
            val_blur = len(list((difficulty_dir / 'val' / 'blur').glob("*.png")))
            test_clean = len(list((difficulty_dir / 'test' / 'clean').glob("*.png")))
            test_blur = len(list((difficulty_dir / 'test' / 'blur').glob("*.png")))

            print(f"\n{difficulty.upper()} 难度:")
            print(f"  训练集: {train_clean} 清晰 + {train_blur} 模糊")
            print(f"  验证集: {val_clean} 清晰 + {val_blur} 模糊")
            print(f"  测试集: {test_clean} 清晰 + {test_blur} 模糊")

            total_train_clean += train_clean
            total_train_blur += train_blur
            total_val_clean += val_clean
            total_val_blur += val_blur
            total_test_clean += test_clean
            total_test_blur += test_blur

    print(f"\n总计:")
    print(f"训练集: {total_train_clean} 清晰 + {total_train_blur} 模糊")
    print(f"验证集: {total_val_clean} 清晰 + {total_val_blur} 模糊")
    print(f"测试集: {total_test_clean} 清晰 + {total_test_blur} 模糊")
    print(f"数据集保存在: {final_dir.absolute()}")

    return True


def train_model(dataset_path: str, model_type: str = "unet",
                difficulty: str = "easy", epochs: int = 50,
                batch_size: int = 8, learning_rate: float = 1e-4):
    """训练去模糊模型"""
    print(f"=== LineFuse 模型训练 ===")
    print(f"数据集: {dataset_path}")
    print(f"模型类型: {model_type}")
    print(f"难度级别: {difficulty}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from pathlib import Path
        import logging

        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        if not torch.cuda.is_available():
            print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")

        # 验证数据集路径
        dataset_dir = Path(dataset_path) / difficulty
        if not dataset_dir.exists():
            print(f"✗ 数据集路径不存在: {dataset_dir}")
            print("请先运行 'python main.py generate' 生成数据集")
            return False

        # 检查数据集结构
        train_clean = dataset_dir / 'train' / 'clean'
        train_blur = dataset_dir / 'train' / 'blur'
        val_clean = dataset_dir / 'val' / 'clean'
        val_blur = dataset_dir / 'val' / 'blur'

        for path in [train_clean, train_blur, val_clean, val_blur]:
            if not path.exists():
                print(f"✗ 缺少数据集目录: {path}")
                return False

        print(f"✓ 数据集验证通过")

        # 导入模型和训练器
        from src.models.unet_baseline import UNetBaseline
        from src.models.trainer import (
            ModelTrainer, DeblurDataset, get_default_transforms,
            create_loss_function, create_optimizer, create_scheduler
        )

        # 创建数据集
        print("准备数据集...")
        transforms = get_default_transforms(image_size=512)

        train_dataset = DeblurDataset(str(train_clean), str(train_blur), transform=transforms)
        val_dataset = DeblurDataset(str(val_clean), str(val_blur), transform=transforms)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

        print(f"训练集: {len(train_dataset)} 图像对")
        print(f"验证集: {len(val_dataset)} 图像对")

        # 创建模型
        if model_type == "unet":
            model = UNetBaseline(in_channels=3, out_channels=3)
            print(f"✓ 创建U-Net模型")
            print(f"模型参数量: {model.get_model_size():,}")
        elif model_type == "diffusion":
            from src.models.diffusion_model import ConditionalDiffusionModel
            model = ConditionalDiffusionModel(
                in_channels=3,
                out_channels=3,
                sample_size=512,
                num_train_timesteps=1000
            )
            print(f"✓ 创建条件扩散模型")
            print(f"模型参数量: {model.get_model_size():,}")
        else:
            print(f"✗ 不支持的模型类型: {model_type}")
            return False

        # 创建损失函数、优化器和调度器
        loss_fn = create_loss_function('combined')
        optimizer = create_optimizer(model, 'adamw', learning_rate)
        scheduler = create_scheduler(optimizer, 'cosine', epochs)

        # 创建训练器
        trainer = ModelTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            mixed_precision=True
        )

        # 设置保存目录
        save_dir = Path(f'models/{model_type}_{difficulty}')
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"开始训练...")
        print(f"模型将保存到: {save_dir.absolute()}")

        # 开始训练
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            save_dir=save_dir,
            save_every=10
        )

        print(f"\n=== 训练完成 ===")
        print(f"最佳验证PSNR: {results['best_val_psnr']:.2f}dB")
        print(f"最佳验证损失: {results['best_val_loss']:.4f}")
        print(f"模型保存在: {save_dir / 'best_model.pth'}")

        return True

    except ImportError as e:
        print(f"✗ 缺少必要依赖: {e}")
        print("请运行: conda install pytorch torchvision -c pytorch")
        print("或安装GPU版本: conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia")
        return False
    except Exception as e:
        print(f"✗ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='LineFuse - 光谱图像去模糊系统')

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 数据生成命令
    gen_parser = subparsers.add_parser('generate', help='生成训练数据集')
    gen_parser.add_argument('--samples', type=int, default=50,
                           help='生成的光谱样本数量 (默认: 50)')
    gen_parser.add_argument('--output', type=str, default='linefuse_dataset',
                           help='输出目录 (默认: linefuse_dataset)')
    gen_parser.add_argument('--no-style-diversity', action='store_true',
                           help='禁用样式多样化 (默认: 启用)')
    gen_parser.add_argument('--style-level', type=float, default=0.8,
                           help='样式多样化程度 0.0-1.0 (默认: 0.8)')
    gen_parser.add_argument('--target-style', type=str, choices=['scan_document', 'academic_paper', 'lab_notebook', 'field_notes', 'mixed'],
                           help='指定特定样式模板 (mixed=随机混合)')
    gen_parser.add_argument('--image-size', type=int, default=1024,
                           help='图像尺寸 (默认: 1024x1024)')
    gen_parser.add_argument('--line-width', type=float, default=0.8,
                           help='线条粗细 (默认: 0.8)')
    gen_parser.add_argument('--pixel-perfect', action='store_true', default=True,
                           help='启用像素完美对齐 (默认: True)')
    gen_parser.add_argument('--no-pixel-perfect', action='store_true',
                           help='禁用像素完美对齐')
    gen_parser.add_argument('--pure-line-only', action='store_true',
                           help='纯线条模式 (仅线条，无坐标轴等)')
    gen_parser.add_argument('--test-new-blur', action='store_true',
                           help='测试新的模糊效果 (线条断续、区域细化等)')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练去模糊模型')
    train_parser.add_argument('--dataset', type=str, required=True,
                             help='训练数据集路径')
    train_parser.add_argument('--model', type=str, default='unet',
                             choices=['unet', 'diffusion'],
                             help='模型类型 (默认: unet)')
    train_parser.add_argument('--difficulty', type=str, default='easy',
                             choices=['easy', 'medium', 'hard'],
                             help='训练难度级别 (默认: easy)')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='训练轮数 (默认: 50)')
    train_parser.add_argument('--batch-size', type=int, default=8,
                             help='批次大小 (默认: 8)')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='学习率 (默认: 1e-4)')

    # 快速演示命令
    demo_parser = subparsers.add_parser('demo', help='快速演示 (10个样本)')

    args = parser.parse_args()

    if args.command == 'generate':
        # 处理pixel_perfect参数
        pixel_perfect = args.pixel_perfect and not args.no_pixel_perfect

        generate_dataset(
            num_samples=args.samples,
            output_dir=args.output,
            enable_style_diversity=not args.no_style_diversity,
            style_diversity_level=args.style_level,
            image_size=args.image_size,
            line_width=args.line_width,
            pixel_perfect=pixel_perfect,
            pure_line_only=args.pure_line_only,
            target_style=args.target_style
        )
    elif args.command == 'train':
        train_model(
            dataset_path=args.dataset,
            model_type=args.model,
            difficulty=args.difficulty,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    elif args.command == 'demo':
        print("=== LineFuse 快速演示 ===")
        generate_dataset(
            num_samples=10,
            output_dir="demo_dataset",
            enable_style_diversity=True,
            style_diversity_level=0.9  # 演示时使用高多样性
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()