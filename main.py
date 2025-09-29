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
from data.blur_generator import BlurGenerator

def create_comprehensive_blur_demo():
    """
    创建完整的模糊效果演示
    展示每种模糊效果在easy/medium/hard难度下的上下限效果
    """
    print("=== LineFuse 模糊效果完整演示 ===")

    # 检查依赖
    try:
        from data.difficulty_config import get_difficulty_config, get_random_value_in_range
        import cv2
    except ImportError as e:
        print(f"❌ 缺少必要依赖: {e}")
        print("请运行: pip install opencv-python")
        return False

    # 检查CSV数据
    csv_dir = Path('dataset/csv_data')
    if not csv_dir.exists() or not list(csv_dir.glob("*.csv")):
        print(f"❌ 未找到CSV数据目录或文件: {csv_dir}")
        print("请先确保有可用的CSV数据文件")
        return False

    # 选择第一个CSV文件作为演示样本
    csv_file = list(csv_dir.glob("*.csv"))[0]
    print(f"📊 使用样本: {csv_file.name}")

    # 创建演示目录
    demo_dir = Path("blur_effects_demo")
    demo_dir.mkdir(exist_ok=True)

    # 生成基础清晰图表
    print("📈 生成基础清晰图表...")
    base_generator = CleanChartGenerator(
        figure_size=(512, 512),
        line_width=0.5,
        enable_style_diversity=False  # 演示使用固定样式便于对比
    )
    base_image_path = demo_dir / "00_base_clean.png"
    base_generator.process_csv_to_chart(csv_file, base_image_path, pixel_perfect=True)
    print(f"✅ 基础图表已保存: {base_image_path}")

    # 定义要演示的所有效果类型
    demo_effects = {
        # 基础必备效果
        'background_variation': '背景颜色变化',
        'line_thinning_fading': '线条变细和变淡',
        'line_discontinuity': '虚线断续效果',
        'print_noise': '打印噪点',

        # 额外模糊效果
        'gaussian': '高斯模糊',
        'motion': '运动模糊',
        'compression': 'JPEG压缩伪影',
        'scan': '打印扫描模拟',
        'lowres': '低分辨率',
        'text': '文本干扰',
        'lines': '线条干扰',
        'print_scan': '高级打印扫描',
        'localblur': '局部模糊退化',
        'scan_lines': '扫描线条伪影',
        'spectral_degradation': '光谱退化'
    }

    difficulties = ['easy', 'medium', 'hard']

    # 加载基础图像
    base_image = cv2.imread(str(base_image_path))
    if base_image is None:
        print(f"❌ 无法加载基础图像: {base_image_path}")
        return False

    print(f"\n🎨 开始生成 {len(demo_effects)} 种效果 × 3个难度 × 2个强度 = {len(demo_effects) * 6} 张演示图...")

    total_generated = 0

    for effect_name, effect_desc in demo_effects.items():
        print(f"\n📝 {effect_desc} ({effect_name}):")

        for difficulty in difficulties:
            # 获取该难度的配置
            config = get_difficulty_config(difficulty)
            blur_generator = BlurGenerator(difficulty=difficulty)

            # 为每个难度生成最小和最大强度的效果
            for intensity_type in ['min', 'max']:
                output_name = f"{effect_name}_{difficulty}_{intensity_type}.png"
                output_path = demo_dir / output_name

                try:
                    result_image = base_image.copy()
                    effect_log = []

                    if effect_name == 'background_variation':
                        # 背景变化效果
                        bg_config = config['background_variation']
                        if intensity_type == 'min':
                            intensity = bg_config['intensity'][0]
                        else:
                            intensity = bg_config['intensity'][1]
                        result_image = blur_generator.background_color_variation(result_image, intensity=intensity)
                        effect_log.append(f"background_variation(intensity={intensity:.3f})")

                    elif effect_name == 'line_thinning_fading':
                        # 线条变细和变淡效果 - 使用新的绘制时变化方法
                        line_config = config['line_thinning_fading']
                        if intensity_type == 'min':
                            thin_strength = line_config['thinning_strength'][0]
                            fade_strength = line_config['fading_strength'][0]
                        else:
                            thin_strength = line_config['thinning_strength'][1]
                            fade_strength = line_config['fading_strength'][1]

                        # 使用新的matplotlib绘制时线条变化方法
                        blur_generator.generate_chart_with_line_variations(
                            csv_data_path=csv_file,  # 使用演示CSV文件
                            output_path=output_path,
                            thinning_strength=thin_strength,
                            fading_strength=fade_strength,
                            dash_density=0.0  # demo中不加虚线效果
                        )
                        # 直接跳过图像保存，因为已经通过matplotlib保存了
                        continue  # 跳过后续的cv2.imwrite，直接进入下一个效果

                    elif effect_name == 'line_discontinuity':
                        # 虚线效果
                        disc_config = config['line_discontinuity']
                        if intensity_type == 'min':
                            gap_density = disc_config['gap_density'][0]
                            gap_size_range = (disc_config['gap_size_range'][0][0], disc_config['gap_size_range'][0][1])
                        else:
                            gap_density = disc_config['gap_density'][1]
                            gap_size_range = (disc_config['gap_size_range'][1][0], disc_config['gap_size_range'][1][1])
                        result_image = blur_generator.line_discontinuity_blur(
                            result_image, gap_density=gap_density, gap_size_range=gap_size_range)
                        effect_log.append(f"line_discontinuity(density={gap_density:.3f}, size={gap_size_range})")

                    elif effect_name == 'print_noise':
                        # 打印噪点效果
                        noise_config = config['print_noise']
                        if intensity_type == 'min':
                            noise_intensity = noise_config['noise_intensity'][0]
                        else:
                            noise_intensity = noise_config['noise_intensity'][1]
                        result_image = blur_generator.add_print_noise(result_image, intensity=noise_intensity)
                        effect_log.append(f"print_noise(intensity={noise_intensity:.3f})")

                    # 配置化额外效果处理
                    elif effect_name == 'gaussian':
                        if 'gaussian_blur' in config:
                            gauss_config = config['gaussian_blur']
                            if intensity_type == 'min':
                                kernel_range = gauss_config['kernel_size_range'][0]
                                sigma_range = gauss_config['sigma_range'][0]
                            else:
                                kernel_range = gauss_config['kernel_size_range'][1]
                                sigma_range = gauss_config['sigma_range'][1]
                            kernel_size = kernel_range[1]  # 使用上限
                            sigma_range = [sigma_range[1], sigma_range[1]]  # 使用上限
                            result_image = blur_generator.gaussian_blur(result_image,
                                                                       kernel_size=kernel_size, sigma_range=sigma_range)
                            effect_log.append(f"gaussian(kernel={kernel_size}, sigma={sigma_range[1]:.2f})")
                        else:
                            result_image = blur_generator.apply_single_blur_effect(result_image, effect_name)
                            effect_log.append(f"gaussian(default)")

                    elif effect_name == 'motion':
                        if 'motion_blur' in config:
                            motion_config = config['motion_blur']
                            if intensity_type == 'min':
                                kernel_range = motion_config['kernel_size_range'][0]
                            else:
                                kernel_range = motion_config['kernel_size_range'][1]
                            kernel_size = kernel_range[1]  # 使用上限
                            result_image = blur_generator.motion_blur(result_image, kernel_size=kernel_size)
                            effect_log.append(f"motion(kernel={kernel_size})")
                        else:
                            result_image = blur_generator.apply_single_blur_effect(result_image, effect_name)
                            effect_log.append(f"motion(default)")

                    elif effect_name == 'compression':
                        if 'compression' in config:
                            comp_config = config['compression']
                            if intensity_type == 'min':
                                quality_range = comp_config['quality_range'][0]
                            else:
                                quality_range = comp_config['quality_range'][1]
                            quality = quality_range[0]  # 使用下限（更低质量=更强压缩）
                            result_image = blur_generator.compression_blur(result_image, quality=quality)
                            effect_log.append(f"compression(quality={quality})")
                        else:
                            result_image = blur_generator.apply_single_blur_effect(result_image, effect_name)
                            effect_log.append(f"compression(default)")

                    elif effect_name == 'lowres':
                        if 'lowres' in config:
                            lowres_config = config['lowres']
                            if intensity_type == 'min':
                                factor_range = lowres_config['downscale_factor_range'][0]
                            else:
                                factor_range = lowres_config['downscale_factor_range'][1]
                            factor = factor_range[1]  # 使用上限（更大下采样=更模糊）
                            result_image = blur_generator.low_resolution_blur(result_image, downscale_factor=factor)
                            effect_log.append(f"lowres(factor={factor})")
                        else:
                            result_image = blur_generator.apply_single_blur_effect(result_image, effect_name)
                            effect_log.append(f"lowres(default)")

                    elif effect_name == 'spectral_degradation':
                        if 'spectral_degradation' in config:
                            spec_config = config['spectral_degradation']
                            if intensity_type == 'min':
                                strength = spec_config['degradation_strength'][0]
                                range_pct = spec_config['range_percentage'][0]
                            else:
                                strength = spec_config['degradation_strength'][1]
                                range_pct = spec_config['range_percentage'][1]
                            # 这里需要计算x_range
                            w = result_image.shape[1]
                            range_width = int(w * range_pct)
                            x_start = random.randint(int(w * 0.1), int(w * 0.5))
                            x_range = (x_start, min(x_start + range_width, w))
                            result_image = blur_generator.spectral_line_degradation(result_image, x_range=x_range)
                            effect_log.append(f"spectral_degradation(strength={strength:.2f}, range={range_pct:.2f})")
                        else:
                            result_image = blur_generator.apply_single_blur_effect(result_image, effect_name)
                            effect_log.append(f"spectral_degradation(default)")

                    else:
                        # 其他效果：大多数效果不需要外部intensity参数，有内置的随机性
                        # 对于需要intensity参数的效果，使用基于配置的合理范围

                        # 为不同效果定义合理的强度范围
                        effect_intensity_ranges = {
                            'text': {'easy': (0.1, 0.2), 'medium': (0.2, 0.4), 'hard': (0.3, 0.5)},
                            'lines': {'easy': (0.1, 0.2), 'medium': (0.2, 0.4), 'hard': (0.3, 0.5)},
                            'scan_lines': {'easy': (0.1, 0.2), 'medium': (0.2, 0.4), 'hard': (0.3, 0.5)},
                            'localblur': {'easy': (0.2, 0.4), 'medium': (0.4, 0.6), 'hard': (0.5, 0.7)},
                        }

                        if effect_name in effect_intensity_ranges:
                            # 使用配置化的强度范围
                            intensity_range = effect_intensity_ranges[effect_name][difficulty]
                            if intensity_type == 'min':
                                effect_intensity = intensity_range[0]  # 使用最小值
                            else:
                                effect_intensity = intensity_range[1]  # 使用最大值

                            result_image = blur_generator.apply_single_blur_effect(result_image, effect_name,
                                                                                 intensity=effect_intensity)
                            effect_log.append(f"{effect_name}(intensity={effect_intensity:.2f})")
                        else:
                            # 对于不需要intensity参数的效果（如scan, print_scan），直接调用
                            result_image = blur_generator.apply_single_blur_effect(result_image, effect_name)
                            effect_log.append(f"{effect_name}(default)")

                    # 保存结果
                    cv2.imwrite(str(output_path), result_image)
                    total_generated += 1

                    print(f"  ✅ {difficulty.upper()} {intensity_type}: {', '.join(effect_log)} → {output_name}")

                except Exception as e:
                    print(f"  ❌ {difficulty.upper()} {intensity_type}: 生成失败 - {str(e)}")

    print(f"\n🎉 演示完成! 共生成 {total_generated} 张图片")
    print(f"📁 所有演示图片保存在: {demo_dir.absolute()}")
    print(f"\n📋 演示内容:")
    print(f"  • 基础清晰图: 00_base_clean.png")
    print(f"  • 每种效果的6个变体: [效果名]_[难度]_[强度].png")
    print(f"  • 难度: easy/medium/hard")
    print(f"  • 强度: min(最小)/max(最大)")

    return True

def generate_dataset(num_samples: int = 10, output_dir: str = "linefuse_dataset",
                    difficulty_levels: list = ["easy", "medium", "hard"],
                    enable_style_diversity: bool = True,
                    style_diversity_level: float = 0.8,
                    image_size: int = 1024,
                    line_width: float = 0.8,
                    pixel_perfect: bool = True,
                    pure_line_only: bool = False,
                    target_style: str = None,
                    clean_only: bool = False):
    """生成完整的训练数据集"""
    print(f"=== LineFuse 数据集生成 ===")
    print(f"使用样本数量: {num_samples}")
    print(f"输出目录: {output_dir}")

    if clean_only:
        print(f"🎯 清晰图表模式: 仅生成清晰图表，跳过模糊处理")
    else:
        print(f"难度级别: {', '.join(difficulty_levels)}")

    print(f"样式多样化: {'启用' if enable_style_diversity else '禁用'}")
    if enable_style_diversity:
        print(f"多样化程度: {style_diversity_level:.1f} (0.0=最低, 1.0=最高)")
        if target_style:
            print(f"目标样式: {target_style}")
    print(f"像素完美对齐: {'启用' if pixel_perfect else '禁用'}")
    print(f"纯线条模式: {'启用' if pure_line_only else '禁用'}")

    # DEPRECATED: 使用新的配置系统 src/data/difficulty_config.py
    # 这个旧配置保留仅用于向后兼容，将来会被移除
    from src.data.difficulty_config import get_difficulty_config, get_global_config

    # 为向后兼容，构建描述映射
    difficulty_descriptions = {
        "easy": "轻度模糊，线条适中",
        "medium": "中度模糊，线条很细",
        "hard": "重度模糊，线条极细"
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

    if clean_only:
        # 清晰模式：直接创建清晰图表目录
        clean_output_dir = Path(output_dir) / 'clean_charts'
        clean_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 正常模式：创建分层数据集目录
        final_dir = Path(output_dir) / 'final_dataset'
        final_dir.mkdir(parents=True, exist_ok=True)

        # 临时目录用于中间处理
        temp_clean_dir = Path(output_dir) / '.temp_clean'
        temp_blur_dir = Path(output_dir) / '.temp_blur'
        temp_clean_dir.mkdir(exist_ok=True)
        temp_blur_dir.mkdir(exist_ok=True)

    if clean_only:
        # 🎯 清晰模式：直接生成清晰图表
        print(f"\n📈 生成清晰光谱图表...")

        total_clean_count = 0

        # 创建清晰图生成器 - 统一使用标准科学图表样式
        clean_generator = CleanChartGenerator(
            figure_size=(image_size, image_size),
            line_width=line_width,
            enable_style_diversity=False,  # 禁用样式多样化，保证统一样式
            style_diversity_level=0.0,     # 确保无随机变化
            target_style='scientific'      # 使用标准科学图表样式（网格背景+完整坐标轴）
        )

        print(f"  📋 样式配置: 统一网格背景 + 完整坐标轴标签 + 'Spectrum Analysis'标题")

        # 为每个CSV文件生成清晰图表
        for csv_file in csv_files:
            output_name = f"{csv_file.stem}_clean.png"
            output_path = clean_output_dir / output_name

            try:
                clean_generator.process_csv_to_chart(csv_file, output_path,
                                                   pure_line_only=pure_line_only,
                                                   pixel_perfect=pixel_perfect)
                total_clean_count += 1
                print(f"  ✅ {output_name}")

            except Exception as e:
                print(f"  ❌ 生成失败 {csv_file.name}: {e}")

        print(f"\n🎉 清晰图表生成完成!")
        print(f"✓ 共生成 {total_clean_count} 张清晰图表")
        print(f"📁 保存位置: {clean_output_dir.absolute()}")

        return True

    else:
        # 🔄 正常模式：生成完整训练数据集
        # 步骤1: 从已有CSV生成清晰图表
        print(f"\n1. 从已有CSV生成清晰光谱图表...")

        # 注意：这里不再直接生成最终的clean图表
        # 而是为每个难度级别生成对应的clean基础图，确保clean/blur背景一致
        print("  清晰图表将按难度级别生成以确保与模糊图表背景一致...")

        # 步骤2: 按难度级别生成统一基础图和配对的clean/blur图
        print(f"\n2. 按难度级别生成统一基础图和配对的clean/blur图...")

    # 检查是否可以生成模糊效果
    can_generate_blur = True

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
        config = get_difficulty_config(difficulty)  # 使用新配置系统
        description = difficulty_descriptions.get(difficulty, f"{difficulty} 难度")
        print(f"\n  生成 {difficulty} 难度 ({description})...")

        difficulty_blur_count = 0
        difficulty_clean_count = 0

        # 如果可以生成模糊效果，初始化模糊生成器
        if can_generate_blur:
            blur_generator = BlurGenerator(difficulty=difficulty)

        # 获取全局配置
        global_config = get_global_config()

        # 为每个CSV文件生成统一基础图，然后配对生成clean/blur
        for csv_file in csv_files:
            # 创建clean图生成器 - 使用统一的line_width
            clean_generator = CleanChartGenerator(
                figure_size=(image_size, image_size),
                line_width=global_config['clean_line_width'],  # 统一的粗线条
                enable_style_diversity=enable_style_diversity,
                style_diversity_level=style_diversity_level,
                target_style=target_style
            )

            # 创建blur图生成器 - 使用难度相关的line_width
            blur_base_generator = CleanChartGenerator(
                figure_size=(image_size, image_size),
                line_width=config['line_width'],  # 难度相关的线条粗细
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

                    # 1. 生成纯线条的clean图 - 使用统一粗线条
                    clean_output_name = f"{csv_file.stem}_{difficulty}_clean.png"
                    clean_output_path = temp_clean_dir / clean_output_name
                    clean_generator.process_csv_to_chart(csv_file, clean_output_path,
                                                       pure_line_only=True,  # clean图纯线条
                                                       pixel_perfect=pixel_perfect)
                    difficulty_clean_count += 1

                    # 2. 如果需要模糊效果，生成有坐标的基础图用于模糊 - 使用难度相关的细线条
                    if can_generate_blur:
                        blur_base_file = temp_clean_dir / f"{csv_file.stem}_{difficulty}_blur_base.png"
                        blur_base_generator.process_csv_to_chart(csv_file, blur_base_file,
                                                               pure_line_only=False,  # blur基础图有坐标
                                                               pixel_perfect=pixel_perfect)

                        # 生成3张不同的模糊图变体
                        for variant in range(3):
                            blur_output_name = f"{csv_file.stem}_{difficulty}_variant_{variant}.png"
                            blur_output_path = temp_blur_dir / blur_output_name

                            try:
                                # 1. 准备基础退化参数（获取线条变化参数，不处理图像）
                                dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
                                _, _ = blur_generator.apply_base_degradation(dummy_image)  # 虚拟调用获取参数

                                # 2. 使用带线条变化的绘制器直接生成模糊图
                                if hasattr(blur_generator, 'line_variation_params'):
                                    # 使用新的绘制时线条变化方法

                                    # 创建带线条变化的生成器
                                    chart_generator = CleanChartGenerator(
                                        figure_size=(image_size, image_size),
                                        line_width=config['line_width'],
                                        enable_style_diversity=enable_style_diversity,
                                        style_diversity_level=style_diversity_level,
                                        target_style=target_style,
                                        enable_line_variations=True
                                    )

                                    # 加载CSV数据
                                    csv_data = chart_generator.load_csv_data(csv_file)
                                    data = csv_data['data']
                                    columns = csv_data['columns']
                                    x_data = data[:, 0]  # 第一列为x轴数据
                                    y_data = data[:, 1]  # 第二列为y轴数据

                                    # 生成带线条变化的图表
                                    chart_generator.generate_clean_chart(
                                        x_data, y_data,
                                        output_path=str(blur_output_path),
                                        pure_line_only=False,  # blur基础图有坐标
                                        pixel_perfect=pixel_perfect,
                                        line_variation_params=blur_generator.line_variation_params
                                    )

                                    # 加载生成的图像以便后续处理
                                    base_degraded = blur_generator.load_image(blur_output_path)
                                    base_effects_log = [f"line_variations(drawing-time)"]
                                else:
                                    # 回退到原来的图像处理方法
                                    base_image = blur_generator.load_image(blur_base_file)
                                    base_degraded, base_effects_log = blur_generator.apply_base_degradation(base_image)

                                # 3. 随机添加额外效果
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

                        # 删除模糊基础图文件
                        if blur_base_file.exists():
                            blur_base_file.unlink()

                else:
                    # 标准模式：分别生成clean和blur图

                    # 1. 生成clean图（使用统一粗线条）
                    clean_output_name = f"{csv_file.stem}_{difficulty}_clean.png"
                    clean_output_path = temp_clean_dir / clean_output_name
                    clean_generator.process_csv_to_chart(csv_file, clean_output_path,
                                                       pure_line_only=False,  # 标准模式有坐标
                                                       pixel_perfect=pixel_perfect)
                    difficulty_clean_count += 1

                    # 2. 生成blur基础图（使用难度相关的细线条）
                    blur_base_generator.process_csv_to_chart(csv_file, base_chart_file,
                                                           pure_line_only=False,  # 标准模式有坐标
                                                           pixel_perfect=pixel_perfect)

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
    gen_parser.add_argument('--clean-only', action='store_true',
                           help='仅生成清晰图表，跳过模糊图生成过程')

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
            target_style=args.target_style,
            clean_only=args.clean_only
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
        create_comprehensive_blur_demo()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()