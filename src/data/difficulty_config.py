"""
Difficulty configuration for blur effects
模糊效果的难度级别配置
"""

# 难度级别配置
DIFFICULTY_CONFIG = {
    'easy': {
        'name': '简单',
        'base_intensity_range': (0.05, 0.15),  # 基础效果强度范围
        'additional_intensity_range': (0.1, 0.2),  # 额外效果强度范围
        'additional_effects_count': (0, 0),  # Easy难度禁用额外效果
        'line_width': 1.0,  # 线条粗细

        # 基础效果参数范围
        'line_thinning_fading': {
            'thinning_strength': (0.1, 0.2),  # Easy: 轻微变细
            'fading_strength': (0.1, 0.2),    # Easy: 轻微变淡
            'num_regions': (0, 1),            # Easy: 0-1个区域，保持大部分正常
        },
        'line_discontinuity': {
            'gap_density': (0.01, 0.05),  # 进一步降低 0.03-0.08 → 0.01-0.05
            'gap_size_range': ((1, 1), (1, 2))  # 更小的间隙
        },
        'print_noise': {
            'noise_intensity': (0.002, 0.008)  # 进一步降低 0.005-0.015 → 0.002-0.008
        },
        'background_variation': {
            'intensity': (0.1, 0.2)
        },

        # 额外效果参数范围
        'gaussian_blur': {
            'kernel_size_range': ((3, 5), (3, 7)),
            'sigma_range': ((0.3, 0.6), (0.5, 0.8))  # 修复：降低重叠，增加区分度
        },
        'motion_blur': {
            'kernel_size_range': ((3, 5), (3, 7))  # 修复：增加kernel范围，确保有模糊效果
        },
        'compression': {
            'quality_range': ((25, 40), (20, 35))
        },
        'lowres': {
            'downscale_factor_range': ((0, 1), (1, 2))
        },
        'spectral_degradation': {
            'degradation_strength': (0.1, 0.2),
            'range_percentage': (0.2, 0.35)
        },
        'threshold': {
            'threshold_range': ((50, 80), (45, 85))  # 更低的阈值，保护线条区域
        },

        # 新增效果配置
        'scan': {
            'blur_strength': 0.8,           # 扫描模糊强度
            'contrast_reduction': 0.9,      # 对比度降低程度
            'noise_level': 0.1              # 扫描噪声等级
        },
        'text': {
            'num_texts': (1, 2),            # 文本数量范围
            'font_size': (12, 18),          # 字体大小范围
            'opacity': (0.3, 0.5)           # 文本透明度范围
        },
        'lines': {
            'num_lines': (1, 2),            # 干扰线数量范围
            'thickness': (1, 2),            # 线条粗细范围
            'opacity': (0.4, 0.6)           # 线条透明度范围
        }
    },

    'medium': {
        'name': '中等',
        'base_intensity_range': (0.15, 0.25),  # 提高基础强度确保足够模糊
        'additional_intensity_range': (0.2, 0.35),  # 提高额外效果强度
        'additional_effects_count': (0, 1),  # 减少额外效果让线条变化更突出
        'line_width': 0.3,

        'line_thinning_fading': {
            'thinning_strength': (0.2, 0.4),  # Medium: 中等变细
            'fading_strength': (0.2, 0.4),    # Medium: 中等变淡
            'num_regions': (1, 2),            # Medium: 1-2个区域
        },
        'line_discontinuity': {
            'gap_density': (0.06, 0.15),  # 提高虚线密度 0.02-0.06 → 0.06-0.15
            'gap_size_range': ((1, 2), (1, 3))  # 增大间隙 (1, 1), (1, 1) → (1, 2), (1, 3)
        },
        'print_noise': {
            'noise_intensity': (0.008, 0.025)  # 提高噪点 0.003-0.012 → 0.008-0.025
        },
        'background_variation': {
            'intensity': (0.4, 0.6)  # 提高背景变化 0.06-0.12 → 0.12-0.22
        },

        'gaussian_blur': {
            'kernel_size_range': ((5, 7), (7, 9)),  # 修复：中等难度增加模糊
            'sigma_range': ((0.8, 1.2), (1.0, 1.5))  # 修复：增加区分度
        },
        'motion_blur': {
            'kernel_size_range': ((5, 7), (5, 9))  # 修复：中等难度增加模糊
        },
        'compression': {
            'quality_range': ((10, 20), (5, 15))  # 降低质量增加压缩伪影
        },
        'lowres': {
            'downscale_factor_range': ((1, 2), (1, 3))  # 增加下采样确保模糊
        },
        'spectral_degradation': {
            'degradation_strength': (0.2, 0.35),  # 增加强度确保明显退化
            'range_percentage': (0.3, 0.45)  # 增加范围影响更多区域
        },
        'threshold': {
            'threshold_range': ((60, 90), (50, 95))  # 更低的阈值，确保不影响线条抗锯齿
        },

        # 新增效果配置 - 中等难度
        'scan': {
            'blur_strength': 1.2,           # 中等扫描模糊强度
            'contrast_reduction': 0.8,      # 更多对比度降低
            'noise_level': 0.15             # 更多扫描噪声
        },
        'text': {
            'num_texts': (2, 3),            # 更多文本数量
            'font_size': (14, 22),          # 更大字体大小
            'opacity': (0.4, 0.6)           # 更高透明度
        },
        'lines': {
            'num_lines': (2, 3),            # 更多干扰线
            'thickness': (2, 3),            # 更粗线条
            'opacity': (0.5, 0.7)           # 更高透明度
        }
    },

    'hard': {
        'name': '困难',
        'base_intensity_range': (0.25, 0.35),  # 提高基础强度确保比medium更模糊
        'additional_intensity_range': (0.35, 0.5),  # 提高额外效果强度
        'additional_effects_count': (0, 2),  # 减少额外效果突出线条变化特征
        'line_width': 0.15,

        'line_thinning_fading': {
            'thinning_strength': (0.3, 0.6),  # Hard: 较强变细但不过度
            'fading_strength': (0.3, 0.6),    # Hard: 较强变淡但不过度
            'num_regions': (1, 3),            # Hard: 1-3个区域
        },
        'line_discontinuity': {
            'gap_density': (0.15, 0.3),  # 提高虚线密度 0.06-0.15 → 0.15-0.3
            'gap_size_range': ((1, 3), (2, 4))  # 增大间隙
        },
        'print_noise': {
            'noise_intensity': (0.02, 0.05)  # 提高噪点 0.008-0.025 → 0.02-0.05
        },
        'background_variation': {
            'intensity': (0.7, 0.9)  # 提高背景变化 0.12-0.22 → 0.18-0.35
        },

        'gaussian_blur': {
            'kernel_size_range': ((7, 11), (9, 13)),  # 修复：困难难度最大模糊
            'sigma_range': ((1.5, 2.0), (1.8, 2.5))  # 修复：最大区分度
        },
        'motion_blur': {
            'kernel_size_range': ((7, 11), (9, 15))  # 修复：困难难度最大模糊
        },
        'compression': {
            'quality_range': ((5, 10), (2, 5))  # 更低质量确保明显压缩伪影
        },
        'lowres': {
            'downscale_factor_range': ((2, 4), (3, 5))  # 更大下采样因子确保更模糊
        },
        'spectral_degradation': {
            'degradation_strength': (0.3, 0.5),  # 更强的退化确保比medium更明显
            'range_percentage': (0.4, 0.6)  # 更大范围影响
        },
        'threshold': {
            'threshold_range': ((70, 100), (60, 105))  # 更低的阈值，避免线条消失
        },

        # 新增效果配置 - 困难难度
        'scan': {
            'blur_strength': 1.8,           # 最强扫描模糊
            'contrast_reduction': 0.7,      # 最大对比度降低
            'noise_level': 0.25             # 最多扫描噪声
        },
        'text': {
            'num_texts': (3, 5),            # 最多文本数量
            'font_size': (16, 28),          # 最大字体范围
            'opacity': (0.5, 0.8)           # 最高透明度
        },
        'lines': {
            'num_lines': (3, 5),            # 最多干扰线
            'thickness': (2, 4),            # 最粗线条
            'opacity': (0.6, 0.8)           # 最高透明度
        }
    }
}

def get_difficulty_config(difficulty: str) -> dict:
    """获取指定难度的配置"""
    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"Unknown difficulty: {difficulty}. Available: {list(DIFFICULTY_CONFIG.keys())}")
    return DIFFICULTY_CONFIG[difficulty]

def get_random_value_in_range(value_range: tuple, is_int: bool = False):
    """在指定范围内随机取值"""
    import random
    min_val, max_val = value_range
    if is_int:
        return random.randint(min_val, max_val)
    else:
        return random.uniform(min_val, max_val)

def get_random_range_in_ranges(ranges: tuple, is_int: bool = False):
    """从两个范围中随机选择一个范围，然后在该范围内生成随机范围"""
    import random
    range1, range2 = ranges

    # 随机选择使用哪个基准范围
    if random.random() < 0.5:
        base_range = range1
    else:
        base_range = range2

    # 在选择的基准范围内生成一个子范围
    if is_int:
        # 对于整数，确保范围有效
        min_possible = base_range[0]
        max_possible = base_range[1]

        min_val = random.randint(min_possible, max_possible)
        max_val = random.randint(min_val, max_possible)  # 确保max >= min
    else:
        # 对于浮点数
        min_possible = base_range[0]
        max_possible = base_range[1]

        min_val = random.uniform(min_possible, max_possible)
        max_val = random.uniform(min_val, max_possible)  # 确保max >= min

    return (min_val, max_val)