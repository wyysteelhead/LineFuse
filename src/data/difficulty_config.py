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
        'regional_thinning': {
            'num_regions': (0, 1),  # 允许0个区域，即可能完全跳过该效果
            'thinning_strength': (0.05, 0.15),  # 进一步降低 0.1-0.25 → 0.05-0.15
            'color_variation': False
        },
        'line_discontinuity': {
            'gap_density': (0.01, 0.05),  # 进一步降低 0.03-0.08 → 0.01-0.05
            'gap_size_range': ((1, 1), (1, 2))  # 更小的间隙
        },
        'print_noise': {
            'noise_intensity': (0.002, 0.008)  # 进一步降低 0.005-0.015 → 0.002-0.008
        },
        'background_variation': {
            'intensity': (0.05, 0.12)  # 降低 0.1-0.2 → 0.05-0.12
        },

        # 额外效果参数范围
        'gaussian_blur': {
            'kernel_size_range': ((3, 5), (3, 7)),
            'sigma_range': ((0.3, 0.8), (0.5, 1.2))
        },
        'motion_blur': {
            'kernel_size_range': ((3, 6), (3, 8))
        },
        'compression': {
            'quality_range': ((50, 80), (40, 70))
        },
        'lowres': {
            'downscale_factor_range': ((2, 3), (2, 4))
        },
        'spectral_degradation': {
            'degradation_strength': (0.1, 0.2),
            'range_percentage': (0.2, 0.35)
        },
        'threshold': {
            'threshold_range': ((50, 80), (45, 85))  # 更低的阈值，保护线条区域
        }
    },

    'medium': {
        'name': '中等',
        'base_intensity_range': (0.15, 0.25),  # 提高基础强度确保足够模糊
        'additional_intensity_range': (0.2, 0.3),  # 提高额外效果强度
        'additional_effects_count': (1, 1),  # 确保每张图都有1个额外效果
        'line_width': 0.3,

        'regional_thinning': {
            'num_regions': (1, 2),  # 确保有区域细化 (0, 1) → (1, 2)
            'thinning_strength': (0.3, 0.5),  # 大幅提高让线条变化明显 0.1-0.25 → 0.3-0.5
            'color_variation': False  # 暂时禁用颜色变化
        },
        'line_discontinuity': {
            'gap_density': (0.08, 0.15),  # 提高虚线密度 0.02-0.06 → 0.08-0.15
            'gap_size_range': ((1, 2), (1, 3))  # 增大间隙 (1, 1), (1, 1) → (1, 2), (1, 3)
        },
        'print_noise': {
            'noise_intensity': (0.01, 0.025)  # 提高噪点 0.003-0.012 → 0.01-0.025
        },
        'background_variation': {
            'intensity': (0.1, 0.2)  # 提高背景变化 0.06-0.12 → 0.1-0.2
        },

        'gaussian_blur': {
            'kernel_size_range': ((3, 5), (3, 7)),  # 减小kernel (3, 6), (5, 9) → (3, 5), (3, 7)
            'sigma_range': ((0.3, 0.6), (0.5, 1.0))  # 减小sigma (0.5, 1.0), (0.8, 1.5) → (0.3, 0.6), (0.5, 1.0)
        },
        'motion_blur': {
            'kernel_size_range': ((3, 5), (3, 7))  # 减小kernel (3, 7), (5, 10) → (3, 5), (3, 7)
        },
        'compression': {
            'quality_range': ((40, 70), (50, 80))  # 提高质量 (25, 50), (40, 65) → (40, 70), (50, 80)
        },
        'lowres': {
            'downscale_factor_range': ((2, 3), (2, 4))  # 减少下采样 (2, 4), (3, 5) → (2, 3), (2, 4)
        },
        'spectral_degradation': {
            'degradation_strength': (0.1, 0.25),  # 减少强度 0.2-0.35 → 0.1-0.25
            'range_percentage': (0.25, 0.35)  # 减少范围 0.3-0.45 → 0.25-0.35
        },
        'threshold': {
            'threshold_range': ((60, 90), (50, 95))  # 更低的阈值，确保不影响线条抗锯齿
        }
    },

    'hard': {
        'name': '困难',
        'base_intensity_range': (0.25, 0.35),  # 提高基础强度确保比medium更模糊
        'additional_intensity_range': (0.3, 0.4),  # 提高额外效果强度
        'additional_effects_count': (1, 2),  # 确保有1-2个额外效果
        'line_width': 0.15,

        'regional_thinning': {
            'num_regions': (2, 3),  # 增加区域数量 (1, 2) → (2, 3)
            'thinning_strength': (0.4, 0.6),  # 提高强度让变化更明显 0.2-0.4 → 0.4-0.6
            'color_variation': False  # 暂时禁用颜色变化
        },
        'line_discontinuity': {
            'gap_density': (0.12, 0.22),  # 提高虚线密度 0.06-0.15 → 0.12-0.22
            'gap_size_range': ((1, 3), (2, 4))  # 增大间隙
        },
        'print_noise': {
            'noise_intensity': (0.015, 0.04)  # 提高噪点 0.008-0.025 → 0.015-0.04
        },
        'background_variation': {
            'intensity': (0.15, 0.28)  # 提高背景变化 0.12-0.22 → 0.15-0.28
        },

        'gaussian_blur': {
            'kernel_size_range': ((3, 7), (5, 9)),  # 减小kernel (5, 9), (7, 13) → (3, 7), (5, 9)
            'sigma_range': ((0.4, 1.0), (0.6, 1.4))  # 大幅降低sigma (0.8, 1.8), (1.2, 2.5) → (0.4, 1.0), (0.6, 1.4)
        },
        'motion_blur': {
            'kernel_size_range': ((3, 8), (5, 12))  # 大幅减小 (5, 12), (8, 16) → (3, 8), (5, 12)
        },
        'compression': {
            'quality_range': ((40, 75), (50, 80))  # 进一步提高质量 (30, 60), (40, 70) → (40, 75), (50, 80)
        },
        'lowres': {
            'downscale_factor_range': ((2, 4), (3, 5))  # 减少下采样 (3, 5), (4, 6) → (2, 4), (3, 5)
        },
        'spectral_degradation': {
            'degradation_strength': (0.15, 0.35),  # 大幅降低 0.3-0.5 → 0.15-0.35
            'range_percentage': (0.3, 0.45)  # 减少范围 0.4-0.6 → 0.3-0.45
        },
        'threshold': {
            'threshold_range': ((70, 100), (60, 105))  # 更低的阈值，避免线条消失
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