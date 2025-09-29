"""
Difficulty configuration for blur effects
模糊效果的难度级别配置
"""

# 全局配置
GLOBAL_CONFIG = {
    'clean_line_width': 1.2,  # Clean图统一使用的线条粗细，便于去模糊任务
}

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
            'gap_density': (0.04, 0.06),  # 温和虚线：5-8%覆盖率
            'gap_size_range': ((1, 1), (1, 2))  # 真正小间隙：1-2像素
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
        'additional_effects_count': (1, 2),  # 增加额外效果确保模糊强度
        'line_width': 0.3,

        'line_thinning_fading': {
            'thinning_strength': (0.3, 0.5),  # 提高下限：0.2→0.3
            'fading_strength': (0.3, 0.5),    # 提高下限：0.2→0.3
            'num_regions': (2, 3),            # 提高下限：1→2
        },
        'line_discontinuity': {
            'gap_density': (0.06, 0.12),  # 提高下限：0.08→0.12
            'gap_size_range': ((1, 1), (1, 2))  # 大幅缩小：真正的小间隙
        },
        'print_noise': {
            'noise_intensity': (0.015, 0.035)  # 提高下限：0.008→0.015
        },
        'background_variation': {
            'intensity': (0.5, 0.7)  # 提高下限：0.4→0.5
        },

        'gaussian_blur': {
            'kernel_size_range': ((5, 7), (7, 9)),  # 适度降低：保持可见效果但不过度
            'sigma_range': ((1.0, 1.5), (1.2, 1.8))  # 适度降低：保持可见效果但不过度
        },
        'motion_blur': {
            'kernel_size_range': ((7, 10), (9, 12))  # 提高下限：5→7, 5→9
        },
        'compression': {
            'quality_range': ((5, 15), (3, 10))  # 降低下限：10→5, 5→3
        },
        'lowres': {
            'downscale_factor_range': ((2, 3), (3, 4))  # 提高下限：1→2, 1→3
        },
        'spectral_degradation': {
            'degradation_strength': (0.3, 0.45),  # 提高下限：0.2→0.3
            'range_percentage': (0.35, 0.5)  # 提高下限：0.3→0.35
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
        'additional_effects_count': (1, 3),  # 增加额外效果确保强烈模糊
        'line_width': 0.15,

        'line_thinning_fading': {
            'thinning_strength': (0.5, 0.7),  # 提高下限：0.3→0.5
            'fading_strength': (0.5, 0.7),    # 提高下限：0.3→0.5
            'num_regions': (2, 4),            # 提高下限：1→2
        },
        'line_discontinuity': {
            'gap_density': (0.12, 0.18),  # 提高下限：0.08→0.12
            'gap_size_range': ((1, 1), (1, 2))  # 大幅缩小：真正的小间隙
        },
        'print_noise': {
            'noise_intensity': (0.03, 0.06)  # 提高下限：0.02→0.03
        },
        'background_variation': {
            'intensity': (0.8, 1.0)  # 提高下限：0.7→0.8
        },

        'gaussian_blur': {
            'kernel_size_range': ((7, 11), (9, 13)),  # 降低上限：避免过度模糊
            'sigma_range': ((1.5, 2.2), (1.8, 2.8))  # 降低上限：避免过度模糊
        },
        'motion_blur': {
            'kernel_size_range': ((9, 13), (12, 18))  # 提高下限：7→9, 9→12
        },
        'compression': {
            'quality_range': ((3, 8), (1, 5))  # 降低下限：5→3, 2→1
        },
        'lowres': {
            'downscale_factor_range': ((3, 5), (4, 6))  # 提高下限：2→3, 3→4
        },
        'spectral_degradation': {
            'degradation_strength': (0.4, 0.6),  # 提高下限：0.3→0.4
            'range_percentage': (0.45, 0.7)  # 提高下限：0.4→0.45
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

def get_global_config() -> dict:
    """获取全局配置"""
    return GLOBAL_CONFIG

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