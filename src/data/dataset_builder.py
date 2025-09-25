import shutil
import random
from pathlib import Path
from typing import Union, List, Tuple, Dict
import logging

class DatasetBuilder:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        
    def create_dataset_structure(self, output_dir: Union[str, Path]) -> None:
        output_path = Path(output_dir)
        
        splits = ['train', 'val', 'test']
        types = ['clean', 'blur']
        
        for split in splits:
            for type_name in types:
                dir_path = output_path / split / type_name
                dir_path.mkdir(parents=True, exist_ok=True)
                
        logging.info(f"Created dataset structure at {output_path}")
    
    def split_data_by_difficulty(self,
                               clean_dir: Union[str, Path],
                               blur_dir: Union[str, Path],
                               output_dir: Union[str, Path],
                               difficulties: List[str] = ["easy", "medium", "hard"],
                               split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> None:
        """按难度级别分层构建数据集"""

        clean_path = Path(clean_dir)
        blur_path = Path(blur_dir)
        output_path = Path(output_dir)

        # 获取所有清晰图片
        clean_files = list(clean_path.glob("*.png"))
        if not clean_files:
            raise ValueError(f"No PNG files found in {clean_path}")

        clean_files.sort()
        random.shuffle(clean_files)

        # 计算分割数量
        total_files = len(clean_files)
        train_count = int(total_files * split_ratios[0])
        val_count = int(total_files * split_ratios[1])

        train_files = clean_files[:train_count]
        val_files = clean_files[train_count:train_count + val_count]
        test_files = clean_files[train_count + val_count:]

        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        # 为每个难度级别创建数据集结构
        for difficulty in difficulties:
            difficulty_dir = output_path / difficulty
            self.create_dataset_structure(difficulty_dir)

            # 为每个分割复制对应难度的文件
            for split_name, file_list in splits.items():
                self._copy_difficulty_files_to_split(
                    file_list, clean_path, blur_path,
                    difficulty_dir, split_name, difficulty
                )

        logging.info(f"Difficulty-based dataset created: {len(difficulties)} difficulties, "
                    f"{len(train_files)} train, {len(val_files)} val, {len(test_files)} test each")

    def _copy_difficulty_files_to_split(self,
                                      file_list: List[Path],
                                      clean_dir: Path,
                                      blur_dir: Path,
                                      output_dir: Path,
                                      split_name: str,
                                      difficulty: str) -> None:
        """复制指定难度级别的文件到对应分割"""

        output_clean_dir = output_dir / split_name / 'clean'
        output_blur_dir = output_dir / split_name / 'blur'

        for clean_file in file_list:
            # 复制清晰图片
            shutil.copy2(clean_file, output_clean_dir / clean_file.name)

            # 复制对应难度的模糊图片
            blur_pattern = f"{clean_file.stem}_{difficulty}_*"
            blur_files = list(blur_dir.glob(blur_pattern))

            for blur_file in blur_files:
                shutil.copy2(blur_file, output_blur_dir / blur_file.name)

        logging.info(f"Copied {len(file_list)} clean files and {difficulty} blur files to {split_name}")

    def split_paired_data_by_difficulty(self,
                                      clean_dir: Union[str, Path],
                                      blur_dir: Union[str, Path],
                                      output_dir: Union[str, Path],
                                      difficulties: List[str] = ["easy", "medium", "hard"],
                                      split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> None:
        """
        按难度级别分层构建数据集 - 新版本，处理配对的clean/blur文件
        文件命名格式: spectrum_*_difficulty_clean.png 和 spectrum_*_difficulty_effect.png
        """

        clean_path = Path(clean_dir)
        blur_path = Path(blur_dir)
        output_path = Path(output_dir)

        # 为每个难度级别处理数据
        for difficulty in difficulties:
            print(f"  处理 {difficulty} 难度数据...")

            # 找到该难度的所有clean文件
            clean_pattern = f"*_{difficulty}_clean.png"
            difficulty_clean_files = list(clean_path.glob(clean_pattern))

            if not difficulty_clean_files:
                print(f"    ⚠️  没有找到 {difficulty} 难度的clean文件")
                continue

            # 按基础名称排序并随机打乱
            difficulty_clean_files.sort()
            random.shuffle(difficulty_clean_files)

            # 计算分割数量
            total_files = len(difficulty_clean_files)
            train_count = int(total_files * split_ratios[0])
            val_count = int(total_files * split_ratios[1])

            train_files = difficulty_clean_files[:train_count]
            val_files = difficulty_clean_files[train_count:train_count + val_count]
            test_files = difficulty_clean_files[train_count + val_count:]

            splits = {
                'train': train_files,
                'val': val_files,
                'test': test_files
            }

            # 创建该难度的目录结构
            difficulty_dir = output_path / difficulty
            self.create_dataset_structure(difficulty_dir)

            # 复制文件到各个分割
            for split_name, file_list in splits.items():
                self._copy_paired_difficulty_files_to_split(
                    file_list, clean_path, blur_path,
                    difficulty_dir, split_name, difficulty
                )

            print(f"    ✓ {difficulty}: {len(train_files)} 训练, {len(val_files)} 验证, {len(test_files)} 测试")

    def _copy_paired_difficulty_files_to_split(self,
                                             file_list: List[Path],
                                             clean_dir: Path,
                                             blur_dir: Path,
                                             output_dir: Path,
                                             split_name: str,
                                             difficulty: str) -> None:
        """复制配对的clean/blur文件到对应分割"""

        output_clean_dir = output_dir / split_name / 'clean'
        output_blur_dir = output_dir / split_name / 'blur'

        for clean_file in file_list:
            # 提取基础名称 (移除 _difficulty_clean.png 后缀)
            base_name = clean_file.stem.replace(f"_{difficulty}_clean", "")

            # 复制清晰图片，重命名为简单格式
            final_clean_name = f"{base_name}.png"
            shutil.copy2(clean_file, output_clean_dir / final_clean_name)

            # 找到对应的模糊图片 (新格式: spectrum_*_difficulty_variant_*.png)
            blur_pattern = f"{base_name}_{difficulty}_variant_*.png"
            blur_files = list(blur_dir.glob(blur_pattern))

            # 复制模糊图片，重命名为简单格式
            for i, blur_file in enumerate(blur_files):
                # 将文件名从 spectrum_00001_easy_variant_0.png
                # 改为 spectrum_00001_0.png (保留变体编号)
                variant_num = blur_file.stem.split('_')[-1]
                final_blur_name = f"{base_name}_{variant_num}.png"
                shutil.copy2(blur_file, output_blur_dir / final_blur_name)

        logging.info(f"Copied {len(file_list)} clean files and corresponding blur files to {split_name}")

    def split_data(self,
                  clean_dir: Union[str, Path],
                  blur_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> None:
        
        clean_path = Path(clean_dir)
        blur_path = Path(blur_dir)
        
        clean_files = list(clean_path.glob("*.png"))
        
        if not clean_files:
            raise ValueError(f"No PNG files found in {clean_path}")
        
        clean_files.sort()
        random.shuffle(clean_files)
        
        total_files = len(clean_files)
        train_count = int(total_files * split_ratios[0])
        val_count = int(total_files * split_ratios[1])
        
        train_files = clean_files[:train_count]
        val_files = clean_files[train_count:train_count + val_count]
        test_files = clean_files[train_count + val_count:]
        
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        self.create_dataset_structure(output_dir)
        
        for split_name, file_list in splits.items():
            self._copy_files_to_split(file_list, clean_path, blur_path, 
                                    output_dir, split_name)
            
        logging.info(f"Dataset split completed: {len(train_files)} train, "
                    f"{len(val_files)} val, {len(test_files)} test")
    
    def _copy_files_to_split(self, 
                           file_list: List[Path],
                           clean_dir: Path,
                           blur_dir: Path,
                           output_dir: Path,
                           split_name: str) -> None:
        
        output_clean_dir = Path(output_dir) / split_name / 'clean'
        output_blur_dir = Path(output_dir) / split_name / 'blur'
        
        for clean_file in file_list:
            shutil.copy2(clean_file, output_clean_dir / clean_file.name)
            
            blur_files = list(blur_dir.glob(f"{clean_file.stem}_*"))
            for blur_file in blur_files:
                shutil.copy2(blur_file, output_blur_dir / blur_file.name)
        
        logging.info(f"Copied {len(file_list)} clean files and corresponding "
                    f"blur files to {split_name} split")
    
    def validate_dataset(self, dataset_dir: Union[str, Path]) -> Dict[str, int]:
        dataset_path = Path(dataset_dir)
        
        stats = {}
        splits = ['train', 'val', 'test']
        
        for split in splits:
            clean_files = list((dataset_path / split / 'clean').glob("*.png"))
            blur_files = list((dataset_path / split / 'blur').glob("*.png"))
            
            stats[f"{split}_clean"] = len(clean_files)
            stats[f"{split}_blur"] = len(blur_files)
            
        total_clean = stats['train_clean'] + stats['val_clean'] + stats['test_clean']
        total_blur = stats['train_blur'] + stats['val_blur'] + stats['test_blur']
        
        stats['total_clean'] = total_clean
        stats['total_blur'] = total_blur
        
        logging.info(f"Dataset validation completed: {stats}")
        return stats