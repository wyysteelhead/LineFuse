import shutil
import random
from pathlib import Path
from typing import Union, List, Tuple
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