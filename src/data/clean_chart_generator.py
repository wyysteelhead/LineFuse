import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging

class CleanChartGenerator:
    def __init__(self, 
                 figure_size: tuple = (512, 512),
                 dpi: int = 150,
                 background_color: str = 'white',
                 line_color: str = 'black',
                 font_family: str = 'Arial'):
        self.figure_size = figure_size
        self.dpi = dpi
        self.background_color = background_color
        self.line_color = line_color
        self.font_family = font_family
        
        plt.rcParams['font.family'] = self.font_family
        
    def load_csv_data(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        try:
            data = pd.read_csv(csv_path)
            return data
        except Exception as e:
            logging.error(f"Error loading CSV file {csv_path}: {e}")
            raise
    
    def generate_clean_chart(self, 
                           x_data: np.ndarray, 
                           y_data: np.ndarray,
                           output_path: Union[str, Path],
                           xlabel: str = "Wavelength",
                           ylabel: str = "Intensity",
                           title: Optional[str] = None) -> None:
        
        fig_width = self.figure_size[0] / self.dpi
        fig_height = self.figure_size[1] / self.dpi
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor(self.background_color)
        ax.set_facecolor(self.background_color)
        
        ax.plot(x_data, y_data, color=self.line_color, linewidth=1.5)
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if title:
            ax.set_title(title, fontsize=12)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, 
                   dpi=self.dpi, 
                   bbox_inches='tight',
                   facecolor=self.background_color,
                   edgecolor='none')
        plt.close()
        
    def process_csv_to_chart(self, 
                           csv_path: Union[str, Path],
                           output_path: Union[str, Path],
                           x_column: str = None,
                           y_column: str = None,
                           **kwargs) -> None:
        
        data = self.load_csv_data(csv_path)
        
        if x_column is None:
            x_column = data.columns[0]
        if y_column is None:
            y_column = data.columns[1]
            
        x_data = data[x_column].values
        y_data = data[y_column].values
        
        self.generate_clean_chart(x_data, y_data, output_path, **kwargs)
        
    def batch_process(self, 
                     input_dir: Union[str, Path],
                     output_dir: Union[str, Path],
                     pattern: str = "*.csv") -> None:
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(input_path.glob(pattern))
        
        for csv_file in csv_files:
            output_file = output_path / f"{csv_file.stem}.png"
            try:
                self.process_csv_to_chart(csv_file, output_file)
                logging.info(f"Generated chart: {output_file}")
            except Exception as e:
                logging.error(f"Failed to process {csv_file}: {e}")