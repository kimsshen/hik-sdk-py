from abc import abstractmethod

import torch
import cv2
import numpy as np
import os
import time
from pathlib import Path
import pandas as pd
import logging
import shutil

from PIL import Image, ImageDraw, ImageFont

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("packaging_inspection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PackagingInspector")

class BasicModel:

    @abstractmethod
    def load_image(self, image_path):
        """
        加载图像并进行预处理
        :param image_path: 图像文件路径
        :return: 预处理后的图像 (RGB格式)
        """
        pass

    @abstractmethod
    def predict_objects(self, image):
        """
        执行物体检测
        :param image: 输入图像 (RGB格式)
        :return: 检测结果DataFrame
        """
        pass

    @abstractmethod
    def analyze_detections(self, detections):
        """
        分析检测结果并判断是否符合条件
        :param detections: 检测结果DataFrame
        :return: 
            status: 检测状态 ("OK" 或 "NG")
            message: 状态描述
            class_counts: 每个类别的检测数量
        """
        pass

    @abstractmethod
    def visualize_results(self, image, detections, status, message, class_counts):
        """
        可视化检测结果
        :param image: 原始图像 (RGB格式)
        :param detections: 检测结果DataFrame
        :param status: 检测状态 ("OK" 或 "NG")
        :param message: 状态描述
        :param class_counts: 每个类别的检测数量
        :return: 可视化后的图像 (RGB格式)
        """
        pass

    @abstractmethod
    def process_image(self, image_path, output_dir="results"):
        """
        处理单张图像并保存结果
        :param image_path: 输入图像路径
        :param output_dir: 输出目录
        :return: 检测状态和结果图像路径
        """
        pass

    @abstractmethod
    def save_results_to_csv(self, image_path, detections, status, output_dir):
        """
        保存检测结果到CSV文件
        :param image_path: 原始图像路径
        :param detections: 检测结果
        :param status: 检测状态
        :param output_dir: 输出目录
        """
        pass

    @abstractmethod
    def process_batch(self, image_dir, output_dir="results"):
        """
        批量处理目录中的所有图像
        :param image_dir: 图像目录路径
        :param output_dir: 输出目录
        :return: 处理结果统计
        """
        pass

    @abstractmethod
    def quick_empty_directory(directory_path):
        """
        快速清空目录 - 删除整个目录后重新创建

        参数:
            directory_path (str): 要清空的目录路径
        """
        pass

