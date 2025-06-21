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

class PackagingDetectionSystem:
    def __init__(self, model_path, class_names, class_mapping, object_class_names, conf_threshold=0.6, iou_threshold=0.45):
        """
        初始化包装检测系统
        :param model_path: 训练好的模型权重路径
        :param class_names: 类别名称列表 (必须与训练时的类别顺序一致)
        :param conf_threshold: 置信度阈值 (0-1)
        :param iou_threshold: IOU阈值 (0-1)
        """
        # 检查模型文件是否存在
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载训练好的YOLOv5模型,custom是指本地文件，model_path是指本地模型的路径
        try:
            self.model = torch.hub.load('yolov5', 'custom', path=model_path, source='local', trust_repo=True)
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            exit(1)
        # 设置模型参数
        self.model.conf = conf_threshold  # 置信度阈值
        self.model.iou = iou_threshold    # IOU阈值
        self.class_names = class_names    # 类别名称
        self.class_mapping = class_mapping  #类别映射关系
        self.object_class_names = object_class_names #不区分正反面的物体类别名称
        self.num_classes = len(class_names)  # 类别数量
       
        
        # 设置设备（自动选择GPU或CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        
        # 创建颜色映射 (每个类别一个颜色)
        self.colors = self._generate_colors(self.num_classes)
        
        logger.info(f"模型加载成功! 使用设备: {self.device}")
        logger.info(f"检测类别: {class_names}")
        logger.info(f"置信度阈值: {conf_threshold}, IOU阈值: {iou_threshold}")

    def _generate_colors(self, n):
        """为每个类别生成一个独特的颜色"""
        return [
            (int(r), int(g), int(b)) 
            for r, g, b in np.random.randint(50, 255, size=(n, 3))
        ]

    def load_image(self, image_path):
        """
        加载图像并进行预处理
        :param image_path: 图像文件路径
        :return: 预处理后的图像 (RGB格式)
        """
        # 检查图像文件是否存在
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise IOError(f"无法读取图像: {image_path}")
        
        # 转换颜色空间 (BGR to RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def detect_objects(self, image):
        """
        执行物体检测
        :param image: 输入图像 (RGB格式)
        :return: 检测结果DataFrame
        """
        # 记录推理开始时间
        start_time = time.time()
        
        # 使用模型进行推理 (禁用梯度计算以提高效率)
        with torch.no_grad():
            results = self.model(image)
        
        # 解析检测结果
        detections = results.pandas().xyxy[0]
        
        # 记录推理时间
        inference_time = time.time() - start_time
        logger.info(f"检测完成! 耗时: {inference_time:.4f}秒, 检测到 {len(detections)} 个物体")
        
        return detections

    def analyze_detections(self, detections):
        """
        分析检测结果并判断是否符合条件
        :param detections: 检测结果DataFrame
        :return: 
            status: 检测状态 ("OK" 或 "NG")
            message: 状态描述
            class_counts: 每个类别的检测数量
        """
        # 过滤低置信度的检测
        valid_detections = detections[detections['confidence'] > self.model.conf]
        
        # 如果没有检测到任何物体
        if len(valid_detections) == 0:
            return "NG", "未检测到任何物体", {}
        
        # 获取检测到的类别ID
        detected_class_ids = valid_detections['class'].astype(int).tolist()
        
        # 统计每个类别的检测数量
        class_counts={}
        for item in self.class_names:
            class_counts[item] = 0
        for class_id in detected_class_ids:
            single_class_name = self.class_names[class_id]
            if single_class_name in class_counts:
                class_counts[single_class_name] += 1
        logger.info("不去分正反面分类后的数据：" + str(class_counts))

        # 统计每个类别（不区分正反面）的检测数量
        object_class_counts = {}
        for item in self.object_class_names:
            object_class_counts[item] = 0
        for key, value in class_counts.items():
            # 获取映射后的物体类别名
            object_class_name = self.class_mapping.get(key)
            if object_class_name in object_class_counts:
                #映射后为同一类物体的药+1
                object_class_counts[object_class_name] = object_class_counts[object_class_name] + value
        logger.info("按照物体种类分类后的数据："+str(object_class_counts))
        
        # 检查三个关键条件：
        # 1. 总检测数量是否为5
        # 2. 是否检测到5个不同类别
        
        total_detections = len(valid_detections)
        
        # 条件1: 检测到5个物体，算上正反面标签就是2*5，实际检测为classes的1/2
        condition1 = (total_detections == 5)
        
        # 条件2: 5个大类别恰好每类一个物体，正反面算一个类别
        condition2 = all(value == 1 for value in object_class_counts.values())
        
        # 判断结果
        if condition1 and condition2:
            return "OK", "所有条件满足", object_class_counts
        else:
            # 详细记录不符合条件的原因
            failure_reasons = []
            
            if not condition1:
                failure_reasons.append(f"物体数量错误: 检测到{total_detections}个物体, 需要{len(self.object_class_names)}个")
            
            if condition1 and not condition2:
                failure_reasons.append(f"物体种类错误: 检测到{total_detections}个,但是有重复种类")

            # 找出重复的类别
            duplicate_classes=[]
            # 找出缺失的类别
            missing_classes=[]
            for key,value in object_class_counts.items():
                if(value > 1):
                    duplicate_classes.append(key)
                if(value == 0):
                    missing_classes.append(key)
             
            if duplicate_classes:
                failure_reasons.append(f"\n重复类别: {', '.join(duplicate_classes)}")
            if missing_classes:
                failure_reasons.append(f"\n缺失类别: {', '.join(missing_classes)}")
            
            return "NG", "; ".join(failure_reasons), object_class_counts

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
        # 将numpy数组转换为PIL图像以便绘制中文
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        
        # 设置字体
        try:
            # 尝试加载中文字体 (需要系统中存在该字体)
            font = ImageFont.truetype("simhei.ttf", 40)
            small_font = ImageFont.truetype("simhei.ttf", 30)
        except:
            # 如果无法加载中文字体，使用默认字体
            font = ImageFont.load_default(100)
            small_font = ImageFont.load_default(80)
        
        # 绘制每个检测框
        for _, det in detections.iterrows():
            # 解析检测框坐标
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            
            # 获取类别信息
            class_id = int(det['class'])
            class_name = self.class_names[class_id]
            confidence = det['confidence']
            
            # 根据状态选择颜色
            if status == "OK":
                color = (0, 255, 0)  # 绿色
            else:
                color = (255, 0, 0)  # 红色
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制类别标签
            label = f"{class_name} {confidence:.2f}"

            # 使用 textbbox 获取文本边界框
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]  # right - left
            text_height = bbox[3] - bbox[1]  # bottom - top
            
            # 绘制文本背景
            draw.rectangle(
                [x1, y1 - text_height - 5, x1 + text_width, y1], 
                fill=color
            )
            
            # 绘制文本
            draw.text(
                (x1, y1 - text_height - 5), 
                label, 
                fill=(255, 255, 255), 
                font=font
            )
        
        # 添加状态文本
        status_color = (0, 200, 0) if status == "OK" else (255, 0, 0)
        draw.text((50, 50), f"检测结果: {status}", fill=status_color, font=font)
        
        # 添加类别统计信息
        stats_y = 100
        for class_name, count in class_counts.items():
            required = "ok" if count == 1 else "ng"
            color = (0, 200, 0) if count == 1 else (255, 0, 0)
            
            stats_text = f"{class_name}: {count}个 ({required})"
            draw.text((50, stats_y), stats_text, fill=color, font=font)
            stats_y += 50
        
        # 添加总检测数量
        total_detections = len(detections)

        draw.text(
            (50, stats_y + 20),
            f"总检测数: {total_detections}个 ({status}- {message})",
            fill=status_color,
            font=font
        )
        
        # 添加时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        draw.text((50, pil_img.height - 60), timestamp, fill=(200, 200, 200), font=small_font)
        
        # 转换回numpy数组
        return np.array(pil_img)

    def process_image(self, image_path, output_dir="results"):
        """
        处理单张图像并保存结果
        :param image_path: 输入图像路径
        :param output_dir: 输出目录
        :return: 检测状态和结果图像路径
        """
        logger.info(f"开始处理图像: {image_path}")
        
        try:
            # 加载图像
            img = self.load_image(image_path)
            
            # 执行物体检测
            detections = self.detect_objects(img)
            
            # 分析检测结果
            status, message, class_counts = self.analyze_detections(detections)
            logger.info(f"检测结果: {status} - {message}")
            
            # 可视化结果
            result_img = self.visualize_results(img, detections, status, message, class_counts)
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            # 保存结果图像
            result_filename = f"result_{Path(image_path).name}"
            result_filepath = output_path / result_filename
            cv2.imwrite(str(result_filepath), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            logger.info(f"结果已保存至: {result_filepath}")
            
            # 保存检测结果到CSV
            self.save_results_to_csv(image_path, detections, status, output_path)
            
            return status, str(result_filepath) ,message
        
        except Exception as e:
            logger.error(f"处理图像时出错: {str(e)}", exc_info=True)
            return "ERROR", None

    def save_results_to_csv(self, image_path, detections, status, output_dir):
        """
        保存检测结果到CSV文件
        :param image_path: 原始图像路径
        :param detections: 检测结果
        :param status: 检测状态
        :param output_dir: 输出目录
        """
        # 创建CSV文件路径
        csv_path = output_dir / "detection_results.csv"
        
        # 准备数据
        data = {
            "timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")],
            "image_path": [image_path],
            "status": [status],
            "total_detections": [len(detections)],
        }
        
        # 添加每个类别的检测数量
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            count = len(detections[detections['class'] == class_id])
            data[f"{class_name}_count"] = [count]
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到CSV (追加模式)
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        
        logger.info(f"检测结果已保存到CSV: {csv_path}")

    def process_batch(self, image_dir, output_dir="results"):
        """
        批量处理目录中的所有图像
        :param image_dir: 图像目录路径
        :param output_dir: 输出目录
        :return: 处理结果统计
        """
        logger.info(f"开始批量处理目录: {image_dir}")
        
        # 获取所有图像文件
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        if not image_files:
            logger.warning(f"目录中没有找到图像文件: {image_dir}")
            return
        
        logger.info(f"找到 {len(image_files)} 张待处理图像")
        
        results = []
        for i, img_path in enumerate(image_files):
            logger.info(f"处理图像 {i+1}/{len(image_files)}: {img_path.name}")
            status, result_path = self.process_image(str(img_path), output_dir)
            results.append({
                "image": img_path.name,
                "status": status,
                "result_path": result_path
            })
        
        # 生成统计报告
        status_counts = pd.Series([r['status'] for r in results]).value_counts()
        logger.info("\n===== 批量处理结果统计 =====")
        logger.info(f"总处理图像: {len(results)}")
        logger.info(f"合格(OK): {status_counts.get('OK', 0)}")
        logger.info(f"不合格(NG): {status_counts.get('NG', 0)}")
        logger.info(f"错误(ERROR): {status_counts.get('ERROR', 0)}")
        
        return results


def quick_empty_directory(directory_path):
    """
    快速清空目录 - 删除整个目录后重新创建

    参数:
        directory_path (str): 要清空的目录路径
    """
    if os.path.exists(directory_path):
        try:
            # 删除整个目录
            shutil.rmtree(directory_path)
            logger.info(f"已删除目录: {directory_path}")

            # 重新创建空目录
            os.makedirs(directory_path)
            logger.info(f"已重新创建目录: {directory_path}")

        except Exception as e:
            logger.error(f"操作失败: {e}")
    else:
        logger.info(f"目录不存在: {directory_path}")

if __name__ == "__main__":
    # 配置参数，需要跟样本中的classes.txt中顺序一致，区分正反面种类
    CLASS_NAMES = ['bone_front', 'bone_back',\
                    'fish_front', 'fish_back', \
                    'hedgehog_front', 'hedgehog_back', \
                    'heart_front', 'heart_back', \
                    'paw']  # 9种包装类型，4*2+1\n\n    
    # 定义映射字典\n    
    CLASS_MAPPING = { 'bone_front': 'bone','bone_back': 'bone',\
                    'fish_front': 'fish','fish_back': 'fish',\
                    'hedgehog_front': 'hedgehog','hedgehog_back': 'hedgehog',\
                    'heart_front': 'heart','heart_back': 'heart',\
                    'paw': 'paw' }

    OBJECT_CLASS_NAMES = ['bone', 
                        'fish', \
                        'hedgehog', \
                        'heart', \
                        'paw']

    #使用预训练模型
    model_path = "packaging_models/best.pt"
    
    #初始化检测系统
    try:
        detector = PackagingDetectionSystem(
            model_path=model_path,
            class_names=CLASS_NAMES,
            class_mapping=CLASS_MAPPING,
            object_class_names=OBJECT_CLASS_NAMES,
            conf_threshold=0.75,
            iou_threshold=0.25
        )
    except Exception as e:
        logger.error(f"初始化检测系统失败: {str(e)}")
        exit(1)

    # 处理单个图像
    test_image = "./Image/HIK_20250621_134234.jpg"
    status, result_path, message = detector.process_image(test_image)
    logger.info(f"单张图像检测结果: {status}，详细信息：{message}")

    # 批量处理图像
    # test_image_dir = Path("./predict")
    # test_image_output_dir = Path("./output")

    #清理output目录
    # quick_empty_directory(test_image_output_dir)
    #
    # detector.process_batch(test_image_dir, test_image_output_dir)

    logger.info("包装检测流程完成!")