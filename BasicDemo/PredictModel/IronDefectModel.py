import torch
import cv2
import numpy as np
import sys, os
import time
from pathlib import Path
import pandas as pd
import logging
import shutil


from PIL import Image, ImageDraw, ImageFont
from .BasicModel import BasicModel

# 将 yolov5 目录添加到模块搜索路径,先退出当前目录的yolov5目录
yolov5_path = os.path.join(os.path.dirname(__file__), "..",'yolov5')
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)

from utils.general import non_max_suppression

# print("当前 Python 模块搜索路径 (sys.path):")
# for i, path in enumerate(sys.path):
#     print(f"{i+1:2d}. {path}")

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("IronDefectModel.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PackagingInspector")

class IronDefectModel(BasicModel):

    def __init__(self, model_path, conf_threshold=0.6, iou_threshold=0.45):
        """
        初始化包装检测系统
        :param model_path: 训练好的模型权重路径
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
            
        # 设置类别参数，使用传入的参数或默认值
        self.class_names = ['Blistering', 'Cloudiness', \
                       'OrangePeel', 'Scrape']
        self.num_classes = len(self.class_names)  # 类别数量
            
        # 设置模型参数
        self.model.conf = conf_threshold  # 置信度阈值
        self.model.iou = iou_threshold    # IOU阈值
       
        # 设置设备（自动选择GPU或CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        
        # 创建颜色映射 (每个类别一个颜色)
        self.colors = self._generate_colors(self.num_classes)
        
        logger.info(f"模型加载成功! 使用设备: {self.device}")
        logger.info(f"检测类别: {self.class_names}")
        logger.info(f"置信度阈值: {conf_threshold}, IOU阈值: {iou_threshold}")

    def _generate_colors(self, n):
        """为每个类别生成一个独特的颜色"""
        return [
            (int(r), int(g), int(b)) 
            for r, g, b in np.random.randint(50, 255, size=(n, 3))
        ]
 

    def load_image(self, image_path,img_size=640):

        """
        加载黑白图像，转换为三通道灰度图以适配 YOLOv5 输入。
        
        Args:
            image_path (str): 图像路径
            img_size (int): 模型输入尺寸
        
        Returns:
            original_img (np.array): 原始灰度图像 (H, W)
            input_tensor (torch.Tensor): 归一化后的三通道输入张量 (1, 3, H, W)
            orig_shape (tuple): 原始图像形状 (H, W)
        """
        # 读取为灰度图
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        orig_shape = original_img.shape  # (H, W)

        rgb_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        # 调整大小
        resized_rgb_img = cv2.resize(rgb_img, (img_size, img_size))


        
        # 转为 Tensor 并归一化 [0,1]
        resized_rgb_img = resized_rgb_img.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(resized_rgb_img).permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        return rgb_img, input_tensor, orig_shape




    def detect_objects(self, input_tensor):
        """
        使用yolov5进行推理
        input_tensor (torch.Tensor): 预处理后的输入张量
        :return: 检测结果DataFrame
        """
        # 记录推理开始时间
        start_time = time.time()
        
        # 使用模型进行推理 (禁用梯度计算以提高效率)
        with torch.no_grad():
            results  = self.model(input_tensor)

        # 后处理：NMS
        pred = non_max_suppression(results, conf_thres=self.model.conf, iou_thres=self.model.iou)[0]
        if pred is not None:
            detections = pred.cpu().numpy()  # shape: (n, 6) -> [x1, y1, x2, y2, conf, cls]
        else:
            detections = np.zeros((0, 6))
        
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
        # 如果没有检测到任何缺陷
        if len(detections) == 0:
            class_counts = {name: 0 for name in self.class_names}
            return "OK", "未检测到任何缺陷", class_counts
        
        # 获取检测到的类别ID（第6列）
        detected_class_ids = detections[:, 5].astype(int).tolist()
        
        # 统计每个类别的检测数量
        class_counts={}
        for item in self.class_names:
            class_counts[item] = 0
        for class_id in detected_class_ids:
            single_class_name = self.class_names[class_id]
            if single_class_name in class_counts:
                class_counts[single_class_name] += 1
        logger.info("分类统计后的数据：" + str(class_counts))
    
            
        return "NG", "存在缺陷", class_counts

    def visualize_results(self, image, detections, status, message, class_counts):
        """
        在图像上绘制检测框和标注，并返回可视化后的图像（PIL Image）
        :param image: 原始图像，numpy.ndarray, shape (H, W, 3) or (H, W), dtype uint8
        :param detections: 检测结果，numpy.ndarray, shape (n, 6) → [x1, y1, x2, y2, conf, cls]
        :param status: "OK" 或 "NG"
        :param message: 状态描述
        :param class_counts: dict, 各类别数量
        :return: PIL.Image 对象
        """

        # ========== 1. 检查并修复图像格式 ==========
        if not isinstance(image, np.ndarray):
            raise TypeError("image 必须是 numpy.ndarray")

        # 如果是 float 类型，转 uint8
        if image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)

        # 如果是单通道灰度图 (H, W, 1) → 去掉最后一维
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        # 如果是单像素图像，放大便于查看（可选）
        if image.shape[0] <= 1 or image.shape[1] <= 1:
            scale = max(100 // image.shape[0], 100 // image.shape[1], 1)
            image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

        # 转 PIL Image
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)

        # 尝试加载中文字体（如无，用默认字体）
        try:
            font = ImageFont.truetype("simhei.ttf", size=20)  # Windows 黑体
        except:
            try:
                font = ImageFont.truetype("arial.ttf", size=20)
            except:
                font = ImageFont.load_default()

        # ========== 2. 绘制检测框和标签 ==========
        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                cls_id = int(cls_id)  # ← 关键！转成整数

                # 防止越界
                if cls_id < 0 or cls_id >= len(self.class_names):
                    continue

                class_name = self.class_names[cls_id]

                # 画框
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # 写标签：类别 + 置信度
                label = f"{class_name} {conf:.2f}"
                text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y1 - 25), label, fill="white", font=font)

        # ========== 3. 在左上角写全局状态信息 ==========
        status_text = f"状态: {status} | {message}"
        draw.text((10, 10), status_text, fill="green" if status == "OK" else "red", font=font)

        # 在左上角下方写类别统计
        y_offset = 40
        for class_name, count in class_counts.items():
            if count > 0:
                draw.text((10, y_offset), f"{class_name}: {count}", fill="blue", font=font)
                y_offset += 25

        return pil_img  # 返回 PIL Image，可 .save() 或 .show()

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
            original_img, input_tensor, orig_shape = self.load_image(image_path)
            
            # 执行物体检测
            detections = self.detect_objects(input_tensor)
            
            # 分析检测结果
            status, message, class_counts = self.analyze_detections(detections)
            logger.info(f"检测结果: {status} - {message}")
            
            # 可视化结果
            result_img = self.visualize_results(original_img, detections, status, message, class_counts)
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            # 保存结果图像
            result_filename = f"result_{Path(image_path).name}"
            result_filepath = output_path / result_filename
            result_img.save(result_filepath)
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
        :param detections: 检测结果(numpy.ndarray)
        :param status: 检测状态
        :param output_dir: 输出目录
        """
        # 创建输出目录（如果不存在）
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建CSV文件路径
        csv_path = output_dir / "detection_results.csv"

        # 准备数据
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": str(image_path),
            "status": status,
            "total_detections": len(detections),
        }

        # 添加每个类别的检测数量
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            count = len(detections[detections[:, 5] == class_id])  # 假设class_id位于detections的最后一列
            data[f"{class_name}_count"] = count

        # 将字典转换为DataFrame
        df = pd.DataFrame([data])  # 注意这里的[]，因为data是一个字典而不是字典列表

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

