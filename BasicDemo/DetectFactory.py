from PredictModel.IronClassifyModel import IronClassifyModel
from PredictModel.IronDefectModel import IronDefectModel
from PredictModel.BasicModel import BasicModel

class DetectFactory:


    @staticmethod
    def create_detector(detector_type: str):
        # 使用默认模型路径
        model_path = "packaging_models/best.pt"
        if detector_type.lower() == "ironclassify":
            return IronClassifyModel(model_path=model_path,conf_threshold=0.75,iou_threshold=0.45)
        elif detector_type.lower() == "irondefect":
            return IronDefectModel(model_path=model_path,conf_threshold=0.75,iou_threshold=0.45)
        else:
            raise ValueError(f"不支持的检测器类型: {detector_type}")