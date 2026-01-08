import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import glob
from dataclasses import dataclass
from transformers import (
    VisionEncoderDecoderModel, 
    TrOCRProcessor
)

# 配置 matplotlib
plt.rcParams['figure.figsize'] = (12, 9)
block_plot = False

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 配置类
@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE:    int = 48
    EPOCHS:        int = 35
    LEARNING_RATE: float = 0.00005

@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT:     str = 'static/scut_data'

@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = 'microsoft/trocr-small-printed'

# 加载模型和处理器
processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)
trained_model = VisionEncoderDecoderModel.from_pretrained(
    'static/seq2seq_model_printed/checkpoint_'+str(7580)
).to(device)

def read_and_show(image_source):
    """
    读取并返回图像
    :param image_source: 可以是文件路径字符串或上传的文件对象
    Returns: PIL Image
    """
    if isinstance(image_source, str):
        image = Image.open(image_source).convert('RGB')
    else:
        image = Image.open(image_source.file).convert('RGB')
    return image

def ocr(image, processor, model):
    """
    执行OCR识别
    :param image: PIL Image
    :param processor: OCR processor
    :param model: OCR model
    Returns: 识别的文本字符串
    """
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

async def eval_new_data(uploaded_file=None, num_samples=5, show_image=True):
    """
    评估新数据的OCR结果
    :param uploaded_file: 上传的文件对象（可选）
    :param num_samples: 处理的最大样本数
    :param show_image: 是否显示图像
    Returns: dict 包含处理结果
    """
    try:
        # 处理单个上传文件的情况
        if uploaded_file:
            image = read_and_show(uploaded_file)
            text = ocr(image, processor, trained_model)
            recognized_text = text.strip()
            
            # 输出结果（参考代码A的详细输出）
            print(f"Processing uploaded file")
            if recognized_text:
                print(f"Recognized text: {recognized_text}")
            else:
                print("No text recognized by the model in the image.")
            
            # 显示图像（参考代码A的可视化）
            if show_image:
                plt.figure(figsize=(7, 4))
                plt.imshow(image)
                plt.title(recognized_text if recognized_text else "No text")
                plt.axis('off')
                plt.show()
            
            return {
                "success": True if recognized_text else False,
                "text": recognized_text if recognized_text else None,
                "message": "No text recognized" if not recognized_text else None
            }
            
        # 处理目录中的多张图片（参考代码A的批量处理逻辑）
        else:
            data_path = os.path.join(DatasetConfig.DATA_ROOT, 'scut_test', '*')
            image_paths = glob.glob(data_path)
            results = []
            
            for i, image_path in tqdm(enumerate(image_paths), total=min(num_samples, len(image_paths))):
                if i >= num_samples:
                    break
                    
                image = read_and_show(image_path)
                text = ocr(image, processor, trained_model)
                recognized_text = text.strip()
                
                print(f"Image path: {image_path}")
                if recognized_text:
                    print(f"Recognized text: {recognized_text}")
                else:
                    print("No text recognized by the model in the image.")
                
                if show_image:
                    plt.figure(figsize=(7, 4))
                    plt.imshow(image)
                    plt.title(recognized_text if recognized_text else "No text")
                    plt.axis('off')
                    plt.show()
                
                results.append({
                    "image_path": image_path,
                    "text": recognized_text if recognized_text else None,
                    "success": bool(recognized_text)
                })
                
            return {"success": True, "results": results}
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import asyncio
    # 测试单个文件模式（需要提供文件对象）
    # result = asyncio.run(eval_new_data(uploaded_file=some_file))
    
    # 测试目录模式
    result = asyncio.run(eval_new_data())
    print(result)