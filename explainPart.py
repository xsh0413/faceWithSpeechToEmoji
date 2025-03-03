import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import LayerGradCam, LayerAttribution
from model import EmojiResNet18  # 确保正确导入您的模型

# 设置设备
device = torch.device("cpu")

# 初始化模型并加载权重
model = EmojiResNet18().to(device)
model.load_state_dict(torch.load('./models/emoji_resnet_FER2013.t7', map_location=device)['model'])
model.eval()

def preprocess_image(image_path):
    """预处理图像以适配模型输入要求"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((48, 48)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为Tensor
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

# 生成热力图和去除边框的热力图
def howtoexplain(image_path):
    input_image = preprocess_image(image_path)

    # Grad-CAM
    grad_cam = LayerGradCam(model, model.conv1)  # 使用模型的第一个卷积层
    output = model(input_image)
    prediction = output.max(1, keepdim=True)[1]

    # 获取属性
    attr = grad_cam.attribute(input_image, target=prediction.item())
    # 上采样热力图以匹配输入图像的尺寸
    attr = LayerAttribution.interpolate(attr, (48, 48))

    # 处理热力图和原始图像以应用阈值和叠加
    attr_vis = np.maximum(attr.cpu().detach().numpy().squeeze(), 0)
    attr_vis = attr_vis / np.max(attr_vis)  # 归一化

    # 设定阈值，低于阈值的部分将使用原始图像的像素
    threshold = 0.2  # 阈值可以根据需要调整
    mask = attr_vis > threshold  # 创建一个掩膜，表示热力图中高于阈值的区域
    masked_attr = np.where(mask, attr_vis, 0)  # 只保留高于阈值的热力图部分

    # 将原始图像调整为与模型输入相同的尺寸
    original_image_resized = np.array(Image.open(image_path).resize((48, 48)))

    # 定义边框大小（单位：像素）
    border_size = 7  # 可以根据需要调整

    # 创建一个与热力图相同尺寸的全1掩膜，然后将边框区域设为0
    mask_border = np.ones_like(attr_vis)
    mask_border[:border_size, :] = 0  # 上边框
    mask_border[-border_size:, :] = 0  # 下边框
    mask_border[:, :border_size] = 0  # 左边框
    mask_border[:, -border_size:] = 0  # 右边框

    # 将边框掩膜应用到已经通过阈值处理的热力图上
    masked_attr_with_border = np.where(mask_border, masked_attr, 0)

    return original_image_resized, masked_attr_with_border

def save_grad_cam_result(image_path, output_path):
    """保存Grad-CAM结果到本地"""
    original_image_resized, masked_attr_with_border = howtoexplain(image_path)

    plt.figure(figsize=(10, 5))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_resized, alpha=1)
    plt.axis('off')
    plt.title('Original Image')

    # 去除边框的热力图叠加
    plt.subplot(1, 2, 2)
    plt.imshow(original_image_resized, alpha=1)
    plt.imshow(masked_attr_with_border, cmap='hot', alpha=0.5)
    plt.axis('off')
    plt.title('After-explained Image')

    # 保存图像

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    # plt.close()
    print(f"Grad-CAM result saved at {output_path}")

# 使用示例
if __name__ == '__main__':
    image_path = './screenshot111.png'
    output_path = './result_screenshot111.png'
    save_grad_cam_result(image_path, output_path)
