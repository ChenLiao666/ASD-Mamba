import io
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from net.models.ASDMamba import ASDMamba
from net.configs.config_setting import setting_config


def setup_model(device):
    config = setting_config
    model_cfg = config.model_config
    if config.network == 'asd':
        model = ASDMamba(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
        )
        model.load_state_dict(torch.load('ASD240.pth', map_location=device))
        model.to(device)
        return model
    else:
        raise NotImplementedError("The specified network configuration is not supported.")


def load_and_preprocess_image(img_path):
    # 打开图像并转换为RGB
    image = Image.open(img_path).convert('RGB')
    orig_size = image.size

    # 定义albumentations的预处理流程
    transform = A.Compose([
        A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),  # 使用 cv2 的插值常量
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 将PIL图像转换为numpy数组
    image_np = np.array(image)

    # 应用albumentations的预处理
    transformed = transform(image=image_np)
    image = transformed['image']

    return image, orig_size


def saliency_map_prediction(img_path, condition, model, device):
    img, orig_size = load_and_preprocess_image(img_path)
    img = img.unsqueeze(0).to(device)
    one_hot_condition = torch.zeros((1, 4), device=device)
    one_hot_condition[0, condition] = 1
    model.eval()
    with torch.no_grad():
        pred_saliency = model(img, one_hot_condition)

    pred_saliency = pred_saliency.squeeze().cpu().numpy()
    return pred_saliency, orig_size


def overlay_heatmap_on_image(original_img_path, heatmap_img_path, output_img_path):
    # Read the original image
    orig_image = cv2.imread(original_img_path)
    orig_size = orig_image.shape[:2]  # Height, Width

    # Read the heatmap image
    overlay_heatmap = cv2.imread(heatmap_img_path, cv2.IMREAD_GRAYSCALE)

    # Resize the heatmap to match the original image size
    overlay_heatmap = cv2.resize(overlay_heatmap, (orig_size[1], orig_size[0]))

    # Apply color map to the heatmap
    overlay_heatmap = cv2.applyColorMap(overlay_heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    overlay_image = cv2.addWeighted(orig_image, 1, overlay_heatmap, 0.8, 0)

    # Save the result
    cv2.imwrite(output_img_path, overlay_image)


def process_directory(img_dir, condition, model, device, output_path, heat_map_type):
    # Check if the path is a directory
    if not os.path.isdir(img_dir):
        print(f"Error: {img_dir} is not a directory")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(img_dir):
        file_path = os.path.join(img_dir, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing image: {filename}")
            pred_saliency, orig_size = saliency_map_prediction(file_path, condition, model, device)

            # Save HOT heatmap
            filename_without_ext = os.path.splitext(filename)[0]
            hot_output_filename = os.path.join(output_path, f'{filename_without_ext}.png')

            # Save HOT heatmap with deartifacting
            plt.figure()
            plt.imshow(pred_saliency, cmap='hot')
            plt.axis('off')

            # Save the heatmap to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close()

            # Open the buffer as an image and apply deartifacting
            img = Image.open(buf)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

            # Resize and apply Gaussian blur to reduce artifacts
            img_resized = cv2.resize(img_cv, orig_size,
                                     interpolation=cv2.INTER_CUBIC)  # Use cubic interpolation for smoother results
            img_smoothed = cv2.bilateralFilter(img_resized, d=9, sigmaColor=75, sigmaSpace=75)
            # img_smoothed = cv2.GaussianBlur(img_resized, (5, 5), sigmaX=1, sigmaY=1)  # Apply Gaussian blur

            # Save the smoothed heatmap
            cv2.imwrite(hot_output_filename, img_smoothed)

            print(f"Saved HOT saliency map to {hot_output_filename}")

            if heat_map_type == 'Overlay':
                overlay_output_filename = os.path.join(output_path, f'{filename_without_ext}_overlay.png')
                overlay_heatmap_on_image(file_path, hot_output_filename, overlay_output_filename)
                print(f"Saved overlay image to {overlay_output_filename}")


def main():
    parser = argparse.ArgumentParser(description='Saliency Map Prediction')
    parser.add_argument('--img_path', type=str, default=r'/home/robot/Images', help='Path to a single image or directory of images')
    parser.add_argument('--condition', type=int, default=1, help='Condition type')
    parser.add_argument('--output_path', type=str, default='/home/robot/output/ASDhot', help='Directory to save output images')
    parser.add_argument('--heat_map_type', type=str, default='Overlay', choices=['HOT', 'Overlay'],
                        help='Type of heatmap: HOT or Overlay')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Setup device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = setup_model(device)

    # If the input path is a directory, process all images in the directory
    if os.path.isdir(args.img_path):
        process_directory(args.img_path, args.condition, model, device, args.output_path, args.heat_map_type)
    else:
        # If it's a single image, process just that one
        pred_saliency, orig_size = saliency_map_prediction(args.img_path, args.condition, model, device)

        filename = os.path.splitext(os.path.basename(args.img_path))[0]
        hot_output_filename = os.path.join(args.output_path, f'{filename}hot.png')

        # Save HOT heatmap
        plt.figure()
        plt.imshow(pred_saliency, cmap='hot')
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()

        img = Image.open(buf)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
        img_resized = cv2.resize(img_cv, orig_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(hot_output_filename, img_resized)

        print(f"Saved HOT saliency map to {hot_output_filename}")

        if args.heat_map_type == 'Overlay':
            overlay_output_filename = os.path.join(args.output_path, f'{filename}_overlay.png')
            overlay_heatmap_on_image(args.img_path, hot_output_filename, overlay_output_filename)
            print(f"Saved overlay image to {overlay_output_filename}")


if __name__ == "__main__":
    main()