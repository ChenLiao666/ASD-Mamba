import io
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    image = Image.open(img_path).convert('RGB')
    orig_size = image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
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


# def save_heatmap(pred_saliency, orig_size, output_path):
#
#     pred_saliency = cv2.GaussianBlur(pred_saliency, (5, 5), 0)
#
#
#     plt.figure()
#     plt.imshow(pred_saliency, cmap='gray')
#     plt.axis('off')
#
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     buf.seek(0)
#     plt.close()
#
#     img = Image.open(buf)
#     img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
#     img_resized = cv2.resize(img_cv, orig_size, interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(output_path, img_resized)
#
#     print(f"save hotmap: {output_path}")
def save_gray(pred_saliency, orig_size, output_path):
    pred_saliency = cv2.GaussianBlur(pred_saliency, (5, 5), 0)

    plt.imshow(pred_saliency, cmap='gray')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    img = Image.open(buf)

    img = img.convert('L')

    img_cv = np.array(img)

    img_resized = cv2.resize(img_cv, orig_size, interpolation=cv2.INTER_NEAREST)
    img_resized = cv2.GaussianBlur(img_resized, (5, 5), 0)

    img_resized = cv2.bilateralFilter(img_resized, d=9, sigmaColor=75, sigmaSpace=75)

    cv2.imwrite(output_path, img_resized)
    print(f"save gray: {output_path}")


def process_directory(img_dir, condition, model, device, output_path):
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

            # 保存灰度热度图
            filename_without_ext = os.path.splitext(filename)[0]
            heatmap_output_filename = os.path.join(output_path, f'{filename_without_ext}.png')

            save_gray(pred_saliency, orig_size, heatmap_output_filename)


def main():
    parser = argparse.ArgumentParser(description='Saliency Map Prediction')
    parser.add_argument('--img_path', type=str, default=r'/home/robot/Images', help='Path to a single image or directory of images')
    parser.add_argument('--condition', type=int, default=1, help='Condition type')
    parser.add_argument('--output_path', type=str, default='/home/robot/output/gray', help='Directory to save output images')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Setup device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = setup_model(device)

    # If the input path is a directory, process all images in the directory
    if os.path.isdir(args.img_path):
        process_directory(args.img_path, args.condition, model, device, args.output_path)
    else:
        # If it's a single image, process just that one
        pred_saliency, orig_size = saliency_map_prediction(args.img_path, args.condition, model, device)

        filename = os.path.splitext(os.path.basename(args.img_path))[0]
        graymap_output_filename = os.path.join(args.output_path, f'{filename}.png')


        save_gray(pred_saliency, orig_size, graymap_output_filename)


if __name__ == "__main__":
    main()