from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
# plt.switch_backend('agg')

def show_image(img_original, is_tensor=True):
    if is_tensor:
        img_original = transforms.functional.to_pil_image(img_original, mode='RGB')
    plt.figure()
    plt.imshow(img_original.convert("RGB"))
    plt.show()

def show_pred_image(img_original, img_masked, img_pred, is_tensor=True):
    if is_tensor:
        img_original = transforms.functional.to_pil_image(img_original.squeeze().detach().cpu(), mode='RGB')
        img_masked = transforms.functional.to_pil_image(img_masked.squeeze().detach().cpu(), mode='RGBA')
        img_pred = transforms.functional.to_pil_image(img_pred.squeeze().detach().cpu(), mode='RGB')
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img_original)
    plt.subplot(1, 3, 2)
    plt.imshow(img_masked)
    plt.subplot(1, 3, 3)
    plt.imshow(img_pred)
    plt.show()

def save_pred_image(img_original, img_masked, img_pred, filename, is_tensor=True):
    if is_tensor:
        img_original = transforms.functional.to_pil_image(img_original.squeeze().detach().cpu(), mode='RGB')
        img_masked = transforms.functional.to_pil_image(img_masked.squeeze().detach().cpu(), mode='RGBA')
        img_pred = transforms.functional.to_pil_image(img_pred.squeeze().detach().cpu(), mode='RGB')
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img_original)
    plt.subplot(1, 3, 2)
    plt.imshow(img_masked)
    plt.subplot(1, 3, 3)
    plt.imshow(img_pred)
    plt.savefig(filename)