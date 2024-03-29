import numpy as np
from safetensors.torch import load_file
import cv2
from PIL import Image
from safetensors.torch import load_file
from sdscripts.networks.lora import create_network_from_weights
import torch

def apply_lora(pipe, lora_path, weight:float = 1.0):
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    sd = load_file(lora_path)
    lora_network, sd = create_network_from_weights(weight, None, vae, text_encoder, unet, sd)
    lora_network.apply_to(text_encoder, unet)
    lora_network.load_state_dict(sd)
    lora_network.to("cuda", dtype=torch.float16)

def create_mask(seg_img, color, dillation_size=5, dillation_iters=2, convex_hull=False):
    
    mask = cv2.inRange(seg_img, np.array(color), np.array(color))
    mask = cv2.dilate(mask, np.ones((dillation_size,dillation_size),np.uint8), iterations = dillation_iters)
    
    if convex_hull:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull_list = []
        for contour in contours:
            hull = cv2.convexHull(contour)
            hull_list.append(hull)
        mask = np.zeros_like(mask)

        for hull in hull_list:
            cv2.fillPoly(mask, [hull], 255)
    
    mask = Image.fromarray(mask).convert('RGB')
    return mask

def paste(output, original, mask):
    mask_np = np.array(mask)

    mask_np = cv2.dilate(mask_np, np.ones((8,8),np.uint8), iterations = 2)
    mask_np  = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
    ksize = (20, 20)
    alpha = cv2.blur(mask_np, ksize)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)/255
    
    output_np = np.array(output)
    original_np = np.array(original)

    masked_fg = alpha * output_np
    masked_bg = (1.0 - alpha)* original_np

    output = np.uint8(cv2.addWeighted(masked_fg, 1, masked_bg, 1, 0.0))

    return Image.fromarray(output)

def apply_mask(image, mask):
    mask_np = np.array(mask)
    mask_np  = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
    ksize = (20, 20)
    alpha = cv2.blur(mask_np, ksize)

    output_np = np.array(image)
    output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2RGBA)
    output_np[:,:,3] = alpha
    
    return Image.fromarray(output_np)

if __name__=="__main__":
    seg_img = cv2.imread("seg.png")
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    mask = create_mask(seg_img, [11,102,255], dillation_size=5, convex_hull=False)
    mask.save("mask2.png")