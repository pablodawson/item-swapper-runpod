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
        hull = cv2.convexHull(contours[0])
        mask = np.zeros_like(mask)
        cv2.fillPoly(mask, [hull], 255)
    
    mask = Image.fromarray(mask).convert('RGB')
    return mask

if __name__=="__main__":
    seg_img = cv2.imread("descarga (6).png")
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    mask = create_mask(seg_img, [11,102,255], dillation_size=5, convex_hull=True)
    mask.save("mask.png")