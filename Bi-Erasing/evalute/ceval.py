import os
import sys
import pandas as pd
import numpy as np
import torch
import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import glob
from pathlib import Path
from cleanfid import fid
def load_coco_prompts(coco_csv_path):
    if not os.path.exists(coco_csv_path):
        print(f"COCO dataset file not found: {coco_csv_path}")
        return None
    
    df = pd.read_csv(coco_csv_path)
    print(f"Loaded {len(df)} COCO prompts")
    case_to_prompt = {}
    for _, row in df.iterrows():
        case_to_prompt[row['case_number']] = row['prompt']
    
    return case_to_prompt

def extract_number(filename):
    """Extract case_number from filename"""
    match = re.search(r'_(\d+)_seed', filename)
    if match:
        case_number = match.group(1)
        return int(case_number)
    return None


@torch.no_grad()
def calculate_clip(image_dir, co_prompts, device="cuda:0"):
    """Calculate CLIP similarity scores between images and prompts"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    model_path = '/data/hao_chen/model/hub/models--openai--clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_path).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    similarities = []
    valid_images = []
    failed_count= 0
    
    for image_path in tqdm(image_files):
        image = Image.open(image_path).convert('RGB')  
        filename = os.path.basename(image_path)
        case_number = extract_number(filename)
        if not case_number:
            print(f"Cannot extract case_number from filename: {filename}")
            failed_count += 1
            continue
        
        if case_number not in co_prompts:
            print(f"Prompt not found for case_number: {case_number}")
            failed_count += 1
            continue        

        matched_prompt = co_prompts[case_number]

        inputs = processor(text=matched_prompt, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        clip_score = outputs.logits_per_image[0][0].detach().cpu().float()
        
        similarities.append(float(clip_score))
        valid_images.append({
            'filename': filename,
            'case_number': case_number,
            'prompt': matched_prompt,
            'clip_score': float(clip_score)
        })
    
    return similarities, valid_images

def compute_fid_score(dir1, dir2, device="cuda:0"):

    score = fid.compute_fid(dir1, dir2,device=device)
        
    return score

def evaluate(image_dir, coco_csv_path , device="cuda:0"):
    print(f"=== Evaluating: {image_dir} ===")

    prompts = load_coco_prompts(coco_csv_path)
    erase_dir = os.path.join(image_dir, "erase_model")
    print(f"Erasing: {erase_dir}")

    fid_score = None
    real_dir ="/data/hao_chen/Co-Erasing-main/real_image/coco_10k"
    print("\nComputing FID...")
    fid_score = compute_fid_score(erase_dir, real_dir ,device=device)
    print(f"\nFID Score: {fid_score:.4f}")

    print("\nComputing CLIP...")
    erase_clip , erase_clip_details  = calculate_clip(erase_dir, prompts, device=device)
    erase_mean = float(np.mean(erase_clip)) if len(erase_clip) > 0 else float('nan')
    print(f"Erasing CLIP Score: {erase_mean:.4f}")

def main():
    parser = argparse.ArgumentParser
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--coco_csv',type=str,default='prompts/coco_10k.csv')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.image_dir
    
    print("=== Basic Evaluation Script ===")
    print(f"Image directory: {args.image_dir}")
    print(f"COCO dataset: {args.coco_csv}")
    print(f"Device: {args.device}")
    print(f"Output directory: {output_dir}")
    
    evaluate(
        image_dir=args.image_dir,
        coco_csv_path=args.coco_csv,
        device=args.device
    )

if __name__ == "__main__":
    main()
