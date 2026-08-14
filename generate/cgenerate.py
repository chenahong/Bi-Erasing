"""
Generate evaluation images for erase model based on COCO-10K
Use random seeds from COCO dataset to ensure reproducibility
"""
import torch
import numpy as np
import random
import pandas as pd
from diffusers import StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler
from PIL import Image
import os
from datetime import datetime
import argparse
from tqdm import tqdm

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_original_model(base_model_path, device="cuda:1"):
    """Load original SD model"""
    print("Loading original Stable Diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    print("Original model loaded")
    return pipeline

def load_erase_model(base_model_path, trained_unet_path, device="cuda:1"):
    """Load trained Erasing model"""
    print("Loading Co-Erasing model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    

    trained_state_dict = torch.load(trained_unet_path, map_location="cpu")
    trained_state_dict = {k: v.float() if v.dtype == torch.float16 else v 
                     for k, v in trained_state_dict.items()} 
    original_state_dict = pipeline.unet.state_dict()
    
    filtered_state_dict = {}
    skipped_keys = []
    
    for key, value in trained_state_dict.items():
        if key in original_state_dict:
            if original_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                skipped_keys.append(f"{key} (shape mismatch)")
        else:
            if any(ip_key in key for ip_key in ["processor.to_k_ip", "processor.to_v_ip"]):
                skipped_keys.append(f"{key} (IP-Adapter)")
            else:
                skipped_keys.append(f"{key} (unknown)")
    
    print(f" Skipped {len(skipped_keys)} incompatible weights")
    
 
    missing_keys, unexpected_keys = pipeline.unet.load_state_dict(filtered_state_dict, strict=False)
    print(f"Co-Erasing model loaded, successfully loaded {len(filtered_state_dict)} weights")
    
    return pipeline

def load_coco_dataset(csv_path, num_samples=None, start_idx=0):
    print(f"Loading COCO dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if num_samples:
        end_idx = min(start_idx + num_samples, len(df))
        df = df.iloc[start_idx:end_idx]
        print(f"Selected samples {start_idx}-{end_idx-1}, total {len(df)} prompts")
    else:
        print(f"Using all {len(df)} prompts")
    
    return df

def generate_coco_images(erase_pipeline, coco_df, device="cuda:1"):
    """Generate images based on COCO dataset"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"comparison_coco/coco_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    erase_dir = os.path.join(output_dir, "erase_model")
    os.makedirs(erase_dir, exist_ok=True)
    
    results = []
    success_count = 0
    total_images = len(coco_df) 
    
    print(f" Starting to generate {total_images} erased model images ")
    
    with tqdm(total=total_images, desc="Generating COCO images for Erase model", unit="images") as pbar:
        for idx, (_, row) in enumerate(coco_df.iterrows()):
            case_number = row['case_number']
            prompt = row['prompt']
            seed = int(row['evaluation_seed'])
            
            set_seed(seed)
            ok = False
            path_out = None
            
            try:
                with torch.no_grad():
                    erase_pipeline.set_progress_bar_config(disable=True)
                    
                    image = erase_pipeline(
                        prompt=prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=torch.Generator(device=device).manual_seed(seed)
                    ).images[0]
                    
                    erase_pipeline.set_progress_bar_config(disable=False)
                
                filename = f"erase_{case_number}_seed{seed}.png"
                path_out = os.path.join(erase_dir, filename)
                image.save(path_out)
                ok = True
                success_count += 1
                
            except Exception as e:
                print(f"\nErasing failed ({case_number}): {e}")
            
            results.append({
                "case_number": case_number,
                "prompt": prompt,
                "seed": seed,
                "erase_image": path_out,
                "erase_success": ok
            })
            
            pbar.update(1)
            pbar.set_postfix({
                'Sample': f"{idx+1}/{len(coco_df)}",
                'Success': f"{success_count}/{idx+1}"
            })

    return results, output_dir

def main():
    
    base_model_path = "/data/hao_chen/model/hub/models--runwayml--stable-diffusion-v1-5"
    coco_csv= "prompts/coco_10k.csv"
    coco_df = load_coco_dataset(coco_csv, num_samples=10000, start_idx=0) 

    possible_paths = ["checkpoints/image/nudity/unet_full_im200_ng1.0_it1500_B/unet_1499.pth"]
    
    trained_unet_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            trained_unet_path = path
            print(f"Found trained model: {trained_unet_path}")
            break
    
    if not trained_unet_path:
        print("Trained model file not found")
        return

    device = "cuda:1"
    erase_pipeline = load_erase_model(base_model_path, trained_unet_path, device = device)
    results, output_dir = generate_coco_images(
        erase_pipeline, 
        coco_df,
        device= device
    )
    
    print(f"\nCOCO generation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Total generated {len(results)} images")
    print("Recommend using coco_eval.py for quantitative evaluation")

if __name__ == "__main__":
    main()