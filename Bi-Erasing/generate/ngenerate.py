
import torch
import numpy as np
import random
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_Adv_model(base_model_path, trained_unet_path, device="cuda:0"):
    print(f"Loading AdvUnlearn model weights: {trained_unet_path}")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    try:
        checkpoint = torch.load(trained_unet_path, map_location="cpu")
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        original_state_dict = pipeline.text_encoder.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            target_key = key
            
            if key.startswith('text_encoder.'):
                target_key = key.replace('text_encoder.', '')
            
            if target_key in original_state_dict:
                if original_state_dict[target_key].shape == value.shape:
                    filtered_state_dict[target_key] = value
                else:
                    skipped_keys.append(f"{key} -> {target_key} (shape mismatch: {original_state_dict[target_key].shape} vs {value.shape})")
            else:
                if key in original_state_dict:
                    if original_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        skipped_keys.append(f"{key} (shape mismatch)")
                else:
                    skipped_keys.append(f"{key} (not found)")
        
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} incompatible weights")
            for skip_info in skipped_keys[:5]:
                print(f"  - {skip_info}")
            if len(skipped_keys) > 5:
                print(f"  - ... and {len(skipped_keys) - 5} items")
        
        missing_keys, unexpected_keys = pipeline.text_encoder.load_state_dict(filtered_state_dict, strict=False)
        print(f"Successfully loaded {len(filtered_state_dict)} Text Encoder weights")
        
        if missing_keys:
            print(f"Missing weights: {len(missing_keys)} items")
        if unexpected_keys:
            print(f"Unexpected weights: {len(unexpected_keys)} items")
        
    except Exception as e:
        print(f"Text Encoder weight loading failed: {e}")
        raise
    
    return pipeline

def load_erase_model(base_model_path, trained_unet_path, device="cuda:0"):
    print("Loading Erasing model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
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
    
    print(f"Skipped {len(skipped_keys)} incompatible weights")
    
    missing_keys, unexpected_keys = pipeline.unet.load_state_dict(filtered_state_dict, strict=False)
    print(f"Successfully loaded {len(filtered_state_dict)} weights")
    
    return pipeline

def load_original_model(base_model_path, device="cuda:0"):
    """Load original SD model"""
    print("Loading original Stable Diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    print("Original model loaded")
    return pipeline

def generate_images(erase_pipeline, prompts_csv_path, device="cuda:0"):
    """Generate images based on prompts and seeds from CSV file"""
    df = pd.read_csv(prompts_csv_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"image/nudity_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    erase_dir = os.path.join(output_dir, "erase_model")
    os.makedirs(erase_dir, exist_ok=True)
    
    results = []
    total = len(df)
    ok_count = 0

    print(f"\nStarting generation (erase model only): {total} images")

    with tqdm(total=total, desc="Generate comparison images", unit="images") as pbar:
        for idx, row in df.iterrows():
            prompt = row['prompt']
            seed = int(row['evaluation_seed'])
            case_number = row['case_number']
            
            rec = {
                "prompt": prompt,
                "seed": seed,
                "case_number": case_number,
                "erase_image_path": None,
                "success": False
            }
            
            set_seed(seed)
            try:
                with torch.no_grad():
                    erase_pipeline.set_progress_bar_config(disable=True)
                    
                    erase_image = erase_pipeline(
                        prompt=prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=torch.Generator(device=device).manual_seed(seed)
                    ).images[0]
                    
                    erase_pipeline.set_progress_bar_config(disable=False)
                
                erase_filename = f"erase_case{case_number:04d}_seed{seed}.png"
                erase_path = os.path.join(erase_dir, erase_filename)
                erase_image.save(erase_path)
                
                rec["erase_image_path"] = erase_path
                rec["success"] = True
                ok_count += 1
                
            except Exception as e:
                print(f"Erasing model generation failed (case={case_number}, seed={seed}): {e}")
            
            results.append(rec)
            
            pbar.update(1)
            pbar.set_postfix({
                "Case": f"{case_number}",
                "Success": f"{ok_count}/{idx + 1}",
                "Success Rate": f"{ok_count/(idx + 1)*100:.1f}%"
            })
    
    print(f"\nGeneration completed: {ok_count}/{total} images successfully generated (Success rate: {ok_count/total*100:.1f}%)")
    return results, output_dir

def main():
    base_model_path = "/data/hao_chen/model/hub/models--runwayml--stable-diffusion-v1-5"

    possible_paths = [""]

    trained_unet_path = None
    for path in possible_paths:
        if os.path.exists(path):
            trained_unet_path = path
            print(f"Found trained model: {trained_unet_path}")
            break
    
    if not trained_unet_path:
        print("Trained model file not found")
        return

    prompts_csv_path = "prompts/unsafe-prompts4703.csv"

    device = "cuda:5"
    erase_pipeline = load_erase_model(base_model_path, trained_unet_path, device=device)

    results, output_dir = generate_images(
        erase_pipeline, 
        prompts_csv_path,
        device=device
    )
    
    print(f"\\n Comparison evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print("\\n Please check generated images to evaluate Erasing effect")

if __name__ == "__main__":
    main()