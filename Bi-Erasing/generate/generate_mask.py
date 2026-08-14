
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


class NudityMaskBatchGenerator:
    
    def __init__(self, method='clipseg', device='cuda'):
        self.method = method
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading CLIPSeg model...")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(self.device)
        self.processor = CLIPSegProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.model.eval()
        print(f"Model loaded, using device: {self.device}")
    
    def generate_mask_clipseg(self, image, prompts):
        """Generate mask using CLIPSeg"""
        inputs = self.processor(
            text=prompts,
            images=[image] * len(prompts),
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        mask = logits.max(dim=0)[0]
        mask = torch.sigmoid(mask)
        
        return mask.cpu().numpy()
    
    def post_process_mask(self, mask, threshold=0.15, min_area_ratio=0.001, 
                         smooth_kernel=3, use_morphology=True):
        """Post-process mask"""
        try:
            from scipy import ndimage
            from skimage import morphology
            has_skimage = True
        except ImportError:
            has_skimage = False
            print(" skimage not installed, using simplified post-processing")
        
        if not has_skimage:
            from scipy import ndimage
            binary_mask = (mask > threshold).astype(np.uint8)
            smooth_mask = ndimage.gaussian_filter(binary_mask.astype(float), sigma=2)
            return smooth_mask
        

        binary_mask = (mask > threshold).astype(np.uint8)

        min_area = int(mask.size * min_area_ratio)
        binary_mask = morphology.remove_small_objects(
            binary_mask.astype(bool), 
            min_size=min_area
        ).astype(np.uint8)

        if use_morphology:
            kernel = morphology.disk(smooth_kernel)
            binary_mask = morphology.binary_closing(binary_mask, kernel).astype(np.uint8)
            binary_mask = morphology.binary_opening(binary_mask, kernel).astype(np.uint8)
        
        smooth_mask = ndimage.gaussian_filter(binary_mask.astype(float), sigma=2)
        
        return smooth_mask
    
    def generate_single_mask(self, image_path, prompts, threshold=0.15):
        """Generate mask for single image"""

        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        mask = self.generate_mask_clipseg(image, prompts)

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_resized = mask_pil.resize(original_size, Image.BILINEAR)
        mask = np.array(mask_resized).astype(float) / 255.0

        coverage_before = (mask > threshold).sum() / mask.size
        if coverage_before < 0.05:  
            mask = self.post_process_mask(mask, threshold=threshold, 
                                         min_area_ratio=0.0005,
                                         smooth_kernel=2,
                                         use_morphology=False)
        else:
            mask = self.post_process_mask(mask, threshold=threshold)
        
        coverage_after = (mask > 0.5).sum() / mask.size
        
        return mask, coverage_before, coverage_after
    
    def batch_generate(self, input_dir, output_dir, prompts, threshold=0.15):
        """Batch generate masks"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            import glob
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        if len(image_files) == 0:
            print(f"No image files found in {input_dir}")
            return
        
        print(f"\nFound {len(image_files)} images")
        print(f"Using prompts: {prompts}")
        print(f"Threshold: {threshold}")
        print(f"Output directory: {output_dir}")
        print("="*60 + "\n")

        stats = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'low_coverage': 0,
            'coverage_sum': 0.0
        }
        
        for i, image_path in enumerate(tqdm(image_files, desc="s")):
            try:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f'{base_name}_mask.npy')
                
                if os.path.exists(output_path):
                    stats['success'] += 1
                    continue
                
                mask, coverage_before, coverage_after = self.generate_single_mask(
                    image_path, prompts, threshold
                )
                
                np.save(output_path, mask)
                
                stats['success'] += 1
                stats['coverage_sum'] += coverage_after
                
                if coverage_after < 0.02:
                    stats['low_coverage'] += 1
                
                if (i + 1) % 100 == 0:
                    avg_coverage = stats['coverage_sum'] / stats['success'] * 100
                    print(f"\n[{i+1}/{len(image_files)}] Average coverage: {avg_coverage:.1f}%")
                
            except Exception as e:
                print(f"\nProcessing failed {os.path.basename(image_path)}: {e}")
                stats['failed'] += 1
                continue
        
        print("\n" + "="*60)
        print("Batch generation completed")
        print("="*60)
        print(f"Total files: {stats['total']}")
        print(f"Successfully generated: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        print(f"Low coverage (<2%): {stats['low_coverage']}")
        
        if stats['success'] > 0:
            avg_coverage = stats['coverage_sum'] / stats['success'] * 100
            print(f"Average coverage: {avg_coverage:.2f}%")
        
        success_rate = stats['success'] / stats['total'] * 100
        print(f"Success rate: {success_rate:.1f}%")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Batch generate nudity image masks')
    parser.add_argument('--input_dir', type=str, 
                       default='generation_dataset_v1_5/clothed',
                       help='Input image directory')
    parser.add_argument('--output_dir', type=str, 
                       default='generation_dataset_v1_5/mask',
                       help='Output mask directory')
    parser.add_argument('--threshold', type=float, default=0.4, 
                       help='threshold: Binarization threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Computing device')
    
    args = parser.parse_args()
    
    nudity_prompts = [
        "woman",
        "man",
        "clothe"
    ]
    
    print("="*60)
    print("Batch Nudity Mask Generator")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print("="*60)
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        return
    
    generator = NudityMaskBatchGenerator(device=args.device)
    
    generator.batch_generate(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompts=nudity_prompts,
        threshold=args.threshold
    )
    
    print("\nAll completed!")
    print(f"Mask files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()