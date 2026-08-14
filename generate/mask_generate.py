
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor, CLIPSegProcessor, CLIPSegForImageSegmentation

def setup_latex_style_fonts():

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Computer Modern'],
        'text.usetex': False, 
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'mathtext.fontset': 'cm',
        'font.weight': 'normal'
    })
    return False 

_latex_available = setup_latex_style_fonts()


class SemanticMaskGenerator:
    def __init__(self, method='clipseg', device='cuda'):
        self.method = method
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if method == 'clipseg':
            self.model = CLIPSegForImageSegmentation.from_pretrained(
                "CIDAS/clipseg-rd64-refined"
            ).to(self.device)
            self.processor = CLIPSegProcessor.from_pretrained(
                "CIDAS/clipseg-rd64-refined"
            )
        elif method == 'clip_ensemble':
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        
        self.model.eval()
    
    def generate_mask_clipseg(self, image, prompts):
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
    
    def generate_mask_clip_ensemble(self, image, prompts):
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        all_masks = []
        
        for prompt in prompts:
            text_inputs = self.processor(
                text=[prompt],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                vision_outputs = self.model.vision_model(pixel_values=image_tensor)
                patch_features = vision_outputs.last_hidden_state[:, 1:, :]
                
                text_outputs = self.model.text_model(**text_inputs)
                text_features = text_outputs.pooler_output

                if hasattr(self.model, 'visual_projection'):
                    patch_features = self.model.visual_projection(patch_features)
                if hasattr(self.model, 'text_projection'):
                    text_features = self.model.text_projection(text_features)

                patch_norm = F.normalize(patch_features, dim=-1)
                text_norm = F.normalize(text_features, dim=-1)

                similarity = torch.matmul(patch_norm, text_norm.T).squeeze()

                grid_size = int(np.sqrt(similarity.shape[0]))
                mask = similarity.reshape(grid_size, grid_size)
                
                all_masks.append(mask.cpu().numpy())

        ensemble_mask = np.maximum.reduce(all_masks)
        
        return ensemble_mask
    
    def post_process_mask(self, mask, threshold=0.5, min_area_ratio=0.01, 
                         smooth_kernel=5, use_morphology=True):

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
    
    def generate(self, image_path, prompts, threshold=0.5, 
                post_process=True, return_binary=False):
        """
        Main function to generate mask
        
        Args:
            image_path: image_path: Image path
            prompts: prompts: Text prompt list, e.g. ["cat", "dog"] or ["naked person", "nude body"]
            threshold: threshold: Binarization threshold
            post_process: post_process: Whether to post-process
            return_binary: return_binary: Whether to return binary mask
            
        Returns:
            mask: [H, W] numpy array, values in [0, 1]
            image: PIL Image object
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        print(f"Using prompts: {prompts}")
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        if self.method == 'clipseg':
            mask = self.generate_mask_clipseg(image, prompts)
        else:
            mask = self.generate_mask_clip_ensemble(image, prompts)
        
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_resized = mask_pil.resize(original_size, Image.BILINEAR)
        mask = np.array(mask_resized).astype(float) / 255.0
        
        print(f"Original mask statistics:")
        print(f"   Range: [{mask.min():.4f}, {mask.max():.4f}]")
        print(f"   Mean: {mask.mean():.4f}")
        print(f"   Coverage: {(mask > threshold).sum() / mask.size * 100:.1f}%")
        
        if post_process:
            print(f"🔧 Post-process mask...")
            coverage = (mask > threshold).sum() / mask.size
            if coverage < 0.05:
                print(f"   Low coverage detected ({coverage*100:.1f}%), using relaxed parameters")
                mask = self.post_process_mask(mask, threshold=threshold, 
                                             min_area_ratio=0.001,
                                             smooth_kernel=3,
                                             use_morphology=False)
            else:
                mask = self.post_process_mask(mask, threshold=threshold)
            
            print(f"Post-processing statistics:")
            print(f"   Range: [{mask.min():.4f}, {mask.max():.4f}]")
            print(f"   Coverage: {(mask > 0.5).sum() / mask.size * 100:.1f}%")
        
        if return_binary:
            mask = (mask > 0.5).astype(np.uint8)
        
        return mask, image


def save_visualization(image, mask, output_path, prompts, alpha=0.6):
    """Save visualization results - save 3 images separately"""
    base_path = os.path.splitext(output_path)[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'{base_path}_original.png', dpi=300, bbox_inches='tight', 
                pad_inches=0, format='png')
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(mask, cmap='jet')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'{base_path}_mask.png', dpi=300, bbox_inches='tight', 
                pad_inches=0, format='png')
    plt.close()
    
    image_np = np.array(image).astype(float) / 255.0
    mask_colored = plt.get_cmap('jet')(mask)[:, :, :3]
    overlay = (1 - alpha) * image_np + alpha * mask_colored
    overlay = np.clip(overlay, 0, 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(overlay)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'{base_path}_overlay.png', dpi=300, bbox_inches='tight', 
                pad_inches=0, format='png')
    plt.close()
    
    print(f"   Saved original: {base_path}_original.png")
    print(f"   Saved mask: {base_path}_mask.png") 
    print(f"   Saved overlay: {base_path}_overlay.png")


def batch_generate_masks(image_dir, output_dir, prompts, method='clipseg', 
                        threshold=0.5, save_format='npy'):
    """Batch generate masks"""
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SemanticMaskGenerator(method=method)
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nFound {len(image_files)} images")
    print("="*60 + "\n")
    
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {image_file}")
        
        image_path = os.path.join(image_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        try:
            mask, image = generator.generate(
                image_path, 
                prompts=prompts,
                threshold=threshold,
                post_process=True,
                return_binary=False
            )
            
            if save_format == 'npy':
                mask_path = os.path.join(output_dir, f'{base_name}_mask.npy')
                np.save(mask_path, mask)
            elif save_format == 'png':
                mask_path = os.path.join(output_dir, f'{base_name}_mask.png')
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img.save(mask_path)
            
            vis_path = os.path.join(output_dir, f'{base_name}_vis.png')
            save_visualization(image, mask, vis_path, prompts)
            
            print(f"   Saved mask: {mask_path}")
            
        except Exception as e:
            print(f"   Failed: {e}\n")
            continue



PRESET_PROMPTS = {
    'cat': ['cat', 'feline', 'kitten'],
    'dog': ['dog', 'canine', 'puppy'],
    'person': ['person', 'human', 'people'],
    'car': ['car', 'vehicle', 'automobile'],
    'bird': ['bird', 'avian'],
    
    'nudity': ['naked person', 'nude body', 'exposed skin', 'unclothed human', 'bare body parts'],
    'violence': ['violence', 'fighting', 'weapon', 'blood'],
    'text': ['text', 'words', 'letters', 'writing'],
    'logo': ['logo', 'brand', 'trademark'],
    
    'red': ['red color', 'red object'],
    'blue': ['blue color', 'blue object'],
    'metal': ['metal surface', 'metallic'],
    'wood': ['wood', 'wooden'],
    
    'background': ['background', 'backdrop'],
    'foreground': ['foreground', 'main subject'],
    'sky': ['sky', 'clouds'],
    'ground': ['ground', 'floor', 'earth'],
}


def main():
    parser = argparse.ArgumentParser(
        description='Universal Semantic Mask Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python semantic_mask_generator.py --image cat.jpg --prompts "cat"
  
  python semantic_mask_generator.py --image img.jpg --prompts "naked person" "nude body" "exposed skin"
  
  python semantic_mask_generator.py --image img.jpg --preset nudity
  
  python semantic_mask_generator.py --image_dir images/ --prompts "person" --output_dir masks/
  
  python semantic_mask_generator.py --list_presets
        """
    )
    
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image_dir', type=str, help='Image directory (batch processing)')
    parser.add_argument('--output_dir', type=str, default='masks_output', help='Output directory')
    
    parser.add_argument('--prompts', type=str, nargs='+', help='Text prompts (multiple allowed)')
    parser.add_argument('--preset', type=str, choices=list(PRESET_PROMPTS.keys()),
                       help='Use preset prompt combination')
    parser.add_argument('--list_presets', action='store_true', help='List all preset prompts')
    
    parser.add_argument('--method', type=str, default='clipseg',
                       choices=['clipseg', 'clip_ensemble'],
                       help='Method selection (clipseg recommended)')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold: Binarization threshold')
    parser.add_argument('--no_post_process', action='store_true', help='Skip post-processing (if dependencies missing)')
    parser.add_argument('--save_format', type=str, default='npy',
                       choices=['npy', 'png'],
                       help='Mask save format')
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("\nPreset Prompt Combinations:\n")
        for name, prompts in PRESET_PROMPTS.items():
            print(f"  {name:15s}: {prompts}")
        print()
        return
    
    if args.preset:
        prompts = PRESET_PROMPTS[args.preset]
        print(f"Using preset '{args.preset}': {prompts}")
    elif args.prompts:
        prompts = args.prompts
    else:
        print("Please specify --prompts or --preset")
        return
    
    if args.image:
        print("="*60)
        print("Universal Semantic Mask Generator")
        print("="*60)
        print(f"Image: {args.image}")
        print(f"Prompts: {prompts}")
        print(f"Method: {args.method}")
        print(f"Threshold: {args.threshold}")
        print("="*60 + "\n")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        generator = SemanticMaskGenerator(method=args.method)
        mask, image = generator.generate(
            args.image,
            prompts=prompts,
            threshold=args.threshold,
            post_process=not args.no_post_process
        )
        
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        
        if args.save_format == 'npy':
            mask_path = os.path.join(args.output_dir, f'{base_name}_mask.npy')
            np.save(mask_path, mask)
        else:
            mask_path = os.path.join(args.output_dir, f'{base_name}_mask.png')
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(mask_path)
        
        vis_path = os.path.join(args.output_dir, f'{base_name}_vis.png')
        save_visualization(image, mask, vis_path, prompts)
        
        print(f"\nmask: {mask_path}")
        
    elif args.image_dir:
        batch_generate_masks(
            args.image_dir,
            args.output_dir,
            prompts=prompts,
            method=args.method,
            threshold=args.threshold,
            save_format=args.save_format
        )
    
    else:
        print("Please specify --image or --image_dir")
        return
    
    print("\n" + "="*60)
    print("Completed!")
    print("="*60)


if __name__ == "__main__":
    main()
