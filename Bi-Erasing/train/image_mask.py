import os
import random
import yaml
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from utils.training_utils import get_training_params
from utils.model_utils import (
    load_unet, load_others,
    ImageProjModel, IPAdapter, get_attn_processor
)
from utils.diffusion_utils import (
    set_scheduler_device, denoise_to_text_timestep,
    predict_image_t_noise, predict_text_t_noise
)

transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def train_image_mode(args):

    device_list = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    device_1 = torch.device(device_list[0])

    if len(device_list) > 1:
        device_2 = torch.device(device_list[1])
        print(f"Using dual-GPU mode: origin model on {device_1}, training model on {device_2}")
    else:
        device_2 = device_1
        print(f"Using single-GPU mode: both models share {device_1}")
        print("Warning: Single-GPU mode may cause OOM, dual-GPU recommended")

    origin_unet = load_unet(args.ckpt_path, requires_grad=False).to(device_1)
    unet = load_unet(args.ckpt_path, requires_grad=True).to(device_2)

    if args.unet_ckpt_path:
        print(f"Loading pretrained UNet checkpoint: {args.unet_ckpt_path}")
        checkpoint = torch.load(args.unet_ckpt_path, map_location='cpu')
        unet.load_state_dict(checkpoint)
        origin_unet.load_state_dict(checkpoint)
        print("Pretrained UNet checkpoint loaded successfully")

    unet.train()
    origin_unet.eval()

    vae, tokenizer, text_encoder, noise_scheduler, _ = load_others(args.ckpt_path, requires_grad=False)
    text_encoder = text_encoder.to(origin_unet.device)

    parameters = get_training_params(unet, args.train_method)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    criterion = torch.nn.MSELoss()
    num_inference_steps = args.num_inference_steps

    noise_scheduler.set_timesteps(num_inference_steps)

    save_dir = args.save_path or os.path.join("checkpoints", args.modality, args.prompt)
    unet_save_path = os.path.join(save_dir, f"unet_{args.train_method}_im{args.image_number}_ng{args.negative_guidance}_it{args.iterations}_B")
    os.makedirs(unet_save_path, exist_ok=True)
    writer = SummaryWriter(unet_save_path)

    print(f"Loading image encoder: {args.image_encoder_path}")
    try:
        if os.path.exists(args.image_encoder_path):
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path, local_files_only=True).to(origin_unet.device)
        else:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).to(origin_unet.device)
        image_encoder.requires_grad_(False)
        print("Image encoder loaded successfully")
    except Exception as e:
        print(f"Failed to load image encoder: {e}")
        raise

    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    origin_image_proj_model = ImageProjModel(
        cross_attention_dim=origin_unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    attn_procs = get_attn_processor(unet)
    unet.set_attn_processor(attn_procs)

    origin_attn_procs = get_attn_processor(origin_unet)
    origin_unet.set_attn_processor(origin_attn_procs)

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    origin_adapter_modules = torch.nn.ModuleList(origin_unet.attn_processors.values())
    ip_adapter_path = args.ip_adapter

    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, ip_adapter_path).to(unet.device)
    origin_ip_adapter = IPAdapter(origin_unet, origin_image_proj_model, origin_adapter_modules, ip_adapter_path).to(origin_unet.device)

    neg_input_ids = tokenizer(args.prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids
    neg_text_embeddings = text_encoder(neg_input_ids.to(origin_unet.device))[0].to(unet.device)

    positive_prompt = args.positive_prompts if args.positive_prompts else "clothed"
    pos_input_ids = tokenizer(positive_prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids
    pos_text_embeddings = text_encoder(pos_input_ids.to(origin_unet.device))[0].to(unet.device)
    print(f"Positive guidance prompt: '{positive_prompt}'")

    uncond_input_ids = tokenizer("", return_tensors="pt", padding="max_length", truncation=True).input_ids
    uncond_text_embeddings = text_encoder(uncond_input_ids.to(origin_unet.device))[0].to(unet.device)

    mask_dir = getattr(args, 'mask_dir', None)
    use_mask = mask_dir is not None and os.path.exists(mask_dir)

    if mask_dir is not None:
        if use_mask:
            print(f"Mask functionality enabled:")
            print(f"   Mask directory: {mask_dir}")
            print(f"   Mode: Image preprocessing (highlight important regions)")
        else:
            print(f"Mask functionality disabled: directory not found {mask_dir}")
    else:
        print("Mask functionality not enabled")


    negative_image_path = args.negative_image_dir
    positive_image_path = args.positive_image_dir

    print(f"Negative image path: {negative_image_path}")
    print(f"Positive image path: {positive_image_path}")

    negative_data = _load_image_and_mask_list(negative_image_path, args.image_number, mask_dir)
    positive_data = _load_image_and_mask_list(positive_image_path, args.image_number, mask_dir)

    if use_mask:
        neg_with_mask = sum(1 for _, mask_path in negative_data if mask_path is not None)
        pos_with_mask = sum(1 for _, mask_path in positive_data if mask_path is not None)
        print(f"Negative images: {len(negative_data)} images ({neg_with_mask} with mask)")
        print(f"Positive images: {len(positive_data)} images ({pos_with_mask} with mask)")
    else:
        print(f"Negative images (erase target): {len(negative_data)} images from {negative_image_path}")
        print(f"Positive images (guide target): {len(positive_data)} images from {positive_image_path}")

    if len(negative_data) == 0:
        raise ValueError(f"No negative images found, check path: {negative_image_path}")
    if len(positive_data) == 0:
        raise ValueError(f"No positive images found, check path: {positive_image_path}")

    negative_weight = getattr(args, 'negative_weight', 0.6)
    positive_weight = getattr(args, 'positive_weight', 0.4)
    print(f"Guidance weights - negative: {negative_weight}, positive: {positive_weight}")

    for idx in tqdm(range(args.iterations), desc="[Bidirectional Image Guidance Training]"):
        optimizer.zero_grad()
        if getattr(args, 'dynamic_weights', False):
            progress = idx / args.iterations

            stage_threshold = getattr(args, 'stage_threshold', 0.2)
            if progress > stage_threshold:
                current_negative_weight = 1.0
                current_positive_weight = 0.0
                stage_name = "Erase stage"
            else:
                current_negative_weight = 0.0
                current_positive_weight = 1.0
                stage_name = "Guide stage"


            neg_eff = current_negative_weight * args.negative_guidance
            pos_eff = current_positive_weight * args.positive_guidance
            w_neg_eff, w_pos_eff = neg_eff , pos_eff

            if idx % 100 == 0:
                print(f"[{stage_name}] Progress: {progress:.1%}, negative weight: {current_negative_weight:.2f}, positive weight: {current_positive_weight:.2f}")
        else:
            current_negative_weight = negative_weight
            current_positive_weight = positive_weight
            w_neg_eff = current_negative_weight
            w_pos_eff = current_positive_weight

        if w_neg_eff > 0:
            negative_image_embeds, negative_mask_applied = _sample_image_embedding_with_optional_mask(
                negative_data, image_encoder, use_mask
            )
        else:
            negative_image_embeds, negative_mask_applied = None, False

        if w_pos_eff > 0:
            positive_image_embeds, positive_mask_applied = _sample_image_embedding_with_optional_mask(
                positive_data, image_encoder, use_mask
            )
        else:
            positive_image_embeds, positive_mask_applied = None, False

        if args.noise_factor > 0:
            negative_noise = torch.randn_like(negative_image_embeds) * args.noise_factor
            negative_image_embeds = negative_image_embeds + negative_noise

            positive_noise = torch.randn_like(positive_image_embeds) * args.noise_factor * 0.5
            positive_image_embeds = positive_image_embeds + positive_noise

        t = torch.randint(num_inference_steps, (1,)).to(unet.device)
        t_ddpm = torch.randint(int(t * 1000 / num_inference_steps), int((t + 1) * 1000 / num_inference_steps), (1,))

        start_code = torch.randn((1, 4, 64, 64)).to(unet.device)

        with torch.no_grad():
            set_scheduler_device(noise_scheduler, unet.device)
            z_neutral = denoise_to_text_timestep(unet, uncond_text_embeddings, t, start_code, noise_scheduler)

            if w_neg_eff > 0:
                uncond_origin_noise_neg = predict_image_t_noise(
                    z_neutral, t_ddpm, origin_unet, uncond_text_embeddings, origin_ip_adapter, negative_image_embeds
                )
                negative_cond_origin_noise = predict_image_t_noise(
                    z_neutral, t_ddpm, origin_unet, neg_text_embeddings, origin_ip_adapter, negative_image_embeds
                )
            else:
                uncond_origin_noise_neg, negative_cond_origin_noise = None, None

            if w_pos_eff > 0:
                uncond_origin_noise_pos = predict_image_t_noise(
                    z_neutral, t_ddpm, origin_unet, uncond_text_embeddings, origin_ip_adapter, positive_image_embeds
                )
                positive_cond_origin_noise = predict_image_t_noise(
                    z_neutral, t_ddpm, origin_unet, pos_text_embeddings, origin_ip_adapter, positive_image_embeds
                )
            else:
                uncond_origin_noise_pos, positive_cond_origin_noise = None, None

        if w_neg_eff > 0:
            negative_cond_noise = predict_image_t_noise(z_neutral, t_ddpm, unet, neg_text_embeddings, ip_adapter, negative_image_embeds)
        else:
            negative_cond_noise = None

        if w_pos_eff > 0:
            positive_cond_noise = predict_image_t_noise(z_neutral, t_ddpm, unet, pos_text_embeddings, ip_adapter, positive_image_embeds)
        else:
            positive_cond_noise = None

        total_loss = 0.0

        if w_neg_eff > 0 and negative_cond_noise is not None:
            negative_cond_origin_noise, uncond_origin_noise_neg, negative_cond_noise = [
                t.to(unet.device) for t in [negative_cond_origin_noise, uncond_origin_noise_neg, negative_cond_noise]
            ]

            negative_target = uncond_origin_noise_neg - args.negative_guidance * (negative_cond_origin_noise - uncond_origin_noise_neg)
            negative_loss = criterion(negative_cond_noise, negative_target)
            total_loss += w_neg_eff * negative_loss

            writer.add_scalar('Loss/negative', negative_loss.item(), idx)
        else:
            negative_loss = torch.tensor(0.0)
            writer.add_scalar('Loss/negative', 0.0, idx)

        if w_pos_eff > 0 and positive_cond_noise is not None:
            positive_cond_origin_noise, uncond_origin_noise_pos, positive_cond_noise = [
                t.to(unet.device) for t in [positive_cond_origin_noise, uncond_origin_noise_pos, positive_cond_noise]
            ]

            positive_target = uncond_origin_noise_pos + args.positive_guidance * (positive_cond_origin_noise - uncond_origin_noise_pos)
            positive_loss = criterion(positive_cond_noise, positive_target)
            total_loss += w_pos_eff * positive_loss

            writer.add_scalar('Loss/positive', positive_loss.item(), idx)
        else:
            positive_loss = torch.tensor(0.0)
            writer.add_scalar('Loss/positive', 0.0, idx)

        if total_loss > 0:
            total_loss.backward()
            optimizer.step()

        writer.add_scalar('Loss/total', float(total_loss), idx)
        writer.add_scalar('Weights/w_neg_eff', float(w_neg_eff), idx)
        writer.add_scalar('Weights/w_pos_eff', float(w_pos_eff), idx)

        if use_mask:
            writer.add_scalar('Mask/negative_applied', float(negative_mask_applied), idx)
            writer.add_scalar('Mask/positive_applied', float(positive_mask_applied), idx)

        if (idx + 1) % args.save_iter == 0:
            os.makedirs(unet_save_path, exist_ok=True)
            state_dict = unet.state_dict().copy()
            keys_to_remove = ['to_k_ip.weight', 'to_v_ip.adapter']
            for key in list(state_dict.keys()):
                if any(sub in key for sub in keys_to_remove):
                    del state_dict[key]
            file = f"unet_{idx}.pth"
            torch.save(state_dict, os.path.join(unet_save_path, file))
            print(f"[Checkpoint] Saved UNet at iteration {idx + 1}")


def _load_image_list(image_path, image_number):

    if os.path.isdir(image_path):
        all_images = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(('png', 'jpg'))]
        return random.sample(all_images, min(image_number, len(all_images)))
    else:
        return [image_path]


def _load_image_and_mask_list(image_path, image_number, mask_dir=None):

    image_list = _load_image_list(image_path, image_number)

    if mask_dir is None:
        return image_list

    image_mask_pairs = []
    for img_path in image_list:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f'{base_name}_mask.npy')

        if os.path.exists(mask_path):
            image_mask_pairs.append((img_path, mask_path))
        else:
            image_mask_pairs.append((img_path, None))

    return image_mask_pairs


def _sample_image_embedding(image_list, image_encoder):

    image_path = random.choice(image_list)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(image_encoder.device)
    return image_encoder(image_tensor).image_embeds


def _sample_image_embedding_with_optional_mask(data_list, image_encoder, use_mask):

    if use_mask:
        image_path, mask_path = random.choice(data_list)

        image = Image.open(image_path).convert("RGB")

        if mask_path is not None:
            mask = _load_mask(mask_path)
            if mask is not None:
                masked_image = _apply_mask_to_image(image, mask)
                image_tensor = transform(masked_image).unsqueeze(0).to(image_encoder.device)
                image_embeds = image_encoder(image_tensor).image_embeds
                return image_embeds, True
        image_tensor = transform(image).unsqueeze(0).to(image_encoder.device)
        image_embeds = image_encoder(image_tensor).image_embeds
        return image_embeds, False
    else:
        image_path = random.choice(data_list)

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(image_encoder.device)
        image_embeds = image_encoder(image_tensor).image_embeds

        return image_embeds, False


def _load_mask(mask_path):

    try:
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask).float()
        return mask
    except Exception as e:
        print(f"Failed to load mask: {mask_path}, error: {e}")
        return None


def _apply_mask_to_image(image, mask):

    image_array = np.array(image)

    if mask.shape != image_array.shape[:2]:
        mask_resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=image_array.shape[:2],
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
    else:
        mask_resized = mask.numpy()

    mask_3d = np.expand_dims(mask_resized, axis=2).repeat(3, axis=2)
    mean_color = image_array.mean(axis=(0, 1), keepdims=True)
    masked_array = mask_3d * image_array + (1 - mask_3d) * mean_color
    masked_array = np.clip(masked_array, 0, 255).astype(np.uint8)
    masked_image = Image.fromarray(masked_array)

    return masked_image


