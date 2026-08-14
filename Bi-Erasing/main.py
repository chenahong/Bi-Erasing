import argparse
from train.text_trainer import train_text_mode
from train.image_mask import train_image_mode


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modality', type=str, choices=['text', 'image'], default='text')
    parser.add_argument('--train_method', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--devices', type=str, default='0')

    parser.add_argument('--ckpt_path', type=str, default='/data/hao_chen/model/hub/models--runwayml--stable-diffusion-v1-5')
    parser.add_argument('--unet_ckpt_path', type=str, default=None)
    parser.add_argument('--image_encoder_path', type=str, default='/data/hao_chen/model/hub/models--runwayml--stable-diffusion-v1-5/image_encoder/')
    parser.add_argument('--ip_adapter', type=str, default='/data/hao_chen/model/hub/models--runwayml--stable-diffusion-v1-5/ip_adapter/ip-adapter_sd15.bin')

    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--save_iter', type=int, default=500)

    parser.add_argument('--negative_guidance', type=float, default=1.0)
    parser.add_argument('--positive_guidance', type=float, default=1.0)
    parser.add_argument('--num_inference_steps', type=int, default=50)

    parser.add_argument('--negative_image_dir', type=str, default=None)
    parser.add_argument('--positive_image_dir', type=str, default=None)
    parser.add_argument('--image_number', type=int, default=100)
    parser.add_argument('--noise_factor', type=float, default=0.0)

    parser.add_argument('--positive_prompts', type=str, default=None)

    parser.add_argument('--negative_weight', type=float, default=0.6)
    parser.add_argument('--positive_weight', type=float, default=0.4)
    parser.add_argument('--dynamic_weights', action='store_true')
    parser.add_argument('--stage_threshold', type=float, default=0.8)

    parser.add_argument('--mask_dir', type=str, default=None)

    return parser.parse_args()


def main():
    args = init_args()

    if args.modality == 'text':
        train_text_mode(args)
    elif args.modality == 'image':
        train_image_mode(args)
    else:
        raise ValueError("Modality must be either 'text' or 'image'")


if __name__ == "__main__":
    main()
