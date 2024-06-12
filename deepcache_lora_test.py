import os
import torch
import argparse
import json

from diffusers import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler

from LoRA_Cache.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


def load_lora_info(image_style):
    with open(f'{image_style}_lora_info.json') as file:
        lora_info = json.load(file)
    return lora_info

def get_prompt(image_style):
    if image_style == 'anime':
        prompt = "masterpiece, best quality"
        negative_prompt = "EasyNegative, extra fingers, extra limbs, fewer fingers, fewer limbs, multiple girls, multiple views, worst quality, low quality, depth of field, blurry, greyscale, 3D face, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, reference sheet, curvy, plump, fat, muscular female, strabismus, clothing cutout, side slit, tattoo, nsfw"
    else:
        prompt = "RAW photo, subject, 8k uhd, dslr, high quality, Fujifilm XT3, half-length portrait from knees up"
        negative_prompt = "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    return prompt, negative_prompt

def main(args):

    lora_path = os.path.join('models/lora', args.image_style)
    lora_info = load_lora_info(args.image_style)

    # Get correct checkpoint
    if args.image_style == 'anime':
        model_name = 'gsdf/Counterfeit-V2.5'
    else:
        model_name = 'SG161222/Realistic_Vision_V5.1_noVAE'

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        # torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # Set VAE
    if args.image_style == "reality":
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            # torch_dtype=torch.float16
        ).to("cuda")
        pipeline.vae = vae

    # Set scheduler
    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    init_prompt, negative_prompt = get_prompt(args.image_style)
    caching_intervals = ()

    if not os.path.exists('images'):
        os.makedirs('images')

    # Go through each LoRA and caching interval
    for category in lora_info:
        for lora in lora_info[category]:
            for interval in caching_intervals:

                pipeline.load_lora_weights(
                    lora_path,
                    weight_name=f'{lora['id']}.safetensors',
                    adapter_name=lora['id']
                )
                prompt = init_prompt + ', ' + ', '.join(lora['trigger'])
                pipeline.set_adapters([lora['id']])

                image = pipeline(
                    prompt=prompt, 
                    negative_prompt=negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.denoise_steps,
                    guidance_scale=args.cfg_scale,
                    generator=args.generator,
                    cross_attention_kwargs={"scale": args.lora_scale},
                    cache_interval=interval,
                    cache_layer_id=0,
                    cache_block_id=1
                ).images[0]
                image.save(os.path.join('images', f'{args.image_style}_{lora['id']}_{interval}.png'), 'PNG')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--lora_scale', default=0.8, type=float)
    parser.add_argument('--switch_step', default=5, type=int)

    # Arguments for generating images
    parser.add_argument('--height', default=512, type=int)
    parser.add_argument('--width', default=512, type=int)
    parser.add_argument('--denoise_steps', default=200, type=int)
    parser.add_argument('--cfg_scale', default=10, type=float)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--image_style', default='anime', choices=['anime', 'reality'], type=str)

    args = parser.parse_args()
    args.generator = torch.manual_seed(args.seed)

    main(args)