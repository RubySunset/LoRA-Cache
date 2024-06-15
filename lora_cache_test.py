import os
import torch
import argparse
import json
import itertools

from diffusers import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler

from LoRA_Cache.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


def make_callback(switch_step, loras):
    def switch_callback(pipeline, step_index, timestep, callback_kwargs):
        callback_outputs = {}
        if step_index > 0 and step_index % switch_step == 0:
            for cur_lora_index, lora in enumerate(loras):
                if lora in pipeline.get_active_adapters():
                    next_lora_index = (cur_lora_index + 1) % len(loras)
                    pipeline.set_adapters(loras[next_lora_index])
                    break
        return callback_outputs
    return switch_callback

def get_prompt(image_style):
    if image_style == 'anime':
        prompt = "masterpiece, best quality"
        negative_prompt = "EasyNegative, extra fingers, extra limbs, fewer fingers, fewer limbs, multiple girls, multiple views, worst quality, low quality, depth of field, blurry, greyscale, 3D face, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, reference sheet, curvy, plump, fat, muscular female, strabismus, clothing cutout, side slit, tattoo, nsfw"
    else:
        prompt = "RAW photo, subject, 8k uhd, dslr, high quality, Fujifilm XT3, half-length portrait from knees up"
        negative_prompt = "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    return prompt, negative_prompt

def generate_combinations(lora_info, compos_num):
    """
    Generate all combinations of LoRA elements ensuring that each combination includes at least one 'character'.

    Args:
    lora_info (dict): A dictionary containing LoRA elements and their instances.
    compos_num (int): The number of elements to be included in each combination.

    Returns:
    list: A list of all possible combinations, each combination is a list of element instances.
    """
    elements = list(lora_info.keys())

    # Check if the composition number is greater than the number of element types
    if compos_num > len(elements):
        raise ValueError("The composition number cannot be greater than the number of elements.")

    all_combinations = []

    # Ensure that 'character' is always included in the combinations
    if 'character' in elements:
        # Remove 'character' from the list to avoid duplicating
        elements.remove('character')

        # Generate all possible combinations of the remaining element types
        selected_types = list(itertools.combinations(elements, compos_num - 1))

        # For each combination of types, generate all possible combinations of instances
        for types in selected_types:
            # Add 'character' to the current combination of types
            current_types = ['character', *types]

            # Gather instances for each type in the current combination
            instances = [lora_info[t] for t in current_types]

            # Create combinations of instances across the selected types
            for combination in itertools.product(*instances):
                all_combinations.append(combination)

    return all_combinations

def main(args):

    lora_path = os.path.join('models/lora', args.image_style)
    with open(f'{args.image_style}_lora_info.json') as file:
        lora_info = json.load(file)

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

    if not os.path.exists('images2'):
        os.makedirs('images2')

    # Load all lora weights
    for category in lora_info:
        for lora in lora_info[category]:
            pipeline.load_lora_weights(
                lora_path,
                weight_name=lora['id']+'.safetensors',
                adapter_name=lora['id']
            )
    
    combinations = generate_combinations(lora_info, args.compos_num)

    for combo in combinations:

        active_loras = [lora['id'] for lora in combo]

        # set prompt
        triggers = [trigger for lora in combo for trigger in lora['trigger']]
        prompt = init_prompt + ', ' + ', '.join(triggers)
        
        # set LoRAs
        if args.method == "switch":
            pipeline.set_adapters([active_loras[0]])
            switch_callback = make_callback(args.switch_step, active_loras)
        elif args.method == "merge":
            pipeline.set_adapters(active_loras)
            switch_callback = None
        else:
            pipeline.set_adapters(active_loras)
            switch_callback = None

        # generate images
        image = pipeline(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.denoise_steps,
            guidance_scale=args.cfg_scale,
            generator=args.generator,
            cross_attention_kwargs={"scale": args.lora_scale},
            callback_on_step_end=switch_callback,
            lora_composite=True if args.method == "composite" else False,
            dom_lora_coeff=args.dom_lora_weight,
            cache_interval=args.cache_interval,
            cache_layer_id=args.cache_layer_id,
            cache_block_id=args.cache_block_id
        ).images[0]

        filename = f'{args.image_style}_{args.method}'
        if args.method == 'composite' and args.cache_layer_id is not None:
            filename += f'_{args.dom_lora_weight}'
        for lora in combo:
            filename += f'_{lora["id"]}'
        image.save(os.path.join('images2', f'{filename}.png'), 'PNG')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    # General options
    parser.add_argument('--height', default=512, type=int)
    parser.add_argument('--width', default=512, type=int)
    parser.add_argument('--denoise_steps', default=200, type=int)
    parser.add_argument('--cfg_scale', default=10, type=float)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--image_style', default='anime', choices=['anime', 'reality'], type=str)

    # LoRA options
    parser.add_argument('--compos_num', default=2, type=int)
    parser.add_argument('--method', default='switch', type=str)
    parser.add_argument('--lora_scale', default=0.8, type=float)
    parser.add_argument('--switch_step', default=5, type=int)

    # Caching options
    parser.add_argument('--dom_lora_weight', default=1.0, type=float)
    parser.add_argument('--cache_interval', default=1, type=int)
    parser.add_argument('--cache_layer_id', default=None, type=int)
    parser.add_argument('--cache_block_id', default=None, type=int)

    args = parser.parse_args()
    args.generator = torch.manual_seed(args.seed)

    main(args)