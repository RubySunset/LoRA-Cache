import json
import torch
import numpy as np
import os
from functools import partial
from diffusers.utils import load_image
from torchmetrics.functional.multimodal import clip_score

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
caching_intervals = (1, 2, 3, 4, 5)

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def load_image(name):
    image = load_image(os.path.join('images', f'{name}.png'))
    return np.array(image, dtype=np.float32)[np.newaxis, ...] / 255.0

scores = {}
for style in ('anime', 'reality'):
    scores[style] = {}
    lora_info = None
    with open(f'{style}_lora_info.json', 'r') as file:
        lora_info = json.load(file)
    for category in lora_info:
        scoreMatrix = np.empty((len(lora_info[category]), len(caching_intervals)))
        for i, lora in enumerate(lora_info[category]):
            prompt = [', '.join(lora['trigger'])]
            for j, interval in enumerate(caching_intervals):
                image = load_image(f'{style}_{lora['id']}_{str(interval)}')
                score = calculate_clip_score(image, prompt)
                scoreMatrix[i, j] = score
        scores[style][category] = scoreMatrix.mean(axis=0).tolist()

with open('results.json', 'w') as file:
    json.dump(scores, file)