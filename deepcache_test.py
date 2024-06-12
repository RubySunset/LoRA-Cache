from LoRA_Cache.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5').to('cuda')
# Without DeepCache
image = pipe(
    prompt='A girl looking up at the stars'
).images[0]
image.save('image.png', 'PNG')
# With DeepCache
image = pipe(
    prompt='A girl looking up at the stars',
    cache_interval=5,
    cache_layer_id=0,
    cache_block_id=1
).images[0]
image.save('image_dc.png', 'PNG')