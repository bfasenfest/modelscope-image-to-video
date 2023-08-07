# Adapted from https://github.com/ExponentialML/Text-To-Video-Finetuning/blob/main/inference.py

import argparse
import warnings
import torch
import random
import subprocess
import json
import numpy as np
import os
import re
import PIL
import torch.nn.functional as F

from PIL import Image
from compel import Compel
from train import export_to_video, load_primary_models, handle_memory_attention
from diffusers import DPMSolverMultistepScheduler
from diffusers import TextToVideoSDPipeline, DiffusionPipeline
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Union
from einops import rearrange
from torch import Tensor
from tqdm import trange
from uuid import uuid4
from diffusers.utils import PIL_INTERPOLATION
from diffusers import StableDiffusionLatentUpscalePipeline

def save_image(tensor, filename):
    tensor = tensor.cpu().numpy()  # Move to CPU
    tensor = tensor.transpose((1, 2, 0))  # Swap tensor dimensions to HWC
    tensor = (tensor * 255).astype('uint8')  # Denormalize
    img = Image.fromarray(tensor)  # Convert to a PIL image
    img.save(filename)  # Save image

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

def image_to_tensor(image_path):
    # Open image file
    image = Image.open(image_path).convert('RGB')

    return preprocess(image)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def encode_video(input_file, output_file, height):
    command = ['ffmpeg',
               '-i', input_file,
               '-c:v', 'libx264',
               '-crf', '23',
               '-preset', 'fast',
               '-c:a', 'aac',
               '-b:a', '128k',
               '-movflags', '+faststart',
               '-vf', f'scale=-1:{height}',
               '-y',
               output_file]
    
    subprocess.run(command, check=True)

def get_video_height(input_file):
    command = ['ffprobe', 
               '-v', 'quiet', 
               '-print_format', 'json', 
               '-show_streams', 
               input_file]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_info = json.loads(result.stdout)
    
    for stream in video_info.get('streams', []):
        if stream['codec_type'] == 'video':
            return stream['height']

    return None

def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model, False)

    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    vae.enable_slicing()

    handle_memory_attention(xformers, sdp, unet)

    return pipe

def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    init_video: Optional[str],
    vae_batch_size: int,
):
    if init_video is None:
        scale = pipe.vae_scale_factor
        shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
        latents = torch.randn(shape, dtype=torch.half)

    else:
        latents = encode(pipe, init_video, vae_batch_size)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)

    return latents


def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents

def decode(pipe: TextToVideoSDPipeline, latents: Tensor, batch_size: int = 8):
    nf = latents.shape[2]
    latents = rearrange(latents, "b c f h w -> (b f) c h w")

    pixels = []
    for idx in trange(
        0, latents.shape[0], batch_size, desc="Decoding to pixels...", unit_scale=batch_size, unit="frame"
    ):
        latents_batch = latents[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = latents_batch.div(pipe.vae.config.scaling_factor)
        pixels_batch = pipe.vae.decode(latents_batch).sample.cpu()
        pixels.append(pixels_batch)
    pixels = torch.cat(pixels)

    pixels = rearrange(pixels, "(b f) c h w -> b c f h w", f=nf)

    return pixels.float()

def primes_up_to(n):
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

@torch.inference_mode()
def diffuse(
    pipe: TextToVideoSDPipeline,
    prompt: Optional[List[str]],
    negative_prompt: Optional[List[str]],
    prompt_embeds: Optional[List[Tensor]],
    negative_prompt_embeds: Optional[List[Tensor]],
    num_inference_steps: int,
    guidance_scale: float,
    encode_to_latent: bool,
    num_frames: int,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    init_image: torch.Tensor = None,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    eta: float = 0.0
):
    device = pipe.device
    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = pipe._encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    # set the scheduler to start at the correct timestep
    # 4. Scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    # 5. Prepare latent variables
    init_latents = init_image.unsqueeze(2).expand(-1, -1, num_frames, -1, -1)
    init_latents = encode(pipe, init_latents, 1) if encode_to_latent else init_latents
    init_latents = init_latents.to(device)

    noisy_latents = torch.randn_like(init_latents)

    noisy_latents = noisy_latents.to(device)
    init_latents = init_latents.to(device)

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            noisy_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
            init_model_input = torch.cat([init_latents] * 2) if do_classifier_free_guidance else init_latents

            noisy_model_input = pipe.scheduler.scale_model_input(noisy_model_input, t)
            init_model_input = pipe.scheduler.scale_model_input(init_model_input, t)

            # predict the noise residual
            noise_pred = pipe.unet(
                noisy_model_input,
                init_model_input[:, :, 0, :, :],
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # reshape latents
            bsz, channel, frames, width, height = noisy_latents.shape
            noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

            # compute the previous noisy sample x_t -> x_t-1
            noisy_latents = pipe.scheduler.step(noise_pred, t, noisy_latents, **extra_step_kwargs).prev_sample

            # reshape latents back
            noisy_latents = noisy_latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, noisy_latents)
    
    return noisy_latents


@torch.inference_mode()
def inference(
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    image_width: int = None,
    image_height: int = None,
    model_2d: str = None,
    num_frames: int = 24,
    vae_batch_size: int = 8,
    num_steps: int = 50,
    guidance_scale: float = 15,
    image_guidance_scale: float = 7.5,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    times: int = 1,
    seed: Optional[int] = None,
    init_image: torch.Tensor = None,
    save_init: bool = False,
    upscale: bool = False
):
    if seed is not None:
        set_seed(seed)

    if init_image is None:
        stable_diffusion_pipe = DiffusionPipeline.from_pretrained(model_2d, torch_dtype=torch.float16).to(device)
        init_image = stable_diffusion_pipe(prompt=prompt, negative_prompt=negative_prompt, width=image_width, height=image_height, guidance_scale=image_guidance_scale, output_type="pt").images[0]
        if (save_init):
            unique_id = str(uuid4())[:8]
            save_image(init_image, f"output/{prompt}-{unique_id}.png")
        init_image = init_image.unsqueeze(0)
        init_image = F.interpolate(init_image, size=(width, height), mode='bilinear', align_corners=False)
        del stable_diffusion_pipe
        torch.cuda.empty_cache()

    with torch.autocast(device, dtype=torch.half):
        # prepare models
        pipe = initialize_pipeline(model, device, xformers, sdp)

        # prepare prompts
        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        prompt_embeds, negative_prompt_embeds = compel(prompt), compel(negative_prompt) if negative_prompt else None

        generator = torch.Generator().manual_seed(seed)
        
        video_latents = []
        # run diffusion
        for t in range(0, times):
            latents = diffuse(
                pipe=pipe,
                init_image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                encode_to_latent=True if t == 0 else False,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                generator=generator
            )

            # Smooth transitions. Suggested by bfasenfest
            # if t != 0:  # not the first iteration
                # average last three frames of previous latent and first three frames of current latent
                # video_latents[-1][:, :, -1:, :, :] = (video_latents[-1][:, :, -1:, :, :] + latents[:, :, :1, :, :]) / 2

            video_latents.append(latents[:, :, 1:, :, :])
            init_image = latents[:, :, -1, :, :]

        # Upscale the latents
        if upscale:
            upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16)
            upscaler.to(device)
            
            concat_videos = torch.cat(video_latents, dim=0)
            batch, channel, frames, W, H = concat_videos.shape

            # Reshape to put frames into the batch dimension
            reshaped_videos = concat_videos.permute(2, 0, 1, 3, 4).reshape(-1, channel, H, W)  # Shape: (B*F, C, H, W)

            upscaled_reshaped_videos = []

            for i in range(0, reshaped_videos.shape[0], (num_frames - 1)):
                reshaped_frames = reshaped_videos[i:i+(num_frames - 1)]
                prompt_repeated = [prompt] * len(reshaped_frames)
                upscaled_batch_frames = upscaler(
                    prompt=prompt_repeated,
                    image=reshaped_frames,
                    num_inference_steps=20,
                    guidance_scale=0,
                    generator=generator,
                    output_type="latent"
                ).images
                upscaled_reshaped_videos.append(upscaled_batch_frames)

            upscaled_reshaped_videos = torch.cat(upscaled_reshaped_videos, dim=0)

            B_times_F, C_prime, H_prime, W_prime = upscaled_reshaped_videos.shape

            # Reshape back to the original structure
            video_latents = upscaled_reshaped_videos.reshape(frames, batch, C_prime, H_prime, W_prime).permute(1, 2, 0, 3, 4)  # Shape: (B, C', F, H', W')
        else:
            video_latents = torch.cat(video_latents, dim=0)

        # decode latents to pixel space
        videos = decode(pipe, video_latents, vae_batch_size)

    return torch.cat(torch.unbind(videos, dim=0), dim=1)

if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="HuggingFace repository or path to model checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Text prompt to condition on")
    parser.add_argument("-o", "--output-dir", type=str, default="./output", help="Directory to save output video to")
    parser.add_argument("-n", "--negative-prompt", type=str, default=None, help="Text prompt to condition against")
    parser.add_argument("-T", "--num-frames", type=int, default=16, help="Total number of frames to generate")
    parser.add_argument("-WI", "--width", type=int, default=512, help="Width of the video to generate (if init image is not provided)")
    parser.add_argument("-HI", "--height", type=int, default=512, help="Height of the video (if init image is not provided)")
    parser.add_argument("-IW", "--image-width", type=int, default=None, help="Width of the image to generate (if init image is not provided)")
    parser.add_argument("-IH", "--image-height", type=int, default=None, help="Height of the image (if init image is not provided)")
    parser.add_argument("-MP", "--model-2d", type=str, default="stabilityai/stable-diffusion-2-1", help="Path to the model for image generation (if init image is not provided)")
    parser.add_argument("-i", "--init-image", type=str, default=None, help="Path to initial image to use")
    parser.add_argument("-VB", "--vae-batch-size", type=int, default=8, help="Batch size for VAE encoding/decoding to/from latents (higher values = faster inference, but more memory usage).")
    parser.add_argument("-s", "--num-steps", type=int, default=25, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-g", "--guidance-scale", type=float, default=14, help="Scale for guidance loss (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-IG", "--image-guidance-scale", type=float, default=7.5, help="Scale for guidance loss for 2d model (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-f", "--fps", type=int, default=12, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed to make generations reproducible.")
    parser.add_argument("-t", "--times", type=int, default=None, help="How many times to continue to generate videos")
    parser.add_argument("-I", "--save-init", action="store_true", help="Save the init image to the output folder for reference")
    parser.add_argument("-N", "--include-model", action="store_true", help="Include the name of the model in the exported file")
    parser.add_argument("-u", "--upscale", action="store_true", help="Use a latent upscaler")

    args = parser.parse_args()
    # fmt: on

    # =========================================
    # ============= sample videos =============
    # =========================================
    init_image = None
    if args.init_image is not None:
        init_image = image_to_tensor(args.init_image)

    image_width = None
    if args.image_width is None:
        image_width = args.width

    image_height = None
    if args.image_height is None:
        image_height = args.height

    videos = inference(
        model=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        image_width=image_width,
        image_height=image_height,
        model_2d=args.model_2d,
        num_frames=args.num_frames,
        init_image=init_image,
        vae_batch_size=args.vae_batch_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        device=args.device,
        seed=args.seed,
        xformers=args.xformers,
        sdp=args.sdp,
        times=args.times,
        save_init=args.save_init,
        upscale=args.upscale
    )

    # =========================================
    # ========= write outputs to file =========
    # =========================================
    os.makedirs(args.output_dir, exist_ok=True)

    for video in [videos]:
        video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)
        video = video.byte().cpu().numpy()
        
        unique_id = str(uuid4())[:8]
        out_file = f"{args.output_dir}/{args.prompt}-{unique_id}.mp4"
        model_name = ""
        if args.include_model:
            model_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "-", os.path.basename(args.model))
        encoded_out_file = f"{args.output_dir}/{args.prompt}-{model_name}-{unique_id}_encoded.mp4"

        export_to_video(video, out_file, args.fps)

        try:
            encode_video(out_file, encoded_out_file, get_video_height(out_file))
            os.remove(out_file)
        except:
            pass
