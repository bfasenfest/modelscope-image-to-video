# Adapted from https://github.com/ExponentialML/Text-To-Video-Finetuning/blob/main/train.py

import argparse
import datetime
import inspect
import math
import os
import gc
import shutil
import deepspeed
import json
import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
import subprocess

from PIL import Image
from typing import Dict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from einops import rearrange
from utils.dataset import VideoFolderDataset
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
from models.unet import UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, DiffusionPipeline
from pipeline.pipeline import TextToVideoSDPipeline
from diffusers.utils import export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0

def save_image(tensor, filename):
    tensor = tensor.cpu().numpy()  # Move to CPU
    tensor = tensor.transpose((1, 2, 0))  # Swap tensor dimensions to HWC
    tensor = (tensor * 255).astype('uint8')  # Denormalize
    img = Image.fromarray(tensor)  # Convert to a PIL image
    img.save(filename)  # Save image

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            for m in module:
                if isinstance(m, BasicTransformerBlock):
                    set_processors([m.attn1, m.attn2])
                    optim_count += 1

    print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def read_deepspeed_config_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def handle_temporal_params(model, is_enabled=True):
    unfrozen_params = 0

    for name, module in model.named_modules():
        if 'conditioning_in' in name or 'conditioning_mid' in name or 'conditioning_out' in name or 'temp_convs' in name or 'temp_attentions' in name:
            for m in module.parameters():
                m.requires_grad_(is_enabled)
                if is_enabled: unfrozen_params +=1

    if unfrozen_params > 0:
        print(f"{unfrozen_params} params have been unfrozen for training.")

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

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)

    return out_dir

def load_primary_models(pretrained_model_path, upgrade_model):
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    if upgrade_model:
        unet = UNet3DConditionModel.from_pretrained_3d(pretrained_model_path, subfolder="unet")
    else:
        unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    print(f'The model has {count_parameters(unet):,} trainable parameters')

    return noise_scheduler, tokenizer, text_encoder, vae, unet

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")

    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

def sample_noise(latents, noise_strength, use_offset_noise):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 1) and validation_data.sample_preview

def save_pipe(
        path, 
        global_step,
        unet, 
        vae, 
        text_encoder, 
        tokenizer,
        scheduler,
        output_dir,
        is_checkpoint=False
    ):

    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)

        existing_checkpoints = [d for d in os.listdir(output_dir) if 'checkpoint-' in d]
        existing_checkpoints = sorted(existing_checkpoints, key=lambda d: os.path.getmtime(os.path.join(output_dir, d)))

        while len(existing_checkpoints) > 2:
            shutil.rmtree(os.path.join(output_dir, existing_checkpoints.pop(0)))
    else:
        save_path = output_dir

    pipeline = TextToVideoSDPipeline.from_pretrained(path, 
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler)
    
    pipeline.save_pretrained(save_path)

    print(f"Saved model at {save_path} on step {global_step}")
    
    del pipeline

    torch.cuda.empty_cache()
    gc.collect()

def main(
    pretrained_3d_model_path: str,
    pretrained_2d_model_path: str,
    upgrade_model: bool,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    epochs: int = 1,
    validation_steps: int = 100,
    checkpointing_steps: int = 500,
    seed: int = 42,
    gradient_checkpointing: bool = False,
    use_offset_noise: bool = False,
    enable_xformers_memory_efficient_attention: bool = False,
    enable_torch_2_attn: bool = True,
    offset_noise_strength: float = 0.1,
    **kwargs
):
    dist.init_process_group(backend='nccl')

    *_, config = inspect.getargvalues(inspect.currentframe())

    if dist.get_rank() == 0:
        output_dir = create_output_folders(output_dir, config)

    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_3d_model_path, upgrade_model)

    data = read_deepspeed_config_file(train_data.deepspeed_config_file)

    unet_engine, _, _, _ = deepspeed.initialize(
        model=unet,
        model_parameters=unet.parameters(),
        config=train_data.deepspeed_config_file,
    )

    pipe = DiffusionPipeline.from_pretrained(pretrained_2d_model_path, torch_dtype=torch.float16)
    
    text_encoder.to(unet_engine.device)
    vae.to(unet_engine.device)
    unet.to(unet_engine.device)
    pipe = pipe.to(unet_engine.device)

    freeze_models([text_encoder, vae, unet])

    vae.enable_slicing()

    train_dataset = VideoFolderDataset(**train_data, tokenizer=tokenizer, device=unet_engine.device)

    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data['train_micro_batch_size_per_gpu'],
        sampler=train_sampler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    global_step = 0

    unet.train()

    if gradient_checkpointing:
        unet._set_gradient_checkpointing(value=True)

    handle_temporal_params(unet, is_enabled=True)

    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)
    
    progress_bar = tqdm(range(global_step, num_update_steps_per_epoch * epochs))
    progress_bar.set_description("Steps")

    def finetune_unet(batch):
        pixel_values = batch["pixel_values"]
        pixel_values = pixel_values.to(unet_engine.device)

        latents = tensor_to_vae_latent(pixel_values, vae)
        init_image = latents[:, :, 0, :, :]

        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]

        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        token_ids = batch['prompt_ids'].to(unet_engine.device)
        encoder_hidden_states = text_encoder(token_ids)[0].detach()  # Detach to avoid training the text encoder
        
        if noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
        
        model_pred = unet(noisy_latents, init_image, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    for _ in range(0, epochs):
        for batch in train_dataloader:
            with autocast():
                loss = finetune_unet(batch)

            unet_engine.backward(loss)
            unet_engine.step()

            '''
            # To avoid using same videos during training
            try:
                with open("videos.txt", 'a') as file:
                    file.write("\n" + "\n".join(batch['file']))
            except:
                pass
            '''

            if dist.get_rank() == 0:
                progress_bar.update(1)
                global_step += 1

                if global_step % checkpointing_steps == 0:
                    save_pipe(
                        pretrained_3d_model_path,
                        global_step,
                        unet,
                        vae,
                        text_encoder,
                        tokenizer,
                        noise_scheduler,
                        output_dir,
                        is_checkpoint=True
                    )

                if should_sample(global_step, validation_steps, validation_data):
                    with autocast():
                        if gradient_checkpointing:
                            unet._set_gradient_checkpointing(value=False)
                        unet.eval()

                        pipeline = TextToVideoSDPipeline.from_pretrained(
                            pretrained_3d_model_path,
                            text_encoder=text_encoder,
                            vae=vae,
                            unet=unet
                        )
                        
                        diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                        pipeline.scheduler = diffusion_scheduler

                        prompt = batch["text_prompt"][0] if len(validation_data.prompt) <= 0 else validation_data.prompt
                        init_image = pipe(prompt, width=validation_data.width, height=validation_data.height, output_type="pt").images[0]

                        save_filename = f"{global_step}-{prompt}"
                        out_file = f"{output_dir}/samples/{save_filename}.mp4"
                        img_file = f"{output_dir}/samples/{save_filename}.png"
                        encoded_out_file = f"{output_dir}/samples/{save_filename}_encoded.mp4"

                        save_image(init_image, img_file)

                        with torch.no_grad():
                            video_frames = pipeline(
                                prompt,
                                width=validation_data.width,
                                height=validation_data.height,
                                init_image=init_image,
                                num_frames=validation_data.num_frames,
                                num_inference_steps=validation_data.num_inference_steps,
                                guidance_scale=validation_data.guidance_scale
                            ).frames

                        export_to_video(video_frames, out_file, validation_data.fps)

                        try:
                            encode_video(out_file, encoded_out_file, get_video_height(out_file))
                            os.remove(out_file)
                        except:
                            pass
                            
                        del pipeline, video_frames
                        torch.cuda.empty_cache()
                        
                        if gradient_checkpointing:
                            unet._set_gradient_checkpointing(value=True)
                        unet.train()

    if dist.get_rank() == 0:
        save_pipe(
            pretrained_3d_model_path,
            global_step,
            unet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            output_dir,
            is_checkpoint=False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training.yaml")
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank of this process. Used for distributed training.')
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
