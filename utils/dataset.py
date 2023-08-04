import os
import decord
import numpy as np
import random
import torchvision.transforms as T
import cv2
import concurrent.futures

from tqdm import tqdm
from .bucketing import sensible_buckets

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange

def process_file(args):
    file, root, existing_files = args
    if file.endswith('.mp4'):
        if file.replace(".mp4", "").replace("_", " ") not in existing_files:
            full_file_path = os.path.join(root, file)
            return full_file_path
    return None

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids


def center_crop(frame, crop_size):
    h, w, _ = frame.shape
    start_x = w//2-(crop_size//2)
    start_y = h//2-(crop_size//2)   

    return frame[start_y:start_y+crop_size, start_x:start_x+crop_size, :]

def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices

def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)
    else:
        vr = decord.VideoReader(vid_path, width=w, height=h)
        video = get_frame_batch(vr, crop=True, w=w)

    return video, vr

class VideoFolderDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        frame_step: int = 4,
        path: str = "./data",
        fallback_prompt: str = "",
        use_bucketing: bool = False,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing

        self.fallback_prompt = fallback_prompt

        self.video_files = []
        self.find_videos(path)
        
        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.frame_step = frame_step
    
    def find_videos(self, path):
        digested = set()
        if os.path.isfile("videos.txt"):
            with open("videos.txt", 'r') as f:
                digested = set(f.read().splitlines())

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    jobs.append(executor.submit(process_file, (file, root, digested)))
            
            for future in tqdm(concurrent.futures.as_completed(jobs), total=len(jobs)):
                result = future.result()
                if result is not None:
                    self.video_files.append(result)

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, w=None, crop=None, resize=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()

        every_nth_frame = max(1, round(self.frame_step * native_fps / 30))

        if len(vr) < n_sample_frames * every_nth_frame:
            return None, None

        effective_length = len(vr) // every_nth_frame
        effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)

        video = vr.get_batch(idxs)

        if video.shape[-1] == 4:
            video = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for frame in video])

        video = rearrange(video, "f h w c -> f c h w")

        if crop is not None:
            video = np.stack([center_crop(frame, w) for frame in video])

        if resize is not None: 
            video = resize(video)
                
        return video, vr
        
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
            vid_path,
            self.use_bucketing,
            self.width, 
            self.height, 
            self.get_frame_buckets, 
            self.get_frame_batch
        )
        return video, vr
    
    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        try:
            video, _ = self.process_video_wrapper(self.video_files[index])
        except:
            return self.__getitem__((index + 1) % len(self))
        
        if video is None or (video and video[0] is None):
            return self.__getitem__((index + 1) % len(self))

        basename = os.path.basename(self.video_files[index]).replace('.mp4', '').replace('_', ' ')
        split_basename = basename.split('-')

        if len(split_basename) > 1:
            prompt = '-'.join(split_basename[:-1])
        else:
            prompt = split_basename[0]

        if not prompt:
            prompt = self.fallback_prompt

        prompt_ids = self.get_prompt_ids(prompt)

        return {"pixel_values": (video[0] / 127.5 - 1.0), "prompt_ids": prompt_ids[0], "text_prompt": prompt, "file": basename, 'dataset': self.__getname__()}
