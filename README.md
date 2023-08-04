# Modelscope text2video transformed to text + img2video
## Finetune the modelscope text to video model to support image conditioning and create infinite length videos

This project involves fine tuning the modelscope damo text to video to support image conditioning

## Examples

![2090-a bouquet of yellow crocuses in a vase 517](https://github.com/motexture/modelscope-img2video/assets/64956941/fe906d55-8f68-4d26-ba11-0e13ee999130)

https://github.com/motexture/modelscope-img2video/assets/64956941/ebb593d9-64f1-460b-9969-bd6b296580f6

![2150-a dolphin swims under the water in a large tank 271](https://github.com/motexture/modelscope-img2video/assets/64956941/1bd66f15-8a2e-4ca4-b599-aa03498c0d5a)

https://github.com/motexture/modelscope-img2video/assets/64956941/9daa50b4-8740-4191-a765-51545db49979
![2200-three white crocuses in the grass with an orange center 143](https://github.com/motexture/modelscope-img2video/assets/64956941/2e06ab0e-9a1b-4735-9253-687e7b51b7c5)

https://github.com/motexture/modelscope-img2video/assets/64956941/415d81b6-8baa-4684-b280-7c0c3e9c1c38


## Getting Started

### Installation
```bash
git clone https://github.com/motexture/modelscope-img2video
cd modelscope-img2video
```

### Python Requirements

```bash
pip install deepspeed
pip install -r requirements.txt
```

On some systems, deepspeed requires installing the CUDA toolkit first in order to properly install. If you do not have CUDA toolkit, or deepspeed shows an error follow the instructions by NVIDIA: https://developer.nvidia.com/cuda-downloads

or on linux systems:
```bash
sudo apt install build-essential
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```

During the installation you only need to install toolkit, not the drivers or documentation.

## Preparing the config file
Open the training.yaml file and modify the parameters according to your needs  <br /> 
If its the first time you are training you want to leave upgrade_model to True  <br /> 
The pretrained_3d_model variable should be your old modelscope text to video model or your new model path if you set upgrade_model to False  <br /> 
The pretrained_2d_model can be any diffusion text2img model (used during validation)

## Train
```python
deepspeed train.py --config training.yaml
```
---

## Running inference
The `inference.py` script can be used to render videos with trained checkpoints.

Example usage, where times is how many times you want to continue video generation using the newly generated last frame as the new image conditioner (potentially infinite length videos)
```
python inference.py \
  --model img2vid \
  --prompt "a fast moving fancy sports car" \
  --init-image "car.png" \
  --num-frames 16 \
  --width 512 \
  --height 512 \
  --times 4 \
  --sdp
```

Or if you prefer to use a 2d text to video model for image conditioning instead of a init-image:
```
python inference.py \
  --model img2vid \
  --prompt "a fast moving fancy sports car" \
  --model-2d "stabilityai/stable-diffusion-2-1" \
  --num-frames 16 \
  --width 512 \
  --height 512 \
  --times 4 \
  --sdp
```
## Shoutouts

- [ExponentialML](https://github.com/ExponentialML/Text-To-Video-Finetuning/) for the original training and inference code
- [Showlab](https://github.com/showlab/Tune-A-Video) and [bryandlee](https://github.com/bryandlee/Tune-A-Video) for their Tune-A-Video contribution
