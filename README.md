# Modelscope text to video synthesis model transformed to be text + image to video
## Training and inference scripts for creating videos from a static image

This project contains the code for transforming the modelscope text to video synthesis model to a text + image to video model and the code for generating videos with the new model.

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
  --model motexture/image-to-video-ms-3.4b \
  --prompt "an astronaut is walking on the moon" \
  --init-image "image.png" \
  --num-frames 16 \
  --width 512 \
  --height 512 \
  --times 4 \
  --sdp
```

Or if you prefer to use a 2d text to image model for image conditioning instead of a init-image:
```
python inference.py \
  --model motexture/image-to-video-ms-3.4b \
  --prompt "an astronaut is walking on the moon" \
  --model-2d "stabilityai/stable-diffusion-2-1" \
  --num-frames 16 \
  --width 512 \
  --height 512 \
  --times 4 \
  --sdp
```
## Shoutouts

- [ExponentialML](https://github.com/ExponentialML/Text-To-Video-Finetuning/) for the original training and inference code
- [bfasenfest](https://github.com/bfasenfest) for his contribution to the training and testing phases
- [Showlab](https://github.com/showlab/Tune-A-Video) and [bryandlee](https://github.com/bryandlee/Tune-A-Video) for their Tune-A-Video contribution
