# InputMix
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the PyTorch implementation of InputMix. Please refer [inputmix.py](./inputmix.py)

# Quick start

## Data preparation
Home page of CompCars http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
Download CompCars dataset following the instruction
http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt
Unzip data.zip, and find the image folder.

# Train

```
python main.py --output_dir exp \
                --data_path your_data_path/compcars/image \
                --fusion_layer 11 \
                --fusion_mix_p 0.7 \
                --fusion_mix_lam 0.5 0.5 \
                --num_inputs 2 \
                --input_ids 2 3
```
Arguments Explanation:
- `fusion_layer`: where start to fuse, range [0, 12] for Vit-small
- `fusion_mix_p`: The InputMix hyper-parameter p. Set fusion_mix_p=0 to train without InputMix
- `fusion_mix_lam`: The InputMix hyper-parameter lam
- `num_inputs`: The number of inputs of the model
- `input_ids`: The input ids, range [0, 4] for CompCar dataset; 0: front view, 1: rear, 2: side view, 3: front-side view, 4: rear-side view,

