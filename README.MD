# InputMix
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the PyTorch implementation of InputMix. Please refer [inputmix.py](./inputmix.py)

# Quick start

## Data preperation
Download CompCars dataset from
http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

# Train

```
python main.py --output_dir exp \
                --data_path datapath/compcars/image \
                --fusion_layer 11 \
                --fusion_mix_p 0.7 \
                --fusion_mix_lam 0.5 0.5 \
                --num_inputs 2 \
                --view_ids 2 3"
```
Arguments Explanation:
- `fusion_layer`: where start to fuse, range [0, 12] for Vit-small
- `fusion_mix_p`: The InputMix hyper-parameter p
- `fusion_mix_lam`: The InputMix hyper-parameter lam
- `num_inputs`: The number of inputs of the model
- `input_ids`: The input ids.

