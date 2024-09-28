# Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance (NeurIPS 2024)

<a href="https://arxiv.org/abs/2406.07540"><img src="https://img.shields.io/badge/arXiv-Paper-red"></a> 
<a href="https://genforce.github.io/ctrl-x"><img src="https://img.shields.io/badge/Project-Page-yellow"></a>
[![GitHub](https://img.shields.io/github/stars/genforce/ctrl-x?style=social)](https://github.com/genforce/ctrl-x)

[Kuan Heng Lin](https://kuanhenglin.github.io)<sup>1*</sup>, [Sicheng Mo](https://sichengmo.github.io/)<sup>1*</sup>, [Ben Klingher](https://bklingher.github.io)<sup>1</sup>, [Fangzhou Mu](https://pages.cs.wisc.edu/~fmu/)<sup>2</sup>, [Bolei Zhou](https://boleizhou.github.io/)<sup>1</sup> <br>
<sup>1</sup>UCLA&emsp;<sup>2</sup>NVIDIA <br>
<sup>*</sup>Equal contribution <br>

![Ctrl-X teaser figure](docs/assets/teaser_github.jpg)

## Getting started

### Environment setup

Our code is built on top of [`diffusers v0.28.0`](https://github.com/huggingface/diffusers). To set up the environment, please run the following.
```
conda env create -f environment.yaml
conda activate ctrlx
```

### Running Ctrl-X

#### Gradio demo

We provide a user interface for testing our method. Running the following command starts the demo.
```bash
python app_ctrlx.py
```

#### Script

We also provide a script for running our method. This is equivalent to the Gradio demo.
```bash
python run_ctrlx.py \
    --structure_image assets/images/horse__point_cloud.jpg \
    --appearance_image assets/images/horse.jpg \
    --prompt "a photo of a horse standing on grass" \
    --structure_prompt "a 3D point cloud of a horse"
```
If `appearance_image` is not provided, then Ctrl-X does *structure-only* control. If `structure_image` is not provided, then Ctrl-X does *appearance-only* control.

#### Optional arguments

There are three optional arguments for both `app_ctrlx.py` and `run_ctrlx.py`:
- `cpu_offload` (flag): If enabled, offloads each component of both the base model and refiner to CPU when not in use, reducing memory usage while slightly increasing runtime (runtime increases by ~30%).
    - To use `cpu_offload`, [`accelerate`](https://github.com/huggingface/accelerate) must be installed. This must be done manually with `pip install accelerate` as `environment.yaml` does *not* have `accelerate` listed.
- `disable_refiner` (flag): If enabled, disables the refiner (and does not load it), reducing memory usage.
- `model` (`str`): When provided a `safetensor` checkpoint path, loads the checkpoint for the base model.

Approximate GPU VRAM usage for the Gradio demo and script is as follows.
| No flags | `cpu_offload` | `disable_refiner` | `cpu_offload` + `disable_refiner` |
| -------- | ------------- | ----------------- | --------------------------------- |
| 19GiB    | 13GiB         | 15GiB             | 8GiB                              |

Have fun playing around! :D

## Contact

For any questions, thoughts, discussions, and any other things you want to reach out for, please contact [Jordan Lin](https://kuanhenglin.github.io) (kuanhenglin@ucla.edu).

## Reference

If you use our code in your research, please cite the following work.

```bibtex
@inproceedings{lin2024ctrlx,
    author = {Lin, {Kuan Heng} and Mo, Sicheng and Klingher, Ben and Mu, Fangzhou and Zhou, Bolei},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance},
    year = {2024}
}
```
