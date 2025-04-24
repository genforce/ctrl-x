# Ctrl-X quantitative evaluation dataset

We publicly release the dataset for quantitative evaluation as described in our [paper](https://arxiv.org/abs/2406.07540). The dataset consists of 177 1024&times;1024 images divided into 16 types and across 7 categories, including both in-the-wild (web) and generated images.

Sources for all non-generated images can be found [here](https://drive.google.com/file/d/10yQqMz0_o3EvouGM9TC_nDVtLQwz08vn/view?usp=sharing). If you are the author of one of these images and would like it removed, or if we missed crediting your work in the linked PDF, please contact **Jordan Lin** at kuanhenglin@ucla.edu.

## Download

To download our dataset, go to [here](https://drive.google.com/file/d/1FzaaLcVK9NBWehdq9eAWnIMtOZkkMz5M/view?usp=sharing), download the file, and move `ctrl-x_data.tar.gz` into the currect folder `data` and inflate with
```
tar -xzvf ctrl-x_data.tar.gz
```
You should see the folder `images` appear.

## Overview

The structure of our image dataset is as follows:

```
images
  animals
    a_black_cat_in_the_rain-hed.jpg
    a_lion_in_the_savanna-painting.jpg
    ...
  buildings
    a_building_in_a_city-normal-map.jpg
    a_church-hed.jpg
    ...
  humans
    ...
  objects
    ...
  rooms
    ...
  scenes
    ...
  vehicles
    ...
  data_prompt-driven.yaml
  data_structure+appearance.yaml
```

The `image` folder is split into 7 folders corresponding to the 7 categories. This is mostly for organization purposes as we do not distinguish between these categories during quantitative evaluation. Moreover, the `image` folder also contains two `yaml` files:

- `data_prompt-driven.yaml`, which contains the file names and prompts for our prompt-driven conditional generation evaluation, and
- `data_structure+appearance.yaml`, which contains the file names and prompts for our structure and appearance control evaluation.

### Structure and appearance control

Pairs, prompts, and other metadata about each of the examples for structure and appearance control quantitative evaluation can be found in `data_structure+appearance.yaml`, which looks like the following:

```yaml
pairs:
- id: vehicles__car-mountain-sam___vehicles__truck.jpg
  f1_data: &id036
    filename: vehicles/car-mountain-sam.jpg
    category: vehicles
    description: a segmentation mask of an SUV in front of a mountain
    type: segmentation mask
    subject: an SUV
    context: in front of a mountain
  f2_data: &id001
    filename: vehicles/truck.jpg
    category: vehicles
    description: a photo of a red pickup truck driving off-road
    type: photo
    subject: a red pickup truck
    context: driving off-road
  target_prompt: a photo of a red pickup truck in front of a mountain
- ...
...
```

For each pair (element in `pairs`), the meaning of each field is as follows:

- `id`: Unique identifier for each pair, can be used as the file name for saving quantitative evaluation results
- `f1_data`: Metadata about the structure image
- `f2_data`: Metadata about the appearance image
- `target_prompt`: Prompt used during generation; in our paper, this is *both* the appearance prompt and the output prompt

Note the initial `f1_data` and `f2_data` entries have anchors (e.g., `&id036`) which follow them. These are `yaml` anchors so later pairs which use the same images as prior pairs can reference the anchors directly to reduce redundancy (e.g., `*id036`).

`f1_data` and `f2_data` have the same format, and the meaning of each field within them is as follows:

- `filename`: Path to the image relative to folder `images`
- `category`: One of the 7 categories mentioned above
- `description`: Prompt of the image, which is of the format `a [type] of [subject] [context]`, see below for `type`, `subject`, and `context`
- `type`: One of the 16 types mentioned above, for distinguishing condition images (ControlNet-supported conditions and in-the-wild conditions) and natural images
- `subject`: The subject of the image
- `context`: The "context" of the image, usually describing the state (e.g., action) and/or location of the subject

For each pair, `target_prompt` is *always* generated with the format `a [type from f2_data] of [subject from f2_data] [context from f1_data]`.

### Prompt-driven conditional generation

Prompts and other metadata about each of the examples for prompt-driven conditional generation quantitative evaluation can be found in `data_prompt-driven.yaml`, which looks like the following:

```yaml
images:
- filename: humans/3d-humanoid-knight.jpg
  category: humans
  description: a 3d humanoid of a knight with sword and shield
  type: 3d humanoid
  subject: a knight
  context: with sword and shield
  prompts:
  - prompt: a painting of a chinese emperor with sword and shield
    subject: a chinese emperor
    id: humans_3d-humanoid-knight___a_painting_of_a_chinese_emperor_with_sword_and_shield
  - prompt: a photo of a sci-fi mecha warrior with a sword and shield
    subject: a sci-fi mecha warrior
    id: humans_3d-humanoid-knight___a_photo_of_a_sci-fi_mecha_warrior_with_a_sword_and_shield
  - prompt: a photo of a roman gladiator with a sword and shield
    subject: a roman gladiator
    id: humans_3d-humanoid-knight___a_photo_of_a_roman_gladiator_with_a_sword_and_shield
- filename: ...
  ...
```

For each image, the meaning/purpose of each field is the same as the fields in `f1_data`/`f2_data` in `data_structure+appearance.yaml` as described above. However, we have one extra field `prompts` here, which includes:

- `prompt`: Prompt for conditional generation, has the format `a [type different from current one] of [subject different from current one] [context]`, `type` and `subject` different from current one are written by the authors
- `subject`: Subject of the generation prompt, a.k.a. `subject different from current one`
- `id`: Unique identifier for each prompt-image pair, can be used as the file name for saving quantitative evaluation results

### Conditional vs. natural images

As mentioned above, the 17 types are used to categorize the images into conditional images and natural images, where conditional images are further categorized into ControlNet-supported conditions and in-the-wild conditions. The `type` field included in each are as follows:

- Conditional images (67 images): `["canny edge map", "metadrive", "3d mesh", "3d humanoid", "depth map", "human pose image", "point cloud", "sketch", "line drawing", "HED edge drawing", "normal map", "segmentation mask"]`
  - ControlNet-supported conditions: `["canny edge map", "depth map", "human pose image", "line drawing", "HED edge drawing", "normal map", and "segmentation mask"]`
  - In-the-wild conditions: `["metadrive", "3D mesh", "3D humanoid", "point cloud", and "sketch"]`
- Natural images (110 images): `["photo", "painting", "cartoon", "birds eye view"]`

For structure and appearance control quantitative evaluation, both conditional and natural images can be structure images, but only natural images can be appearance images.

## Quantitative evaluation

Code for quantitative evaluation is coming soon, but details of the evaluation can be found in our [paper](https://arxiv.org/abs/2406.07540) in the mean time.

## Contact

For any questions, thoughts, and discussions about the dataset, please contact [Jordan Lin](https://kuanhenglin.github.io) (kuanhenglin@ucla.edu).