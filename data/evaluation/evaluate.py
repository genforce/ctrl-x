from argparse import ArgumentParser
from os import listdir, path

from PIL import Image
from tqdm.auto import tqdm
import yaml

from .score import LossG


# Categorization of Ctrl-X evaluation dataset
GROUPED_STRUCTAPP = {
    "natural": ["photo", "cartoon", "painting", "birds eye view"],
    "controlnet": ["segmentation mask", "canny edge map", "depth map", "normal map",
                   "line drawing", "HED edge drawing", "human pose image"],
    "new": ["birds eye view metadrive", "3d mesh", "3d humanoid", "point cloud", "sketch"],
}
GROUPED_PROMPTDRIVEN = {
    "controlnet": ["segmentation mask", "canny edge map", "depth map", "normal map",
                   "line drawing", "HED edge drawing", "human pose image"],
    "new": ["birds eye view metadrive", "3d mesh", "3d humanoid", "point cloud", "sketch"],
}

# Change the following if necessary (the .yaml files are necessarily to get the type group of each result)
with open("./data/images/data_structure+appearance.yaml") as file:
    YAML_STRUCTAPP = yaml.safe_load(file)["pairs"]
with open("./data/images/data_prompt-driven.yaml") as file:
    YAML_PROMPTDRIVEN = yaml.safe_load(file)["images"]


def get_prompt_group_structapp(id):
    id_ = id + ".jpg"
    category = None
    prompt = None
    for pair in YAML_STRUCTAPP:
        if pair["id"] == id_:
            category = pair["f1_data"]["type"]
            prompt = pair["target_prompt"]
            break
    if category is None:
        raise ValueError(f"ID {id} not in yaml")
    for group in GROUPED_STRUCTAPP:
        if category in GROUPED_STRUCTAPP[group]:
            return prompt, group
    raise ValueError(f"ID {id} with group {group} not in GROUPED_STRUCTAPP")


def get_prompt_group_promptdriven(id):
    category = None
    prompt = None
    for image in YAML_PROMPTDRIVEN:
        for prompt_info in image["prompts"]:
            if prompt_info["id"] == id:
                category = image["type"]
                prompt = prompt_info["prompt"]
                break
    if category is None:
        raise ValueError(f"ID {id} not in yaml")
    for group in GROUPED_PROMPTDRIVEN:
        if category in GROUPED_PROMPTDRIVEN[group]:
            return prompt, group
    raise ValueError(f"ID {id} with group {group} not in GROUPED_PROMPTDRIVEN")


def evaluate_structapp(args):
    lossg = LossG(cfg=None)

    file_prefix = "result_refiner" if args.refiner else "result"
    filename = file_prefix + "__" + str(args.results_seed) + ".jpg"
    
    groups = {group: {"results": [], "structures": [], "appearances": []} for group in GROUPED_STRUCTAPP}
    progress = tqdm(total=256, desc="Loading")
    for category in listdir(args.results_folder):
        for run in listdir(path.join(args.results_folder, category)):
            dir = path.join(args.results_folder, category, run)
            
            try:
                result = Image.open(path.join(dir, filename)).resize((args.size,) * 2, resample=Image.BILINEAR)
                structure = Image.open(path.join(dir, "structure.jpg")).resize((args.size,) * 2, resample=Image.BILINEAR)
                appearance = Image.open(path.join(dir, "appearance.jpg")).resize((args.size,) * 2, resample=Image.BILINEAR)
            except:
                print("Missing files (skipping):", dir)
                continue

            prompt, group = get_prompt_group_structapp(run)
            groups[group]["results"].append(result)
            groups[group]["structures"].append(structure)
            groups[group]["appearances"].append(appearance)
            
            progress.update(1)
    
    losses = {group: {"count": 0} for group in ["all"] + list(GROUPED_STRUCTAPP.keys())}
    
    count_total = 0
    for group in groups:
        count = len(groups[group]["results"])
        count_total += count
        
        losses[group]["count"] = count
        # Structure alignment: DINO feature self-similarity (lower the better)
        losses[group]["self_sim"] = lossg.calculate_self_sim_loss(
            groups[group]["results"], groups[group]["structures"]) / count
        # Appearance alignment: DINO-I, a.k.a. CLS token cosine similarity (higher the better)
        losses[group]["dino_i"] = lossg.calculate_dino_i_loss(
            groups[group]["results"], groups[group]["appearances"]) / count
    
    losses["all"]["count"] = count_total
    for group in groups:
        for loss in losses[group]:
            if loss == "count":
                continue
            if loss not in losses["all"]:
                losses["all"][loss] = 0.0
            losses["all"][loss] += losses[group][loss] * losses[group]["count"] / count_total
        
    print(
        f"\n[Natural image (n={losses['natural']['count']})]           "
        f"Self-sim: {losses['natural']['self_sim']:.7f}   DINO-I: {losses['natural']['dino_i']:.7f}\n"
        f"[ControlNet-supported (n={losses['controlnet']['count']})]   "
        f"Self-sim: {losses['controlnet']['self_sim']:.7f}   DINO-I: {losses['controlnet']['dino_i']:.7f}\n"
        f"[New condition (n={losses['new']['count']})]           "
        f"Self-sim: {losses['new']['self_sim']:.7f}   DINO-I: {losses['new']['dino_i']:.7f}\n"
        f"[All (n={losses['all']['count']})]                    "
        f"Self-sim: {losses['all']['self_sim']:.7f}   DINO-I: {losses['all']['dino_i']:.7f}\n"
    )

    return losses


def evaluate_promptdriven(args):
    lossg = LossG(cfg=None)

    file_prefix = "result_refiner" if args.refiner else "result"
    filename = file_prefix + "__" + str(args.results_seed) + ".jpg"
    
    groups = {group: {"results": [], "structures": [], "prompts": []} for group in GROUPED_PROMPTDRIVEN}
    progress = tqdm(total=175, desc="Loading")
    for category in listdir(args.results_folder):
        for run in listdir(path.join(args.results_folder, category)):
            dir = path.join(args.results_folder, category, run)
            
            try:
                result = Image.open(path.join(dir, filename)).resize((args.size,) * 2, resample=Image.BILINEAR)
                structure = Image.open(path.join(dir, "structure.jpg")).resize((args.size,) * 2, resample=Image.BILINEAR)
            except:
                print("Missing files (skipping):", dir)
                continue

            prompt, group = get_prompt_group_promptdriven(run)
            groups[group]["results"].append(result)
            groups[group]["structures"].append(structure)
            groups[group]["prompts"].append(prompt)
            
            progress.update(1)
    
    losses = {group: {"count": 0} for group in ["all"] + list(GROUPED_PROMPTDRIVEN.keys())}
    
    count_total = 0
    for group in groups:
        count = len(groups[group]["results"])
        count_total += count
        
        losses[group]["count"] = count
        # Structure alignment: DINO feature self-similarity (lower the better)
        losses[group]["self_sim"] = lossg.calculate_self_sim_loss(
            groups[group]["results"], groups[group]["structures"]) / count
        # Prompt alignment: CLIP similarity score (higher the better)
        losses[group]["clip"] = lossg.calculate_clip_text_loss(
            groups[group]["results"], groups[group]["prompts"]) / count
        # Appearance leakage from structure: LPIPS (higher the better)
        losses[group]["lpips"] = lossg.calculate_LPIPS_distance(
            groups[group]["results"], groups[group]["structures"]) / count
    
    losses["all"]["count"] = count_total
    for group in groups:
        for loss in losses[group]:
            if loss == "count":
                continue
            if loss not in losses["all"]:
                losses["all"][loss] = 0.0
            losses["all"][loss] += losses[group][loss] * losses[group]["count"] / count_total
        
    print(
        f"\n[ControlNet-supported (n={losses['controlnet']['count']})]   "
        f"Self-sim: {losses['controlnet']['self_sim']:.7f}   CLIP score: {losses['controlnet']['clip']:.7f}   "
        f"LPIPS: {losses['controlnet']['lpips']:.7f}\n"
        f"[New condition (n={losses['new']['count']})]           "
        f"Self-sim: {losses['new']['self_sim']:.7f}   CLIP score: {losses['new']['clip']:.7f}   "
        f"LPIPS: {losses['new']['lpips']:.7f}\n"
        f"[All (n={losses['all']['count']})]                    "
        f"Self-sim: {losses['all']['self_sim']:.7f}   CLIP score: {losses['all']['clip']:.7f}   "
        f"LPIPS: {losses['all']['lpips']:.7f}\n"
    )

    return losses
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_folder", "-rf", type=str, default="./results")
    parser.add_argument("--results_seed", "-rs", type=int, default=90095)
    parser.add_argument("--evaluation_type", "-t", type=str, default="structure+appearance",
                        choices=["structure+appearance", "prompt-driven"])
    parser.add_argument("--size", "-s", type=int, default=512)
    parser.add_argument("--refiner", "-r", action="store_true")
    args = parser.parse_args()
    
    args.results_folder = path.join(args.results_folder, args.evaluation_type)

    eval_fn = {"structure+appearance": evaluate_structapp, "prompt-driven": evaluate_promptdriven}[args.evaluation_type]
    eval_fn(args)
