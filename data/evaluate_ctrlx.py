from argparse import ArgumentParser
from os import makedirs, path
import subprocess

from tqdm.auto import tqdm
import yaml
    
    
def wrap_quotes(s):
    s = f"\"{s}\""
    return s


def run_structapp(args):
    pairs = args.yaml_data["pairs"]
    args.batch_size = len(pairs) if args.batch_size <= 0 else args.batch_size

    progress = tqdm(total=len(pairs))
    for i in range(0, len(pairs), args.batch_size):
        pairs_batch = pairs[i:i + args.batch_size]
        
        output_paths = []
        structure_images = []
        appearance_images = []
        prompts = []
        structure_prompts = []
        appearance_prompts = []
        
        # Wrap quotes around *everything* to prevent spaces from screwing things up
        for pair in pairs_batch:
            structure = pair["f1_data"]
            appearance = pair["f2_data"]
            
            id_stripped = path.splitext(pair["id"])[0]
            output_path = path.join(args.output_path, appearance["category"], id_stripped)
            output_paths.append(wrap_quotes(output_path))
            
            structure_images.append(wrap_quotes(path.join(args.data_path, structure["filename"])))
            appearance_images.append(wrap_quotes(path.join(args.data_path, appearance["filename"])))
            
            prompts.append(wrap_quotes(pair["target_prompt"]))
            
            structure_prompts.append(wrap_quotes(structure["description"]))
            if args.use_app_prompt_from_data:
                appearance_prompts.append(wrap_quotes(appearance["description"]))
            else:
                appearance_prompts.append(wrap_quotes(""))
        
        command_args = (
            " --output_folder " + " ".join(output_paths) +
            " --structure_image " + " ".join(structure_images) +
            " --appearance_image " + " ".join(appearance_images) +
            " --prompt " + " ".join(prompts) +
            " --structure_prompt " + " ".join(structure_prompts) +
            " --appearance_prompt " + " ".join(appearance_prompts) +
            " --seed " + " ".join(args.seed) +
            " --disable_refiner --benchmark"
        )
            
        command = f"python3 -m data.run_ctrlx_batch {command_args}"
        
        subprocess.run(command, shell=True, check=True)
        
        progress.update(args.batch_size)
        
        
def run_promptdriven(args):
    images = args.yaml_data["images"]
    args.batch_size = len(images) if args.batch_size <= 0 else args.batch_size

    progress = tqdm(total=len(images))
    for i in range(0, len(images), args.batch_size):
        images_batch = images[i:i + args.batch_size]
        
        output_paths = []
        structure_images = []
        prompts = []
        structure_prompts = []
        
        # Wrap quotes around *everything* to prevent spaces from screwing things up
        for image in images_batch:
            for prompt_data in image["prompts"]:
                output_path = path.join(args.output_path, image["category"], prompt_data["id"])
                output_paths.append(wrap_quotes(output_path))
                
                structure_images.append(wrap_quotes(path.join(args.data_path, image["filename"])))
                structure_prompts.append(wrap_quotes(image["description"]))

                prompts.append(wrap_quotes(prompt_data["prompt"]))
        
        command_args = (
            " --output_folder " + " ".join(output_paths) +
            " --structure_image " + " ".join(structure_images) +
            " --prompt " + " ".join(prompts) +
            " --structure_prompt " + " ".join(structure_prompts) +
            " --seed " + " ".join(args.seed) +
            " --disable_refiner --benchmark"
        )
            
        command = f"python3 -m data.run_ctrlx_batch {command_args}"
        
        subprocess.run(command, shell=True, check=True)
        
        progress.update(args.batch_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    # General
    parser.add_argument("--data_path", "-b", type=str, default="./data/images")
    parser.add_argument(
        "--use_app_prompt_from_data", action="store_true",
        help=("Use appearance prompt from the evaluation dataset .yaml file. "
              "Otherwise (default) use (output) prompt as appearance prompt.")
    )
    parser.add_argument("--evaluation_type", "-e", type=str, default="structure+appearance",
                        choices=["structure+appearance", "prompt-driven"])
    parser.add_argument("--output_base_path", "-o", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--seed", "-s", nargs="+", type=int, default=[90095])

    args = parser.parse_args()

    with open(path.join(args.data_path, f"data_{args.evaluation_type}.yaml")) as file:
        yaml_data = yaml.safe_load(file)
    args.yaml_data = yaml_data

    args.output_path = path.join(args.output_base_path, args.evaluation_type)
    
    args.seed = [str(seed) for seed in args.seed]

    run_fn = {"structure+appearance": run_structapp, "prompt-driven": run_promptdriven}[args.evaluation_type]
    run_fn(args)
