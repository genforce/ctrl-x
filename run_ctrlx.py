from argparse import ArgumentParser
from datetime import datetime
from os import makedirs, path
from time import sleep

from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import torch
import yaml

from ctrl_x.pipelines.pipeline_sdxl import CtrlXStableDiffusionXLPipeline
from ctrl_x.utils import *
from ctrl_x.utils.sdxl import *


JPEG_QUALITY = 100


@torch.no_grad()
def inference(
    pipe, refiner, device,
    structure_image, appearance_image,
    prompt, structure_prompt, appearance_prompt,
    positive_prompt, negative_prompt,
    guidance_scale, structure_guidance_scale, appearance_guidance_scale,
    num_inference_steps, eta, seed,
    width, height,
    structure_schedule, appearance_schedule,
):  
    torch.manual_seed(seed)
    
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    control_config = get_control_config(structure_schedule, appearance_schedule)
    print(f"\nUsing the following control config:\n{control_config}\n")
    
    config = yaml.safe_load(control_config)
    register_control(
        model = pipe,
        timesteps = timesteps,
        control_schedule = config["control_schedule"],
        control_target = config["control_target"],
    )
    
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    self_recurrence_schedule = get_self_recurrence_schedule(config["self_recurrence_schedule"], num_inference_steps)
    
    pipe.set_progress_bar_config(desc="Ctrl-X inference")
    result, structure, appearance = pipe(
        prompt = prompt,
        structure_prompt = structure_prompt,
        appearance_prompt = appearance_prompt,
        structure_image = structure_image,
        appearance_image = appearance_image,
        num_inference_steps = num_inference_steps,
        negative_prompt = negative_prompt,
        positive_prompt = positive_prompt,
        height = height,
        width = width,
        guidance_scale = guidance_scale,
        structure_guidance_scale = structure_guidance_scale,
        appearance_guidance_scale = appearance_guidance_scale,
        eta = eta,
        output_type = "pil",
        return_dict = False,
        control_schedule = config["control_schedule"],
        self_recurrence_schedule = self_recurrence_schedule,
    )
    
    if refiner is not None:
        refiner.set_progress_bar_config(desc="Refiner")
        result_refiner = refiner(
            image = pipe.refiner_args["latents"],
            prompt = pipe.refiner_args["prompt"],
            negative_prompt = pipe.refiner_args["negative_prompt"],
            height = height,
            width = width,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            guidance_rescale = 0.7,
            num_images_per_prompt = 1,
            eta = eta,
            output_type = "pil",
        ).images
    
    else:
        result_refiner = [None]

    del pipe.refiner_args
    
    return result[0], result_refiner[0], structure[0], appearance[0]


def ctrlx_sdxl(pipe, structure_image, appearance_image, prompt, structure_prompt, device):
    torch.manual_seed(90095)
    
    pipe.scheduler.set_timesteps(50, device=device)
    timesteps = pipe.scheduler.timesteps
    
    control_config = get_control_config(0.6, 0.6)
    config = yaml.safe_load(control_config)
    register_control(  # TODO: Maybe just pass the config file in :P
        model = pipe,
        timesteps = timesteps,
        control_schedule = config["control_schedule"],
        control_target = config["control_target"],
    )
    
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    self_recurrence_schedule = get_self_recurrence_schedule(config["self_recurrence_schedule"], 50)

    result, structure, appearance = pipe(
        prompt = prompt,
        structure_prompt = structure_prompt,
        appearance_prompt = "",
        structure_image = structure_image,
        appearance_image = appearance_image,
        num_inference_steps = 50,
        negative_prompt = "ugly, blurry, dark, low res, unrealistic",
        positive_prompt = "high quality",
        height = 1024,
        width = 1024,
        guidance_scale = 5.0,
        structure_guidance_scale = 5.0,
        appearance_guidance_scale = 5.0,
        eta = 1.0,
        output_type = "pil",
        return_dict = False,
        control_schedule = config["control_schedule"],
        self_recurrence_schedule = self_recurrence_schedule,
        decode_structure = True,
        decode_appearance = True,
    )


@torch.no_grad()
def main(args):
    
    structure_image = None
    if args.structure_image is not None:
        structure_image = load_image(args.structure_image)
        
    appearance_image = None
    if args.appearance_image is not None:
        appearance_image = load_image(args.appearance_image)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant = "fp16" if device == "cuda" else "fp32"

    scheduler = DDIMScheduler.from_config(model_id_or_path, subfolder="scheduler")  # TODO: Support schedulers beyond DDIM

    if args.model is None:
        pipe = CtrlXStableDiffusionXLPipeline.from_pretrained(
            model_id_or_path, scheduler=scheduler, torch_dtype=torch_dtype, variant=variant, use_safetensors=True,
        )
    else:
        print(f"Using weights {args.model} for SDXL base model.")
        pipe = CtrlXStableDiffusionXLPipeline.from_single_file(args.model, scheduler=scheduler, torch_dtype=torch_dtype)
    
    refiner = None
    if not args.disable_refiner:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_id_or_path, scheduler=scheduler, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae,
            torch_dtype=torch_dtype, variant=variant, use_safetensors=True,
        )
    
    if args.cpu_offload:
        try:
            import accelerate  # Checking if accelerate is installed for CPU offloading
        except:
            raise ModuleNotFoundError("`accelerate` must be installed for CPU offloading.")
        pipe.enable_model_cpu_offload()
        if refiner is not None:
            refiner.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
        if refiner is not None:
            refiner = refiner.to(device)

    print(f"Base model + refiner loaded. Running on device: {device}.")
    
    result, result_refiner, structure, appearance = inference(
        pipe = pipe,
        refiner = refiner,
        device = device,
        structure_image = structure_image,
        appearance_image = appearance_image,
        prompt = args.prompt,
        structure_prompt = args.structure_prompt,
        appearance_prompt = args.appearance_prompt,
        positive_prompt = args.positive_prompt,
        negative_prompt = args.negative_prompt,
        guidance_scale = args.guidance_scale,
        structure_guidance_scale = args.structure_guidance_scale,
        appearance_guidance_scale = args.appearance_guidance_scale,
        num_inference_steps = args.num_inference_steps,
        eta = args.eta,
        seed = args.seed,
        width = args.width,
        height = args.height,
        structure_schedule = args.structure_schedule,
        appearance_schedule = args.appearance_schedule,
    )
    
    makedirs(args.output_folder, exist_ok=True)
    prefix = "ctrlx__" + datetime.now().strftime("%Y%m%d_%H%M%S")
    structure.save(path.join(args.output_folder, f"{prefix}__structure.jpg"), quality=JPEG_QUALITY)
    appearance.save(path.join(args.output_folder, f"{prefix}__appearance.jpg"), quality=JPEG_QUALITY)
    result.save(path.join(args.output_folder, f"{prefix}__result.jpg"), quality=JPEG_QUALITY)
    if result_refiner is not None:
        result_refiner.save(path.join(args.output_folder, f"{prefix}__result_refiner.jpg"), quality=JPEG_QUALITY)
    
    print("Done.")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--structure_image", "-si", type=str, default=None)
    parser.add_argument("--appearance_image", "-ai", type=str, default=None)
    
    parser.add_argument("--prompt", "-p", type=str, required=True)
    parser.add_argument("--structure_prompt", "-sp", type=str, default="")
    parser.add_argument("--appearance_prompt", "-ap", type=str, default="")
    
    parser.add_argument("--positive_prompt", "-pp", type=str, default="high quality")
    parser.add_argument("--negative_prompt", "-np", type=str, default="ugly, blurry, dark, low res, unrealistic")
    
    parser.add_argument("--guidance_scale", "-g", type=float, default=5.0)
    parser.add_argument("--structure_guidance_scale", "-sg", type=float, default=5.0)
    parser.add_argument("--appearance_guidance_scale", "-ag", type=float, default=5.0)
    
    parser.add_argument("--num_inference_steps", "-n", type=int, default=50)
    parser.add_argument("--eta", "-e", type=float, default=1.0)
    parser.add_argument("--seed", "-s", type=int, default=90095)
    
    parser.add_argument("--width", "-W", type=int, default=1024)
    parser.add_argument("--height", "-H", type=int, default=1024)
    
    parser.add_argument("--structure_schedule", "-ss", type=float, default=0.6)
    parser.add_argument("--appearance_schedule", "-as", type=float, default=0.6)
    
    parser.add_argument("--output_folder", "-o", type=str, default="./results")
    
    parser.add_argument(
        "-c", "--cpu_offload", action="store_true",
        help="Model CPU offload, lowers memory usage with slight runtime increase. `accelerate` must be installed.",
    )
    parser.add_argument("-r", "--disable_refiner", action="store_true")
    parser.add_argument("-m", "--model", type=str, default=None, help="Optionally, load model safetensors")
    
    args = parser.parse_args()
    main(args)
    