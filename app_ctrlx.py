from argparse import ArgumentParser

from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline
import gradio as gr
import torch
import yaml

from ctrl_x.pipelines.pipeline_sdxl import CtrlXStableDiffusionXLPipeline
from ctrl_x.utils import *
from ctrl_x.utils.sdxl import *


parser = ArgumentParser()
parser.add_argument("-m", "--model", type=str, default=None)  # Optionally, load model checkpoint from single file
args = parser.parse_args()

torch.backends.cudnn.enabled = False  # Sometimes necessary to suppress CUDNN_STATUS_NOT_SUPPORTED

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
variant = "fp16" if device == "cuda" else "fp32"

scheduler = DDIMScheduler.from_config(model_id_or_path, subfolder="scheduler")  # TODO: Support other schedulers
if args.model is None:
    pipe = CtrlXStableDiffusionXLPipeline.from_pretrained(
        model_id_or_path, scheduler=scheduler, torch_dtype=torch_dtype, variant=variant, use_safetensors=True
    )
else:
    print(f"Using weights {args.model} for SDXL base model.")
    pipe = CtrlXStableDiffusionXLPipeline.from_single_file(args.model, scheduler=scheduler, torch_dtype=torch_dtype)
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    refiner_id_or_path, scheduler=scheduler, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae,
    torch_dtype=torch_dtype, variant=variant, use_safetensors=True,
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    refiner = refiner.to("cuda")
    

def get_control_config(structure_schedule, appearance_schedule):
    s = structure_schedule
    a = appearance_schedule
    
    control_config =\
f"""control_schedule:
    #       structure_conv   structure_attn   appearance_attn  conv/attn
    encoder:                                                # (num layers)
        0: [[             ], [             ], [             ]]  # 2/0
        1: [[             ], [             ], [{a}, {a}     ]]  # 2/2
        2: [[             ], [             ], [{a}, {a}     ]]  # 2/2
    middle: [[            ], [             ], [             ]]  # 2/1
    decoder:
        0: [[{s}          ], [{s}, {s}, {s}], [0.0, {a}, {a}]]  # 3/3
        1: [[             ], [             ], [{a}, {a}     ]]  # 3/3
        2: [[             ], [             ], [             ]]  # 3/0

control_target:
    - [output_tensor]  # structure_conv   choices: {{hidden_states, output_tensor}}
    - [query, key]     # structure_attn   choices: {{query, key, value}}
    - [before]         # appearance_attn  choices: {{before, value, after}}

self_recurrence_schedule:
    - [0.1, 0.5, 2]  # format: [start, end, num_recurrence]"""
    
    return control_config
    

css = """
.config textarea {font-family: monospace; font-size: 80%; white-space: pre}
.mono {font-family: monospace}
"""

title = """
<div style="display: flex; align-items: center; justify-content: center;margin-bottom: -15px">
    <h1 style="margin-left: 12px;text-align: center;display: inline-block">
        Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance
    </h1>
    <h3 style="display: inline-block; margin-left: 10px; margin-top: 7.5px; font-weight: 500">
        SDXL v1.0
    </h3>
</div>
<div style="display: flex; align-items: center; justify-content: center;margin-bottom: 25px">
    <h3 style="text-align: center">
        [<a href="https://genforce.github.io/ctrl-x/">Page</a>]
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        [<a href="https://arxiv.org/abs/2406.07540">Paper</a>]
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        [<a href="https://github.com/genforce/ctrl-x">Code</a>]
    </h3>
</div>
<div>
    <p>
        <b>Ctrl-X</b> is a simple training-free and guidance-free framework for text-to-image (T2I) generation with 
        structure and appearance control. Given structure and appearance images, Ctrl-X designs feedforward structure 
        control to enable structure alignment with the arbitrary structure image and semantic-aware appearance transfer 
        to facilitate the appearance transfer from the appearance image.
    </p>
    <p>
        Here are some notes and tips for this demo:
    </p>
    <ul>
        <li> On input images:
            <ul>
                <li>
                    If both the structure and appearance images are provided, then Ctrl-X does <i>structure and 
                    appearance</i> control.
                </li>
                <li>
                    If only the structure image is provided, then Ctrl-X does <i>structure-only</i> control and the 
                    appearance image is jointly generated with the output image.
                </li>
                <li>
                    Similarly, if only the appearance image is provided, then Ctrl-X does <i>appearance-only</i> 
                    control.
                </li>
            </ul>
        </li>
        <li> On prompts:
            <ul>
                <li>
                    Though the output prompt can affect the output image to a noticeable extent, the "accuracy" of the 
                    structure and appearance prompts are not impactful to the final image.
                </li>
                <li>
                    If the structure or appearance prompt is left blank, then it uses the (non-optional) output prompt 
                    by default.
                </li>
            </ul>
        </li>
        <li> On control schedules:
            <ul>
                <li>
                    When "Use advanced config" is <b>OFF</b>, the demo uses the structure guidance 
                    (<span class="mono">structure_conv</span> and <span class="mono">structure_attn</span> 
                    in the advanced config) and appearance guidance (<span class="mono">appearance_attn</span> in the 
                    advanced config) sliders to change the control schedules.
                </li>
                <li>
                    Otherwise, the demo uses "Advanced control config," which allows per-layer structure and 
                    appearance schedule control, along with self-recurrence control. <i>This should be used 
                    carefully</i>, and we recommend switching "Use advanced config" <b>OFF</b> in most cases. (For the 
                    examples provided at the bottom of the demo, the advanced config uses the default schedules that 
                    may not be the best settings for these examples.)
                </li>
            </ul>
        </li>
    </ul>
    <p>
        Have fun! :D
    </p>
</div>
"""


def inference(
    structure_image, appearance_image,
    prompt, structure_prompt, appearance_prompt,
    positive_prompt, negative_prompt,
    guidance_scale, structure_guidance_scale, appearance_guidance_scale,
    num_inference_steps, eta, seed,
    width, height,
    structure_schedule, appearance_schedule, use_advanced_config,
    control_config,
):
    torch.manual_seed(seed)
    
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    print(f"\nUsing the following control config (use_advanced_config={use_advanced_config}):")
    if not use_advanced_config:
        control_config = get_control_config(structure_schedule, appearance_schedule)
    print(control_config, end="\n\n")
    
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
    refiner.set_progress_bar_config(desc="Refiner")
    
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
    del pipe.refiner_args
    
    return [result[0], result_refiner[0], structure[0], appearance[0]]
    
    
with gr.Blocks(theme=gr.themes.Default(), css=css, title="Ctrl-X (SDXL v1.0)") as app:
    gr.HTML(title)
    
    with gr.Row():
        
        with gr.Column(scale=55):
            with gr.Group():
                kwargs = {}  # {"width": 400, "height": 400}
                with gr.Row():
                    result = gr.Image(label="Output image", format="jpg", **kwargs)
                    result_refiner = gr.Image(label="Output image w/ refiner", format="jpg", **kwargs)
                with gr.Row():
                    structure_recon = gr.Image(label="Structure image", format="jpg", **kwargs)
                    appearance_recon = gr.Image(label="Style image", format="jpg", **kwargs)
                with gr.Row():
                    structure_image = gr.Image(label="Upload structure image (optional)", type="pil", **kwargs)
                    appearance_image = gr.Image(label="Upload appearance image (optional)", type="pil", **kwargs)
        
        with gr.Column(scale=45):
            with gr.Group():
                with gr.Row():
                    structure_prompt = gr.Textbox(label="Structure prompt (optional)", placeholder="Prompt which describes the structure image")
                    appearance_prompt = gr.Textbox(label="Appearance prompt (optional)", placeholder="Prompt which describes the style image")
                with gr.Row():
                    prompt = gr.Textbox(label="Output prompt", placeholder="Prompt which describes the output image")
                with gr.Row():
                    positive_prompt = gr.Textbox(label="Positive prompt", value="high quality", placeholder="")
                    negative_prompt = gr.Textbox(label="Negative prompt", value="ugly, blurry, dark, low res, unrealistic", placeholder="")
                with gr.Row():
                    guidance_scale = gr.Slider(label="Target guidance scale", value=5.0, minimum=1, maximum=10)
                    structure_guidance_scale = gr.Slider(label="Structure guidance scale", value=5.0, minimum=1, maximum=10)
                    appearance_guidance_scale = gr.Slider(label="Appearance guidance scale", value=5.0, minimum=1, maximum=10)
                with gr.Row():
                    num_inference_steps = gr.Slider(label="# inference steps", value=50, minimum=1, maximum=200, step=1)
                    eta = gr.Slider(label="Eta (noise)", value=1.0, minimum=0, maximum=1.0, step=0.01)
                    seed = gr.Slider(0, 2147483647, label="Seed", value=90095, step=1)
                with gr.Row():
                    width = gr.Slider(label="Width", value=1024, minimum=256, maximum=2048, step=pipe.vae_scale_factor)
                    height = gr.Slider(label="Height", value=1024, minimum=256, maximum=2048, step=pipe.vae_scale_factor)
                with gr.Row():
                    structure_schedule = gr.Slider(label="Structure schedule", value=0.6, minimum=0.0, maximum=1.0, step=0.01, scale=2)
                    appearance_schedule = gr.Slider(label="Appearance schedule", value=0.6, minimum=0.0, maximum=1.0, step=0.01, scale=2)
                    use_advanced_config = gr.Checkbox(label="Use advanced config", value=False, scale=1)
                with gr.Row():
                    control_config = gr.Textbox(
                        label="Advanced control config", lines=20, value=get_control_config(0.6, 0.6), elem_classes=["config"], visible=False,
                    )
                    use_advanced_config.change(
                        fn=lambda value: gr.update(visible=value), inputs=use_advanced_config, outputs=control_config,
                    )
                with gr.Row():
                    generate = gr.Button(value="Run")
                    
    inputs = [
        structure_image, appearance_image,
        prompt, structure_prompt, appearance_prompt,
        positive_prompt, negative_prompt,
        guidance_scale, structure_guidance_scale, appearance_guidance_scale,
        num_inference_steps, eta, seed,
        width, height,
        structure_schedule, appearance_schedule, use_advanced_config,
        control_config,
    ]
    outputs = [result, result_refiner, structure_recon, appearance_recon]
    
    generate.click(inference, inputs=inputs, outputs=outputs)

    examples = gr.Examples(
        [
            [
                "assets/images/horse__point_cloud.jpg",
                "assets/images/horse.jpg",
                "a 3D point cloud of a horse",
                "",
                "a photo of a horse standing on grass",
                0.6, 0.6,
            ],
            [
                "assets/images/cat__mesh.jpg",
                "assets/images/tiger.jpg",
                "a 3D mesh of a cat",
                "",
                "a photo of a tiger standing on snow",
                0.6, 0.6,
            ],
            [
                "assets/images/dog__sketch.jpg",
                "assets/images/squirrel.jpg",
                "a sketch of a dog",
                "",
                "a photo of a squirrel",
                0.6, 0.6,
            ],
            [
                "assets/images/living_room__seg.jpg",
                "assets/images/van_gogh.jpg",
                "a segmentation map of a living room",
                "",
                "a Van Gogh painting of a living room",
                0.6, 0.6,
            ],
            [
                "assets/images/bedroom__sketch.jpg",
                "assets/images/living_room_modern.jpg",
                "a sketch of a bedroom",
                "",
                "a photo of a modern bedroom during sunset",
                0.6, 0.6,
            ],
            [
                "assets/images/running__pose.jpg",
                "assets/images/man_park.jpg",
                "a pose image of a person running",
                "",
                "a photo of a man running in a park",
                0.4, 0.6,
            ],
            [
                "assets/images/fruit_bowl.jpg",
                "assets/images/grapes.jpg",
                "a photo of a bowl of fruits",
                "",
                "a photo of a bowl of grapes in the trees",
                0.6, 0.6,
            ],
            [
                "assets/images/bear_avocado__spatext.jpg",
                None,
                "a segmentation map of a bear and an avocado",
                "",
                "a realistic photo of a bear and an avocado in a forest",
                0.6, 0.6,
            ],
            [
                "assets/images/cat__point_cloud.jpg",
                None,
                "a 3D point cloud of a cat",
                "",
                "an embroidery of a white cat sitting on a rock under the night sky",
                0.6, 0.6,
            ],
            [
                "assets/images/library__mesh.jpg",
                None,
                "a 3D mesh of a library",
                "",
                "a Polaroid photo of an old library, sunlight streaming in",
                0.6, 0.6,
            ],
            [
                "assets/images/knight__humanoid.jpg",
                None,
                "a 3D model of a person holding a sword and shield",
                "",
                "a photo of a medieval soldier standing on a barren field, raining",
                0.6, 0.6,
            ],
            [
                "assets/images/person__mesh.jpg",
                None,
                "a 3D mesh of a person",
                "",
                "a photo of a Karate man performing in a cyberpunk city at night",
                0.5, 0.6,
            ],
        ],
        [
            structure_image,
            appearance_image,
            structure_prompt,
            appearance_prompt,
            prompt,
            structure_schedule,
            appearance_schedule,
        ],
        examples_per_page=50,
    )

app.launch(debug=False, share=False)
