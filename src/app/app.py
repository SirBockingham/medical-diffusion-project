import io
import sys
import zipfile
from pathlib import Path

import streamlit as st
import torch

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from diffusers import UNet2DModel  # type: ignore
from PIL import Image

from generation.generate import (
    denormalize,
    generate_batch,
    load_class_names,
    load_run_info,
)
from models.scheduler import build_scheduler_from_config
from utils.config import load_config


def resolve_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    return torch.device(device_setting)



def list_checkpoints(checkpoint_root: Path) -> dict:
    runs = {}
    
    if not checkpoint_root.exists():
        return runs
    
    direct_epochs = sorted(
        [item for item in checkpoint_root.iterdir()
         if item.is_dir() and item.name.startswith("epoch_")],
        key=lambda folder: folder.name,
    )
    if direct_epochs:
        runs["direct epochs"] = direct_epochs
    
    
    run_folders = sorted(
        [item for item in checkpoint_root.iterdir()
         if item.is_dir() and not item.name.startswith("epoch_")],
        key=lambda folder: folder.name,
        reverse=True
    )
 
    for run_folder in run_folders:
        epoch_folders = sorted(
            [item for item in run_folder.iterdir()
             if item.is_dir() and item.name.startswith("epoch_")],
            key=lambda folder: folder.name
        )
        if epoch_folders:
            runs[run_folder.name] = epoch_folders
 
    return runs



@st.cache_resource(show_spinner=False)
def load_model_cached(checkpoint_path_str: str, device_str: str):
    model = UNet2DModel.from_pretrained(checkpoint_path_str)
    model.to(torch.device(device_str))
    model.eval()
    return model



def images_to_zip(images_uint8: torch.Tensor, metadata_text: str) -> bytes:
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for index in range(images_uint8.shape[0]):
            single = images_uint8[index]
            array = single.permute(1, 2, 0).numpy()
 
            if array.shape[2] == 3:
                pil_image = Image.fromarray(array, mode="RGB").convert("L")
            else:
                pil_image = Image.fromarray(array[:, :, 0], mode="L")
 
            image_buffer = io.BytesIO()
            pil_image.save(image_buffer, format="PNG")
            zip_file.writestr(f"sample_{index:04d}.png", image_buffer.getvalue())
 
        zip_file.writestr("parameters.txt", metadata_text)
 
    return zip_buffer.getvalue()



def tensor_to_display_arrays(images: torch.Tensor) -> list:
    images_uint8 = denormalize(images.cpu())
 
    display_arrays = []
    for index in range(images_uint8.shape[0]):
        array = images_uint8[index].permute(1, 2, 0).numpy()
        if array.shape[2] == 3:
            display_arrays.append(array[:, :, 0])
        else:
            display_arrays.append(array[:, :, 0])
 
    return display_arrays



st.set_page_config(page_title="Diffusion - Medical Image Generator", layout="wide")

config = load_config()

st.title("Diffusion-based Medical Image Generator")

st.sidebar.header("Model")

checkpoint_root = config["paths"]["checkpoints"]
available_runs = list_checkpoints(checkpoint_root)

if not available_runs:
    st.sidebar.error(f"No checkpoints found in: {checkpoint_root}")
    st.info("The model needs to be trained first\n")
    st.stop()

selected_run = st.sidebar.selectbox(
    "Run",
    options=list(available_runs.keys()),
    help="The folder name contains the timestamp, mode (cond/uncond), and the image size"
)

epoch_folders = available_runs[selected_run]
selected_epoch_path = st.sidebar.selectbox(
    "Checkpoint",
    options=epoch_folders,
    format_func=lambda path: path.name,
    index=len(epoch_folders) - 1 
)

device_setting = st.sidebar.selectbox(
    "Device",
    options=["auto", "cuda", "cpu"],
)
device = resolve_device(device_setting)
st.sidebar.caption(f"Active device: {device}")

class_names = load_class_names(selected_epoch_path)
run_info = load_run_info(selected_epoch_path)
is_conditional = class_names is not None

if is_conditional:
    st.sidebar.success(f"Conditional model, {len(class_names)} categories")
else:
    st.sidebar.info("Unconditional model (no category selection)")
    


tab_generate, tab_model, tab_help = st.tabs(
    ["Generation", "Model details", "Help"]
)


with tab_generate:
    st.subheader("Image Generation")
 
    settings_column, preview_column = st.columns([1, 2])
 
    with settings_column:
        if is_conditional:
            label_options = ["Any"] + class_names
            selected_label = st.selectbox(
                "Category",
                options=label_options,
                help="Which diagnosis the model should generate.",
            )
            
            if selected_label == "Any":
                class_index = len(class_names)
                null_class_index = None
            else:
                class_index = class_names.index(selected_label)
                null_class_index = len(class_names)
 
            guidance_scale = st.slider(
                "Guidance scale",
                min_value=1.0,
                max_value=10.0,
                value=float(config["conditional"]["guidance_scale"]),
                step=0.5,
                disabled=(selected_label == "Any"),
                help="How strongly the model should follow the requested category. "
                     "1.0 = no guidance. Higher values follow the label more closely, but produce less varied images. Not applicable when Category is set to Any."
            )
        else:
            selected_label = None
            class_index = None
            null_class_index = None
            guidance_scale = 1.0
 
        num_images = st.number_input(
            "Number of images",
            min_value=1,
            max_value=16,
            value=4,
            step=1,
        )
 
        num_inference_steps = st.slider(
            "Denoising steps",
            min_value=10,
            max_value=1000,
            value=200,
            step=10,
            help="More steps = better quality, but proportionally slower."
        )
 
        use_fixed_seed = st.checkbox(
            "Fixed seed",
            value=True,
            help="When enabled, the same seed produces the same images"
        )
        if use_fixed_seed:
            seed = st.number_input("Seed", value=int(config["seed"]), step=1)
        else:
            seed = None
 
        generate_clicked = st.button("Generate", type="primary", use_container_width=True)
        
    
    with preview_column:
        if generate_clicked:
            progress_bar = st.progress(0.0, text="Loading model...")
 
            try:
                model = load_model_cached(str(selected_epoch_path), str(device))
                noise_scheduler = build_scheduler_from_config(config)
 
                image_size = config["data"]["image_size"]
                in_channels = config["model"]["in_channels"]
 
                if seed is not None:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(seed))
                else:
                    generator = None
 
                def update_progress(current_step: int, total_steps: int):
                    fraction = current_step / total_steps
                    progress_bar.progress(
                        fraction,
                        text=f"Denoising: step {current_step}/{total_steps}",
                    )
 
                images = generate_batch(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    batch_size=int(num_images),
                    image_size=image_size,
                    in_channels=in_channels,
                    num_inference_steps=int(num_inference_steps),
                    device=device,
                    generator=generator,
                    class_index=class_index,
                    null_class_index=null_class_index,
                    guidance_scale=guidance_scale,
                    progress_callback=update_progress,
                )
 
                progress_bar.empty()
 
                st.session_state["generated_images"] = images.cpu()
                st.session_state["generation_params"] = {
                    "checkpoint": str(selected_epoch_path),
                    "category": selected_label if selected_label else "(unconditional)",
                    "guidance_scale": guidance_scale,
                    "denoising_steps": int(num_inference_steps),
                    "seed": seed if seed is not None else "(not fixed)",
                    "image_size": image_size,
                }
 
            except Exception as error:
                progress_bar.empty()
                st.error(f"Error during generation: {error}")
 
        if "generated_images" in st.session_state:
            images = st.session_state["generated_images"]
            params = st.session_state["generation_params"]
 
            display_arrays = tensor_to_display_arrays(images)
 
            columns_per_row = min(4, len(display_arrays))
            for row_start in range(0, len(display_arrays), columns_per_row):
                row_arrays = display_arrays[row_start:row_start + columns_per_row]
                image_columns = st.columns(columns_per_row)
 
                for column, array in zip(image_columns, row_arrays):
                    with column:
                        st.image(array, use_container_width=True, clamp=True)
 
            st.caption(" | ".join(f"{key}: {value}" for key, value in params.items()))
 
            metadata_lines = [f"{key}: {value}" for key, value in params.items()]
            zip_bytes = images_to_zip(
                denormalize(images),
                "\n".join(metadata_lines),
            )
 
            st.download_button(
                "Download images (ZIP)",
                data=zip_bytes,
                file_name="generated_images.zip",
                mime="application/zip",
            )
        elif not generate_clicked:
            st.info("Set the parameters, then click Generate.")
        
        
with tab_model:
    st.subheader("Selected checkpoint")
 
    st.write(f"**Path:** `{selected_epoch_path}`")
 
    if run_info is not None:
        st.write("**Training settings:**")
        st.json(run_info)
 
        mismatches = []
        if run_info.get("image_size") != config["data"]["image_size"]:
            mismatches.append(
                f"image_size: checkpoint {run_info.get('image_size')}, "
                f"config {config['data']['image_size']}"
            )
        if run_info.get("conditional") != config["conditional"]["enabled"]:
            mismatches.append(
                f"conditional: checkpoint {run_info.get('conditional')}, "
                f"config {config['conditional']['enabled']}"
            )
        if run_info.get("beta_schedule") != config["scheduler"]["beta_schedule"]:
            mismatches.append(
                f"beta_schedule: checkpoint '{run_info.get('beta_schedule')}', "
                f"config '{config['scheduler']['beta_schedule']}'"
            )
 
        if mismatches:
            st.error(
                "The checkpoint settings are different from the config.yaml:\n\n"
                + "\n".join(f"- {item}" for item in mismatches)
                + "\n\nGeneration can lead errors or wrong results."
            )
    else:
        st.info(
            "This checkpoint has no `run_info.json`"
        )
 
    if is_conditional:
        st.write("**Available categories:**")
        for index, class_name in enumerate(class_names):
            st.write(f"- `[{index}]` {class_name}")
            
            
with tab_help:
    st.subheader("Usage Instructions")
 
    st.markdown(
        """
**Denoising steps.** The number of steps used to clean the noise during
generation. More steps give better quality but take proportionally longer.
For a quick check 50-100 steps are enough; for final images it is worth
going above 500.
 
**Guidance scale** (conditional models only). Controls how strongly the
model follows the requested category. A value of 1.0 disables guidance.
The usual range is 3-7: higher values follow the label more closely but
produce less varied images, and very high values give artificial,
over-driven results. Note that guidance doubles the generation time,
because the network runs twice per step.
 
**Seed.** With a fixed seed, the same checkpoint and the same parameters
always return exactly the same images.
        """
    )
 
    st.subheader("Command-line equivalents")
    st.code(
        "# generate images\n"
        "python src/generation/generate.py --label Pneumonia --num-images 8\n\n"
        "# list available categories\n"
        "python src/generation/generate.py --list-labels\n\n"
        "# FID evaluation\n"
        "python src/evaluation/evaluate_fid.py --num-samples 500",
        language="bash",
    )