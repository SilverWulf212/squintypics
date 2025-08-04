# app.py
"""
SquintyPics – Streamlit MVP wrapper for Hugging Face IllusionDiffusion
=====================================================================

Kid-safe image remixer with a hidden Admin panel.
Built as a single file for dead-simple deployment.

────────────────────────────────────────────────────────
HOW TO RUN (local Windows example)
---------------------------------------------------------------------
1.  Install Python 3.10+ (https://python.org).
2.  Create and activate a virtual environment:

        python -m venv .venv
        .venv\\Scripts\\activate

3.  Install deps:

        pip install -r requirements.txt

4.  Clone IllusionDiffusion to:  C:\ai\squintypics
    (Or change `MODEL_PATH` below.)

5.  Put a starter image (PNG/JPG) inside an `input_images` folder
    beside this script.

6.  Launch Streamlit:

        streamlit run app.py

7.  Open the printed local URL in a browser.
    The newest image in `input_images` loads automatically.
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import smtplib
import sys
import time
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image

# Add IllusionDiffusion to path
ILLUSION_PATH = Path("IllusionDiffusion")
if ILLUSION_PATH.exists():
    sys.path.insert(0, str(ILLUSION_PATH))

# ───────────────────────────── Model Setup ───────────────────────────── #

MODEL_PATH = Path(r"C:\ai\squintypics\IllusionDiffusion")  # ← change if your repo lives elsewhere

try:
    import torch
    from diffusers.pipelines import (
        StableDiffusionControlNetPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
    )
    from diffusers import (
        ControlNetModel,
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler
    )
    import random

    BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"

    if "main_pipe" not in st.session_state or "image_pipe" not in st.session_state:
        with st.spinner("Loading IllusionDiffusion models…"):
            # Initialize components
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
            controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)

            # Initialize the main pipeline
            st.session_state.main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                BASE_MODEL,
                controlnet=controlnet,
                vae=vae,
                safety_checker=None,
                torch_dtype=torch.float16,
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize image-to-image pipeline
            st.session_state.image_pipe = StableDiffusionControlNetImg2ImgPipeline(**st.session_state.main_pipe.components)

except (ImportError, OSError) as err:
    st.error(
        "❌ IllusionDiffusion model failed to load. "
        "Verify MODEL_PATH, torch, and diffusers.\n\n"
        f"Details: {err}"
    )
    st.stop()

main_pipe = st.session_state.main_pipe
image_pipe = st.session_state.image_pipe

# ───────────────────────────── Config Files ──────────────────────────── #

CONFIG_FILE = Path("prompts.json")
DEFAULT_CONFIG = {
    "Imaginative": [
        "A fantasy playground with twisting slides, floating swings, and candy-colored trampolines in the clouds",
        "A looping toy train running through tunnels made of blocks, letters, and crayons in a vibrant kid's world",
        "A futuristic underwater city with dome houses, jellyfish streetlights, and floating bubble cars",
        "A floating castle made of clouds and rainbow-glass bridges, with children riding balloon creatures",
    ],
    "Realistic": [
        "Dinosaur dig — a dusty excavation with partially uncovered fossils, brushes, and crates",
        "Mountain bliss — a serene mountain-lake sunset scene",
        "Swampy magic — a photorealistic bayou with cypress trees and an alligator",
    ],
    "_settings": {
        "watch_folder": "input_images",  # relative to script
        "smtp": {
            "host": "smtp.example.com",
            "port": 587,
            "user": "username",
            "password": "password",
        },
    },
    "_admin_password": "letmein",  # ⚠ replace ASAP
}


def load_config() -> Dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


config = load_config()

# ───────────────────────────── Utilities ─────────────────────────────── #

def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size
    new_dimension = min(width, height)
    left = (width - new_dimension)/2
    top = (height - new_dimension)/2
    right = (width + new_dimension)/2
    bottom = (height + new_dimension)/2
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)
    return img

def common_upscale(samples, width, height, upscale_method, crop=False):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:,:,y:old_height-y,x:old_width-x]
    else:
        s = samples
    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

def upscale(samples, upscale_method, scale_by):
    width = round(samples["images"].shape[3] * scale_by)
    height = round(samples["images"].shape[2] * scale_by)
    s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
    return (s)

def get_latest_image(folder: Path) -> Tuple[Image.Image | None, Path | None]:
    """Return newest PNG/JPG/JPEG file in *folder*."""
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = sorted([p for e in exts for p in folder.glob(e)], key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None, None
    return Image.open(files[0]).convert("RGB"), files[0]


def remix(base_img: Image.Image, prompt: str, steps: int = 25) -> Image.Image:
    """Run IllusionDiffusion with the chosen prompt."""
    # Convert to proper format and size
    control_image_small = center_crop_resize(base_img)
    control_image_large = center_crop_resize(base_img, (1024, 1024))
    
    # Set scheduler
    main_pipe.scheduler = EulerDiscreteScheduler.from_config(main_pipe.scheduler.config)
    
    # Generate seed
    my_seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(my_seed)
    
    # First pass - generate latents
    out = main_pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, distorted",
        image=control_image_small,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        generator=generator,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        num_inference_steps=15,
        output_type="latent"
    )
    
    # Upscale latents
    upscaled_latents = upscale(out, "nearest-exact", 2)
    
    # Second pass - refine with img2img
    out_image = image_pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, distorted",
        control_image=control_image_large,        
        image=upscaled_latents,
        guidance_scale=7.5,
        generator=generator,
        num_inference_steps=20,
        strength=0.5,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        controlnet_conditioning_scale=0.8
    )
    
    return out_image["images"][0] if "images" in out_image else out_image.images[0]


def email_image(to_addr: str, img: Image.Image, smtp_cfg: Dict) -> None:
    """Send *img* as PNG attachment."""
    msg = EmailMessage()
    msg["Subject"] = "Your SquintyPics Creation"
    msg["From"] = smtp_cfg["user"]
    msg["To"] = to_addr
    msg.set_content("Enjoy your magical remix! 🎨")

    buf = BytesIO()
    img.save(buf, format="PNG")
    msg.add_attachment(buf.getvalue(), maintype="image", subtype="png", filename="creation.png")

    with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
        server.starttls()
        server.login(smtp_cfg["user"], smtp_cfg["password"])
        server.send_message(msg)


def big_btn(label: str, key: str):
    return st.button(label, key=key)


# ───────────────────────────── Streamlit UI ──────────────────────────── #

st.set_page_config(page_title="SquintyPics", layout="wide")

st.markdown(
    """
    <style>
    button[kind="primary"] {font-size:22px !important; padding:0.4em 1.2em;}
    div.stRadio>label {font-size:20px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar (Admin toggle) ---------------- #
with st.sidebar:
    st.header("🔒 Admin Login")
    pwd = st.text_input("Password", type="password")
    admin_mode = pwd == config["_admin_password"]

    if admin_mode:
        st.success("Admin mode enabled")
    else:
        st.caption("Enter password to edit settings")

# ---------------- Main Kid-Mode ---------------- #
st.title("🎈 SquintyPics – Remix Your Pic!")

# Load newest base image
watch_path = Path(config["_settings"]["watch_folder"]).expanduser()
base_img, base_path = get_latest_image(watch_path)
if base_img is None:
    st.warning(f"No images in {watch_path.resolve()}")
    st.info("Add some PNG or JPG images to the input_images folder to get started!")
    st.stop()

col_img, col_ctrl = st.columns([3, 2], gap="large")

with col_img:
    st.image(base_img, caption=f"Latest image · {base_path.name}", use_column_width=True)

with col_ctrl:
    cat = st.radio("Choose a theme ▲", ["Imaginative", "Realistic"], horizontal=True)
    prompt_choice = st.radio("Pick a prompt ▶", config[cat], key="prompt_radio")

    if big_btn("✨ Remix!", key="run"):
        with st.spinner("Dreaming…"):
            try:
                st.session_state.generated = remix(base_img, prompt_choice)
                st.success("Done!")
            except Exception as e:
                st.error(f"Generation failed: {e}")

    if "generated" in st.session_state:
        st.image(st.session_state.generated, caption="Your masterpiece!", use_column_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if big_btn("🖨️ Print", key="print"):
                st.markdown("<script>window.print();</script>", unsafe_allow_html=True)
        with col_b:
            eaddr = st.text_input("Email me:", key="email")
            if big_btn("📧 Send", key="mail") and eaddr:
                try:
                    email_image(eaddr, st.session_state.generated, config["_settings"]["smtp"])
                    st.success("Sent! Check your inbox.")
                except Exception as e:
                    st.error(f"Email failed → {e}")

# ---------------- Admin Panel ---------------- #
if admin_mode:
    st.divider()
    st.header("🛠️ Admin Panel")

    # Editable prompts
    st.subheader("Prompts JSON")
    prompts_json = st.text_area(
        "Edit and press ⌘/Ctrl-Enter to save",
        json.dumps({k: v for k, v in config.items() if not k.startswith("_")}, indent=2),
        height=300,
    )
    if big_btn("💾 Save Prompts", key="save_prompts"):
        try:
            new_prompts = json.loads(prompts_json)
            for k in list(config.keys()):
                if not k.startswith("_"):
                    config.pop(k)
            config.update(new_prompts)
            save_config(config)
            st.rerun()
        except json.JSONDecodeError as err:
            st.error(f"JSON error → {err}")

    # Folder path
    st.subheader("Watch Folder")
    new_folder = st.text_input("Path to folder with new images", value=config["_settings"]["watch_folder"])
    if big_btn("Update Folder", key="upd_folder"):
        config["_settings"]["watch_folder"] = new_folder
        save_config(config)
        st.rerun()

    # SMTP creds
    st.subheader("SMTP Settings")
    smtp_cfg = config["_settings"]["smtp"]
    smtp_cfg["host"] = st.text_input("SMTP host", smtp_cfg["host"])
    smtp_cfg["port"] = st.number_input("SMTP port", value=smtp_cfg["port"], step=1)
    smtp_cfg["user"] = st.text_input("User", smtp_cfg["user"])
    smtp_cfg["password"] = st.text_input("Password", smtp_cfg["password"], type="password")
    if big_btn("Save SMTP", key="save_smtp"):
        save_config(config)
        st.success("SMTP saved")

# Remove auto-refresh as it causes issues with Streamlit
# Auto-refresh can be enabled manually with st.rerun() when needed