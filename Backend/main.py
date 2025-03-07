import io
import uuid
import time
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image

import clip  # official OpenAI CLIP (install via GitHub)
import open_clip  # for OpenCLIP models

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for DAS results. Keys are UUID strings.
das_cache = {}

# Use CUDA if available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Serve index.html by reading from file.
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def read_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading index.html: {e}")

# -------------------------------
# 1. Model Loading and Utilities
# -------------------------------
print("Loading models. This might take a minute...")
models_to_load = [
    ("ViT-B-32", "laion400m_e32"),
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("OpenAI-ViT-B/32", None),
]

models_and_tokenizers = []
for model_str, data_str in models_to_load:
    print(f"Loading {model_str} on data {data_str} ...")
    if data_str is not None:  # OpenCLIP models
        model, _, preprocess = open_clip.create_model_and_transforms(model_str, pretrained=data_str)
        model.to(device)
        try:
            model = torch.compile(model)
        except Exception as e:
            print("torch.compile not supported for OpenCLIP model:", e)
        tokenizer = open_clip.get_tokenizer(model_str)
        models_and_tokenizers.append((model, tokenizer, preprocess.transforms[-1].mean, preprocess.transforms[-1].std))
    else:  # Official CLIP model
        model, preprocess = clip.load(model_str.split("OpenAI-")[1], device=device, jit=False)
        model.eval()
        try:
            model = torch.compile(model)
        except Exception as e:
            print("torch.compile not supported for official CLIP model:", e)
        tokenizer = clip.tokenize
        models_and_tokenizers.append((model, tokenizer, preprocess.transforms[-1].mean, preprocess.transforms[-1].std))

# For this demo, we use all three models.
chosen_model_ids = [0, 1, 2]
models_and_tokenizers_to_use = [x for i, x in enumerate(models_and_tokenizers) if i in chosen_model_ids]
normalize_fns_list = [
    lambda x, m=mean, s=std: (x - torch.Tensor(m).reshape([1, 3, 1, 1]).to(device)) / torch.Tensor(s).reshape([1, 3, 1, 1]).to(device)
    for (_, _, mean, std) in models_and_tokenizers_to_use
]

def get_many_text_features(model, tokenizer, texts):
    tokenized_text = tokenizer(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokenized_text)
    return text_features

def loss_between_images_and_text(model, batch_of_images, text_features, target_values=None):
    text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = model.encode_image(batch_of_images)
    image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)
    scores = image_features_normed @ text_features_normed.T
    if target_values is None:
        return torch.mean(scores, axis=1)
    else:
        return torch.mean(scores * torch.Tensor(target_values).to(device).reshape([1, -1]), axis=1)

def raw_to_real_image(raw_image):
    return (torch.tanh(raw_image) + 1.0) / 2.0

def real_to_raw_image(real_image, eps=1e-5):
    tanh_result = torch.clip(real_image, eps, 1 - eps) * 2.0 - 1.0
    return torch.arctanh(tanh_result)

# --------------------------
# 2. Augmentation Functions
# --------------------------
def add_jitter(x, size=3, res=224):
    in_res = x.shape[2]
    x_shift = np.random.choice(range(-size, size + 1)) if size > 0 else 0
    y_shift = np.random.choice(range(-size, size + 1)) if size > 0 else 0
    x = torch.roll(x, shifts=(x_shift, y_shift), dims=(-2, -1))
    x = x[:, :, (in_res - res) // 2:(in_res - res) // 2 + res,
             (in_res - res) // 2:(in_res - res) // 2 + res]
    return x

def add_noise(x, scale=0.1):
    return x + torch.rand_like(x) * scale

def make_image_augmentations(image_in, count=1, jitter_scale=3, noise_scale=0.1, clip_output=True):
    images_collected = []
    for _ in range(count):
        image_aug = add_jitter(image_in, size=jitter_scale) if jitter_scale is not None else image_in
        image_aug = add_noise(image_aug, scale=noise_scale) if noise_scale is not None else image_aug
        images_collected.append(image_aug)
    images_cat = torch.concatenate(images_collected, axis=0)
    return torch.clip(images_cat, 0, 1) if clip_output else images_cat

# -------------------------------
# 3. DAS Image Generation Method
# -------------------------------
def generate_image(models_and_tokenizers_to_use,
                   target_texts_and_values,
                   starting_image=None,
                   original_resolution=224,
                   large_resolution=336,  # 224 + 2*56
                   resolutions=range(1, 337),
                   batch_size=16,
                   lr=0.1,
                   steps=100,
                   jitter_scale=28,
                   noise_scale=0.2,
                   augmentation_copies=8,
                   multiple_generations_at_once=1,
                   attack_size_factor=None,
                   step_to_show=10,
                   guiding_images_tensor=None,
                   inpainting_mask=None):
    """
    Runs full DAS optimization and returns a dict with:
      - final_image: final image tensor,
      - intermediate_images: list of tensors for each step,
      - loss_curve: list of average losses.
    """
    target_texts = [x[1] for x in target_texts_and_values]
    target_values = [x[0] for x in target_texts_and_values]
    
    # Encode target texts.
    text_features_list = []
    for (model, tokenizer, _, _) in models_and_tokenizers_to_use:
        features = get_many_text_features(model, tokenizer, target_texts)
        text_features_list.append(features.to(device))
    
    # If no starting image, create a blank gray image.
    if starting_image is None:
        np_image_now = np.ones((multiple_generations_at_once, 3, large_resolution, large_resolution)) * 0.5
    else:
        np_image_now = starting_image

    torch_image_raw = real_to_raw_image(torch.Tensor(np_image_now).to(device))
    original_image = torch.Tensor(np_image_now).to(device)
    
    # Create learnable perturbations at multiple resolutions.
    resolutions = sorted(list(set(resolutions)))
    all_image_perturbations = [torch.zeros((multiple_generations_at_once, 3, res, res), device=device, requires_grad=True)
                               for res in resolutions]
    
    optimizer = torch.optim.SGD(all_image_perturbations, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    
    collected_images = []
    loss_curve = []
    
    for step in tqdm.tqdm(range(steps), desc="Optimizing", ncols=80):
        total_perturbation = 0.0
        for i, p in enumerate(all_image_perturbations):
            upscaled = F.interpolate(p, size=(large_resolution, large_resolution),
                                     mode='bicubic' if resolutions[i] > 1 else 'nearest')
            total_perturbation += upscaled

        image_perturbation = total_perturbation

        if inpainting_mask is not None:
            image_perturbation.register_hook(lambda grad: grad * inpainting_mask)
        
        current_image = raw_to_real_image(torch_image_raw + image_perturbation)
        collected_images.append(current_image.detach().cpu())
        
        optimizer.zero_grad()
        step_losses = []
        for _ in range(int(np.ceil(augmentation_copies / batch_size))):
            for i_model, (model, tokenizer, _, _) in enumerate(models_and_tokenizers_to_use):
                model.eval()
                aug_images = make_image_augmentations(current_image, count=batch_size,
                                                      jitter_scale=jitter_scale, noise_scale=noise_scale)
                aug_images = aug_images.to(device)
                loss = -loss_between_images_and_text(
                    model,
                    normalize_fns_list[i_model](torch.clip(aug_images, 0, 1)),
                    text_features_list[i_model],
                    target_values=target_values,
                )
                loss = loss.mean() * multiple_generations_at_once
                loss.backward(retain_graph=True)
                step_losses.append(loss.item())
        optimizer.step()
        scheduler.step()
        
        avg_loss = np.mean(step_losses)
        loss_curve.append(avg_loss)
        
        if step % step_to_show == 0:
            ell_inf = np.max(np.abs(collected_images[-1].numpy() - collected_images[0].numpy())) * 255
            tqdm.tqdm.write(f"Step {step}: Loss = {avg_loss:.4f}, ℓ∞ = {ell_inf:.2f}/255")
    
    return {"final_image": collected_images[-1],
            "intermediate_images": collected_images,
            "loss_curve": loss_curve}

# -------------------------------
# 4. API Data Models
# -------------------------------
class DASRequest(BaseModel):
    prompt: str
    weight: float = 0.5
    steps: int = 100

# -------------------------------
# 5. API Endpoints
# -------------------------------
@app.post("/run_das")
def run_das(request: DASRequest):
    """
    Run the DAS optimization given a prompt, weight, and steps.
    Here we perform dual prompt interpolation: the provided prompt (content)
    is blended with a hardcoded style prompt ("vibrant colorful background, high saturation, dynamic lighting")
    based on the weight slider. When weight=0, only the content prompt is used;
    when weight=1, only the style prompt influences the generation.
    Returns a unique run_id that can be used to fetch results.
    """
    run_id = str(uuid.uuid4())
    
    # Dual prompt interpolation: blend content prompt and style prompt.
    # Weight determines the contribution of the style prompt.
    target_texts_and_values = [
        (1.0 - request.weight, request.prompt),
        (request.weight, "vibrant colorful background, high saturation, dynamic lighting")
    ]
    
    start_time = time.time()
    results = generate_image(
        models_and_tokenizers_to_use,
        target_texts_and_values=target_texts_and_values,
        starting_image=None,
        steps=request.steps,
        augmentation_copies=8,
        jitter_scale=28,
        noise_scale=0.2,
        lr=0.1,
        batch_size=16,
        multiple_generations_at_once=1
    )
    elapsed = time.time() - start_time
    print(f"DAS optimization completed in {elapsed:.1f} seconds for run_id: {run_id}")
    das_cache[run_id] = results
    return {"run_id": run_id, "elapsed": elapsed}

def tensor_to_pil(tensor):
    """
    Convert a tensor (shape [1, 3, H, W] with values in [0,1]) to a PIL Image.
    Performs a center crop to 224x224.
    """
    offset = (tensor.shape[-1] - 224) // 2
    cropped = tensor[0, :, offset:offset+224, offset:offset+224]
    np_img = (cropped.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

@app.get("/get_final/{run_id}")
def get_final(run_id: str):
    """
    Return the final generated image as a PNG.
    """
    if run_id not in das_cache:
        raise HTTPException(status_code=404, detail="Run ID not found")
    final_tensor = das_cache[run_id]["final_image"]
    pil_img = tensor_to_pil(final_tensor)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/get_intermediate/{run_id}")
def get_intermediate(run_id: str, step: int = Query(0, ge=0)):
    """
    Return the intermediate image at a given step as a PNG.
    """
    if run_id not in das_cache:
        raise HTTPException(status_code=404, detail="Run ID not found")
    intermediate_images = das_cache[run_id]["intermediate_images"]
    if step >= len(intermediate_images):
        raise HTTPException(status_code=400, detail="Step index out of range")
    tensor = intermediate_images[step]
    pil_img = tensor_to_pil(tensor)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/get_loss/{run_id}")
def get_loss(run_id: str):
    """
    Return the loss curve as a JSON list.
    """
    if run_id not in das_cache:
        raise HTTPException(status_code=404, detail="Run ID not found")
    loss_curve = das_cache[run_id]["loss_curve"]
    return JSONResponse(content={"loss_curve": loss_curve})
