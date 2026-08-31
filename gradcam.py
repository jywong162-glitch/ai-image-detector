import numpy as np
import cv2
import torch
from PIL import Image

def generate_heatmap(model, input_tensor, target_layer):
    """
    Generates a Grad-CAM heatmap for the input image.
    """
    model.eval()
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def hook_fn(module, _input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    # Attach hook to target layer
    handle = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    output = model(input_tensor)
    idx = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, idx].backward()

    handle.remove()

    # Calculate Grad-CAM
    grads = gradients[0].detach().cpu().numpy()[0]
    f_maps = activations[0].detach().cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * f_maps[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    return cam

def clip_attention_rollout(clip_detector, input_tensor):
    """
    'Where did CLIP look?' heatmap for the frozen-CLIP model.

    CLIP is a Vision Transformer, so it has no conv feature map for Grad-CAM.
    Instead we use **attention rollout** (Abnar & Zuidema, 2020): capture the
    self-attention weights from every transformer block, add the residual
    connection, and multiply them together to trace how much each image patch
    ultimately influences the CLS token. No gradients needed — we just read the
    attention the model already computed. Returns a 0..1 heatmap (same format as
    generate_heatmap) or None if the CLIP internals don't match what we expect.
    """
    visual = clip_detector.clip.visual
    blocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
    if blocks is None:
        return None

    attn_weights = []
    saved = []

    def make_wrapper(orig_forward):
        def wrapper(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True   # average over heads -> [B, S, S]
            out = orig_forward(*args, **kwargs)
            if isinstance(out, tuple) and out[1] is not None:
                attn_weights.append(out[1].detach())
            return out
        return wrapper

    try:
        for blk in blocks:
            attn = blk.attn
            saved.append((attn, attn.forward))
            attn.forward = make_wrapper(attn.forward)

        with torch.no_grad():
            clip_detector.clip.encode_image(input_tensor)
    finally:
        for attn, orig in saved:            # always restore the real forwards
            attn.forward = orig

    if not attn_weights:
        return None

    # Rollout: start from identity, fold in each layer's (attention + residual).
    seq = attn_weights[0].shape[-1]
    result = torch.eye(seq)
    for w in attn_weights:
        a = w[0].cpu()                      # [S, S] for the single image
        a = a + torch.eye(seq)              # residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    # CLS token is index 0; the rest are image patches laid out on a square grid.
    num_patches = seq - 1
    grid = int(round(num_patches ** 0.5))
    if grid * grid != num_patches:
        return None
    mask = result[0, 1:].reshape(grid, grid).numpy()

    cam = cv2.resize(mask, (input_tensor.shape[3], input_tensor.shape[2]))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    return cam


def overlay_heatmap(pil_image, heatmap):
    """
    Overlays the normalized heatmap onto the original PIL image.
    """
    img = np.array(pil_image)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(overlay)
