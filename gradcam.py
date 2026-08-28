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

def overlay_heatmap(pil_image, heatmap):
    """
    Overlays the normalized heatmap onto the original PIL image.
    """
    img = np.array(pil_image)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(overlay)
