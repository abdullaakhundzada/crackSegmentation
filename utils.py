import torch
from torch import nn
from model import AttentionGate
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def resize_tensor_transform(target_shape : tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(target_shape, 
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()])

def print_model_summary(model, input_size=(3, 256, 256)):
    def hook(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        input_shape = str(tuple(input[0].shape))
        output_shape = str(tuple(output.shape))
        print(f"{class_name:20} | Input: {input_shape:20} | Output: {output_shape:20}")

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, AttentionGate)):
            hooks.append(module.register_forward_hook(hook))
    
    x = torch.randn((1, *input_size)) # dummy input
    print("\nModel Feature Dimensions:")
    print("-" * 90)
    print(f"{'Layer':20} | {'Input Shape':35} | {'Output Shape':35}")
    print("-" * 90)
    
    model.to("cpu")
    model(x)
    
    for hook in hooks:
        hook.remove()

def visualize_attention(model, image_tensor, device):
    """
    Visualize attention maps for a given image
    """
    model.eval()
    with torch.no_grad():
        # Get attention maps
        attention_maps = []
        def hook_fn(module, input, output):
            attention_maps.append(output.cpu().numpy())
        
        # Register hooks for attention gates
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, AttentionGate):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        _ = model(image_tensor.to(device))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps

def apply_kmeans(prediction, n_clusters=2):
    """
    Apply K-means clustering to refine the model"s predictions
    """
    # Convert prediction to numpy array
    pred_np = prediction.cpu().numpy().reshape(-1, 1)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pred_np)
    
    # Reshape back to original dimensions
    refined_mask = clusters.reshape(prediction.shape)
    
    # Convert to binary mask (assuming crack cluster has higher mean value)
    cluster_means = [np.mean(pred_np[clusters == i]) for i in range(n_clusters)]
    crack_cluster = np.argmax(cluster_means)
    binary_mask = (refined_mask == crack_cluster).astype(np.float32)
    
    return torch.from_numpy(binary_mask)

def predict_image(model, image_path, device):
    """
    Predict cracks in a single image with K-means refinement
    """
    # Load and preprocess image
    # TODO: fix the transform in the prediction function. 
    # TODO: build a unified pipeline for the image pre-processing and past-processing
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
        # Apply K-means refinement
        refined_prediction = apply_kmeans(prediction.squeeze())
    
    return refined_prediction

def predict_and_visualize(model, image_path, device):
    """
    Predict cracks and visualize attention maps
    """
    # Load and preprocess image
    # TODO: remove the old transform method, use unified pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    # Get prediction and attention maps
    attention_maps = visualize_attention(model, image_tensor, device)
    prediction = predict_image(model, image_path, device)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(prediction.cpu().numpy(), cmap="gray")
    axes[0, 1].set_title("Prediction")
    
    # Display attention maps
    for i, attention_map in enumerate(attention_maps[:3]):
        axes[1, i].imshow(attention_map[0, 0], cmap="gray")
        axes[1, i].set_title(f"Attention Map {i+1}")
    
    plt.tight_layout()
    plt.show()