import torch
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==============================
# Load Images
# ==============================
image1_path = input("Enter path of Image 1: ")
image2_path = input("Enter path of Image 2: ")

image1 = Image.open(image1_path).convert("RGB")
image2 = Image.open(image2_path).convert("RGB")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CLIP Embeddings
# ============================================================
print("\nLoading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    # Normalize embedding
    features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0]

clip_emb1 = get_clip_embedding(image1)
clip_emb2 = get_clip_embedding(image2)

print("\n========== CLIP Embeddings ==========")
print("\nImage 1 CLIP Embedding:")
print(clip_emb1)

print("\nImage 2 CLIP Embedding:")
print(clip_emb2)

clip_similarity = cosine_similarity(
    clip_emb1.reshape(1, -1),
    clip_emb2.reshape(1, -1)
)[0][0]

# ============================================================
# ResNet50 Embeddings
# ============================================================
print("\nLoading ResNet50 model...")

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove classification layer
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_resnet_embedding(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(img_tensor)

    embedding = embedding.squeeze().cpu().numpy()

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

resnet_emb1 = get_resnet_embedding(image1)
resnet_emb2 = get_resnet_embedding(image2)

print("\n========== ResNet50 Embeddings ==========")

print("\nImage 1 ResNet50 Embedding:")
print(resnet_emb1)

print("\nImage 2 ResNet50 Embedding:")
print(resnet_emb2)

resnet_similarity = cosine_similarity(
    resnet_emb1.reshape(1, -1),
    resnet_emb2.reshape(1, -1)
)[0][0]

# ============================================================
# Results
# ============================================================
print("\n" + "="*50)
print("COSINE SIMILARITY RESULTS")
print("="*50)

print(f"CLIP Similarity     : {clip_similarity:.4f}")
print(f"ResNet50 Similarity : {resnet_similarity:.4f}")

# Percentage form
print("\nSimilarity Percentage:")
print(f"CLIP     : {clip_similarity*100:.2f}%")
print(f"ResNet50 : {resnet_similarity*100:.2f}%")
