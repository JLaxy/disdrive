import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a photo of a person driving safely with both hands on the steering wheel",
                     " photo of a person texting using their mobile phone while driving", "a photo of a person with their head down while driving", "a photo of a person drinking while driving"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(logits_per_image.shape)

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
