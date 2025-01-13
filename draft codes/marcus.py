import os
import json
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

# Start timer
start_time = time.time()
print("Processing started...")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device configured successfully.")

# 1. Load Dataset
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Paths
train_path = r"C:\Users\genes\Desktop\Thesis\Facebook Hateful Meme Dataset\data\train.jsonl"
dev_path = r"C:\Users\genes\Desktop\Thesis\Facebook Hateful Meme Dataset\data\dev.jsonl"
image_folder = r"C:\Users\genes\Desktop\Thesis\Facebook Hateful Meme Dataset\data"

# Load and map datasets
train_data = load_jsonl(train_path)
dev_data = load_jsonl(dev_path)
print("Datasets loaded successfully.")

def map_image_paths(data, img_folder):
    for item in data:
        img_path = os.path.join(img_folder, item["img"])
        item["img_path"] = img_path if os.path.exists(img_path) else None
    return [item for item in data if item["img_path"]]

train_data = map_image_paths(train_data, image_folder)
dev_data = map_image_paths(dev_data, image_folder)
print("Image paths mapped successfully.")

# 2. Initialize CLIP Model and Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model and processor initialized successfully.")

# Prompts
PROMPT_HATEFUL = "A hateful meme containing racism, sexism, nationality, religion, and disability."
PROMPT_NON_HATEFUL = "A non-hateful meme that is good."

# 3. Dataset Class
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["img_path"]).convert("RGB")
        text = item["text"]

        # Process text and image
        processed = self.processor(
            text=[PROMPT_HATEFUL, PROMPT_NON_HATEFUL],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "label": torch.tensor(item["label"], dtype=torch.float),
        }

train_dataset = CLIPDataset(train_data, processor)
dev_dataset = CLIPDataset(dev_data, processor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=16)
print("Datasets prepared successfully.")

# 4. Evaluation Function with Cosine Similarity
def evaluate(model, dataloader, device, threshold=0.2):
    model.eval()
    all_labels = []
    all_preds = []
    all_similarities = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)  # Shape: [batch_size, 2, seq_len]
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Separate prompts for hateful and non-hateful
            input_ids_hateful = input_ids[:, 0, 🙂  # First prompt
            attention_mask_hateful = attention_mask[:, 0, 🙂

            input_ids_non_hateful = input_ids[:, 1, 🙂  # Second prompt
            attention_mask_non_hateful = attention_mask[:, 1, 🙂

            # Get embeddings for each prompt
            outputs_hateful = model(
                pixel_values=pixel_values,
                input_ids=input_ids_hateful,
                attention_mask=attention_mask_hateful,
            )
            outputs_non_hateful = model(
                pixel_values=pixel_values,
                input_ids=input_ids_non_hateful,
                attention_mask=attention_mask_non_hateful,
            )

            # Cosine similarity for each prompt
            similarities_hateful = torch.nn.functional.cosine_similarity(
                outputs_hateful.image_embeds, outputs_hateful.text_embeds, dim=1
            )
            similarities_non_hateful = torch.nn.functional.cosine_similarity(
                outputs_non_hateful.image_embeds, outputs_non_hateful.text_embeds, dim=1
            )

            # Compare similarities to threshold
            preds = (similarities_hateful > similarities_non_hateful).int()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_similarities.extend(zip(similarities_hateful.cpu().numpy(), similarities_non_hateful.cpu().numpy()))

    print("Evaluation completed successfully.")
    return all_labels, all_preds, all_similarities

# 5. Evaluation and Visualization
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Hateful", "Hateful"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    print("Confusion matrix plotted successfully.")

# 6. Run Evaluation
labels, preds, similarities = evaluate(model, dev_loader, device, threshold=0.2)

# Metrics
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average="binary", zero_division=0)
recall = recall_score(labels, preds, average="binary", zero_division=0)
f1 = f1_score(labels, preds, average="binary", zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Metrics calculated successfully.")

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"Time elapsed: {hours} hours, {minutes} minutes, {seconds} seconds")

# Confusion Matrix
plot_confusion_matrix(labels, preds)