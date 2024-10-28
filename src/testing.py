import pandas as pd
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model import AttentionUNet
from custom_utils import predict_image, transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset_path = "/home/abdulla/Documents/Untitled Folder/data/Conglomerate Concrete Crack Detection/Test"

results_df = pd.read_csv("/home/abdulla/Documents/Untitled Folder/data/results.csv")
checkpoint_id = results_df["val_loss"].idxmin()

model_checkpoints_dir = f"/home/abdulla/Documents/Untitled Folder/data/checkpoints/model_{checkpoint_id}.pth"
model_state_dict = torch.load(model_checkpoints_dir, "cuda:0", weights_only=True)

test_results = {"IoU" : [], "sample_path" : []}

test_results_save_path = "/home/abdulla/Documents/Untitled Folder/data/test_results.csv"

model = AttentionUNet().to(device)
model.load_state_dict(model_state_dict)

image_names = os.listdir(f"{test_dataset_path}/images")

def iou(prediction, target):
    intersection = torch.logical_and(target, prediction)
    union = torch.logical_or(target, prediction)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score

with torch.no_grad():
    for image_name in tqdm(iter(image_names), maxinterval=len(image_names), desc="Testing"):
        prediction = predict_image(model, os.path.join(f"{test_dataset_path}/images", image_name), device)
        mask = transform(Image.open(os.path.join(f"{test_dataset_path}/masks", image_name)).convert("L"))
        iou_score = iou(prediction, mask).item()
        test_results["IoU"].append(iou_score)
        test_results["sample_path"].append(os.path.join(f"{test_dataset_path}/images", image_name))

# As some of the masks are empty, they result in IoU of 0. Hence, they must be excluded for averge IoU calculation
existing_iou_scores = []
for i in test_results["IoU"]:
    if i > 0:
        existing_iou_scores.append(i)
    
average_iou = sum(existing_iou_scores) / len(existing_iou_scores)
print(f"Average IoU score: {average_iou}")

pd.DataFrame(test_results).to_csv(test_results_save_path)