from typing import List
import sys
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViltConfig
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering

import numpy as np
import cv2
import matplotlib.pyplot as plt

colors = [(255,0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (0, 255, 255), (255, 0, 255),
          (255, 255, 255), (128, 0, 0), (0, 128, 0),
          (0, 0, 128)]
thicknesses = [3,3,3, 3, 3, 3, 3, 3, 3, 3]
# Helper function
def is_single_groundings(predicted_grounding_masks: List[np.ndarray], threshold=0.3):
    if len(predicted_grounding_masks) == 0:
        return False
    reference_mask = predicted_grounding_masks[0]
    for mask in predicted_grounding_masks[1:]:
        if mask.shape != reference_mask.shape:
            return False
        intersection = np.logical_and(mask, reference_mask)
        union = np.logical_or(mask, reference_mask)
        iou = np.sum(intersection) / np.sum(union)
        if iou < threshold:
            return False
    return True

def draw_boundaries(image, mask, color=(55, 0, 200)):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundaries_image = cv2.drawContours(image.copy(), contours, -1, color, 2)
    return boundaries_image

def coordinates_to_mask(coordinates, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(coordinates, dtype=np.int32)
    cv2.fillPoly(mask, [points], color=255)
    return mask

def calculate_iou_mask(ground_truth_mask, predicted_mask):
    intersection = np.logical_and(ground_truth_mask, predicted_mask)
    union = np.logical_or(ground_truth_mask, predicted_mask)
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    iou = intersection_count / union_count
    return iou


def plot_grid(result_images, num_rows, num_cols=1):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8 * num_rows))
    for i, ax in enumerate(axes.flat):
        fig_header = "Prediction"
        ax.imshow(result_images[i])
        ax.set_title(fig_header, fontsize=10, pad=5)
        ax.axis('off')
    fig.tight_layout()
    figure_path = './eval_samples.png'
    plt.savefig(figure_path)


class VQADataset(torch.utils.data.Dataset):
    """VQA dataset."""

    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get image + text
        data = self.data[idx]

        image_id = str(data["image_id"])
        if "jpg" not in image_id:
            image_id = image_id.zfill(12) + '.jpg'
        img_path = os.path.join(vqa_datatrain_image_dir, image_id)
        image = Image.open(img_path)
        if image.mode == "L":
            # If image is in grayscale, convert to RGB
            image = image.convert("RGB")
        text = data['question']
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        # add labels
        labels = []
        for ans in data["answers"]:
            labels.append(config.label2id[ans])
        scores = [0.8] * len(labels)

        # labels = annotation['labels']
        # scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))

        for label, score in zip(labels, scores):
            targets[label] = score
        encoding["labels"] = targets

        return encoding


def handle_dataset(df):
    print("Handle dataset .... ")
    label_set = set()
    for ans_list in df["answers"]:
        for ans in ans_list:
            label_set.add(ans)
    unique_labels = list(label_set)
    config.label2id = {label: idx for idx, label in enumerate(unique_labels)}
    config.id2label = {idx: label for label, idx in config.label2id.items()}


def generate_dataset(file_path) -> VQADataset:
    dataframe = pd.read_json(file_path)

    dataframe_dict = dataframe.to_dict(orient='records')
    handle_dataset(dataframe)

    dataset = VQADataset(data=dataframe_dict,
                         processor=processor)

    return dataset


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)

    return batch


def train_model(model, dataset, num_epochs=50):
    train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Epoch: {epoch}")
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader):
            # get the inputs;
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            epoch_loss = loss.item()
            print("Loss:", epoch_loss)
            epoch_loss += epoch_loss
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_dataloader)
        # Store loss value
        losses.append(epoch_loss)

        # Plot and save the loss graph
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('loss_graph.png')
    print("Total missing words", len(missing_words_in_vocab))

    # Save the model weights
    model.save_pretrained(finetune_folder)
    # Save the feature extractor
    processor.save_pretrained(finetune_folder)

def get_model():
    print("Get model from pretrained: ", model_folder)
    model = ViltForQuestionAnswering.from_pretrained(
        pretrained_model_name_or_path=model_folder,
        id2label=config.id2label,
        label2id=config.label2id,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    model.to(device)

    return model
#     dataset_vqa = generate_dataset(vqa_datatrain_annotation_path)
#     my_model = get_model()
#     train_model(my_model, dataset_vqa, 10)

# def inference_vilt(input_image, input_text, threshold=0.1, print_preds=True):
#     print("Inference finetune.....")
#     fintune_processor = ViltProcessor.from_pretrained(finetune_folder)
#     model = ViltForQuestionAnswering.from_pretrained(finetune_folder)
#
#     # prepare inputs
#     encoding = fintune_processor(input_image, input_text, return_tensors="pt")
#
#     # forward pass
#     outputs = model(**encoding)
#     logits = outputs.logits
#     predicted_classes = torch.sigmoid(logits)
#     probs, classes = torch.topk(predicted_classes, 5)
#
#     results = []
#     for prob, class_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
#         if prob >= threshold:
#             results.append(model.config.id2label[class_idx])
#         if print_preds:
#             print(prob, model.config.id2label[class_idx])
#
#     if not results:
#         class_idx = logits.argmax(-1).item()
#         results.append(model.config.id2label[class_idx])
#     return results

def inference_vilt(input_image, input_text, threshold=0.01, print_preds=True):
    # print("Inference finetune.....")
    fintune_processor = ViltProcessor.from_pretrained(finetune_folder)
    model = ViltForQuestionAnswering.from_pretrained(finetune_folder)
    # prepare inputs
    encoding = fintune_processor(input_image, input_text, return_tensors="pt")
    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    predicted_classes = torch.sigmoid(logits)
    probs, classes = torch.topk(predicted_classes, 5)
    results = []
    num_detected = 0
    for prob, class_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
        if prob >= threshold:
            results.append(model.config.id2label[class_idx])
            num_detected += 1
            if num_detected >= 3:
                break
    if not results:
        try:
            class_idx = logits.argmax(-2).item()
        except:
            class_idx = logits.argmax(-1).item()
        results.append(model.config.id2label[class_idx])
    return results
