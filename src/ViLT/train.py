import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViltConfig
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering
import matplotlib.pyplot as plt
import numpy as np
import cv2

vizwiz_data_base_path = '..'


viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'val')
viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'VizWiz_val.json')

vqa_datatrain_image_dir = os.path.join(vizwiz_data_base_path, 'coco/images/train2017')
vqa_datatrain_annotation_path = os.path.join(vizwiz_data_base_path, 'VQA_val.json')
pretrained_model = 'vilt-b32-finetuned-vqa'
model_folder = os.path.join('..', pretrained_model)
finetune_folder = os.path.join('..', "using__" + pretrained_model)

config = ViltConfig.from_pretrained(os.path.join(model_folder, "config.json"))
processor = ViltProcessor.from_pretrained(model_folder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

missing_words_in_vocab = []


def get_model():
    model = ViltForQuestionAnswering.from_pretrained(model_folder,
                                                     id2label=config.id2label,
                                                     label2id=config.label2id)
    model.to(device)

    return model


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
            # print("convert to RGB")
            image = image.convert("RGB")
        text = data['question']
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        # add labels
        labels = []
        for ans in data["answers"]:
            for c in ans.split():
                try:
                    labels.append(config.label2id[c])
                except:
                    # print(f"{c} is not exist in vocab ... ")
                    missing_words_in_vocab.append(c)
                    labels.append(0)
        scores = [0.4] * len(labels)

        # labels = annotation['labels']
        # scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))

        for label, score in zip(labels, scores):
            targets[label] = score
        encoding["labels"] = targets

        return encoding


def generate_dataset(file_path) -> VQADataset:
    dataframe = pd.read_json(file_path)

    dataframe = dataframe.to_dict(orient='records')
    dataset = VQADataset(data=dataframe,
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


def inference(input_image, input_text):
    print("Inference .....")
    processor = ViltProcessor.from_pretrained(finetune_folder)
    model = ViltForQuestionAnswering.from_pretrained(finetune_folder)

    # prepare inputs
    encoding = processor(input_image, input_text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]


def visualize_prediction(image, text, output_image_path="output.jpg"):
    """
    Visualize prediction
        image: PIL image
        text: prediction output string
    """
    print("Visualize text", text)
    image_array = np.array(image)
    if isinstance(image, Image.Image):
        # Convert the image array to BGR format for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    # Create a copy of the BGR image
    image_with_text = image_array.copy()

    padding_bottom = 80  # Adjust the padding size as needed

    # Add padding to the image
    height, width = image_with_text.shape[:2]
    padded_image = np.pad(image_with_text, ((0, padding_bottom), (0, 0), (0, 0)), mode='constant')

    # Set the font properties
    font_size = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    thickness = 1

    # Calculate the position to place the text
    text_width, text_height = cv2.getTextSize(text, font, font_size, thickness)[0]
    text_position1 = (10, height + 20)  # Position below the image

    # Draw the text on the image
    cv2.putText(padded_image, text, text_position1, font, font_size, text_color, thickness, cv2.LINE_AA)

    # Save the image to a file
    cv2.imwrite(output_image_path, padded_image)

    return output_image_path


if __name__ == "__main__":
    dataset_vqa = generate_dataset(vqa_datatrain_annotation_path)
    my_model = get_model()
    train_model(my_model, dataset_vqa, num_epochs=10)

