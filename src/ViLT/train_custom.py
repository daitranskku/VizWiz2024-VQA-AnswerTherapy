import os
import pandas as pd
from datetime import datetime
import torch
import copy
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViltConfig
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List


files = os.listdir('..')
val_files = os.listdir('...')



vizwiz_data_base_path = '..'


vizwiz_datatrain_image_dir = os.path.join(vizwiz_data_base_path, 'train') 
vizwiz_datatrain_annotation_path =os.path.join(vizwiz_data_base_path, 'VizWiz_train.json')

vqa_datatrain_image_dir = os.path.join(vizwiz_data_base_path, 'train2017') 
vqa_datatrain_annotation_path =os.path.join(vizwiz_data_base_path, 'VQA_train.json')

eval_image_dir = os.path.join(vizwiz_data_base_path, 'val') 
eval_annotation_path = os.path.join(vizwiz_data_base_path, 'VizWiz_val.json')




pretrained_model = 'vilt-b32-finetuned-vqa'
model_folder = os.path.join('..', pretrained_model)
finetune_folder = os.path.join('..', "custom_vqa_" + pretrained_model)



class CustomConfig:
    id2label: dict
    label2id: dict


# config = CustomConfig()

config = ViltConfig.from_pretrained(model_folder)
processor = ViltProcessor.from_pretrained(model_folder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

missing_words_in_vocab = []




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


class VQADataset(torch.utils.data.Dataset):
    """VQA dataset."""

    def __init__(self, processor):
        
        dataframe = pd.read_json(vqa_datatrain_annotation_path)
        dataframe_dict = dataframe.to_dict(orient='records')
#         handle_dataset(dataframe)
#         print("Total dataframe_dict ", len(dataframe_dict))
        
        print("Init VQA dataset ...")
        self.data = self.init_data(dataframe_dict)
        print("VQA Dataset size: ", self.__len__())
        self.processor = processor
        self.datatrain_image_dir = vqa_datatrain_image_dir
        
    
    def init_data(self, data):
        vqa_data = []
        for sample in data:
            valid_labels = []
            for ans in sample["answers"]:
                if ans in list(config.label2id.keys()):
                    valid_labels.append(ans)
            if len(valid_labels) > 0:
                vqa_data.append(sample)
        return vqa_data
            
            

    def __len__(self):
        return len(self.data)
    
    def get_score(self, count: int) -> float:
        return min(1.0, count / 2)

    def __getitem__(self, idx):
        # get image + text
        data = self.data[idx]

        image_id = str(data["image_id"])
        if "jpg" not in image_id:
            image_id = image_id.zfill(12) + '.jpg'


        img_path = os.path.join(self.datatrain_image_dir, image_id)
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
        answer_count = dict()
        for answer_ in data["answers"]:
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        scores = []
        for ans in data["answers"]:
            if ans not in list(config.label2id.keys()):
                missing_words_in_vocab.append(ans)
                continue
            labels.append(config.label2id[ans])
            score = self.get_score(answer_count[ans])
            scores.append(score)
        if len(labels) == 0:
            print("Exception lable leng = 0, exit")
            exit(0)

        # labels = annotation['labels']
        # scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))

        for label, score in zip(labels, scores):
            targets[label] = score
        encoding["labels"] = targets

        return encoding



class VizWizDataset(torch.utils.data.Dataset):
    """VizWiz dataset."""

    def __init__(self, processor):
        dataframe = pd.read_json(vizwiz_datatrain_annotation_path)
        dataframe_dict = dataframe.to_dict(orient='records')
#         handle_dataset(dataframe)
#         print("Total dataframe_dict ", len(dataframe_dict))
        print("Init VizWiz dataset ...")
        self.data = self.init_data(dataframe_dict)
        print("VizWiz Dataset size: ", self.__len__())
        self.processor = processor
        self.datatrain_image_dir = vizwiz_datatrain_image_dir
        
    
    def init_data(self, data):
        vqa_data = []
        for sample in data:
            valid_labels = []
            for ans in sample["answers"]:
                if ans in list(config.label2id.keys()):
                    valid_labels.append(ans)
            if len(valid_labels) > 0:
                vqa_data.append(sample)
        return vqa_data
            

    def __len__(self):
        return len(self.data)
    
    def get_score(self, count: int) -> float:
        return min(1.0, count / 2)

    def __getitem__(self, idx):
        # get image + text
        data = self.data[idx]

        image_id = str(data["image_id"])
        if "jpg" not in image_id:
            image_id = image_id.zfill(12) + '.jpg'


        img_path = os.path.join(self.datatrain_image_dir, image_id)
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
        answer_count = dict()
        for answer_ in data["answers"]:
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        scores = []
        for ans in data["answers"]:
            if ans not in list(config.label2id.keys()):
                missing_words_in_vocab.append(ans)
                continue
            labels.append(config.label2id[ans])
            score = self.get_score(answer_count[ans])
            scores.append(score)
        if len(labels) == 0:
            print("Exception lable leng = 0, exit")
            exit(0)

        # labels = annotation['labels']
        # scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))

        for label, score in zip(labels, scores):
            targets[label] = score
        encoding["labels"] = targets

        return encoding

    
class CombinedVQADataset(torch.utils.data.Dataset):
    def __init__(self, dataset1:VQADataset, dataset2:VizWizDataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
        print("CombinedVQADatasetsize: ", self.__len__())
    def __getitem__(self, index):
        if index < len(self.dataset1):
            return self.dataset1[index]
        else:
            return self.dataset2[index - len(self.dataset1)]

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    
    
    

def handle_dataset(df):
    print("Handle dataset .... ")
    label_set = set()
    for ans_list in df["answers"]:
        for ans in ans_list:
            label_set.add(ans)
    unique_labels = list(label_set)
#     pretrained_model = ViltConfig.from_pretrained(model_folder)
#     pretrained_config = copy.deepcopy(pretrained_model.config)
    pretrained_config = copy.deepcopy(ViltConfig.from_pretrained(model_folder))
    existing_id2label = pretrained_config.id2label
    existing_label2id = pretrained_config.label2id
    max_label_id = len(existing_label2id.values())
    
#     for label in unique_labels:
#         if existing_label2id.get(label) is None:
#             existing_label2id[label] = max_label_id
#             existing_id2label[max_label_id] = label
#             max_label_id += 1
            
    config.label2id = existing_label2id
    config.id2label = existing_id2label
    print(len(config.label2id.keys()))
    
#     config.label2id = {label: idx for idx, label in enumerate(unique_labels)}
#     config.id2label = {idx: label for label, idx in config.label2id.items()}


def generate_dataset() -> CombinedVQADataset:
#     dataframe = pd.read_json(file_path)
#     dataframe_dict = dataframe.to_dict(orient='records')
#     handle_dataset(dataframe)
#     print("Total dataframe_dict ", len(dataframe_dict))
    dataset1 = VQADataset(processor=processor)
    dataset2 = VizWizDataset(processor=processor)
    dataset = CombinedVQADataset(dataset1, dataset2)


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
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"Epoch: {epoch} -- Current time: {current_time}")
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


def inference(input_image, input_text, threshold=0.2, print_preds=True):
    print("Inference .....")
    processor = ViltProcessor.from_pretrained(finetune_folder)
    model = ViltForQuestionAnswering.from_pretrained(finetune_folder)

    # prepare inputs
    encoding = processor(input_image, input_text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    predicted_classes = torch.sigmoid(logits)
    probs, classes = torch.topk(predicted_classes, 5)

    results = []
    for prob, class_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
        if prob >= threshold:
            results.append(model.config.id2label[class_idx])
        if print_preds:
            print(prob, model.config.id2label[class_idx])

    if not results:
        class_idx = logits.argmax(-1).item()
        results.append(model.config.id2label[class_idx])
    return results


def inference_pretrained(input_image, input_text, threshold=0.2, print_preds=True):
    print("Inference pretrained model.....")
    processor = ViltProcessor.from_pretrained(model_folder)
    model = ViltForQuestionAnswering.from_pretrained(model_folder)

    # prepare inputs
    encoding = processor(input_image, input_text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    predicted_classes = torch.sigmoid(logits)
    probs, classes = torch.topk(predicted_classes, 5)
    results = []
    for prob, class_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
        if prob >= threshold:
            results.append(model.config.id2label[class_idx])
        if print_preds:
            print(prob, model.config.id2label[class_idx])

    if not results:
        class_idx = logits.argmax(-1).item()
        results.append(model.config.id2label[class_idx])
    return results


def visualize_prediction(image, question: str, answer_texts: List[str], save_img=False, fixed_width=600):
    """
    Visualize prediction
        image: PIL image
        text: prediction output string
    """
    init_pad_answer = 40
    image_array = np.array(image)
    if isinstance(image, Image.Image):
        # Convert the image array to BGR format for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    # Create a copy of the BGR image
    image_with_text = image_array.copy()
    height, width = image_with_text.shape[:2]
    new_height = int(fixed_width * height / width)
    image_with_text = cv2.resize(image_with_text, (fixed_width, new_height))

    padding_bottom = 120  # Adjust the padding size as needed

    # Add padding to the image
    height, width = image_with_text.shape[:2]
    padded_image = np.pad(image_with_text, ((0, padding_bottom), (0, 0), (0, 0)), mode='constant')

    # Set the font properties
    font_size = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    thickness = 1

    # Calculate the position to place the text
    question_position = (10, height + 20)  # Position below the image

    # Draw the text on the image
    cv2.putText(padded_image, question, question_position, font, font_size, text_color, thickness, cv2.LINE_AA)


    for ans in answer_texts:
        text_position = (10, height + init_pad_answer)  # Position below the question
        cv2.putText(padded_image, f">> {str(ans)}", text_position, font, font_size, text_color, thickness, cv2.LINE_AA)
        init_pad_answer += 30

    if save_img:
        output_image_path = f'{text}.jpg'
        # Save the image to a file
        cv2.imwrite(output_image_path, padded_image)

    return padded_image


def eval_samples(sample_sets: List[dict]):
    result_images = []
    for sample in sample_sets:
        image_id = sample['image_id']
        question = sample['question']

        img_path = os.path.join(eval_image_dir, image_id)
        input_image = Image.open(img_path)

        result_pretrained: List[str] = inference_pretrained(input_image, question, print_preds=False)
        result_finetune: List[str] = inference(input_image, question, print_preds=False)
        ground_truth: List[str] = sample['answers']

        visualize_pretrained = visualize_prediction(input_image, question, result_pretrained)
        visualize_finetune = visualize_prediction(input_image, question, result_finetune)
        visualize_GT = visualize_prediction(input_image, question, ground_truth)

        result_images.append(visualize_pretrained)
        result_images.append(visualize_finetune)
        result_images.append(visualize_GT)

    # Create a grid of subplots
    num_rows = len(sample_sets)
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8 * num_rows))

    # Plot each image in a subplot
    for i, ax in enumerate(axes.flat):
        fig_header = ''
        if i % 3 == 0:
            fig_header = 'Pretrained'
        if i % 3 == 1:
            fig_header = 'Finetune'
        if i % 3 == 2:
            fig_header = 'Ground truth'
        ax.imshow(result_images[i])  # Display the image
        ax.set_title(fig_header, fontsize=10, pad=5)
        ax.axis('off')  # Turn off axis labels

    # Adjust spacing between subplots
    fig.tight_layout()

    # Save the figure as an image
    figure_path = './eval_samples.png'  # Path to save the figure
    plt.savefig(figure_path)


if __name__ == "__main__":
    dataset = generate_dataset()
    my_model = get_model()
    train_model(my_model, dataset, 10)

    df = pd.read_json(eval_annotation_path)
    sample_df = df.sample(n=4)
    dataframe_dict = sample_df.to_dict(orient='records')
    eval_samples(dataframe_dict)

