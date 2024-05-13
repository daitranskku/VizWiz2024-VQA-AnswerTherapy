from typing import List
import sys
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViltConfig
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering
from inference_utils import *

sys.path.append('../polygon-transformer/')
sys.path.append('../polygon-transformer/fairseq')
from demo import visual_grounding

vizwiz_data_base_path = '../dataset'

viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'val')
viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'VizWiz_val.json')

pretrained_model = 'vilt-b32-finetuned-vqa'
model_folder = os.path.join('../ViLT', pretrained_model)
finetune_folder = '../ViLT/my_finetune/custom_vqa_vilt-b32-finetuned-vqa'

processor = ViltProcessor.from_pretrained(model_folder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

missing_words_in_vocab = []

COMBINED_IMAGE_DIR = '../test_dataset/combined_VQA_VIZWIZ'
question_image_data = '../test_dataset/combined_question_image.json'
question_image_data = json.load(open(question_image_data))
print(len(question_image_data['questions']))

num = 111
for i in range(20):
    i = i + num
    print("=====Input information: {}".format(i))
    question = question_image_data['questions'][i]
    print(question)
    image_path = os.path.join(COMBINED_IMAGE_DIR, question_image_data['images'][i])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
    print("Inference using finetune model")
    pred_answers: List[str] = inference_vilt(input_image=image,
                                             input_text=question,
                                             print_preds=False)
    predicted_grounding_masks: list = []
    image = Image.open(image_path)
    for i, ans in enumerate(pred_answers):
        # input
        input_text = question + " answer:" + ans
        # input_text = ans

        # input_text = question + " answer: sweet potato" + ans
        print(input_text)
        pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
        plt.imshow(pred_mask)
        plt.show()
        predicted_grounding_masks.append(pred_mask)
        # save to mask
    predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
    print("Final predicted label: ", predicted_label)
    # prediction
    break
print("Done")

## INFERENCE FOR SUBMISSION
naive_path = '../test_dataset/submission.json'
naive_data = json.load(open(naive_path))
results = []
# num = 694
maximum_length = 35
for i in tqdm(range(len(question_image_data['questions']))):
    # i = i + num
    raw_question = question_image_data['questions'][i]
    # print("=====Input information: {}".format(i))
    # print(raw_question)
    # print(question_image_data['images'][i])
    number_of_chars = len(raw_question)
    if number_of_chars > maximum_length:
        question = raw_question[:maximum_length]
        # print("Truncated question: ", question)
    else:

        question = raw_question
        # print("Original question: ", question)
    image_path = os.path.join(COMBINED_IMAGE_DIR, question_image_data['images'][i])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # visualize
    # plt.imshow(image)
    # plt.show()
    #
    pred_answers: List[str] = inference_vilt(input_image=image,
                                             input_text=question,
                                             print_preds=False)
    predicted_grounding_masks: list = []
    image = Image.open(image_path)
    for j, ans in enumerate(pred_answers):
        # input
        input_text = question + " answer:" + ans
        pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
        predicted_grounding_masks.append(pred_mask)
        # print(input_text)
        # plt.imshow(pred_overlayed)
        # plt.show()
        # save to mask
    predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
    temp = {}
    temp['question_id'] = question_image_data['images'][i]
    temp['single_grounding'] = 1 if predicted_label == 'single' else 0
    results.append(temp)

    # break
# save to json
with open('submission.json', 'w') as f:
    json.dump(results, f)
