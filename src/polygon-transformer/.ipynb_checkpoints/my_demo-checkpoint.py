import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List
from PIL import Image
from tqdm import tqdm

from demo import visual_grounding

FONTCOLOR = [(0, 69, 255),
             (255, 0, 0),
             (0, 255, 0),
             (0, 0, 255),
             (0, 0, 0),
             (200, 55, 0),
             (55, 200, 0),
             (0, 200, 55),
             (0, 55, 200),
             (200, 0, 55),
             (55, 0, 200)]


def draw_boundaries(image, mask, color=(55, 0, 200)):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert the mask to uint8 if it's in a different dtype
    mask = mask.astype(np.uint8)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the boundaries on the image
    boundaries_image = cv2.drawContours(image.copy(), contours, -1, color, 2)

    return boundaries_image


def coordinates_to_mask(coordinates, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    # Convert the coordinates to a NumPy array
    points = np.array(coordinates, dtype=np.int32)
    # Draw a filled polygon on the mask image
    cv2.fillPoly(mask, [points], color=255)
    return mask


def calculate_iou_mask(ground_truth_mask, predicted_mask):
    intersection = np.logical_and(ground_truth_mask, predicted_mask)
    union = np.logical_or(ground_truth_mask, predicted_mask)
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    iou = intersection_count / union_count
    return iou


def visualize_prediction(image, question: str, answer_texts: List[str], predicted_grounding_masks: List[np.ndarray],
                         gt_masks: List[np.ndarray], binary_label: str, save_img=False, fixed_width=600):
    """
    Visualize prediction for each answers and their boundaries
        image: PIL image
        text: prediction output string
    """
    image_array = np.array(image)
    if isinstance(image, Image.Image):
        # Convert the image array to BGR format for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    IoU_values = list()

    for idx, ans in enumerate(answer_texts):
        # Draw grounding labels
        font_color = FONTCOLOR[idx]
        mask = predicted_grounding_masks[idx]
        gt_mask = gt_masks[idx]
        image_array = draw_boundaries(image_array, mask, color=font_color)
        image_array = draw_boundaries(image_array, gt_mask, color=font_color)
        IoU_values.append(calculate_iou_mask(gt_mask, mask))

    # Create a copy of the BGR image
    image_with_text = image_array.copy()
    height, width = image_with_text.shape[:2]
    new_height = int(fixed_width * height / width)
    image_with_text = cv2.resize(image_with_text, (fixed_width, new_height))

    padding_bottom = 120  # Adjust the padding size as needed
    padding_bottom = init_pad_answer * len(answer_texts) + 100  # Adjust the padding size as needed

    # Add padding to the image
    height, width = image_with_text.shape[:2]
    padded_image = np.pad(image_with_text, ((0, padding_bottom), (0, 0), (0, 0)), mode='constant', constant_values=255)

    # Set the font properties
    font_size = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    question_text_color = (0, 0, 0)
    thickness = 1

    # Calculate the position to place the text
    question_position = (10, height + 20)  # Position below the image

    # Draw the text on the image
    cv2.putText(padded_image, question, question_position, font, font_size, question_text_color, thickness, cv2.LINE_AA)

    init_pad_answer = 40
    for idx, ans in enumerate(answer_texts):
        font_color = FONTCOLOR[idx]
        text_position = (10, height + init_pad_answer)  # Position below the question
        ans_text = f"{str(ans)}  - IoU: {str(round(IoU_values[idx], 7))}"
        cv2.putText(padded_image, f">> (GT) {ans_text}", text_position, font, font_size, font_color, thickness,
                    cv2.LINE_AA)
        init_pad_answer += 30

    text_position = (10, height + init_pad_answer)  # Position below the question
    cv2.putText(padded_image, f"*Label (GT) {str(binary_label)}", text_position, font, font_size, question_text_color,
                thickness, cv2.LINE_AA)

    if save_img:
        output_image_path = f'{text}.jpg'
        # Save the image to a file
        cv2.imwrite(output_image_path, padded_image)

    return padded_image


def plot_grid(result_images, num_rows, num_cols=1):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8 * num_rows))

    # Plot each image in a subplot
    for i, ax in enumerate(axes.flat):
        fig_header = "Prediction"
        ax.imshow(result_images[i])  # Display the image
        ax.set_title(fig_header, fontsize=10, pad=5)
        ax.axis('off')  # Turn off axis labels

    # Adjust spacing between subplots
    fig.tight_layout()

    # Save the figure as an image
    figure_path = './eval_samples.png'  # Path to save the figure
    plt.savefig(figure_path)


def inference_samples(sample_sets: List[dict]):
    result_images = []
    for row in sample_sets:
        image_id = row['image_id']
        question = row['question']
        binary_label = row['binary_label']
        answers: List[str] = row['answers']
        grounding_labels: list = row['grounding_labels']

        img_path = os.path.join('/DATA2/Han/VizWiz/dataset/val', image_id)
        input_image = Image.open(img_path)
        gt_masks = []

        for gl in grounding_labels:
            coordinates = [(item['x'], item['y']) for item in gl]
            height, width = np.array(input_image).shape[:2]
            gt_masks.append(np.array(coordinates_to_mask(coordinates, (height, width))))

        predicted_grounding_masks: list = []
        for ans in answers:
            pred_overlayed, pred_mask = visual_grounding(image=input_image, text=ans)
            predicted_grounding_masks.append(pred_mask)

        visualize_result = visualize_prediction(
            image=input_image,
            question=question,
            answer_texts=answers,
            predicted_grounding_masks=predicted_grounding_masks,
            gt_masks=gt_masks,
            binary_label=binary_label)
        result_images.append(visualize_result)

    plot_grid(result_images, len(sample_sets))


if __name__ == '__main__':
    # img_path = '/DATA2/Han/VizWiz/ViLT/sample_image.jpg'
    # input_image = Image.open(img_path)
    # description = 'a cow'

    # pred_overlayed, pred_mask = visual_grounding(image=input_image, text=description)
    # print("pred_mask ",pred_overlayed)

    # # Draw the boundaries on the image
    # # Convert PIL image to OpenCV format
    # # boundaries_image = draw_boundaries(input_image, pred_mask)
    # boundaries_image = draw_boundaries(pred_overlayed, pred_mask)
    # output_file = "result_with_boundaries_2.jpg"
    # # cv2.imwrite(output_file, boundaries_image)
    # cv2.imwrite(output_file, boundaries_image)

    df = pd.read_json('/DATA2/Han/VizWiz/dataset/VizWiz_val.json')
    sample_df = df.sample(n=2, random_state=33)
    dataframe_dict = sample_df.to_dict(orient='records')
    inference_samples(dataframe_dict)

