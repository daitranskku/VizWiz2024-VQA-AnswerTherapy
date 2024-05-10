import json
import os
from tqdm import tqdm
import random
import pickle

# set up image paths
imgsfile = dict(
    coco='mscoco/train2014',
    vg='visual-genome',
    saiaprtc12='saiaprtc12',
    flickr='flickr30k'
)
vizwiz_data_base_path = '/DATA2/Han/VizWiz/dataset'
viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'val')
viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'VizWiz_val.json')

# load annotation files
f = open(viz_wiz_data_train_annotation_path)
print("Loading annotation file")
data = json.load(f)
f.close()

# load the validation and test image list of refcoco, refcoco+, and refcocog
# val_test_files = pickle.load(open("data/val_test_files.p", "rb"))

# create result folder
os.makedirs("datasets/pretrain_vizwiz", exist_ok=True)

# generate training tsv file
tsv_filename = "datasets/pretrain/train_shuffled_vizwiz.tsv"
writer = open(tsv_filename, 'w')
print("generating ", tsv_filename)


def find_bbox(segmentation_points):
    # Extract the x and y coordinates into separate lists
    x_coordinates, y_coordinates = zip(*segmentation_points)

    # Find the minimum and maximum coordinates
    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)

    # Create the bounding box coordinates
    bounding_box = (min_x, min_y, max_x, max_y)
    return   bounding_box



lines = []
for i, data_i in enumerate(tqdm(data)):
    data_source = data_i['data_source']

    image_id = data_i['image_id']
    ground_labels = data_i['ground_labels']
    answers = data_i['answers']
    filepath = os.path.join(viz_wiz_data_train_image_dir, image_id)
    for _i, ans in enumerate(answers):
        ans = str(ans)
        bbox = find_bbox(ground_labels[_i])
        x, y, w, h = bbox
        box_string = f'{x},{y},{x + w},{y + h}'
        line = '\t'.join([str(i), ans.replace('\n', ''), box_string, filepath]) + '\n'
        lines.append(line)

# shuffle the training set
random.shuffle(lines)

# write training tsv file
writer.writelines(lines)
writer.close()

