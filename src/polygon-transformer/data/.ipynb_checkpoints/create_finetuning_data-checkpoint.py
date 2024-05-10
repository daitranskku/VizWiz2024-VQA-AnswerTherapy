import sys
sys.path.append('./refer')
from refer import REFER
import numpy as np
from PIL import Image
import cv2
import random
import os
from tqdm import tqdm

import pickle
from poly_utils import is_clockwise, revert_direction, check_length, reorder_points, \
    approximate_polygons, interpolate_polygons, image_to_base64, polygons_to_string


max_length = 400

data_root = './refer/data'
datasets = ['refcoco', 'refcoco+', 'refcocog']

image_dir = './datasets/images/mscoco/train2014'
val_test_files = pickle.load(open("data/val_test_files.p", "rb"))

combined_train_data = []


def coordinates_to_mask(coordinates, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    # Convert the coordinates to a NumPy array
    points = np.array(coordinates, dtype=np.int32)
    # Draw a filled polygon on the mask image
    cv2.fillPoly(mask, [points], color=255)
    return mask


def get_mask_custom(segmentation: list, width: int, height: int) -> Image:
    n = len(segmentation) // 2
    xs = segmentation[::2]
    ys = segmentation[1::2]
    coordinates = []
    for i in range(n):
        coordinates.append((xs[i], ys[i]))
    mask = coordinates_to_mask(
        coordinates=coordinates,
        image_shape=(height, width)
    )
    # Convert the mask array to a PIL image
    pil_image = Image.fromarray(mask, mode="P")
    return pil_image


for dataset in datasets:
    if dataset == 'refcoco':
        splits = ['train', 'val', 'testA', 'testB']
        splitBy = 'unc'
    elif dataset == 'refcoco+':
        splits = ['train', 'val', 'testA', 'testB']
        splitBy = 'unc'
    elif dataset == 'refcocog':
        splits = ['train', 'val']
        splitBy = 'umd'

    save_dir = f'datasets/finetune/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    for split in splits:
        num_pts = []
        max_num_pts = 0
        file_name = os.path.join(save_dir, f"{dataset}_{split}.tsv")
        print("creating ", file_name)

        uniq_ids = []
        image_ids = []
        sents = []
        coeffs_strings = []
        img_strings = []

        writer = open(file_name, 'w')
        refer = REFER(data_root, dataset, splitBy)

        ref_ids = refer.getRefIds(split=split)

        for this_ref_id in tqdm(ref_ids):
            this_img_id = refer.getImgIds(this_ref_id)
            this_img = refer.Imgs[this_img_id[0]]
            fn = this_img['file_name']
            img_id = fn.split(".")[0].split("_")[-1]
            print("img_id ", img_id)

            # load image
#             img = Image.open(os.path.join(image_dir, this_img['file_name'])).convert("RGB")

            # convert image to string
#             img_base64 = image_to_base64(img, format='jpeg')

            # load mask
            ref = refer.loadRefs(this_ref_id)
            ref_mask = np.array(refer.getMask(ref[0])['mask'])
            annot = np.zeros(ref_mask.shape)
            annot[ref_mask == 1] = 1  # 255
            annot_img = Image.fromarray(annot.astype(np.uint8), mode="P")
            annot_img.save(f"./annot_img_{img_id}.png")
            annot_base64 = image_to_base64(annot_img, format='png')

            polygons = refer.getPolygon(ref[0])['polygon']

            polygons_processed = []
            i = 1
            for polygon in polygons:
                mask_custom = get_mask_custom(polygon, ref_mask.shape[1], ref_mask.shape[0])
                mask_custom.save(f"./annot_img_{img_id}_mask_{i}.png")
                i+=1
                # make the polygon clockwise
                if not is_clockwise(polygon):
                    polygon = revert_direction(polygon)

                # reorder the polygon so that the first vertex is the one closest to image origin
                polygon = reorder_points(polygon)
                polygons_processed.append(polygon)

            polygons = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
            polygons_interpolated = interpolate_polygons(polygons)

            polygons = approximate_polygons(polygons, 5, max_length)

            pts_string = polygons_to_string(polygons)
            pts_string_interpolated = polygons_to_string(polygons_interpolated)

            # load box
            box = refer.getRefBox(this_ref_id)  # x,y,w,h
            x, y, w, h = box
            box_string = f'{x},{y},{x + w},{y + h}'

            max_num_pts = max(max_num_pts, check_length(polygons))

            num_pts.append(check_length(polygons))
            print("polygons", polygons)
            print("box_string", box_string)
            print("pts_string", pts_string)
            print("pts_string_interpolated", pts_string_interpolated)

            # load text
            ref_sent = refer.Refs[this_ref_id]
            for i, (sent, sent_id) in enumerate(zip(ref_sent['sentences'], ref_sent['sent_ids'])):
                uniq_id = f"{this_ref_id}_{i}"
                instance = '\t'.join(
                    [uniq_id, str(this_img_id[0]), sent['sent'], box_string, pts_string, img_base64, annot_base64,
                     pts_string_interpolated]) + '\n'
                writer.write(instance)

                if img_id not in val_test_files and split == 'train':  # filtered out val/test files
                    combined_train_data.append(instance)
                    
        exit(0)
        writer.close()

random.shuffle(combined_train_data)
file_name = os.path.join("datasets/finetune/refcoco+g_train_shuffled.tsv")
print("creating ", file_name)
writer = open(file_name, 'w')
writer.writelines(combined_train_data)
writer.close()




