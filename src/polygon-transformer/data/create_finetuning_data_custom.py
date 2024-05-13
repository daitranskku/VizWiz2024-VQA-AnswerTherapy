import random
import os
from tqdm import tqdm
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw


from poly_utils import is_clockwise, revert_direction, check_length, reorder_points, \
    approximate_polygons, interpolate_polygons, image_to_base64, polygons_to_string

vizwiz_data_base_path = '../dataset'
viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'train')
viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'VizWiz_train.json')

# load annotation files
f = open(viz_wiz_data_train_annotation_path)
print("Loading annotation file at ", viz_wiz_data_train_annotation_path)
data = json.load(f)
f.close()


def find_bbox(segmentation_points):
    points = [(point['x'], point['y']) for point in segmentation_points]
    pts = np.array(points, np.int32)

    x, y, w, h = cv2.boundingRect(pts)

    return (x, y, x + w, y + h)



def coordinates_to_mask(coordinates, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    # Convert the coordinates to a NumPy array
    points = np.array(coordinates, dtype=np.int32)
    # Draw a filled polygon on the mask image
#     cv2.fillPoly(mask, [points], color=255)
    cv2.fillPoly(mask, [points], color=1)
    return mask



def get_mask(segmentation: list, width: int, height: int) -> Image:
    coordinates = [(int(item['x']), int(item['y'])) for item in segmentation]
    mask = coordinates_to_mask(
        coordinates=coordinates,
        image_shape=(height, width)
    )
    # Convert the mask array to a PIL image
    pil_image = Image.fromarray(mask, mode="P")
    return pil_image



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





lines = []
i = 0
max_length = 400

combined_train_data = []

for _, data_i in enumerate(tqdm(data)):
    image_id = data_i['image_id']
    ground_labels = data_i['grounding_labels']
    answers = data_i['answers']
    question = data_i['question']
    width = data_i['width']
    height = data_i['height']
    filepath = os.path.join(viz_wiz_data_train_image_dir, image_id)

    # load image
    img = Image.open(filepath).convert("RGB")
    # convert image to string
    img_base64 = image_to_base64(img, format='jpeg')


    for _i, ans in enumerate(answers):
        uniq_id = f"{image_id}_{_i}"
        ans = str(ans)
        # Grounding label [ {x, y} {x, y}]
        ground_label_i = ground_labels[_i]
        bbox = find_bbox(ground_label_i)
        x1, y1, x2, y2 = bbox
        # Bbox
        box_string = f'{x1},{y1},{x2},{y2}'

        # load mask
        annot_img = get_mask(segmentation=ground_label_i, width=width, height=height)
#         annot_img.save(f"./annot_img_{image_id}_mask.png")
#         img.save(f"./img_{image_id}_raw.jpg")
        
        annot_base64 = image_to_base64(annot_img, format='png')

        # Get polygon
        polygons = []
        polygon_item = []
        for point in ground_label_i:
            polygon_item.append(int(point['x']))
            polygon_item.append(int(point['y']))
        polygons.append(polygon_item)
        polygons_processed = []
        for polygon in polygons:
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
#         print("polygons", polygons)
#         print("box_string", box_string)
#         print("pts_string", pts_string)
#         print("pts_string_interpolated", pts_string_interpolated)
#         exit(0)

        i += 1
#         selected_cols=0,5,6,2,4,3,7
        instance = '\t'.join([uniq_id, str(image_id), ans, box_string, pts_string, img_base64, annot_base64,
                              pts_string_interpolated]) + '\n'
    combined_train_data.append(instance)

random.shuffle(combined_train_data)

# create result folder
os.makedirs("datasets/finetune_vqa", exist_ok=True)

# generate training tsv file
tsv_filename = "datasets/finetune_vqa/train_shuffled_vqa.tsv"


file_name = os.path.join(tsv_filename)
print("creating ", file_name)
writer = open(file_name, 'w')
writer.writelines(combined_train_data)
writer.close()


