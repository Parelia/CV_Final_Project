###########################
# Author: Capuzzo Daniele #
###########################

import os
import sys
import keras
import numpy as np
from PIL import Image

img_height = 256
img_width = 256
labels_map = {0: 'Parasitized', 1: 'Uninfected'}

if len(sys.argv) < 3:
    raise ValueError("Usage: python name.py image_path list_of_bounding_boxes.txt")

input_image_path = sys.argv[1]
bounding_boxes_path = sys.argv[2]

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'saved_model/malaria_det.keras')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the model has been built and saved correctly.")

if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Input image file '{input_image_path}' not found.")

if not os.path.exists(bounding_boxes_path):
    raise FileNotFoundError(f"Bounding boxes file '{bounding_boxes_path}' not found.")

# load the model
model = keras.models.load_model(model_path)

# load the input image
input_image = Image.open(input_image_path)

# read bounding boxes
bounding_boxes = []
with open(bounding_boxes_path, 'r') as f:
    for line in f:
        x, y, width, height = map(int, line.strip().split())
        bounding_boxes.append((x, y, width, height))

# prepare the output file
output_file_path = os.path.join(os.path.dirname(bounding_boxes_path), "predictions.txt")
with open(output_file_path, 'w') as output_file:
    for box in bounding_boxes:
        x, y, width, height = box

        # crop the image based on the bounding box
        cropped_image = input_image.crop((x, y, x + width, y + height))

        # resize the cropped image to the model's input size
        cropped_image = cropped_image.resize((img_height, img_width))
        img_array = keras.utils.img_to_array(cropped_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # predict diagnosis
        predictions = model.predict(img_array)
        score = predictions[0][0]

        # determine the label
        if score > 0.5:
            label = labels_map[1]
            probability = score
        else:
            label = labels_map[0]
            probability = 1 - score

        # determine diagnosis confidence
        if 0.4 <= score < 0.6:
            confidence = "UNCERTAIN"
        else:
            confidence = "CONFIDENT"

        # write result: Label, Probability, BBox, Confidence
        output_file.write(f"{label}, {probability:.2%}, {x} {y} {width} {height}, {confidence}\n")

print(f"Predictions saved to {output_file_path}")