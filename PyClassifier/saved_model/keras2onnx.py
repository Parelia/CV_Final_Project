#############################
# Author: Gabriele Trevisan #
#############################

import subprocess
import sys
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("malaria_det.keras")

# Strip augmentation layers (only needed for training, not inference)
inference_model = tf.keras.Sequential([
    layer for layer in model.layers
    if not any(x in layer.name for x in ["random_flip", "random_rotation", "random_zoom"])
])
inference_model.build(model.input_shape)
inference_model.set_weights(model.get_weights())

# Export clean model as SavedModel
inference_model.export("malaria_saved_model_clean")

# Convert from SavedModel to ONNX
subprocess.run([
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", "malaria_saved_model_clean",
    "--output", "malaria_det.onnx",
    "--opset", "17"
], check=True)


# subprocess.run([
#     sys.executable, "-m", "onnxsim",
#     "malaria_det.onnx",
#     "malaria_det_simplified.onnx"
# ], check=True)

print("Done! Use malaria_det_simplified.onnx in OpenCV C++")