#############################
# Author: Gabriele Trevisan #
#############################

import tensorflow as tf
import tf2onnx
import onnx
import subprocess

# Load model
model = tf.keras.models.load_model("malaria_det.keras")

# Save as SavedModel first
model.export("malaria_saved_model")

# Convert from SavedModel to ONNX
subprocess.run([
    "python", "-m", "tf2onnx.convert",
    "--saved-model", "malaria_saved_model",
    "--output", "malaria_det.onnx",
    "--opset", "17"
], check=True)

print("Done! Model saved as malaria_det.onnx")