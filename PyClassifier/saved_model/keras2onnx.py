import tensorflow as tf
import keras
import onnx 
from onnx import TensorProto  

model = keras.models.load_model("malaria_det.keras")

inference_model = keras.Sequential()

for layer in model.layers:
    if not any(aug in layer.name.lower() for aug in ['random_flip', 'random_rotation', 'random_zoom']):
        inference_model.add(layer)

dummy_input = tf.zeros((1, 256, 256, 3)) 
_ = inference_model(dummy_input)

inference_model.export("malaria_det_v2.onnx", format="onnx")

# In case OpenCV wants one with UINT32 tensors
# model = onnx.load("malaria_det_v2.onnx")  
# for tensor in model.graph.initializer:     
#     if tensor.data_type == TensorProto.UINT64:         
#         tensor.data_type = TensorProto.INT32     
#     elif tensor.data_type == TensorProto.INT64:         
#         tensor.data_type = TensorProto.INT32  
            
# onnx.save(model, "malaria_det_v2-32.onnx")


print("Clean inference model exported successfully!")