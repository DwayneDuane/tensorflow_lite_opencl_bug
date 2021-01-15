#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import python as tf_python
import numpy as np

tf_model = tf.keras.models.load_model('./dummy_model')
#Convert to TFLite model:
#   Set batch dimensions of inputs to 1

for i in range(3):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

model_func = tf.function(lambda a: tf_model(a))
concrete_func = model_func.get_concrete_function([tf.TensorSpec(tf_model.inputs[0].shape, tf_model.inputs[0].dtype), tf.TensorSpec(tf_model.inputs[1].shape, tf_model.inputs[1].dtype), tf.TensorSpec(tf_model.inputs[2].shape, tf_model.inputs[2].dtype)])

tf_out = concrete_func([gru_h_in, gru1_h_in, input_1])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

#save TFLite model
open("dummy.tflite", "wb").write(tflite_model)

