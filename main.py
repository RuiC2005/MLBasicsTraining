import tensorflow as tf 

print("Tensorflow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))