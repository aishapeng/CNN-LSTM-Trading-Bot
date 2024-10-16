import tensorflow as tf
from tensorflow.python.client import device_lib

print("TensorFlow version:", tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())  

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("GPUs are available.")
    print(f"Number of GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU available. Using CPU.")