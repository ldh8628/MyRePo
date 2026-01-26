import tensorflow as tf

print("built with cuda: ", tf.test.is_built_with_cuda())
print("gpu available: ", tf.test.is_gpu_available())