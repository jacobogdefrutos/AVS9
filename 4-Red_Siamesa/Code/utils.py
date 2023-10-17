import tensorflow as tf

def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    return image
    