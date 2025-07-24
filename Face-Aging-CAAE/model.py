# This code is an alternative modernised tensorflow implementation of the paper by
# Zhifei Zhang, Yang Song, and Hairong Qi. "Age Progression/Regression by Conditional Adversarial Autoencoder."
# IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
#
# Date:     July. 24th, 2025
# Please cite above paper if you use this code

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input

def Encoder(size_image=128, size_kernel=5, num_input_channels=3, num_encoder_channels=64, num_z_channels=50):
    """Builds the Encoder model."""
    image_input = Input(shape=(size_image, size_image, num_input_channels))
    num_layers = int(tf.math.log(float(size_image)) / tf.math.log(2.0)) - int(size_kernel / 2)
    
    current = image_input
    for i in range(num_layers):
        filters = num_encoder_channels * (2 ** i)
        current = layers.Conv2D(filters, size_kernel, strides=2, padding='same', name=f'E_conv{i}')(current)
        current = layers.ReLU()(current)
        
    current = layers.Flatten()(current)
    current = layers.Dense(num_z_channels, name='E_fc')(current)
    output = layers.Activation('tanh', name='E_output')(current)
    
    return Model(inputs=image_input, outputs=output, name='Encoder')

def Generator(size_image=128, size_kernel=5, num_z_channels=50, num_categories=10, num_gen_channels=1024, num_input_channels=3):
    """Builds the Generator model with correct label tiling."""
    z_input = Input(shape=(num_z_channels,))
    age_label_input = Input(shape=(num_categories,))
    gender_label_input = Input(shape=(2,))

    # This amplifies the effect of the labels, which is crucial for the model to learn.
    tile_ratio = 1.0 
    
    # Tile age labels
    age_duplicate = int(num_z_channels * tile_ratio / num_categories)
    age_tiled = layers.Lambda(lambda x: tf.tile(x, [1, age_duplicate]))(age_label_input)
    
    # Tile gender labels
    gender_duplicate = int(num_z_channels * tile_ratio / 2)
    gender_tiled = layers.Lambda(lambda x: tf.tile(x, [1, gender_duplicate]))(gender_label_input)

    # Concatenate the z vector with the TILED labels
    z_and_labels = layers.concatenate([z_input, age_tiled, gender_tiled])
    # --------------------------------------------------------------------
    
    num_layers = int(tf.math.log(float(size_image)) / tf.math.log(2.0)) - int(size_kernel / 2)
    size_mini_map = int(size_image / 2 ** num_layers)
    
    # Fully connected layer and reshape
    current = layers.Dense(num_gen_channels * size_mini_map * size_mini_map, name='G_fc')(z_and_labels)
    current = layers.Reshape((size_mini_map, size_mini_map, num_gen_channels))(current)
    current = layers.ReLU()(current)
    
    # Deconvolution layers
    for i in range(num_layers):
        filters = int(num_gen_channels / 2 ** (i + 1))
        current = layers.Conv2DTranspose(filters, size_kernel, strides=2, padding='same', name=f'G_deconv{i}')(current)
        current = layers.ReLU()(current)
        
    # Final layers to get to the output image
    current = layers.Conv2DTranspose(int(num_gen_channels / 2 ** (num_layers + 1)), size_kernel, strides=1, padding='same', name='G_deconv_final1')(current)
    current = layers.ReLU()(current)
    current = layers.Conv2DTranspose(num_input_channels, size_kernel, strides=1, padding='same', name='G_deconv_final2')(current)
    output = layers.Activation('tanh', name='G_output')(current)
    
    return Model(inputs=[z_input, age_label_input, gender_label_input], outputs=output, name='Generator')

def DiscriminatorZ(num_z_channels=50, num_hidden_layer_channels=(64, 32, 16)):
    """Builds the Discriminator for the latent space z."""
    z_input = Input(shape=(num_z_channels,))
    current = z_input
    
    for i, filters in enumerate(num_hidden_layer_channels):
        current = layers.Dense(filters, name=f'D_z_fc{i}')(current)
        current = layers.BatchNormalization(scale=False, name=f'D_z_bn{i}')(current)
        current = layers.ReLU()(current)
        
    logits = layers.Dense(1, name='D_z_logits')(current)
    output = layers.Activation('sigmoid', name='D_z_output')(logits)
    
    return Model(inputs=z_input, outputs=[output, logits], name='Discriminator_Z')

def DiscriminatorImg(size_image=128, size_kernel=5, num_input_channels=3, num_categories=10, num_hidden_layer_channels=(16, 32, 64, 128)):
    """Builds the Discriminator for the image space."""
    image_input = Input(shape=(size_image, size_image, num_input_channels))
    age_label_input = Input(shape=(num_categories,))
    gender_label_input = Input(shape=(2,))
    
    current = image_input
    for i, filters in enumerate(num_hidden_layer_channels):
        current = layers.Conv2D(filters, size_kernel, strides=2, padding='same', name=f'D_img_conv{i}')(current)
        current = layers.BatchNormalization(scale=False, name=f'D_img_bn{i}')(current)
        current = layers.ReLU()(current)
        
        if i == 0: # Concatenate labels after the first conv layer
            def tile_labels(inputs):
                label_reshaped, feature_map = inputs
                target_shape = tf.shape(feature_map)
                return tf.tile(label_reshaped, [1, target_shape[1], target_shape[2], 1])

            age_label_reshaped = layers.Reshape((1, 1, num_categories))(age_label_input)
            gender_label_reshaped = layers.Reshape((1, 1, 2))(gender_label_input)
            
            age_label_tiled = layers.Lambda(tile_labels)([age_label_reshaped, current])
            gender_label_tiled = layers.Lambda(tile_labels)([gender_label_reshaped, current])
            
            current = layers.concatenate([current, age_label_tiled, gender_label_tiled])

    current = layers.Flatten()(current)
    current = layers.Dense(1024, name='D_img_fc1')(current)
    current = layers.LeakyReLU(negative_slope=0.2)(current)
    logits = layers.Dense(1, name='D_img_logits')(current)
    output = layers.Activation('sigmoid', name='D_img_output')(logits)
    
    return Model(inputs=[image_input, age_label_input, gender_label_input], outputs=[output, logits], name='Discriminator_Image')
