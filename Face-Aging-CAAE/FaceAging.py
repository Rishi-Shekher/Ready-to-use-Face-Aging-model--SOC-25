# This code is an alternative modernised tensorflow implementation of the paper by
# Zhifei Zhang, Yang Song, and Hairong Qi. "Age Progression/Regression by Conditional Adversarial Autoencoder."
# IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
#
# Date:     July. 24th, 2025
# Please cite above paper if you use this code

import tensorflow as tf
from tensorflow import keras
from keras import optimizers, losses, utils
import numpy as np
import os
import time
from glob import glob
from utils import load_image, save_batch_images
from model import Encoder, Generator, DiscriminatorZ, DiscriminatorImg

class FaceAging(object):
    def __init__(self, is_training=True, save_dir='./FaceAging_CAAE_save', dataset_name='UTKFace'):
        self.size_image = 128
        self.size_kernel = 5
        self.size_batch = 100
        self.num_input_channels = 3
        self.num_z_channels = 50
        self.num_categories = 10
        self.image_value_range = (-1, 1)
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.is_training = is_training

        # Initialize models
        self.encoder = Encoder()
        self.generator = Generator()
        self.discriminator_z = DiscriminatorZ()
        self.discriminator_img = DiscriminatorImg()

        # Initialize optimizers
        self.eg_optimizer = optimizers.Adam()
        self.d_z_optimizer = optimizers.Adam()
        self.d_img_optimizer = optimizers.Adam()
        
        # Checkpoint manager
        self.checkpoint = tf.train.Checkpoint(
            encoder=self.encoder,
            generator=self.generator,
            discriminator_z=self.discriminator_z,
            discriminator_img=self.discriminator_img,
            eg_optimizer=self.eg_optimizer,
            d_z_optimizer=self.d_z_optimizer,
            d_img_optimizer=self.d_img_optimizer
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, os.path.join(self.save_dir, 'checkpoint'), max_to_keep=2
        )

    def _initialize_optimizers(self):
        """Applies zero gradients to the optimizers to build their state."""
        print("Warming up optimizers...")
        eg_vars = self.encoder.trainable_variables + self.generator.trainable_variables
        dz_vars = self.discriminator_z.trainable_variables
        di_vars = self.discriminator_img.trainable_variables

        zero_grads_eg = [tf.zeros_like(v) for v in eg_vars]
        zero_grads_dz = [tf.zeros_like(v) for v in dz_vars]
        zero_grads_di = [tf.zeros_like(v) for v in di_vars]

        self.eg_optimizer.apply_gradients(zip(zero_grads_eg, eg_vars))
        self.d_z_optimizer.apply_gradients(zip(zero_grads_dz, dz_vars))
        self.d_img_optimizer.apply_gradients(zip(zero_grads_di, di_vars))
        print("Optimizers warmed up.")

    def get_age_category(self, age):
        if 0 <= age <= 5: return 0
        if 6 <= age <= 10: return 1
        if 11 <= age <= 15: return 2
        if 16 <= age <= 20: return 3
        if 21 <= age <= 30: return 4
        if 31 <= age <= 40: return 5
        if 41 <= age <= 50: return 6
        if 51 <= age <= 60: return 7
        if 61 <= age <= 70: return 8
        return 9

    def create_dataset(self, file_paths):
        """Creates a tf.data.Dataset for training from a given list of file paths."""
        
        def generator():
            for filename in file_paths:
                image = load_image(image_path=filename, image_size=self.size_image, image_value_range=self.image_value_range)
                if image is None: continue
                try:
                    fname_only = os.path.basename(filename)
                    parts = fname_only.split('_')
                    age = int(parts[0])
                    gender = int(parts[1])
                    age_cat = self.get_age_category(age)
                    age_label = utils.to_categorical(age_cat, num_classes=self.num_categories)
                    gender_label = utils.to_categorical(gender, num_classes=2)
                    yield image, age_label, gender_label
                except (ValueError, IndexError):
                    continue

        output_signature = (
            tf.TensorSpec(shape=(self.size_image, self.size_image, self.num_input_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(self.num_categories,), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.shuffle(buffer_size=1000).batch(self.size_batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return dataset

    @tf.function
    def train_step(self, images, age_labels, gender_labels, z_prior, weights):
        with tf.GradientTape() as eg_tape, tf.GradientTape() as dz_tape, tf.GradientTape() as di_tape:
            z_encoded = self.encoder(images, training=True)
            generated_images = self.generator([z_encoded, age_labels, gender_labels], training=True)
            
            _, d_z_logits = self.discriminator_z(z_encoded, training=True)
            _, d_g_logits = self.discriminator_img([generated_images, age_labels, gender_labels], training=True)
            _, d_z_prior_logits = self.discriminator_z(z_prior, training=True)
            _, d_input_logits = self.discriminator_img([images, age_labels, gender_labels], training=True)

            bce = losses.BinaryCrossentropy(from_logits=True)
            eg_loss_l1 = tf.reduce_mean(tf.abs(images - generated_images))
            d_z_loss_prior = bce(tf.ones_like(d_z_prior_logits), d_z_prior_logits)
            d_z_loss_z = bce(tf.zeros_like(d_z_logits), d_z_logits)
            loss_dz = d_z_loss_prior + d_z_loss_z
            d_img_loss_input = bce(tf.ones_like(d_input_logits), d_input_logits)
            d_img_loss_g = bce(tf.zeros_like(d_g_logits), d_g_logits)
            loss_di = d_img_loss_input + d_img_loss_g
            e_z_loss = bce(tf.ones_like(d_z_logits), d_z_logits)
            g_img_loss = bce(tf.ones_like(d_g_logits), d_g_logits)
            loss_eg = eg_loss_l1 + weights[0] * g_img_loss + weights[1] * e_z_loss

        eg_grads = eg_tape.gradient(loss_eg, self.encoder.trainable_variables + self.generator.trainable_variables)
        dz_grads = dz_tape.gradient(loss_dz, self.discriminator_z.trainable_variables)
        di_grads = di_tape.gradient(loss_di, self.discriminator_img.trainable_variables)
        
        self.eg_optimizer.apply_gradients(zip(eg_grads, self.encoder.trainable_variables + self.generator.trainable_variables))
        self.d_z_optimizer.apply_gradients(zip(dz_grads, self.discriminator_z.trainable_variables))
        self.d_img_optimizer.apply_gradients(zip(di_grads, self.discriminator_img.trainable_variables))
        
        return loss_eg, loss_dz, loss_di

    def train(self, num_epochs, use_trained_model=True, use_init_model=True, weights=(0.0001, 0, 0)):
        start_epoch = 0
        restored = False
        if use_trained_model:
            status = self.checkpoint.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print(f"Restored from {self.manager.latest_checkpoint}")
                restored = True
            elif use_init_model:
                print("No checkpoint found. Trying to load from init_model...")
                init_manager = tf.train.CheckpointManager(self.checkpoint, 'init_model', max_to_keep=1)
                if init_manager.latest_checkpoint:
                    self.checkpoint.restore(init_manager.latest_checkpoint)
                    print(f"Restored from initial model: {init_manager.latest_checkpoint}")
                    restored = True
                else:
                    print("No initial model found. Initializing from scratch.")
            else:
                print("Initializing from scratch.")
        
        if restored:
            self._initialize_optimizers()

        summary_writer = tf.summary.create_file_writer(os.path.join(self.save_dir, 'summary'))

        file_paths = glob(os.path.join('./data', self.dataset_name, '*.jpg.chip.jpg'))
        if not file_paths:
            print(f"ERROR: No images found at ./data/{self.dataset_name}/*.jpg.chip.jpg")
            return

        file_paths.sort()
        np.random.seed(seed=2017)
        np.random.shuffle(file_paths)

        sample_files = file_paths[0:self.size_batch]
        train_files = file_paths
        
        sample_images, sample_age_labels, sample_gender_labels = self.load_files_as_batch(sample_files)
        
        dataset = self.create_dataset(train_files)
        if dataset is None: return

        #Calculate starting epoch from checkpoint
        num_batches = len(train_files) // self.size_batch
        if restored:
            # The optimizer's iteration count is saved in the checkpoint
            completed_steps = self.eg_optimizer.iterations.numpy()
            start_epoch = completed_steps // num_batches
            print(f"Resuming training from epoch {start_epoch + 1}")
        
        for epoch in range(start_epoch, num_epochs):
            # Shuffle training files each epoch for consistency
            np.random.shuffle(train_files)
            dataset = self.create_dataset(train_files)

            for batch_idx, (batch_images, batch_age, batch_gender) in enumerate(dataset):
                start_time = time.time()
                batch_z_prior = np.random.uniform(-1, 1, [self.size_batch, self.num_z_channels]).astype(np.float32)
                
                eg_err, dz_err, di_err = self.train_step(batch_images, batch_age, batch_gender, batch_z_prior, weights)
                
                with summary_writer.as_default():
                    step = self.eg_optimizer.iterations
                    tf.summary.scalar('loss/EG_err', eg_err, step=step)
                    tf.summary.scalar('loss/D_z_err', dz_err, step=step)
                    tf.summary.scalar('loss/D_img_err', di_err, step=step)

                print(f"Epoch: [{epoch+1}/{num_epochs}] Batch: [{batch_idx}/{num_batches}] Time: {time.time() - start_time:.4f}s")
                print(f"\tEG_err={eg_err:.4f}, D_z_err={dz_err:.4f}, D_img_err={di_err:.4f}")

            name = f'{epoch+1:02d}.png'
            self.sample(sample_images, sample_age_labels, sample_gender_labels, name)
            self.test(sample_images, sample_gender_labels, name)

            if np.mod(epoch, 5) == 4:
                self.manager.save()
        
        self.manager.save()

    def load_files_as_batch(self, file_list):
        """Helper function to load a list of files into a NumPy batch."""
        images, age_labels, gender_labels = [], [], []
        for file_path in file_list:
            image = load_image(image_path=file_path, image_size=self.size_image, image_value_range=self.image_value_range)
            if image is None: continue
            try:
                fname = os.path.basename(file_path)
                parts = fname.split('_')
                age, gender = int(parts[0]), int(parts[1])
                age_cat = self.get_age_category(age)
                age_label = utils.to_categorical(age_cat, num_classes=self.num_categories)
                gender_label = utils.to_categorical(gender, num_classes=2)
                
                images.append(image)
                age_labels.append(age_label)
                gender_labels.append(gender_label)
            except (ValueError, IndexError):
                continue
        return np.array(images), np.array(age_labels), np.array(gender_labels)

    def sample(self, images, age_labels, gender_labels, name):
        """Saves a grid of reconstructed images."""
        sample_dir = os.path.join(self.save_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        z_vectors = self.encoder.predict(images)
        generated_images = self.generator.predict([z_vectors, age_labels, gender_labels])
        
        save_batch_images(
            batch_images=generated_images,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[int(np.sqrt(self.size_batch)), int(np.sqrt(self.size_batch))]
        )

    def test(self, images, gender_labels, name):
        """Saves a grid of age-progressed images, with aging down columns."""
        test_dir = os.path.join(self.save_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        
        num_samples = int(np.sqrt(self.size_batch))
        images = images[:num_samples]
        gender_labels = gender_labels[:num_samples]

        tiled_images = np.tile(images, [self.num_categories, 1, 1, 1])
        tiled_gender_labels = np.tile(gender_labels, [self.num_categories, 1])
        
        age_labels_raw = np.arange(self.num_categories)
        repeated_age_labels_raw = np.repeat(age_labels_raw, num_samples)
        tiled_age_labels = utils.to_categorical(repeated_age_labels_raw, num_classes=self.num_categories)

        z_vectors = self.encoder.predict(tiled_images)
        generated_images = self.generator.predict([z_vectors, tiled_age_labels, tiled_gender_labels])
        
        save_batch_images(
            batch_images=generated_images,
            save_path=os.path.join(test_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[self.num_categories, num_samples]
        )

    def custom_test(self, testing_samples_dir):
        """Modern TF2 implementation for testing the model."""
        status = self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(f"Restored from {self.manager.latest_checkpoint}")
            print("\tSUCCESS ^_^")
        else:
            print("ERROR: No checkpoint found to restore from.")
            return

        num_samples = int(np.sqrt(self.size_batch))
        
        file_names = glob(testing_samples_dir)
        if len(file_names) < num_samples:
            print(f'The number of testing images must be larger than {num_samples}')
            return
            
        sample_files = file_names[0:num_samples]
        sample_images = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
        ) for sample_file in sample_files]
        
        images = np.array(sample_images).astype(np.float32)

        gender_male = np.zeros(shape=(num_samples, 2), dtype=np.float32)
        gender_male[:, 0] = 1.0
        gender_female = np.zeros(shape=(num_samples, 2), dtype=np.float32)
        gender_female[:, 1] = 1.0

        self.test_and_save(images, gender_male, 'test_as_male.png')
        self.test_and_save(images, gender_female, 'test_as_female.png')

        print(f'\n\tDone! Results are saved as {os.path.join(self.save_dir, "CUSTOM TESTS", "test_as_xxx.png")}\n')

    def test_and_save(self, images, gender_labels, output_filename):
        """Helper function to run inference and save a grid of images."""
        num_samples = images.shape[0]
        
        tiled_images = np.tile(images, [self.num_categories, 1, 1, 1])
        tiled_gender_labels = np.tile(gender_labels, [self.num_categories, 1])
        
        age_labels_raw = np.arange(self.num_categories)
        repeated_age_labels_raw = np.repeat(age_labels_raw, num_samples)
        tiled_age_labels = utils.to_categorical(repeated_age_labels_raw, num_classes=self.num_categories)

        z_vectors = self.encoder.predict(tiled_images)
        generated_images = self.generator.predict([z_vectors, tiled_age_labels, tiled_gender_labels])
        
        test_dir = os.path.join(self.save_dir, 'CUSTOM TESTS')
        os.makedirs(test_dir, exist_ok=True)
        
        save_batch_images(
            batch_images=generated_images,
            save_path=os.path.join(test_dir, output_filename),
            image_value_range=self.image_value_range,
            size_frame=[self.num_categories, num_samples]
        )
        
        save_batch_images(
            batch_images=images,
            save_path=os.path.join(test_dir, f'input_for_{output_filename}'),
            image_value_range=self.image_value_range,
            size_frame=[1, num_samples]
        )
