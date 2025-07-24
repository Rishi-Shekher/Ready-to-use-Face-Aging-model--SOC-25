# SOC'25 - A Ready-to-Use Face Aging Model

This project is a modernized and ready-to-use implementation of a face aging model based on the Conditional Adversarial Autoencoder (CAAE) architecture. 

The model is trained on UTKFace dataset on Google colabs, python T4 GPU. Average training time for 50 epochs was 2.25hr.

The model can take a facial photo and generate realistic aged or de-aged versions of that person across 10 different age categories, from 0-5 years up to 61-70+ years.

This is the link to the notion page (provided by my mentor) for learning concepts related to GANs and CNNs.
https://happy-armchair-815.notion.site/Face-Aging-Model-1f816f25632d80f79f38ede6cd083d38



---
First change your working directory to Face-Aging-CAAE.
## Pre-trained Model

I provide a fully trained ready to use model (50 epochs), allowing for immediate testing and inference without the need for training.

* **Download Link:** [**FaceAging_CAAE_save_final.zip**](https://drive.google.com/file/d/1xrUv58UmB1rBpoUFPx465HOXbB-GkZAi/view?usp=sharing)

### Instructions to Use the Pre-trained Model

1.  Click the link above to download the `FaceAging_CAAE_save_final.zip` file.
2.  Unzip the file. This will create a folder named `FaceAging_CAAE_save`.
3.  Copy this entire `FaceAging_CAAE_save` folder and place it inside the Face-Aging-CAAE folder.

The folder contains:
* `checkpoint/`: The final saved weights of the trained models.
* `samples/`: Reconstructed faces generated at each of the 50 training epochs.
* `test/`: Age-progressed faces generated at each of the 50 training epochs.
* `summary/`: TensorBoard logs for visualizing the training losses.

---

## Setup and Installation

This project uses a Conda environment.

1.  **Clone the repository**
 
2.  **Create the Conda environment:**
    ```bash
    # It is recommended to use Python 3.13
    conda create --name face-aging-tf2 python=3.13
    conda activate face-aging-tf2
    ```
3.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy imageio Pillow scipy mtcnn
    ```

---

## Usage

The project can be run in two modes: testing (inference) or training.

### Testing the Pre-trained Model

This is the default and recommended way to use the model. It is a two-step process.

#### Step 1: Align Your Custom Photos

The model was trained on perfectly centered and cropped faces. To get good results, you must pre-process your own photos in the same way.However ,quality of images might be compromised if alignment is performed.

1.  Create a folder for your raw, unaligned photos (e.g., `my_raw_photos`).
2.  Place your `.jpg` face photos inside this folder. Ensure that the number of images must be greater than 10.
3.  Create an empty folder to store the aligned results (e.g., `tests`).
4.  Run the alignment script:
    ```bash
    python align_face.py --input_dir my_raw_photos --output_dir tests
    ```
    This will detect, crop, and resize the faces, saving the results in the `tests` folder.

#### Step 2: Run the Main Test Script

Now that you have a folder of aligned faces, you can run the main testing script.

1.  **Prerequisites:**
    * Ensure you have downloaded and placed the `FaceAging_CAAE_save` folder in your project directory.
    * Ensure you have run the `align_face.py` script and have aligned images in your `tests` folder.

2.  **Run the Test Command:**
    Execute the following command from your terminal:
    ```bash
    python main.py --is_train False --testdir tests
    ```

3.  **View Results:** The script will load the trained model, age your aligned photos, and save the output grids in the `FaceAging_CAAE_save/CUSTOM TESTS/` directory.

One can also directly add their images into the test folder and run the model , without alignment, provided the image is of a good resolution.

### Training the Model from Scratch

If you wish to train the model yourself, you can use the provided `init_model` as a starting point.

1.  **Prerequisites:**

    * Download the Initial Model: The `init_model` provides a starting point for training, leading to faster convergence.
        * **Download Link:** [**init_model.zip**](https://drive.google.com/file/d/1l20h8wQm6HY_yj4A2EkNDove_NRJHSP4/view?usp=sharing)
        * Unzip the file and place the resulting `init_model` folder inside your main project directory.
    * Ensure you have the UTKFace dataset located at `./data/UTKFace/`.
    * Ensure that the image extensions of the UTKFace dataset are .jpg.chip.jpg(default) .


2.  **Start a Fresh Training Run:**
    To start a new training run from the initial model, you must first delete any existing save directory. Then, run the training command.
    ```bash
    # Clear old results
    rm -rf FaceAging_CAAE_save
    
    # Start training for 50 epochs
    python main.py --is_train True --epoch 50 
    ```
    The script will automatically load the `init_model` and save all results to the `./FaceAging_CAAE_save` directory.

3.  **Resume Training:**
    If your training is interrupted, you can simply run the same command again to resume from the last saved checkpoint:
    ```bash
    python main.py --is_train True --epoch 50
    ```

4.  **Monitor with TensorBoard:**
    To visualize the training losses, run the following command in a separate terminal:
    ```bash
    tensorboard --logdir=./FaceAging_CAAE_save/summary
    ```

---

## File Descriptions

* `main.py`: The main entry point for the project. It handles command-line arguments and starts the training or testing process.
* `FaceAging.py`: The core class that encapsulates the entire model. It manages the data pipeline, training loop, checkpointing, and testing logic.
* `model.py`: Defines the four neural network architectures (Encoder, Generator, DiscriminatorZ, DiscriminatorImg) using the Keras Functional API.
* `utils.py`: Contains helper functions for loading, preprocessing, and saving images.
* `align_face.py`: A crucial pre-processing utility. This script takes raw input images, detects the faces, and then crops and aligns them to match the format of the training data, which is essential for getting  test results.


---
## Citation and Background

This work is based on the following seminal paper. Please cite their work if you use this model.

> Zhifei Zhang, Yang Song, and Hairong Qi. "Age Progression/Regression by Conditional Adversarial Autoencoder." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.*

### Advantages of CAAE over standard GANs

As described in the paper, this Conditional Adversarial Autoencoder (CAAE) architecture has several key advantages over a standard Generative Adversarial Network (GAN) for the task of face aging:

1.  **Superior Identity Preservation:** The core of the model is an autoencoder (Encoder -> Z -> Generator). This structure is trained with a direct reconstruction loss (L1 loss), which forces the model to learn a compressed representation (`z` vector) of a person's unique identity. This provides a much stronger and more direct constraint for preserving identity compared to helper losses that are often added to standard GANs.

2.  **Smoother Latent Space:** The CAAE uses a second discriminator on the latent space (`z`). This forces the encoder to produce `z` vectors that match a simple prior distribution (like uniform noise). This regularization prevents the encoder from simply "memorizing" inputs and ensures that the space of identities is smooth and well-behaved, leading to more robust and realistic transformations.

3.  **More Stable Training:** The combination of a reconstruction loss and two separate adversarial losses (one for image realism and one for latent space distribution) creates a more stable training dynamic. It provides clearer gradients and helps avoid the mode collapse issues that can plague standard GANs.

---



## Acknowledgements

This project is an alternative implementation of the CAAE model by the paper's authors. The original code can be found at:

* [**ZZUTK/Face-Aging-CAAE**](https://github.com/ZZUTK/Face-Aging-CAAE)

