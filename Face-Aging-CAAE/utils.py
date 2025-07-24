import numpy as np
import imageio.v2 as imageio
from PIL import Image

def load_image(
        image_path,
        image_size=128,
        image_value_range=(-1, 1),
        is_gray=False,
):
    """
    Loads and preprocesses an image file.
    """
    try:
        pil_mode = 'L' if is_gray else 'RGB'
        pil_image = Image.open(image_path).convert(pil_mode)
        
        resized_pil_image = pil_image.resize([image_size, image_size], Image.Resampling.BICUBIC)
        image = np.array(resized_pil_image)
        
        # Normalize the image to the specified range
        image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def save_batch_images(
        batch_images,
        save_path,
        image_value_range=(-1, 1),
        size_frame=None
):
    """
    Saves a batch of images into a single grid image.
    """
    # Transform the pixel value to 0-1
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
    
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
        
    imageio.imwrite(save_path, (frame * 255).astype(np.uint8))
