import argparse
import os
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN

def align_and_crop_face(image_path, output_path, image_size=128):
    """
    Detects a face in an image, aligns it, and saves the cropped result.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the aligned and cropped image.
        image_size (int): The final size of the output image (width and height).
    """
    try:
        # Load the image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
    except FileNotFoundError:
        print(f"Error: Input image not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # Initialize the MTCNN detector
    detector = MTCNN()

    # Detect faces in the image
    results = detector.detect_faces(image_np)

    if not results:
        print(f"Warning: No face detected in {image_path}. Skipping.")
        return

    # The detector can find multiple faces; we'll use the one with the highest confidence.
    best_face = max(results, key=lambda r: r['confidence'])
    
    x1, y1, width, height = best_face['box']
    x2, y2 = x1 + width, y1 + height

    # Crop the face from the original image
    cropped_face = image.crop((x1, y1, x2, y2))

    # Resize the cropped face to the target size (e.g., 128x128)
    aligned_face = cropped_face.resize((image_size, image_size), Image.Resampling.LANCZOS)

    # Save the final aligned image
    aligned_face.save(output_path)
    print(f"Successfully aligned and saved face to {output_path}")

def main(args):
    """
    Processes all images in an input directory and saves the aligned
    faces to an output directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all image files in the input directory
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} images to process...")
    for filename in image_files:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        align_and_crop_face(input_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Align and crop faces from a directory of images.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the raw input images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the aligned and cropped faces will be saved.')
    args = parser.parse_args()
    main(args)
