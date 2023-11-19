from PIL import Image
import os

def resize_images(input_path, output_path, new_size):
    """
    Resize all images in the input path and save them with the same name in the output path.

    Parameters:
    - input_path: Path to the directory containing the original images.
    - output_path: Path to the directory where resized images will be saved.
    - new_size: Tuple (width, height) specifying the new size of the images.
    """
    # Ensure the output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Loop through all files in the input directory
    for filename in os.listdir(input_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more extensions if needed
            # Construct the full file paths
            input_file_path = os.path.join(input_path, filename)
            output_file_path = os.path.join(output_path, filename)

            # Open the image
            with Image.open(input_file_path) as img:
                # Resize the image
                resized_img = img.resize(new_size)

                # Save the resized image with the same name in the output directory
                resized_img.save(output_file_path)
