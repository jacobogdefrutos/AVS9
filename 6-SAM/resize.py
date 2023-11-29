from PIL import Image
import os
import numpy as np
def id_creation(filename):
    a=404
    b=404
    c=404
    #Left or right eye
    if 'OI' in filename:
        a=1
    elif 'OD' in filename:
        a=2
    #Quadrant type
    if 'TEM' in filename:
        b=1
    elif 'SUP'in filename:
        b=2
    elif 'INF' in filename:
        b=3
    elif 'NAS'in filename:
        b=4
    # Diseas type
    if 'SANO' in filename:
        c=1
    elif "CMV" in filename:
        c=2
    elif "SUF" in filename:
        c=3
    elif "SURV" in filename:
        c=4
    elif "POSTNER" in filename:
        c=5
    id= str(a) + str(b)+str(c)


    return id
def resize_image_with_pading(input_folder,output_folder,target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Adjust the file extensions as needed
            # Load the original image
            original_image_path = os.path.join(input_folder, filename)
            original_image = Image.open(original_image_path)

            # Create a new image with the target size and fill with black
            new_image = Image.new("RGB", target_size, "black")
            # Calculate the position to paste the original image centered on the new image
            paste_position = (
                (target_size[0] - original_image.width) // 2,
                (target_size[1] - original_image.height) // 2)
            # Paste the original image onto the new image
            new_image.paste(original_image, paste_position)

            # Save the result in the output folder
            unique_id=id_creation(filename)
            result_path = os.path.join(output_folder, f"{unique_id}{filename[8:]}")
            new_image.save(result_path)

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
def remove_padding(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    for image_file in image_files:
        # Construct the full path for the input image
        input_image_path = os.path.join(input_folder, image_file)

        # Open the image
        img = Image.open(input_image_path)

        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Find the coordinates of the non-black pixels
        non_black_pixels = np.any(img_array != [0, 0, 0], axis=2)
        coordinates = np.argwhere(non_black_pixels)

        # Get the cropping coordinates
        (y_min, x_min), (y_max, x_max) = coordinates.min(0), coordinates.max(0) + 1

        # Crop the image
        cropped_img_array = img_array[y_min:y_max, x_min:x_max, :]

        # Convert the NumPy array back to an image
        cropped_img = Image.fromarray(cropped_img_array)

        # Construct the full path for the output image
        output_image_path = os.path.join(output_folder, image_file)

        # Save the cropped image
        cropped_img.save(output_image_path)

    print(f"Cropping completed. Cropped images saved in {output_folder}")
def crop_images_from_folder(input_folder, output_folder, target_size=(800, 312)):
     # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Open the image
        img = Image.open(input_path)

        # Get the center coordinates
        center_x, center_y = img.size[0] // 2, img.size[1] // 2

        # Calculate the crop box
        left = max(0, center_x - target_size[0] // 2)
        upper = max(0, center_y - target_size[1] // 2)
        right = min(img.size[0], center_x + target_size[0] // 2)
        lower = min(img.size[1], center_y + target_size[1] // 2)

        # Crop the image to the target size
        cropped_img = img.crop((left, upper, right, lower))

        # Save the cropped image to the output folder
        output_path = os.path.join(output_folder, filename)
        cropped_img.save(output_path)
#if __name__ == "__main__":
#    input_folder='/home/jacobo15defrutos/AVS9/Data/Data_new_SAM/train/labels'
#    output_folder='/home/jacobo15defrutos/AVS9/Data/Data_new_SAM_unpadded/train/labels'
#    crop_images_from_folder(input_folder,output_folder)