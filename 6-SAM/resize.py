from PIL import Image
import os
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
            result_path = os.path.join(output_folder, f"{unique_id}{filename[3:]}")
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
