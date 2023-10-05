import os
import cv2
import random

TRAIN_IMG_DIR = r"D:\Users\jacob\AVS9\Data\train\Imag" #LOCAL
TRAIN_MASK_DIR = r"D:\Users\jacob\AVS9\Data\train\Labels"#LOCAL
OUTPUT_IMG_DIR = r"D:\Users\jacob\AVS9\Code\Data_Augmentation\Imag"
OUTPUT_MASK_DIR = r"D:\Users\jacob\AVS9\Code\Data_Augmentation\Labels"
def augment_image(image,label):
    # Randomly select the augmentation to apply
    augmentation = random.randint(0, 1)
    if augmentation == 0:
        # Rotate the image by a random angle between -30 and 30 degrees
        angle = random.randint(-30, 30)
        height, width = image.shape[:2]
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        # Apply the rotation to the image
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
        label = cv2.warpAffine(label, rotation_matrix, (width, height))
        #image = cv2.rotate(image, angle)
        #label= cv2.rotate(label, angle)
    elif augmentation == 1:
        # Flip the image horizontally
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    return image,label, augmentation


def main():
    file_img_list = os.listdir(TRAIN_IMG_DIR)
    for img in file_img_list:
        # Read the image
        image = cv2.imread(os.path.join(TRAIN_IMG_DIR, img))
        label = cv2.imread(os.path.join(TRAIN_MASK_DIR, img))
        if image  is not None and label is not None:
            augmented_image, augmented_label,i  = augment_image(image,label)
            name = img.split(".png")[0]
            output_img_path = os.path.join(OUTPUT_IMG_DIR, f"{name}_aug.png")
            output_label_path = os.path.join(OUTPUT_MASK_DIR, f"{name}_aug.png")
            cv2.imwrite(output_img_path, augmented_image)
            cv2.imwrite(output_label_path, augmented_label)


    print("End of main")
if __name__ == "__main__":
    main()