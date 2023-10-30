import os
import cv2
import random

def augment_image(image):
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
        #image = cv2.rotate(image, angle)
        #label= cv2.rotate(label, angle)
    elif augmentation == 1:
        # Flip the image horizontally
        image = cv2.flip(image, 1)
    return image, augmentation

TRAIN_IMG_DIR='5-Pruebas/Total_images'
OUTPUT_IMG_DIR= '5-Pruebas/Total_images'
def main():
    file_img_list = os.listdir(TRAIN_IMG_DIR)
    for img in file_img_list:
        # Read the image
        image = cv2.imread(os.path.join(TRAIN_IMG_DIR, img))
        if image  is not None and 'SANO' not in img:#('SUF'  in img or 'POSTNER' in img):
            augmented_image,i  = augment_image(image)
            output_img_path = os.path.join(OUTPUT_IMG_DIR, f"{img[:-4]}_aug.png")
            cv2.imwrite(output_img_path, augmented_image)


    print("End of main")
if __name__ == "__main__":
    main()