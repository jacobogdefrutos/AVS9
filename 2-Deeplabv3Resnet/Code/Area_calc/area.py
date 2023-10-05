import cv2

def get_area(image):
    print("Calculating Area")
    #we want a greyscale image so we get the second channel from the image since the mask are in green (RGB)
    image_gray= image[:,:,1]
    ner_pixels= len([i for i in image_gray.ravel() if i ==255])
    contours,_= cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #area= cv2.contourArea(contours[0]) No saca el total. Mejor contar el numero de pixels y ay
    image2= cv2.drawContours(image, contours[0], -1, (0,0,255), 3)
    return ner_pixels, image2
