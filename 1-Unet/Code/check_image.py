import cv2

def main():
    img= cv2.imread('D:\Users\jacob\Project_Cracks\DeepLabv3FineTuning-master\CrackForest\Masks\001_label.PNG.jpg')
    print(img.shape)
    print('End of main')
if __name__ == "__main__":
    main()