from utils import get_test
import numpy as np
import torch
import cv2
from sklearn.metrics import f1_score,jaccard_score,recall_score,precision_score 
import torch.nn as nn
from torchvision import transforms
import segmentation_models_pytorch as smp
from main import(
    ACTIVATION, ENCODER_NAME, ENCODER_WEIGHTS, NUM_WORKERS, IMAGE_HEIGHT,
    IMAGE_WIDTH,  PIN_MEMORY )

BEST_M_CHECKPOINT_DIR = r"1-Unet/Code/saved_models/best_model_unet_aug_N_6.pth.tar"
TEST_IMG_DIR = r"Fotos_clasificadas/Imag" 
TEST_MASK_DIR = r"Fotos_clasificadas/Labels" #Dejo este path pero no vamos a necesitarlo
def output(loader,model,folder, device):
    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(iter(loader)):
            x = sample['image'].to(device=device)
            preds = model(x)
            output= preds#preds['out']    
            seg_map = output[0] # use only one output from the batch
            seg_map = torch.argmax(seg_map, dim=0).detach().cpu().numpy()

            image = x[0]
            image = np.array(image.cpu())
            image = np.transpose(image, (1, 2, 0))
            # unnormalize the image (important step)
            mean = np.array([0.0, 0.0, 0.0])
            std = np.array([1.0, 1.0, 1.0])
            image = std * image + mean
            image = np.array(image, dtype=np.float32)
            image = image * 255
            #ahora pintamos de negro el background y mantenemos la segmentacion
            label_color_map = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            red_map = image[:,:,0]
            green_map = image[:,:,1]
            blue_map = image[:,:,2]
            for label_num in range(0, 1):
                index = seg_map == label_num
                red_map[index] = np.array(label_color_map)[0]
                green_map[index] = np.array(label_color_map)[1]
                blue_map[index] = np.array(label_color_map)[2]
        
            rgb = np.stack([red_map, green_map, blue_map], axis=2)
            rgb = np.array(rgb, dtype=np.float32)
            name= sample['image_name']
            output= cv2.resize(rgb,(1080,720))
            cv2.imwrite(f"{folder}/{name[0]}_seg.png", output)

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=2,
        activation=ACTIVATION,
    ).to(DEVICE)

    best_checkpoint= torch.load(BEST_M_CHECKPOINT_DIR)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    print("Loading model is done.")

    # Preparing the test data
    transform_imag = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH),antialias=True),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])
    transform_mask = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH),antialias=True),
    #transforms.Normalize(mean=[0.449], std=[0.226])
    ])
    testloader, names =get_test(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        NUM_WORKERS,
        transform_imag,
        transform_mask,
        PIN_MEMORY
        )
    print("Generate segmentations")
    folder_test2 =r"D:\Users\jacob\AVS9\Data\saved_test_clas_images"
    output(testloader,model,folder_test2,device=DEVICE)
if __name__ == "__main__":
    main()