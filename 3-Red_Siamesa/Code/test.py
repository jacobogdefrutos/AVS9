import numpy as np
import torch
import torch.nn.functional as F #de aqui podemos coger la funcion pairwise_distance para calcula la ecludian_distance
from utils import get_test,save_best_model
from torchvision import transforms
from lenet5 import LeNet5
import pandas as pd
import matplotlib.pyplot as plt
from main import( NUM_WORKERS, IMAGE_HEIGHT,
    IMAGE_WIDTH )

BEST_M_CHECKPOINT_DIR = '3-Red_Siamesa/saved_models/best_model_LeNet5RS.pth.tar'
TEST_IMG_DIR = r'Data/saved_seg_class_images/test'
test_csv='Iris_test_seg_list.csv'



def main():
    data = {
    'OI Name': [],
    'OD Name': [],
    'Euclidean Distance': []
    }
    df = pd.DataFrame(data)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model= LeNet5().to(device=DEVICE)

    best_checkpoint= torch.load(BEST_M_CHECKPOINT_DIR)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    print("Loading model is done.")

    # Preparing the test data
    transform_imag = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH),antialias=True),
    transforms.Normalize(mean=[0.0], std=[1.0])
    ])
    testloader =get_test(
        test_csv,
        TEST_IMG_DIR,
        transform_imag,
        NUM_WORKERS
        )
    with torch.no_grad():
        for i,sample in enumerate(testloader,0):
            img_OI,img_OD, label,oi_name,od_name =sample
            img_OI, img_OD, label = img_OI.to(device=DEVICE), img_OD.to(device=DEVICE), label.to(device=DEVICE)
            out1,out2 = model(img_OI,img_OD)
            euclidean_distance = F.pairwise_distance(out1, out2)
            print(f"OI: { oi_name[0].split('/')[-1][:-4]}  OD: {od_name[0].split('/')[-1][:-4]}")
            print("Ecludian Distance: ",euclidean_distance.item())
            df = df._append({
                'OI Name': oi_name[0].split('/')[-1][:-4],
                'OD Name': od_name[0].split('/')[-1][:-4],
                'Euclidean Distance': euclidean_distance.item()
                }, ignore_index=True)
    df.to_excel('3-Red_Siamesa/euclidean_distances.xlsx', index=False)
if __name__ == "__main__":
    main()