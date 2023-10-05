from utils import get_test,check_accuracy
import torch
from sklearn.metrics import f1_score,jaccard_score,recall_score,precision_score 
import torch.nn as nn
from torchvision import transforms
import segmentation_models_pytorch as smp
from main import(
    ACTIVATION, ENCODER_NAME, ENCODER_WEIGHTS, NUM_WORKERS, IMAGE_HEIGHT,
    IMAGE_WIDTH,  PIN_MEMORY )

BEST_M_CHECKPOINT_DIR = r"1-Unet/Code/saved_models/best_model_unet_aug_N.pth.tar"
TEST_IMG_DIR = r"Data/test/Imag" 
TEST_MASK_DIR = r"Data/test/Labels"
BACTH_SIZE=1
def main():
    #Load the saved model for testing
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
    print("Testing model on test data")
    loss_fn = nn.CrossEntropyLoss()
    epoch= 300 # for test data when saved will have a 300. (me da pereza hacer una nueva funcion)
    metrics= {'f1_score': f1_score, 'PPV': precision_score, 'IoU': jaccard_score, 'Recall': recall_score }
    folder_test =r"/home/jacobo15defrutos/AVS9/Data/saved_test_images"
    test_loss = check_accuracy(testloader,metrics, model, epoch, loss_fn,folder_test, device=DEVICE)
    print("End of testing")
if __name__ == "__main__":
    main()