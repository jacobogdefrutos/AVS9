from ultralytics import YOLO
import numpy as np
import torch
import cv2
from utils import get_test, get_totalmask,draw_translucent_seg_maps,find_number_in_string
from sklearn.metrics import f1_score,jaccard_score,recall_score,precision_score
from model import ModelSimple
import matplotlib.pyplot as plt
from collections import defaultdict
from resize import resize_image_with_pading



BEST_M_CHECKPOINT_DIR = r"/home/jacobo15defrutos/AVS9/6-SAM/saved_best_model/best_model_SAM_11_epochs.pth.tar"
TEST_IMG_DIR = r"/home/jacobo15defrutos/AVS9/Data/Data_new_SAM/test/Imag" 
TEST_MASK_DIR = r"/home/jacobo15defrutos/AVS9/Data/Data_new_SAM/test/Labels" #Dejo este path pero no vamos a necesitarlo
NUM_WORKERS=8
PIN_MEMORY=True
NEW_SIZE=(800,800)
def output(loader,model_sam,transform,boxes_dic,names,folder, device):
    IoU_iris_list =[]
    IoU_iris_list_sam=[]
    PPV_iris_list =[]
    Recall_iris_list =[]
    f1_score_iris_list=[]
    model_sam.eval()
    with torch.no_grad():
        for i, sample in enumerate(iter(loader)):
            image = sample['image'].squeeze(1).to(device)
            mask= sample['mask'].squeeze(1).long().to(device)
            idx= sample['idx'].item()
            og_y,og_x= sample['original_image_size']
            original_image_size=(og_y.item(),og_x.item())
            prompt_box=boxes_dic[idx]['cords']
            box = transform.apply_boxes(prompt_box, original_image_size)#esto revisarlo porque no se si lo hace bien
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]
            mask=mask[0]
            total_mask = get_totalmask(mask)
            total_mask = total_mask.to(device)
            preds, iou = model_sam(image,box_torch)
            
            #ahora pintamos de negro el background y mantenemos la segmentacion
            preds_prob = torch.sigmoid(preds.squeeze(1))# shape (1,1024,1024)
            preds_prob_numpy = preds_prob.cpu().numpy().squeeze()
            preds_binary = (preds_prob_numpy > 0.85).astype(np.uint8)
            test_image= image[0].numpy()
            test_image = np.array(test_image, dtype=np.uint8)
            test_image = np.transpose(test_image, (1, 2, 0))
            """""
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # Plot the first image on the left
            axes[0].imshow(np.array(test_image),cmap='gray')  # Assuming the first image is grayscale
            axes[0].set_title("Image")

            # Plot the second image on the right
            axes[1].imshow(preds_binary, cmap='gray')  # Assuming the second image is grayscale
            axes[1].set_title("Mask")

            # Plot the second image on the right
            axes[2].imshow(preds_prob_numpy)  # Assuming the second image is grayscale
            axes[2].set_title("Probability Map")

            # Hide axis ticks and labels
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            # Display the images side by side
            plt.show()
            plt.close('all')
            """
            draw_translucent_seg_maps(image, preds_binary, 1000,idx,folder)
            """""
            if len(np.unique(preds_binary))>1:
                ppv=precision_score(total_mask.numpy(),preds_binary,average='micro')
                recall= recall_score(total_mask.numpy(),preds_binary,average='micro')
                iou_sklearn= jaccard_score(total_mask.numpy(),preds_binary,average='micro')
                f_score=f1_score(total_mask.numpy(),preds_binary,average='micro')
            else:
                ppv=0
                recall=0
                iou_sklearn=0
                f_score=0
            IoU_iris_list_sam.append(iou.item())
            IoU_iris_list.append(iou_sklearn)
            PPV_iris_list.append(ppv)
            Recall_iris_list.append(recall)
            f1_score_iris_list.append(f_score)
    mean_IoU_iris = np.mean(IoU_iris_list)
    mean_PPV_iris = np.mean(PPV_iris_list)
    mean_Recall_iris = np.mean(Recall_iris_list)
    mean_f1_score_iris=np.mean(f1_score_iris_list)
    print("     Iris IoU sklearn: ", mean_IoU_iris*100, "%")
    print("     Irris PPV: ", mean_PPV_iris*100, "%")
    print("     Iris Recall: ", mean_Recall_iris*100, "%")
    print(f"    Iris F1_score: {mean_f1_score_iris*100} %")
    """
def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_yolo = YOLO("/home/jacobo15defrutos/AVS9/6-SAM/saved_best_model/best.pt")
    model_sam = ModelSimple()
    model_sam.setup()
    transform = model_sam.transform
    best_checkpoint= torch.load(BEST_M_CHECKPOINT_DIR)
    model_sam.load_state_dict(best_checkpoint['model_state_dict'])
    model_sam.to(DEVICE)
    model_sam.eval()
    resize_image_with_pading(TEST_IMG_DIR,TEST_IMG_DIR,NEW_SIZE)
    resize_image_with_pading(TEST_MASK_DIR,TEST_MASK_DIR,NEW_SIZE)
    print("Loading model is done.")
    preds= model_yolo.predict(TEST_IMG_DIR)
    boxes_dic= defaultdict(dict)
    for pred in preds:
        file_name = pred.path
        number= find_number_in_string(str(file_name).split('/')[-1][:-4])
        #por ahora nos centramos solo en coger un box, la de mayor conf
        if len(pred.boxes.data)!=0:#There is a detection
            _,max_index = torch.max(pred.boxes.conf, dim=0)
            box= pred.boxes[max_index.item()]
            cords = box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(int, cords)
            box_sam=[x_min, y_min, x_max, y_max]
            boxes_dic[number]['cords']=np.array(box_sam)
        else:
            x_min,y_min,x_max,y_max =[0,0,pred.orig_shape[1],pred.orig_shape[0]]
            box_sam=[x_min, y_min, x_max, y_max]
            boxes_dic[number]['cords']=np.array(box_sam)

    # Preparing the test data
   
    testloader,names =get_test(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        NUM_WORKERS,
        transform,
        PIN_MEMORY
        )

    print("Generate segmentations")
    folder_test2 =r"/home/jacobo15defrutos/AVS9/6-SAM/Results_test_seg"
    output(testloader,model_sam,transform,boxes_dic,names,folder_test2,device=DEVICE)
if __name__ == "__main__":
    main()