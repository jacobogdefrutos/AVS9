import os 
import csv

def main():
    directory = r'5-Pruebas/Total_images'  # Replace with your directory path
    csv_file_name = r'4-Red_Siamesa/Code/total_images_RS.csv'
    file_train_list = os.listdir(directory)
    #Lo dividimos en tres listas: OD, OI y Label
    file_OD_list= [name for name in file_train_list if 'OD' in name ]
    file_OI_list= [name for name in file_train_list if 'OI' in name ]
    # cuatro clases: 0 SANO, 1 CMV, 2 SURV, 3 POSTNER, 4 SUF
    with open(csv_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for idx,name in enumerate(file_OD_list):
            label=101
            if 'SANO' in name and 'SANO' in file_OI_list[idx]:
                label=0
                csv_writer.writerow([file_OI_list[idx], name, label])
            elif ('SANO' in name and 'CMV' in file_OI_list[idx]) or ('CMV' in name and 'SANO' in file_OI_list[idx]):
                label=1
                csv_writer.writerow([file_OI_list[idx], name, label])
            elif ('SANO' in name and 'SURV' in file_OI_list[idx]) or ('SURV' in name and 'SANO' in file_OI_list[idx]):
                label=2
                csv_writer.writerow([file_OI_list[idx], name, label])

if __name__ == "__main__":
    main() 
