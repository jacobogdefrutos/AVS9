import os 
import csv

def main():
    directory = r'5-Pruebas/test'  # Replace with your directory path
    csv_file_name = r'5-Pruebas/test.csv'
    file_train_list = os.listdir(directory)
    #Lo dividimos en tres listas: OD, OI y Label
    file_OD_list= [name for name in file_train_list if 'OD' in name ]
    file_OI_list= [name for name in file_train_list if 'OI' in name ]
    file_list= file_OD_list+file_OI_list
    # cuatro clases: 0 SANO, 1 CMV, 2 SURV
    with open(csv_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for idx,name in enumerate(file_list):
            if 'SANO'  in name:
                label=0
                csv_writer.writerow( [name, label])
            elif 'CMV'in name:
                label=1
                csv_writer.writerow( [name, label])
            elif 'SURV'in name:
                label=2
                csv_writer.writerow( [name, label])

if __name__ == "__main__":
    main() 
