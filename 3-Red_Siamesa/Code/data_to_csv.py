import os 
import csv
#------------Cuando se runee el codigo, se debe de borrar del csv resultante las ultimas lineas hasta la 120 (longitud de la lista)
def main():
    directory = r'Data\saved_test_clas_images\val'  # Replace with your directory path
    csv_file_name = 'Iris_val_seg_list.csv'
    file_train_list = os.listdir(directory)
    #Lo dividimos en tres listas: OD, OI y Label
    file_OD_list= [name for name in file_train_list if 'OD' in name ]
    file_OI_list= [name for name in file_train_list if 'OI' in name ]
    # cuatro clases: 0 SANO, 1 CMV, 2 SURV, 3 POSTNER, 4 SUF
    with open(csv_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for idx,name in enumerate(file_OD_list):
            label=101# en case de que haya algun error
            label2=101
            if 'SANO' in name and 'SANO' in file_OI_list[idx]:
                label=0
                label2=0#si el paciente es sano, es decir ambos iris son"iguales"--> label=0
            elif ('SANO' in name and 'CMV' in file_OI_list[idx]) or ('CMV' in name and 'SANO' in file_OI_list[idx]):
                label=1
                label2=1
            elif ('SANO' in name and 'SURV' in file_OI_list[idx]) or ('SURV' in name and 'SANO' in file_OI_list[idx]):
                label=2
                label2=1
            elif ('SANO' in name and 'POSTNER' in file_OI_list[idx]) or ('POSTNER' in name and 'SANO' in file_OI_list[idx]):
                label=3
                label2=1
            elif ('SANO' in name and 'SUF' in file_OI_list[idx]) or ('SUF' in name and 'SANO' in file_OI_list[idx]):
                label=4
                label2=1
            csv_writer.writerow([file_OI_list[idx], name, label,label2])

if __name__ == "__main__":
    main() 
