import json
import shutil
import os

WLASL_JSON_FILE = "/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL2000/data/splits/asl100.json"
WLASL_CLASSES = "/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL_Utilities/WLASL_classes.json"

NEW_WLASL_LOCATION = "/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL-100"
EXISTING_WLASL_LOCATION = "/home/izzy/Documents/UoA/Sem_1_2022/P4P/WLSLR/RGB/WLASL2000"

def write_classes():
    classes_f = open(WLASL_CLASSES, "w")
    classes_f.write("[")

    f = open(WLASL_JSON_FILE)
    dataset_json = json.load(f)

    for data in dataset_json:
        classes_f.write("\"{}\",\r\n".format(data['gloss']))

    classes_f.write("]")
    classes_f.close()
    f.close()

def copy_videos():
    f = open(WLASL_JSON_FILE)
    dataset_json = json.load(f)

    label_num = 0

    for data in dataset_json:
        pad_label = str(label_num).zfill(4)
        for instance in data['instances']:
            files = os.listdir(NEW_WLASL_LOCATION + "/" + instance['split'])
            sample_num = [file for file in files if file.startswith(pad_label)]
                        
            try:
                pad_sample = str(len(sample_num)).zfill(3)
                shutil.copy(EXISTING_WLASL_LOCATION + "/" + instance['video_id'] + ".mp4", 
                NEW_WLASL_LOCATION + "/" + instance['split'] + "/" + pad_label + pad_sample + ".mp4")
            except FileNotFoundError:
                print("Can't find video " + instance['video_id'] + ".mp4")
        label_num += 1
    
    f.close()

if __name__ == '__main__':
    copy_videos()