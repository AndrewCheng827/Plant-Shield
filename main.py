import cv2
import os
from Preprocessing.SizeUnification import *

input_data = {}

def loadData(dataset_folder, num_classes):
    entries = os.listdir(dataset_folder)
    categories = [entry for entry in entries if os.path.isdir(os.path.join(dataset_folder, entry))]

    for category in categories:
        cur_category = []
        class_folder_path = os.path.join(dataset_folder, category)
        entries = os.listdir(class_folder_path)

        for entry in tqdm(entries):
            image_path = os.path.join(class_folder_path, entry)
            cur_category.append(cv2.imread(image_path))

        input_data[category] = cur_category

if __name__ == "__main__":
    dataset_folder = "datasets/Sugarcane_Five_Classes"
    num_classes = 5
    size_unification(dataset_folder)
    loadData(dataset_folder, num_classes)