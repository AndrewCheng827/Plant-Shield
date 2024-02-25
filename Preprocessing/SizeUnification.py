import cv2
import os
import json
from tqdm import tqdm


"""
Get size distribution of the dataset

Params:
- dataset_folder: Path to the root folder of the dataset
- categories: a list of categories of the data
"""
def get_size_distribution(dataset_folder):    
    possible_sizes = {}
    possible_aspect_ratios = {}
    total = 0 # Used to calculate the weighted average of aspect ratioss
    entries = os.listdir(dataset_folder)
    categories = [entry for entry in entries if os.path.isdir(os.path.join(dataset_folder, entry))]

    for category in categories:
        class_folder_path = os.path.join(dataset_folder, category)
        entries = os.listdir(class_folder_path)
    
        for entry in tqdm(entries):
            image_path = os.path.join(class_folder_path, entry)
            image = cv2.imread(image_path)
            dimensions = image.shape
            if len(dimensions) == 3:
                height, width, _ = dimensions
            else:
                height, width = dimensions

            current_size = f'{width} X {height}'
            if current_size not in possible_sizes:
                possible_sizes[current_size] = 1
            else:
                possible_sizes[current_size] = possible_sizes[current_size] +1

            current_aspect_ratio = max(height, width) / min(height, width)
            if current_aspect_ratio not in possible_aspect_ratios:
                possible_aspect_ratios[current_aspect_ratio] = 1
            else:
                possible_aspect_ratios[current_aspect_ratio] = possible_aspect_ratios[current_aspect_ratio] + 1

    sorted_possible_sizes = sorted(possible_sizes.items(), key=lambda item: item[1], reverse=True)
    sorted_possible_aspect_ratios = sorted(possible_aspect_ratios.items(), key=lambda item: item[0], reverse=True)
    
    pretty_possible_sizes = json.dumps(sorted_possible_sizes, indent=4, sort_keys=True)
    print(pretty_possible_sizes)

    pretty_possible_aspect_ratios = json.dumps(sorted_possible_aspect_ratios, indent=4, sort_keys=True)
    #print(pretty_possible_aspect_ratios)

    for key, value in possible_aspect_ratios.items():
        total = total + key * value
    weighted_aspect_ratio_avg = total / 2569.0 # There are a total of 2569 files in the entire dataset
    #print(weighted_aspect_ratio_avg)

# Resizes all images into the same size
def size_unification(dataset_folder):
    # We will do the following preprocessing:
    # 1. Rotate images such that width >= height

    # 2. Scale images to 1920 * 1080 px
    #    We choose this since the aspect ratio 1920 / 1080 is close to the weighted average
    #    of the aspect ratios of images in the dataset
    target_height = 1080
    target_width = 1920
    entries = os.listdir(dataset_folder)
    categories = [entry for entry in entries if os.path.isdir(os.path.join(dataset_folder, entry))]

    for category in categories:
        class_folder_path = os.path.join(dataset_folder, category)
        entries = os.listdir(class_folder_path)

        for entry in tqdm(entries):
            image_path = os.path.join(class_folder_path, entry)
            image = cv2.imread(image_path)
            
            # Rotating the image if necessary
            dimensions = image.shape
            if len(dimensions) == 3:
                height, width, _ = dimensions
            else:
                height, width = dimensions
            if height > width:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                height, width = width, height

            # Rescaling
            #print(f'TARGET HEIGHT: {target_height}, CORRESPONDING WIDTH: {width * target_aspect_ratio}')
            image = cv2.resize(image, (int(width * target_width / width), target_height))
            if len(dimensions) == 3:
                height, width, _ = dimensions
            else:
                height, width = dimensions
            
            # Padding / Cropping
            if width < target_width:
                left_pad = (target_width - width) // 2
                right_pad = target_width - width - left_pad
                image = cv2.copyMakeBorder(image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0,0,0))
            elif width > target_width:
                left_crop = (width - target_width) // 2
                right_crop = width - target_width - left_crop
                image = image[:, left_crop:width-right_crop]
                
            # Save the image
            image_save_path = os.path.abspath(os.path.join(os.path.join(dataset_folder, category), entry))
            #print(f'PATH: {image_save_path}')
            cv2.imwrite(image_save_path, image)

# if __name__ == "__main__":
#     Sugarcane_Five_Classes dataset

#     Uncomment the line below to see the image size distribution of the dataset
#     This is used to decide how we should preprocess the given images
#     get_size_distribution(dataset_folder)
#     size_unification(dataset_folder)