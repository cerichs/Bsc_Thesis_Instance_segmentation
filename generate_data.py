import os
import sys

from preprocessing.rename_hsi_files import rename_files
from preprocessing.random_original_HSI import extract_subwindow_main
from preprocessing.simple_object_placer import create_synt_images



def load_paths(base_dir):
    """
    Load paths for the dataset, images, and annotations.

    Returns:
        dict: A dictionary containing paths for train/Validation annotations and images.
    """
    paths =  {
        
        'train_annotation_path': os.path.join(base_dir, "data", "PseudoRGB", "Training", "COCO_Training.json"),
        'train_image_dir': os.path.join(base_dir, "data", "PseudoRGB", "Training", "images"),
        
        'test_annotation_path': os.path.join(base_dir, "data", "PseudoRGB", "Test", "COCO_Test.json"),
        'test_image_dir': os.path.join(base_dir, "data", "PseudoRGB", "Test", "images"),
        
        'Validation_annotation_path': os.path.join(base_dir,"data", "PseudoRGB", "Validation", "COCO_Validation.json"),
        'Validation_image_dir': os.path.join(base_dir, "data", "PseudoRGB", "Validation", "images"),
        
        'hyperspectral_path_train': os.path.join(base_dir, "data", "e4Wr5LFI4L", "Training"),
        'hyperspectral_path_test': os.path.join(base_dir, "data", "e4Wr5LFI4L", "Test"),
        'hyperspectral_path_Validation': os.path.join(base_dir, "data", "e4Wr5LFI4L", "Validation")
    }
    
    for key, value in paths.items():
       if not os.path.exists(value):
           print(f"Path {value} does not exist. Please ensure that your data is correctly inserted (refer to readme)")
           sys.exit()
           
    return paths
  
                                              
                                             


def get_user_input():
    while True:
        try:
            n = int(input("Please enter the size of your dataset: "))
            print(f"You entered {n}")
            print("")
            print(f"This will generate:")
            print(f"{int(n*0.8)} training images,") 
            print(f"{int(n*0.1)} validation images,")
            print(f"{int(n*0.1)} test images,")
            print("")
            return n
        except ValueError:
            print("Invalid input. Please try again.")


def get_synthetic_input(N):
    while True:
        print("")
        print("Do you want to create synthetic images? The amount of time to generate n synthetic images would be: 2 minutes pr. image")
        response = input("Answer y/n:")
        if response.lower() not in ['y', 'n']:
            print("Invalid input, please enter 'y' or 'n'")
            continue
        else:
            return response.lower()

def main():
    """
    Main function that loads the dataset and performs PLS classification on the training and Validation sets.
    """
    
    
    base_dir = os.path.dirname(os.path.abspath(__file__))  # gets the location of the current script file
    
    
    paths = load_paths(base_dir)
    
    # Generate data in appropriate format
    rename_files(paths['hyperspectral_path_train'])
    rename_files(paths['hyperspectral_path_Validation'])
    rename_files(paths['hyperspectral_path_test'])
    
    
    # Extracting path to COCO-json files of each data split
    train_dataset = paths['train_annotation_path']
    test_dataset = paths['test_annotation_path']
    validation_dataset = paths['Validation_annotation_path']
    
    
    
    
    # user-input to specify the number of cropped data images
    n = get_user_input()
    
    # Create cropped images
    extract_subwindow_main(train_dataset, paths['hyperspectral_path_train'], paths['train_image_dir'], n=int(n*0.8))
    extract_subwindow_main(validation_dataset, paths['hyperspectral_path_Validation'], paths['Validation_image_dir'], n=int(n*0.1))
    extract_subwindow_main(test_dataset, paths['hyperspectral_path_test'], paths['test_image_dir'], n=int(n*0.1))
    
    # user input to generate synthetic images
    synthetic_response = get_synthetic_input(n)

    # Calculate estimated time if response is 'y'
    if synthetic_response == 'y':
        print(f"Creating {int(n*0.8)} synthetic training images:")
        print("")
        create_synt_images(train_dataset, paths['hyperspectral_path_train'], paths['train_image_dir'], n=int(n*0.8))
        print(f"Creating {int(n*0.1)} synthetic validaiton images:")
        print("")
        create_synt_images(validation_dataset, paths['hyperspectral_path_Validation'], paths['Validation_image_dir'], n=int(n*0.1))
        print(f"Creating {int(n*0.1)} synthetic test images:")
        print("")
        create_synt_images(test_dataset, paths['hyperspectral_path_test'], paths['test_image_dir'], n=int(n*0.1))
        
    print("")
    print("Done creating images...")
        
if __name__ == "__main__":
    main()