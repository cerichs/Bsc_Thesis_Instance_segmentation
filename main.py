
from preprocessing.Display_mask import load_coco
from preprocessing.extract_process_grains import process_data, find_imgs, extract_specifics
from two_stage.pls_watershed import spectra_plot, PLS_evaluation, PLS_show
from two_stage.HSI_mean_msc import mean_centering, msc_hyp, median_centering
from two_stage.output_results import create_dataframe
import pandas as pd



def load_paths2():
    """
    Load paths for the dataset, images, and annotations.

    Returns:
        dict: A dictionary containing paths for train/Validation annotations and images.
    """
    return {
        'train_annotation_path': r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\COCO_Training.json",
        'train_image_dir': r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\images",
        'Validation_annotation_path': r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Validation\COCO_Validation.json",
        'Validation_image_dir': r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Validation\images",
        'hyperspectral_path_train': r"C:\Users\jver\Desktop\Training",
        'hyperspectral_path_Validation': r"C:\Users\jver\Desktop\Validation",
    }


def load_paths():
    """
    Load paths for the dataset, images, and annotations.

    Returns:
        dict: A dictionary containing paths for train/Validation annotations and images.
    """
    return {
        'train_annotation_path': r"C:\Users\jver\Desktop\Training\windows\COCO_HSI_windowed.json",
        'train_image_dir': r"C:\Users\jver\Desktop\Training\windows\PLS_eval_img_rgb",
        
        'test_annotation_path': r"C:\Users\jver\Desktop\Test\windows\COCO_HSI_windowed.json",
        'test_image_dir': r"C:\Users\jver\Desktop\Test\windows\PLS_eval_img_rgb",
        
        'Validation_annotation_path': r"C:\Users\jver\Desktop\Validation\windows\COCO_HSI_windowed.json",
        'Validation_image_dir': r"C:\Users\jver\Desktop\Validation\windows\PLS_eval_img_rgb",
        
        'hyperspectral_path_train': r"C:\Users\jver\Desktop\Training\windows\PLS_eval_img",
        'hyperspectral_path_test': r"C:\Users\jver\Desktop\Test\windows\PLS_eval_img",
        'hyperspectral_path_Validation': r"C:\Users\jver\Desktop\Validation\windows\PLS_eval_img",
    }


def load_data(paths):
    """
    Load training and Validation datasets.

    Args:
        paths (dict): A dictionary containing paths for train/Validation annotations.

    Returns:
        tuple: A tuple containing train_dataset and Validation_dataset.
    """
    train_dataset = load_coco(paths['train_annotation_path'])
    test_dataset = load_coco(paths['test_annotation_path'])
    Validation_dataset = load_coco(paths['Validation_annotation_path'])
    return train_dataset, Validation_dataset, test_dataset



def train_PLS_classifier(train_dataset, paths, whole_img_average=True):
    """
    Train PLS classifiers using different data preprocessing methods.

    Args:
        train_dataset (COCO): COCO format dataset for training.
        paths (dict): A dictionary containing paths for train/Validation annotations and images.
        whole_img_average (bool, optional): Use whole image averages if True, pixel averages otherwise. Defaults to True.

    Returns:
        tuple: A tuple containing two dictionaries: trained classifiers and reference spectra for MSC and median centering.
    """
    
    hyperspectral_path_train, train_image_dir = paths['hyperspectral_path_train'], paths['train_image_dir']
    
    # Find images and process data
    train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids = find_imgs(train_dataset, hyperspectral_path_train, train_image_dir)
    mask_id, pixel_averages, pixel_medians, labels, HSI_train, split = process_data(train_dataset, train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids, whole_img=whole_img_average)
     
    #split = "median" + split
    # Create dataframes for training
    train_dataframe = create_dataframe(mask_id, labels, pixel_averages, split=f"train_{split}")
    #train_dataframe_median = create_dataframe(mask_id, labels, pixel_medians, split=f"train_median{split}")
    
    # Prepare data for training
    X_train = train_dataframe.iloc[:,2:]
    y_train = train_dataframe.label
    #X_train = train_dataframe_median.iloc[:,2:]
    #y_train = train_dataframe_median.label
    
    # Perform different data preprocessing methods
    mean_data = mean_centering(X_train)
    _ = create_dataframe(mask_id, labels, mean_data, split=f"train_mean_{split}")

    msc_data, ref_train = msc_hyp(X_train, ref=None)
    pd.DataFrame(ref_train).to_csv(f"{split}_MSC.csv")
    msc_mean = mean_centering(msc_data, ref=None)
    _ = create_dataframe(mask_id, labels, msc_mean, split=f"train_meanMSC_{split}")
    
    _, _, train_classifier  = PLS_evaluation(X_train, y_train, type_classifier=f"Train Original {split}")
    _, _, train_classifier_mean  = PLS_evaluation(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")
    _, _, train_classifier_msc  = PLS_evaluation(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")


    # We only want to show spectra's on image-crops and not on every grain kernel
    if whole_img_average:
        spectra_plot(X_train, y_train, type_classifier=f"Train Original {split}")
        spectra_plot(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")
        spectra_plot(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")
        

    return {
        'original': train_classifier,
        'mean_centered': train_classifier_mean,
        'msc_mean_centered': train_classifier_msc
    }, {
        'msc_ref': ref_train
    }


def validation_PLS_classifier(Validation_dataset, paths, classifiers, refs, whole_img_average=True):
    """
    Validation PLS classifiers using different data preprocessing methods.

    Args:
        Validation_dataset (COCO): COCO format dataset for validation.
        paths (dict): A dictionary containing paths for train/test/Validation annotations and images.
        classifiers: Fitted PLS-classifiers from training (different based on the preprocessing methods)
        refs: Reference spectra retained from the training (different based on the preprocesisng methods)
        whole_img_average (bool, optional): Use whole image averages if True, pixel averages otherwise. Defaults to True.

    Returns:
        tuple: Results
    """
    
    #os.chdir("../")
    
    hyperspectral_path_Validation, Validation_image_dir = paths['hyperspectral_path_Validation'], paths['Validation_image_dir']
    
    # Find images and process data
    Validation_hyper_imgs, Validation_pseudo_imgs, Validation_img_names, Validation_image_ids = find_imgs(Validation_dataset, hyperspectral_path_Validation, Validation_image_dir)
    mask_id, pixel_averages, pixel_medians, labels, HSI_Validation, split = process_data(Validation_dataset, Validation_hyper_imgs, Validation_pseudo_imgs, Validation_img_names, Validation_image_ids, whole_img=whole_img_average)
    
    #split = "median" + split
    # Create dataframes for validation
    Validation_dataframe = create_dataframe(mask_id, labels, pixel_averages, split=f"Validation_{split}")
    #Validation_dataframe_median = create_dataframe(mask_id, labels, pixel_medians, split=f"Validation_median{split}")
    
    # Prepare data for validation
    X_Validation = Validation_dataframe.iloc[:,2:]
    y_Validation = Validation_dataframe.label
    #X_Validation = Validation_dataframe_median.iloc[:,2:]
    #y_Validation = Validation_dataframe_median.label
    
    # Perform different data preprocessing methods
    mean_data = mean_centering(X_Validation, ref=refs['msc_ref'])
    _ = create_dataframe(mask_id, labels, mean_data, split=f"Validation_mean_{split}")

    msc_data, _ = msc_hyp(X_Validation, ref=refs['msc_ref'])
    msc_mean = mean_centering(msc_data, ref=refs['msc_ref'])
    _ = create_dataframe(mask_id, labels, msc_mean, split=f"Validation_meanMSC_{split}")
    
    
    results = {}

    _, results['original'], _ = PLS_evaluation(X_Validation, y_Validation, classifier=classifiers['original'], type_classifier=f"Validation Original {split}")
    _, results['mean_centered'], _ = PLS_evaluation(mean_data, y_Validation, classifier=classifiers['mean_centered'], type_classifier=f"Validation Mean-Centered {split}")
    _, results['msc_mean_centered'], _ = PLS_evaluation(msc_mean, y_Validation, classifier=classifiers['msc_mean_centered'], type_classifier=f"Validation MSC Mean-Centered {split}")
    
    PLS_show(classifiers['original'], X_Validation, y_Validation, HSI_Validation, results['original'], Validation_dataset, Validation_pseudo_imgs, Validation_img_names, ref=refs['msc_ref'], variation="Orig", type_classifier=f"Validation Original {split}")
    PLS_show(classifiers['mean_centered'], mean_data, y_Validation, HSI_Validation, results['mean_centered'], Validation_dataset, Validation_pseudo_imgs, Validation_img_names, ref=refs['msc_ref'], variation="Mean", type_classifier=f"Validation Mean-Centered {split}")
    PLS_show(classifiers['msc_mean_centered'], msc_mean, y_Validation, HSI_Validation, results['msc_mean_centered'], Validation_dataset, Validation_pseudo_imgs, Validation_img_names, ref=refs['msc_ref'], variation="MSC", type_classifier=f"Validation MSC Mean-Centered {split}")

    return results

# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:01:01 2023

@author: jver
"""

def test_PLS_classifier(test_dataset, paths, classifiers, refs, RMSE_validation, whole_img_average=True):
    """
    test PLS classifiers using different data preprocessing methods.

    Args:
        test_dataset (COCO): COCO format dataset for test.
        paths (dict): A dictionary containing paths for train/test/test annotations and images.
        classifiers: Fitted PLS-classifiers from training (different based on the preprocessing methods)
        refs: Reference spectra retained from the training (different based on the preprocesisng methods)
        whole_img_average (bool, optional): Use whole image averages if True, pixel averages otherwise. Defaults to True.

    Returns:
        tuple: Results
    """
    #os.chdir("../")
    # Find images and process data
    hypersepctral_path_test, test_image_dir = paths['hyperspectral_path_test'], paths['test_image_dir']
    
    # Create dataframes for test
    test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids = find_imgs(test_dataset, hypersepctral_path_test, test_image_dir)
    mask_id, pixel_averages, pixel_medians, labels, HSI_test, split = process_data(test_dataset, test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids, whole_img=whole_img_average)
    
    #split = "median" + split
    # Create dataframes for test
    test_dataframe = create_dataframe(mask_id, labels, pixel_averages, split=f"test_{split}")
    #test_dataframe_median = create_dataframe(mask_id, labels, pixel_medians, split=f"test_median{split}")
    
    # Prepare data for test
    X_test = test_dataframe.iloc[:,2:]
    y_test = test_dataframe.label
    #X_test = test_dataframe_median.iloc[:,2:]
    #y_test = test_dataframe_median.label
    
    # Perform different data preprocessing methods
    mean_data = mean_centering(X_test, ref=refs['msc_ref'])
    _ = create_dataframe(mask_id, labels, mean_data, split=f"test_mean_{split}")
    msc_data, _ = msc_hyp(X_test, ref=refs['msc_ref'])
    msc_mean = mean_centering(msc_data, ref=refs['msc_ref'])
    _ = create_dataframe(mask_id, labels, msc_mean, split=f"test_meanMSC_{split}")
    
    results = {}

    #_, _, _ = PLS_evaluation(X_test, y_test, classifier=classifiers['original'], type_classifier=f"test Original {split}")
    #_, _, _ = PLS_evaluation(mean_data, y_test, classifier=classifiers['mean_centered'], type_classifier=f"test Mean-Centered {split}")
    #_, _, _ = PLS_evaluation(msc_mean, y_test, classifier=classifiers['msc_mean_centered'], type_classifier=f"test MSC Mean-Centered {split}")
    PLS_show(classifiers['original'], X_test, y_test, HSI_test, RMSE_validation['original'], test_dataset, test_pseudo_imgs, test_img_names, ref=refs['msc_ref'], variation="Orig", type_classifier=f"test Original {split}")
    PLS_show(classifiers['mean_centered'], mean_data, y_test, HSI_test, RMSE_validation['mean_centered'], test_dataset, test_pseudo_imgs, test_img_names, ref=refs['msc_ref'], variation="Mean", type_classifier=f"test Mean-Centered {split}")
    PLS_show(classifiers['msc_mean_centered'], msc_mean, y_test, HSI_test, RMSE_validation['msc_mean_centered'], test_dataset, test_pseudo_imgs, test_img_names, ref=refs['msc_ref'], variation="MSC", type_classifier=f"test MSC Mean-Centered {split}")

    return results


def main():
    """
    Main function that loads the dataset and performs PLS classification on the training and Validation sets.
    """
    paths = load_paths()
    
    # Load training and Validation datasets
    train_dataset, validation_dataset, test_dataset = load_data(paths)

    # Train PLS classifier
    classifiers, refs = train_PLS_classifier(train_dataset, paths, whole_img_average=False)

    # Validation PLS classifier
    RMSE = validation_PLS_classifier(validation_dataset, paths, classifiers, refs, whole_img_average=False)
    
    
    # Validation PLS classifier
    test_PLS_classifier(test_dataset, paths, classifiers, refs, RMSE, whole_img_average=False)
    
    






if __name__ == "__main__":
    main()
