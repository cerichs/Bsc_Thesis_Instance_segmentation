from preprocessing.Display_mask import load_coco
from preprocessing.extract_process_grains import process_data, find_imgs, extract_specifics
from two_stage.pls_watershed import spectra_plot, PLS_evaluation, PLS_show
from two_stage.HSI_mean_msc import mean_centering, msc_hyp, median_centering
from two_stage.output_results import create_dataframe
import pandas as pd

def load_paths():
    """
    Load paths for the dataset, images, and annotations.

    Returns:
        dict: A dictionary containing paths for train/test annotations and images.
    """
    return {
        'train_annotation_path': r"C:\Users\jver\Desktop\Training\windows\COCO_HSI_windowed.json",
        'train_image_dir': r"C:\Users\jver\Desktop\Training\windows\PLS_eval_img_rgb",
        'test_annotation_path': r"C:\Users\jver\Desktop\Validation\windows\COCO_HSI_windowed.json",
        'test_image_dir': r"C:\Users\jver\Desktop\Validation\windows\PLS_eval_img_rgb",
        'hyperspectral_path_train': r"C:\Users\jver\Desktop\Training\windows\PLS_eval_img",
        'hyperspectral_path_test': r"C:\Users\jver\Desktop\Validation\windows\PLS_eval_img",
    }


def load_data(paths):
    """
    Load training and test datasets.

    Args:
        paths (dict): A dictionary containing paths for train/test annotations.

    Returns:
        tuple: A tuple containing train_dataset and test_dataset.
    """
    train_dataset = load_coco(paths['train_annotation_path'])
    test_dataset = load_coco(paths['test_annotation_path'])
    return train_dataset, test_dataset



def train_PLS_classifier(train_dataset, paths, whole_img_average=True):
    """
    Train PLS classifiers using different data preprocessing methods.

    Args:
        train_dataset (COCO): COCO format dataset for training.
        paths (dict): A dictionary containing paths for train/test annotations and images.
        whole_img_average (bool, optional): Use whole image averages if True, pixel averages otherwise. Defaults to True.

    Returns:
        tuple: A tuple containing two dictionaries: trained classifiers and reference spectra for MSC and median centering.
    """
    
    hyperspectral_path_train, train_image_dir = paths['hyperspectral_path_train'], paths['train_image_dir']
    
    # Find images and process data
    train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids = find_imgs(train_dataset, hyperspectral_path_train, train_image_dir)
    mask_id, pixel_averages, pixel_medians, labels, HSI_train, split = process_data(train_dataset, train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids, whole_img=whole_img_average)
     
    # Create dataframes for training
    train_dataframe = create_dataframe(mask_id, labels, pixel_averages, split=f"train_{split}")
    train_dataframe_median = create_dataframe(mask_id, labels, pixel_medians, split=f"train_median{split}")
    
    # Prepare data for training
    X_train = train_dataframe.iloc[:,2:]
    y_train = train_dataframe.label
    #X_train = train_dataframe_median.iloc[:,2:]
    #y_train = train_dataframe_median.label
    
    # Perform different data preprocessing methods
    mean_data = mean_centering(X_train)
    _ = create_dataframe(mask_id, labels, mean_data, split=f"train_mean_{split}")

    msc_data, ref_train = msc_hyp(X_train, ref=None)
    pd.DataFrame(ref_train).to_csv("MSC.csv")
    msc_mean = mean_centering(msc_data, ref=None)
    _ = create_dataframe(mask_id, labels, msc_mean, split=f"train_meanMSC_{split}")
    
    median_data, ref_median = median_centering(X_train, ref=None)
    pd.DataFrame(ref_median).to_csv("median.csv")
    _ = create_dataframe(mask_id, labels, median_data, split=f"train_median_{split}")
    
    
    spectra_plot(X_train, y_train, type_classifier=f"Train Original {split}")
    _, _, train_classifier  = PLS_evaluation(X_train, y_train, type_classifier=f"Train Original {split}")
    
    spectra_plot(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")
    _, _, train_classifier_mean  = PLS_evaluation(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")

    spectra_plot(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")
    _, _, train_classifier_msc  = PLS_evaluation(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")

    spectra_plot(median_data, y_train, type_classifier=f"Train Median {split}")
    _, _, train_classifier_median  = PLS_evaluation(median_data, y_train, type_classifier=f"Train Median {split}")

    return {
        'original': train_classifier,
        'mean_centered': train_classifier_mean,
        'msc_mean_centered': train_classifier_msc,
        'median_centered': train_classifier_median
    }, {
        'msc_ref': ref_train,
        'median_ref': ref_median
    }


def test_PLS_classifier(test_dataset, paths, classifiers, refs, whole_img_average=True):
    """
    Test PLS classifiers using different data preprocessing methods.

    Args:
        test_dataset (COCO): COCO format dataset for training.
        paths (dict): A dictionary containing paths for train/test annotations and images.
        classifiers: Fitted PLS-classifiers (different based on the preprocessing methods)
        refs: Reference spectra retained from the training (different based on the preprocesisng methods)
        whole_img_average (bool, optional): Use whole image averages if True, pixel averages otherwise. Defaults to True.

    Returns:
        tuple: Results
    """
    hyperspectral_path_test, test_image_dir = paths['hyperspectral_path_test'], paths['test_image_dir']
    
    test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids = find_imgs(test_dataset, hyperspectral_path_test, test_image_dir)

    mask_id, pixel_averages, pixel_medians, labels, HSI_test, split = process_data(test_dataset, test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids, whole_img=whole_img_average)
        
    test_dataframe = create_dataframe(mask_id, labels, pixel_averages, split=f"test_{split}")
    test_dataframe_median = create_dataframe(mask_id, labels, pixel_medians, split=f"test_median{split}")
    

    X_test = test_dataframe.iloc[:,2:]
    y_test = test_dataframe.label
    #X_test = test_dataframe_median.iloc[:,2:]
    #y_test = test_dataframe_median.label

    mean_data = mean_centering(X_test, ref=refs['msc_ref'])
    _ = create_dataframe(mask_id, labels, mean_data, split=f"test_mean_{split}")

    msc_data, _ = msc_hyp(X_test, ref=refs['msc_ref'])
    msc_mean = mean_centering(msc_data, ref=refs['msc_ref'])
    _ = create_dataframe(mask_id, labels, msc_mean, split=f"test_meanMSC_{split}")
    
    median_data, _ = median_centering(X_test, ref=refs['median_ref'])
    _ = create_dataframe(mask_id, labels, median_data, split=f"test_median_{split}")
    
    results = {}

    _, results['original'], _ = PLS_evaluation(X_test, y_test, classifier=classifiers['original'], type_classifier=f"Test Original {split}")
    _, results['mean_centered'], _ = PLS_evaluation(mean_data, y_test, classifier=classifiers['mean_centered'], type_classifier=f"Test Mean-Centered {split}")
    _, results['msc_mean_centered'], _ = PLS_evaluation(msc_mean, y_test, classifier=classifiers['msc_mean_centered'], type_classifier=f"Test MSC Mean-Centered {split}")
    _, results['median_centered'], _ = PLS_evaluation(test_dataframe_median.iloc[:,2:], y_test, classifier=classifiers['median_centered'], type_classifier=f"Test Median {split}")

    PLS_show(classifiers['original'], X_test, y_test, HSI_test, results['original'], test_dataset, test_pseudo_imgs, test_img_names, ref=refs['msc_ref'], variation="Orig", type_classifier=f"Test Original {split}")
    PLS_show(classifiers['mean_centered'], mean_data, y_test, HSI_test, results['mean_centered'], test_dataset, test_pseudo_imgs, test_img_names, ref=refs['msc_ref'], variation="Mean", type_classifier=f"Test Mean-Centered {split}")
    PLS_show(classifiers['msc_mean_centered'], msc_mean, y_test, HSI_test, results['msc_mean_centered'], test_dataset, test_pseudo_imgs, test_img_names, ref=refs['msc_ref'], variation="MSC", type_classifier=f"Test MSC Mean-Centered {split}")
    PLS_show(classifiers['median_centered'], median_data, y_test, HSI_test, results['median_centered'], test_dataset, test_pseudo_imgs, test_img_names, ref=refs['median_ref'], variation="Orig", type_classifier=f"Test Original Median {split}")
    
    return results


def main():
    """
    Main function that loads the dataset and performs PLS classification on the training and test sets.
    """
    paths = load_paths()
    
    # Load training and test datasets
    train_dataset, test_dataset = load_data(paths)

    # Train PLS classifier
    classifiers, refs = train_PLS_classifier(train_dataset, paths)

    # Test PLS classifier
    test_PLS_classifier(test_dataset, paths, classifiers, refs)
    
    
    
    






if __name__ == "__main__":
    main()
