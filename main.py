from preprocessing.Display_mask import load_coco
from preprocessing.extract_process_grains import process_data, find_imgs
from two_stage.pls_watershed import spectra_plot, PLS_evaluation, PLS_show
from two_stage.HSI_mean_msc import mean_centering, msc_hyp
from two_stage.output_results import create_dataframe


def main():
    """
    Main function that loads the dataset and performs PLS classification on the training and test sets.
    """
    # Load paths
    train_annotation_path =r"C:\Users\jver\Desktop\Training\rgb\COCO_rgb_windowed.json"
    train_image_dir = r"C:\Users\jver\Desktop\Training\rgb"
    #test_annotation_path = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Test\COCO_Test.json"
    #test_image_dir = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Test\images"
    test_annotation_path = r"C:\Users\jver\Desktop\Validation\rgb\COCO_rgb_windowed.json"
    test_image_dir = r"C:\Users\jver\Desktop\Validation\rgb"
    
    hyperspectral_path_train = r"C:\Users\jver\Desktop\Training\windows"
    #hyperspectral_path_test = r"C:\Users\jver\Desktop\Test"
    hyperspectral_path_test = r"C:\Users\jver\Desktop\Validation\windows"
    
    # Load training and test datasets
    train_dataset = load_coco(train_annotation_path)
    test_dataset = load_coco(test_annotation_path)


    # Train PLS classifier
    train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids = find_imgs(train_dataset, hyperspectral_path_train, train_image_dir)

    whole_img_average = True
    image_ids, pixel_averages, labels, HSI_train, split = process_data(train_dataset, train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids, whole_img=whole_img_average)
        
    train_dataframe = create_dataframe(image_ids, labels, pixel_averages, split=f"train_{split}")
    
    X_train = train_dataframe.iloc[:,2:]
    y_train = train_dataframe.label
    
    
    mean_data = mean_centering(X_train)
    msc_data, ref_train = msc_hyp(X_train, ref=None)
    msc_mean = mean_centering(msc_data, ref=None)
    
    spectra_plot(X_train, y_train, type_classifier=f"Train Original {split}")
    _, _, train_classifier = PLS_evaluation(X_train, y_train, type_classifier=f"Train Original {split}")
    PLS_show(train_classifier, X_train, y_train, HSI_train, 21, train_dataset, train_pseudo_imgs, train_img_names, type_classifier=f"Train Original {split}")
    
    
    spectra_plot(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")
    _, _, train_classifier_mean = PLS_evaluation(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")
    PLS_show(train_classifier_mean, mean_data, y_train, HSI_train, 21, train_dataset, train_pseudo_imgs, train_img_names, type_classifier=f"Train Mean-Centered {split}")

    
    spectra_plot(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")
    _ , _ , train_classifier_msc = PLS_evaluation(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")
    PLS_show(train_classifier_msc, msc_mean, y_train, HSI_train, 21, train_dataset, train_pseudo_imgs, train_img_names, type_classifier=f"Train MSC Mean-Centered {split}")
    
        
    # Test PLS classifier
    test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids = find_imgs(test_dataset, hyperspectral_path_test, test_image_dir)
    image_ids, pixel_averages, labels, HSI_test, split = process_data(test_dataset, test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids, whole_img=whole_img_average)
    test_dataframe = create_dataframe(image_ids, labels, pixel_averages, split=f"test_{split}")
    
    X_test = test_dataframe.iloc[:,2:]
    y_test = test_dataframe.label
    
    
    mean_data = mean_centering(X_test, ref=ref_train)
    msc_data, _ = msc_hyp(X_test, ref=ref_train)
    msc_mean = mean_centering(msc_data, ref=ref_train)
    
    spectra_plot(X_test, y_test, type_classifier=f"Test Original {split}")
    _, RMSE, _ = PLS_evaluation(X_test, y_test, classifier=train_classifier, type_classifier=f"Test Original {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier, X_test, y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, type_classifier=f"Test Original {split}", train=False)
    
    
    spectra_plot(mean_data, y_test, type_classifier=f"Test Mean-Centered {split}")
    _, RMSE, _ = PLS_evaluation(mean_data,  y_test, classifier=train_classifier_mean, type_classifier=f"Test Mean-Centered {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier_mean, mean_data, y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, type_classifier=f"Test Mean-Centered {split}", train=False)
    
    
    spectra_plot(msc_mean, y_test, type_classifier=f"Test MSC Mean-Centered {split}")
    _ , RMSE, _ = PLS_evaluation(msc_mean, y_test, classifier=train_classifier_msc, type_classifier=f"Test Mean-Centered {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier_msc, msc_mean, y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, type_classifier=f"Test MSC Mean-Centered {split}", train=False)
    



if __name__ == "__main__":
    main()
