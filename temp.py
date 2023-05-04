# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:50:14 2023

@author: jver
"""

def main():
    """
    Main function that loads the dataset and performs PLS classification on the training and test sets.
    """
    # Load paths
    train_annotation_path =r"C:\Users\jver\Desktop\Training\windows\COCO_HSI_windowed.json"
    train_image_dir = r"C:\Users\jver\Desktop\Training\windows\PLS_eval_img_rgb"
    #test_annotation_path = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Test\COCO_Test.json"
    #test_image_dir = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Test\images"
    test_annotation_path = r"C:\Users\jver\Desktop\Validation\windows\COCO_HSI_windowed.json"
    test_image_dir = r"C:\Users\jver\Desktop\Validation\windows\PLS_eval_img_rgb"
    
    hyperspectral_path_train = r"C:\Users\jver\Desktop\Training\windows\PLS_eval_img"
    #hyperspectral_path_test = r"C:\Users\jver\Desktop\Test"
    hyperspectral_path_test = r"C:\Users\jver\Desktop\Validation\windows\PLS_eval_img"
    
    # Load training and test datasets
    train_dataset = load_coco(train_annotation_path)
    test_dataset = load_coco(test_annotation_path)


    # Train PLS classifier
    train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids = find_imgs(train_dataset, hyperspectral_path_train, train_image_dir)

    whole_img_average = False
    mask_id, pixel_averages, pixel_medians, labels, HSI_train, split = process_data(train_dataset, train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids, whole_img=whole_img_average)
        
    train_dataframe = create_dataframe(mask_id, labels, pixel_averages, split=f"train_{split}")
    train_dataframe_median = create_dataframe(mask_id, labels, pixel_medians, split=f"train_median{split}")
    
    X_train = train_dataframe.iloc[:,2:]
    y_train = train_dataframe.label

    
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
    _, _, train_classifier = PLS_evaluation(X_train, y_train, type_classifier=f"Train Original {split}")
    #PLS_show(train_classifier, X_train, y_train, HSI_train, 21, train_dataset, train_pseudo_imgs, train_img_names, ref = None, variation="Orig", type_classifier=f"Train Original {split}")
    
    
    spectra_plot(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")
    _, _, train_classifier_mean = PLS_evaluation(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")
    #PLS_show(train_classifier_mean, mean_data, y_train, HSI_train, 21, train_dataset, train_pseudo_imgs, train_img_names, ref = None, variation="Mean", type_classifier=f"Train Mean-Centered {split}")

    
    spectra_plot(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")
    _ , _ , train_classifier_msc = PLS_evaluation(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")
    #PLS_show(train_classifier_msc, msc_mean, y_train, HSI_train, 21, train_dataset, train_pseudo_imgs, train_img_names, ref = None, variation="MSC", type_classifier=f"Train MSC Mean-Centered {split}")
    
    
    _ , _ , train_classifier_median = PLS_evaluation(train_dataframe_median.iloc[:,2:], y_train, type_classifier=f"Train Median {split}")
    
    



    
    # Test PLS classifier
    test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids = find_imgs(test_dataset, hyperspectral_path_test, test_image_dir)
    mask_id, pixel_averages, pixel_medians, labels, HSI_test, split = process_data(test_dataset, test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids, whole_img=whole_img_average)
        
    test_dataframe = create_dataframe(mask_id, labels, pixel_averages, split=f"test_{split}")
    test_dataframe_median = create_dataframe(mask_id, labels, pixel_medians, split=f"test_median{split}")
    
    X_test = test_dataframe.iloc[:,2:]
    y_test = test_dataframe.label
    
    
    mean_data = mean_centering(X_test, ref=ref_train)
    _ = create_dataframe(mask_id, labels, mean_data, split=f"test_mean_{split}")
    msc_data, _ = msc_hyp(X_test, ref=ref_train)
    msc_mean = mean_centering(msc_data, ref=ref_train)
    _ = create_dataframe(mask_id, labels, msc_mean, split=f"test_meanMSC_{split}")
    
    median_data, _ = median_centering(X_test, ref=ref_median)
    _ = create_dataframe(mask_id, labels, median_data, split=f"test_median_{split}")
    
    spectra_plot(X_test, y_test, type_classifier=f"Test Original {split}")
    _, RMSE, _ = PLS_evaluation(X_test, y_test, classifier=train_classifier, type_classifier=f"Test Original {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier, X_test, y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, ref = ref_train, variation="Orig", type_classifier=f"Test Original {split}")
    
    
    spectra_plot(mean_data, y_test, type_classifier=f"Test Mean-Centered {split}")
    _, RMSE, _ = PLS_evaluation(mean_data,  y_test, classifier=train_classifier_mean, type_classifier=f"Test Mean-Centered {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier_mean, mean_data, y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, ref = ref_train, variation="Mean", type_classifier=f"Test Mean-Centered {split}")
    
    
    spectra_plot(msc_mean, y_test, type_classifier=f"Test MSC Mean-Centered {split}")
    _ , RMSE, _ = PLS_evaluation(msc_mean, y_test, classifier=train_classifier_msc, type_classifier=f"Test MSC Mean-Centered {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier_msc, msc_mean, y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, ref = ref_train, variation="MSC", type_classifier=f"Test MSC Mean-Centered {split}")
    
    
    _ , RMSE, _ = PLS_evaluation(test_dataframe_median.iloc[:,2:], y_test, classifier=train_classifier_median, type_classifier=f"Test Median {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier_median, test_dataframe_median.iloc[:,2:], y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, ref = ref_median, variation="Orig", type_classifier=f"Test Median {split}")
    
    

    r"""
    df = pd.read_csv(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\two_stage\pls_results\Pixel_grain_avg_dataframe_train_grain.csv")
    
    grain_specifics_rye = extract_specifics(df, "rye")
    grain_specifics_h1 = extract_specifics(df, "h1")
    grain_specifics_h3 = extract_specifics(df, "h3")
    grain_specifics_h4 = extract_specifics(df, "h4")
    grain_specifics_h5 = extract_specifics(df, "h5")
    grain_specifics_halland = extract_specifics(df, "halland")
    grain_specifics_oland = extract_specifics(df, "oland")
    grain_specifics_spelt = extract_specifics(df, "spelt")
    
    
    all_grains = [grain_specifics_rye, grain_specifics_h1, grain_specifics_h3, grain_specifics_h5, grain_specifics_halland, grain_specifics_oland, grain_specifics_spelt ]
    
    
    
    df_test = pd.read_csv(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\two_stage\pls_results\Pixel_grain_avg_dataframe_test_grain.csv")
    grain_specifics_rye = extract_specifics(df_test, "rye")
    grain_specifics_h1 = extract_specifics(df_test, "h1")
    grain_specifics_h3 = extract_specifics(df_test, "h3")
    grain_specifics_h4 = extract_specifics(df_test, "h4")
    grain_specifics_h5 = extract_specifics(df_test, "h5")
    grain_specifics_halland = extract_specifics(df_test, "halland")
    grain_specifics_oland = extract_specifics(df_test, "oland")
    grain_specifics_spelt = extract_specifics(df_test, "spelt")
    
    
    all_grains = [grain_specifics_rye, grain_specifics_h1, grain_specifics_h3, grain_specifics_h5, grain_specifics_halland, grain_specifics_oland, grain_specifics_spelt ]
    
    for i, grain in enumerate(all_grains):
        grain = grain.head()
        spectra_plot(grain.iloc[:,2:], grain.label, type_classifier = f"type_{i}_")
    """