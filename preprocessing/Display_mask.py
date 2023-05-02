import matplotlib.pyplot as plt
import json


def load_coco(path):
    """
    Load COCO dataset from a JSON file.

    Args:
        path (str): Path to the JSON file containing the COCO dataset.

    Returns:
        dict: Loaded COCO dataset.
    """
    
    with open(path) as json_file:
        masks = json.load(json_file)
    ## TODO: add to dict, feature that shows tracks all the annotations belonging to 1 image
    return masks

def load_annotation(dataset, annotation_numb,image_numb):
    """
    Load annotation data for a given annotation number and image number.

    Args:
        dataset (dict): COCO dataset.
        annotation_numb (int): Annotation number.
        image_numb (int): Image number.

    Returns:
        tuple: Bounding box and annotation data.
    """
    temp = dataset["annotations"]
    image_mask = temp[annotation_numb]
    bbox = image_mask["bbox"]
    annotation = image_mask["segmentation"]

    return bbox, annotation

def find_image(dataset,annotation_numb):
    """
    Find image data for a given annotation number.

    Args:
        dataset (dict): COCO dataset.
        annotation_numb (int): Annotation number.

    Returns:
        tuple: Image name and image ID.
    """
    image_id = dataset["annotations"][annotation_numb]["image_id"]
    for i in range(len(dataset["images"])):
        if dataset["images"][i]["id"] == image_id:
            image_name = dataset["images"][i]["file_name"]

    return image_name,image_id

def draw_img(dataset,image_numb,annote_ids, image_dir):
    """
    Draw image with bounding boxes and annotations.

    Args:
        dataset (dict): COCO dataset.
        image_numb (int): Image number.
        annote_ids (list): List of annotation IDs.
        image_dir (str): Directory containing the image files.

    """
    image_name, image_id = find_image(dataset, annote_ids[0])
    img = plt.imread(image_dir + image_name)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for ids in annote_ids:
        bbox, annotation = load_annotation(dataset, ids, image_numb)
        bbox_x, bbox_y, width, height = bbox
        ax.add_patch(plt.Rectangle((bbox_x, bbox_y), width, height, linewidth=1, edgecolor='b', facecolor='none'))
        ax.axis('off')
        x, y = annotation[0][0::2], annotation[0][1::2]
        plt.fill(x, y, alpha=.7)
    plt.show()
    
    
if __name__ == "__main__":
    annotation_path = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/COCO_export.json'
    image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    image_numb = 1
    dataset = load_coco(annotation_path)
    annote_ids = [i for i, annote in enumerate(dataset["annotations"]) if annote["image_id"] == image_numb]
    draw_img(dataset, image_numb, annote_ids, image_dir)
