import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Constants
# Single objects
SINGLE_OBJ = 0
# Multiple objects
MULTIPLE_OBJ = 1


def mask_to_annotation(mask, object_configuration, do_cvt):
    # checking the configuration
    if object_configuration == SINGLE_OBJ:
        return ah.single_object_bounding_box(mask, do_cvt)
    elif object_configuration == MULTIPLE_OBJ:
        return ah.multiple_objects_bounding_box(mask, do_cvt)
    else:
        pass


def display(im_dict, annotation_color, object_configuration):
    # FIXME: Correctly parse the new contour format with class specification
    raise NotImplementedError('Correctly parse the new contour format with class specification')


def save(im_dict):
    # creating directories if they don't exist
    if not os.path.exists(im_dict['directory']):
        os.makedirs(im_dict['directory'])

    # saving the annotation in YOLO text file format
    file_path = os.path.join(
        "./"+im_dict['directory'], str(im_dict['file_name']) + '.txt')

    with open(file_path, 'w') as f:
        for count, el in enumerate(im_dict['contours']):
            # If class_id info is available, parse it correctly. Else just use index as class id
            if len(el) == 2:
                class_id, contour = el
            else:
                contour = el
                class_id = count

            # formatting to account for YOLO format
            x, y, w, h = contour
            x = x/im_dict['image'].shape[1]
            y = y/im_dict['image'].shape[0]
            w = w/im_dict['image'].shape[1]
            h = h/im_dict['image'].shape[0]

            x = x + w / 2
            y = y + h / 2

            f.write(str(class_id)+" " + str(x) + " " +
                    str(y) + " " + str(w) + " " + str(h)+"\n")
        f.close()

    # saving the category in a text file
    if im_dict['category'] is not None:
        labels_file_path = os.path.join(
            "./"+im_dict['directory']+"/"+im_dict['file_name'], 'labels.txt')
        with open(labels_file_path, 'w') as f:
            for count in range(len(im_dict['contours'])):
                f.write(im_dict['category']+" "+str(count)+"\n")
            f.close()


def annotate(im, do_display=True, do_save=True, annotation_color=(0, 255, 0), object_configuration=SINGLE_OBJ, do_cvt=True):
    # retrieving parameters from the tuple
    id_, name, image, project_name, category, directory = im

    # creating a dictionary to store the image and its annotations
    im_dict = {}
    im_dict['file_name'] = os.path.splitext(name)[0]
    im_dict['image'] = image
    im_dict['category'] = category
    im_dict['contours'] = mask_to_annotation(
        image, object_configuration, do_cvt)
    im_dict['directory'] = directory

    # displaying and saving the image, depending on the passed parameters
    if do_display:
        display(im_dict, annotation_color, object_configuration)

    if do_save:
        save(im_dict)

    return im_dict


def process_directory(mask_dir:str, annotations_dir:str, color:int = cv2.IMREAD_GRAYSCALE):
    """
    Iterates over the images in the directory and generates annotations for each file.

    mask_dir: directory where segmentation masks are stored
    annotations_dir: directory where generated text annotations will be stored. Typically also contains the images.
    color: cv2 image read color. Default is grayscale.
    """
    # Verify that the mask directory exists
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory '{mask_dir}' does not exist.")

    # Ensure the output directory exists
    os.makedirs(annotations_dir, exist_ok=True)

    # Iterate over all files in the mask directory
    for filename in tqdm(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, filename)

        # Skip directories or non-file entries
        if not os.path.isfile(mask_path):
            continue

        # Read the mask image using OpenCV
        try:
            mask = cv2.imread(mask_path, color)
            if mask is None:
                raise Exception('Error while reading the file')

            # Annotate the mask
            #im=(img_id, img_name, mask_isolated, project_name, category, yolo_output_dir)
            im=(None, filename, mask, None, None, annotations_dir)
            annotate(im=im,
                     do_display=False,
                     do_save=True,
                     object_configuration=MULTIPLE_OBJ)

        except Exception as e:
            # Log errors but continue processing
            print(f"Failed to annotate '{mask_path}': {e}")