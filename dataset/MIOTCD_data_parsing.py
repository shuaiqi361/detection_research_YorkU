import json
import os
import xml.etree.ElementTree as ET
import cv2

traffic_labels = ['car', 'pickup', 'truck', 'van', 'bus']
traffic_label_map = {k: v + 1 for v, k in enumerate(traffic_labels)}
traffic_label_map['background'] = 0
rev_traffic_label_map = {v: k for k, v in traffic_label_map.items()}  # Inverse mapping

MIO_label_map = {'articulated_truck': 'truck', 'bus': 'bus', 'car': 'car',
                 'pickup_truck': 'pickup', 'single_unit_truck': 'truck', 'work_van': 'van'}

MIO_classes = ['articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle',
               'motorized_vehicle', 'non-motorized_vehicle', 'pedestrian',
               'pickup_truck', 'single_unit_truck', 'work_van']
'''
There are 107446 training images containing a total of 312129 objects. 
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/MIOTCD
No ground truth for testing images
'''


def create_data_lists_MIOTCD(root_path, output_folder):
    annotation_path = os.path.join(root_path, 'gt_train.csv')
    if not os.path.exists(annotation_path):
        print('annotation_path not exist, this folder should contain 4 annotation files.')
        raise FileNotFoundError

    image_folder = os.path.join(root_path, 'train')

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    all_images_dict = {}
    n_objects = 0
    for line in lines:
        elements = line.strip('\n').split(',')
        image_id = elements[0]
        # image_name = image_id + '.jpg'
        obj_class = elements[1]
        if obj_class in MIO_label_map.keys():
            label = traffic_label_map[MIO_label_map[obj_class]]
            n_objects += 1
        else:
            continue

        xmin = int(elements[2])
        ymin = int(elements[3])
        xmax = int(elements[4])
        ymax = int(elements[5])

        difficult = 1

        if image_id not in all_images_dict.keys():
            all_images_dict[image_id] = {'labels': [label],
                                         'boxes': [[xmin, ymin, xmax, ymax]], 'difficulties': [difficult]}
        else:
            all_images_dict[image_id]['labels'].append(label)
            all_images_dict[image_id]['boxes'].append([xmin, ymin, xmax, ymax])
            all_images_dict[image_id]['difficulties'].append(difficult)

    train_images = list()
    train_objects = list()
    for k, img in all_images_dict.items():
        image_path = os.path.join(image_folder, k + '.jpg')
        train_images.append(image_path)
        train_objects.append(img)

    assert len(train_objects) == len(train_images)
    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(traffic_label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))


if __name__ == '__main__':
    root_path = '/media/keyi/Data/Research/traffic/data/MIOTCD/MIO-TCD-Localization'
    output_folder = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/MIOTCD'

    create_data_lists_MIOTCD(root_path, output_folder)
