import json
import os
import xml.etree.ElementTree as ET
import cv2

traffic_labels = ['car', 'pickup', 'truck', 'van', 'bus']
traffic_label_map = {k: v + 1 for v, k in enumerate(traffic_labels)}
traffic_label_map['background'] = 0
rev_traffic_label_map = {v: k for k, v in traffic_label_map.items()}  # Inverse mapping

citycam_label_map = {1: {'name': 'taxi', 'uni_name': 'car'}, 2: {'name': 'black_sedan', 'uni_name': 'car'}
    , 3: {'name': 'other_cars', 'uni_name': 'car'}, 4: {'name': 'little_truck', 'uni_name': 'pickup'}
    , 5: {'name': 'middle_truck', 'uni_name': 'truck'}, 6: {'name': 'big_truck', 'uni_name': 'truck'}
    , 7: {'name': 'van', 'uni_name': 'van'}, 8: {'name': 'middle_bus', 'uni_name': 'bus'}
    , 9: {'name': 'big_bus', 'uni_name': 'bus'}}

citycam_scene_type = ['Downtown', 'Parkway']


'''
There are 37222 training images containing a total of 111666 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/citycam.

There are 17791 validation images containing a total of 53373 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/citycam.
Total skipped file: 929 due to weather section in xml has & symbol.
'''


def parse_annotation(annotation_path):
    # parser = ET.XMLParser(recover=True)
    try:
        tree = ET.parse(annotation_path)
    except:
        return {}

    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter('vehicle'):

        difficult = 1

        label = int(object.find('type').text)
        if label not in citycam_label_map.keys():
            continue

        uni_name = citycam_label_map[label]['uni_name']
        uni_label = traffic_label_map[uni_name]

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(uni_label)
        difficulties.append(difficult)

    if len(boxes) == 0:
        print('Images with no objects: ', annotation_path)
        return {}
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists_citycam(root_path, output_folder):
    annotation_path = os.path.join(root_path, 'train_test_separation')
    if not os.path.exists(annotation_path):
        print('annotation_path not exist, this folder should contain 4 annotation files.')
        raise FileNotFoundError

    # training data
    n_skipped = 0
    train_images = list()
    train_objects = list()
    n_object = 0

    dataType = 'Train'
    for scene_type in citycam_scene_type:

        annFile = '{}/{}_{}.txt'.format(annotation_path, scene_type, dataType)
        if not os.path.exists(annFile):
            print('annotation file not found: ', annFile)
            raise FileNotFoundError

        with open(annFile, 'r') as f:
            lines = f.readlines()

        for line in lines:
            sequence_folder = line.split('-')[0]
            frame_folder = line.strip('\n')
            frame_path = os.path.join(root_path, "{}/{}".format(sequence_folder, frame_folder))

            for frame in os.listdir(frame_path):
                if frame.endswith('.xml') and os.path.exists(os.path.join(frame_path, frame)):
                    # print('Loading annotation:', os.path.join(frame_path, frame))
                    objects = parse_annotation(os.path.join(frame_path, frame))
                    if len(objects) == 0:
                        n_skipped += 1
                        continue
                    else:
                        n_object += len(objects)

                    train_objects.append(objects)
                    train_images.append(os.path.join(frame_path, frame).strip('.xml') + '.jpg')

    assert len(train_objects) == len(train_images)
    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(traffic_label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_object, os.path.abspath(output_folder)))

    # test data
    test_images = list()
    test_objects = list()
    n_object = 0

    dataType = 'Test'
    for scene_type in citycam_scene_type:

        annFile = '{}/{}_{}.txt'.format(annotation_path, scene_type, dataType)
        if not os.path.exists(annFile):
            print('annotation file not found: ', annFile)
            raise FileNotFoundError

        with open(annFile, 'r') as f:
            lines = f.readlines()

        for line in lines:
            sequence_folder = line.split('-')[0]
            frame_folder = line.strip('\n')
            frame_path = os.path.join(root_path, "{}/{}".format(sequence_folder, frame_folder))

            for frame in os.listdir(frame_path):
                if frame.endswith('.xml'):
                    objects = parse_annotation(os.path.join(frame_path, frame))
                    if len(objects) == 0:
                        n_skipped += 1
                        continue
                    else:
                        n_object += len(objects)

                    test_objects.append(objects)
                    test_images.append(os.path.join(frame_path, frame).strip('.xml') + '.jpg')

    assert len(test_objects) == len(test_images)
    # Save to file
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_object, os.path.abspath(output_folder)))
    print('Total skipped file:', n_skipped, 'due to weather section in xml has & symbol.')


if __name__ == '__main__':
    root_path = '/home/keyi/research/data/CityCam'
    output_folder = '/home/keyi/research/code/traffic/detection_research_YorkU/dataset/Citycam'
    # root_path = '/media/keyi/Data/Research/traffic/data/Citycam/CityCam'
    # output_folder = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/Citycam'

    create_data_lists_citycam(root_path, output_folder)
