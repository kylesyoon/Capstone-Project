import argparse
import glob
import os
import csv

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input-path', type=str, default='../detectiondata/',
                help='(optional) the path to the parent directory of the training and testing images')
ap.add_argument('-a', '--test-annotations-path', type=str, default='../annotations/',
                help='(optional) the directory to the testing annotations')
ap.add_argument('-o', '--output-path', type=str, default='./',
                help='(optional) the directory to the output csv files')
args = vars(ap.parse_args())

def extract_box(path):
    """extract_box
    Extract annotation box positions for each labels from VIVA hand dataset.
    output is a list of tuples.

    :param path: text file path
    """

    with open(path) as temp:
        output = []

        for i, line in enumerate(temp):

            if i != 0 and line:
                label, x_1, y_1, x_off, y_off, *_ = line.split()
                pt_1 = (int(x_1), int(y_1))
                pt_2 = (pt_1[0] + int(x_off), (pt_1[1] + int(y_off)))
                output.append((label, pt_1, pt_2))

    return output


def create_csv(image_dir, annotation_dir, csv_out_path):
    image_paths = sorted(glob.glob(image_dir + '*'))
    annotations_paths = sorted(glob.glob(annotation_dir + '*'))

    with open(csv_out_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for image_path, annotations_path in zip(image_paths, annotations_paths):
            annotations = extract_box(annotations_path)
            for annotation in annotations:
                # label, (x1, y1), (x2, y2)
                writer.writerow([image_path,
                                 annotation[1][0], annotation[1][1],
                                 annotation[2][0], annotation[2][1],
                                 annotation[0]])

output_path = args['output_path']
input_path = args['input_path']

train_images_path = input_path + 'train/pos/'
train_annotations_path = input_path + 'train/posGt/'
test_images_path = input_path + 'test/pos/'

create_csv(train_images_path, train_annotations_path, output_path + 'train.csv')

create_csv(test_images_path, args['test_annotations_path'], output_path + 'test.csv')
