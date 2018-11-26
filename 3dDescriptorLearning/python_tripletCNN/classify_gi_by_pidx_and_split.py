# This script classifies geometry images according to point indices and apply train-validation-test split.

import argparse
import os
from os.path import join

from tqdm import tqdm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='This script classifies geometry images according to point indices and apply '
                                             'train-validation-test split.')

parser.add_argument('--n_models', '--nm', default=100, type=int,
                    help='number of calculated models')
parser.add_argument('--n_points', '--np', default=6890, type=int,
                    help='number of calculated points')
parser.add_argument('--source_dir', '-s',
                    default=r'/media/disk_add/hywang/project/3dDescriptor2018/dataset/faust/gi', type=str,
                    help='directory of source geometry images')
parser.add_argument('--destination_dir', '-d',
                    default=r'/media/disk_add/hywang/project/3dDescriptor2018/dataset/faust/gi_classified', type=str,
                    help='directory to store classified geometry images')
parser.add_argument('--percentage_train', '--ptr', default=0.75, type=float,
                    help='percentage of training set')
parser.add_argument('--percentage_val', '--pv', default=0.1, type=float,
                    help='percentage of validation set')
parser.add_argument('--percentage_test', '--pte', default=0.15, type=float,
                    help='percentage of testing set')

args = parser.parse_args()

geoimg_dir = args.source_dir
result_pre_dir = args.destination_dir

percentage_valtest = args.percentage_val + args.percentage_test
test_of_valtest = args.percentage_test / percentage_valtest

if not os.path.exists(geoimg_dir):
    print('ERROR: Geometry image path not found.')
    exit(-1)

model_ids = list(range(args.n_models))
train_ids, valtest_ids = train_test_split(model_ids, test_size=percentage_valtest)
val_ids, test_ids = train_test_split(valtest_ids, test_size=test_of_valtest)

train_target_dir = join(result_pre_dir, 'train')
val_target_dir = join(result_pre_dir, 'val')
test_target_dir = join(result_pre_dir, 'test')

train_result_dir_list = [join(train_target_dir, 'pidx_%04d' % i) for i in range(args.n_points)]
val_result_dir_list = [join(val_target_dir, 'pidx_%04d' % i) for i in range(args.n_points)]
test_result_dir_list = [join(test_target_dir, 'pidx_%04d' % i) for i in range(args.n_points)]


for rd in train_result_dir_list:
    os.makedirs(rd, exist_ok=True)
for rd in val_result_dir_list:
    os.makedirs(rd, exist_ok=True)
for rd in test_result_dir_list:
    os.makedirs(rd, exist_ok=True)

geoimg_path_list = []

sub_dirs = os.listdir(geoimg_dir)
for sub_dir in sub_dirs:
    model_abs_dir = join(geoimg_dir, sub_dir)
    for gi_name in os.listdir(model_abs_dir):
        geoimg_path_list.append(join(model_abs_dir, gi_name))


for gi_path in tqdm(geoimg_path_list):
    if not os.path.isfile(gi_path) or len(gi_path.split(os.path.sep)[-1]) < 3 or gi_path.split('.')[-1] != 'gi':
        continue

    class_id = int(gi_path.split('_')[-1].split('.')[0])
    model_id = int(gi_path.split('_')[-3])

    if model_id in train_ids:
        os.rename(gi_path, join(train_result_dir_list[class_id], gi_path.split(os.path.sep)[-1]))
    elif model_id in val_ids:
        os.rename(gi_path, join(val_result_dir_list[class_id], gi_path.split(os.path.sep)[-1]))
    elif model_id in test_ids:
        os.rename(gi_path, join(test_result_dir_list[class_id], gi_path.split(os.path.sep)[-1]))
    else:
        raise IOError('Unexpected model id')

with open(join(result_pre_dir, 'train_val_test.txt'), 'w') as txt:
    txt.write('training_set: \n')
    txt.write(str(set(train_ids)))
    txt.write('\n\n')
    txt.write('validation_set: \n')
    txt.write(str(set(val_ids)))
    txt.write('\n\n')
    txt.write('testing_set: \n')
    txt.write(str(set(test_ids)))
    txt.write('\n\n')