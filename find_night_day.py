from utils import *
from nuscenes import NuScenes
import string
import os

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
version = 'v1.0'
data_path = '/data/laiyan/datasets/nuscenes/'
case='night_or_rain'#night,rain,night_or_rain

#
if case == 'night':
    words = ['night', 'nights', 'Night', 'Nights']
elif case == 'rain':
    words = ['rain','Rain','Rainy','rainy']
elif case == 'night_or_rain':
    words = ['rain', 'Rain', 'Rainy', 'rainy','night', 'nights', 'Night', 'Nights']
else:
    print('wrong type')
    exit()

if version == 'v1.0-mini':
    train_file = '/data/laiyan/airs/My_tests/my_splits/v1.0-mini_train_keyframes.txt'
    val_file = '/data/laiyan/airs/My_tests/my_splits/v1.0-mini_val_keyframes.txt'
    train_filenames = readlines(train_file)
    val_filenames = readlines(val_file)
    data_root = os.path.join(data_path, version)
    nusc = NuScenes(version='{}'.format(version), dataroot=data_root, verbose=True)
elif version == 'v1.0':
    train_file = '/data/laiyan/airs/My_tests/my_splits/train.txt'
    val_file = '/data/laiyan/airs/My_tests/my_splits/val.txt'
    # val_file = '/data/laiyan/airs/My_tests/my_splits/v1.0_val_keyframes.txt'
    train_filenames = readlines(train_file)
    val_filenames = readlines(val_file)
    data_root = os.path.join(data_path, 'v1.0')
    nusc = NuScenes(version='{}-trainval'.format(version), dataroot=data_root, verbose=True)
else:
    print('wrong version')
    exit()

def check_exists(tokens,words):
    for word in words:
        if word in tokens:
            return True
    return False

def save_lines(filenames,fileeee):
    saved = []
    not_saved = []

    new_description = None
    not_description = None
    for file in filenames:
        line = file.strip()
        sample = nusc.get('sample', line)
        scene = nusc.get('scene', token=sample['scene_token'])
        description = scene['description']
        # tokens = nltk.word_tokenize(description)
        tokens = [word.strip(string.punctuation) for word in description.split()]
        if check_exists(tokens, words):
            saved.append(line)
            new_description = description
        else:
            not_saved.append(line)
            not_description = description

    print(len(saved), len(not_saved), len(filenames))
    print(new_description)
    print(not_description)

    with open(fileeee[:-4] + '_no_' + case + '.txt', 'w') as f:
        print(fileeee[:-4] + '_no_' + case + '.txt')
        for line in not_saved:
            f.write(f"{line}\n")

    with open(fileeee[:-4] + '_' + case + '.txt', 'w') as f:
        print(fileeee[:-4] + '_' + case + '.txt')
        for line in saved:
            f.write(f"{line}\n")

# save_lines(train_filenames,train_file)
save_lines(val_filenames,val_file)