import os
import shutil
import ipdb

data_file = ['clipart', 'painting', 'real', 'sketch', 'infograph', 'quickdraw']
current_path = os.getcwd()
for task in data_file:
    source_path = current_path + '/' + task
    source_train_path = current_path + '/' + task + '_train'
    if not os.path.isdir(source_train_path):
        os.makedirs(source_train_path)
    txt_train = current_path + '/' + task + '_train.txt'
    txt_file = open(txt_train)
    line = txt_file.readline()
    while line:
        image_path = line.split(' ')[0]
        image_path_split_list = image_path.split('/')
        source_image_path = source_path + '/' + image_path_split_list[1] + '/' + image_path_split_list[2]
        source_train_category = source_train_path + '/' + image_path_split_list[1]
        if not os.path.isdir(source_train_category):
            os.makedirs(source_train_category)
        source_train_image_path = source_train_category + '/' + image_path_split_list[2]
        print('copy image from %s -> %s' % (source_image_path, source_train_image_path))
        shutil.copyfile(source_image_path, source_train_image_path)
        line = txt_file.readline()



    source_test_path = current_path + '/' + task + '_test'
    if not os.path.isdir(source_test_path):
        os.makedirs(source_test_path)
    txt_test = current_path + '/' + task + '_test.txt'
    txt_file = open(txt_test)
    line = txt_file.readline()
    while line:
        image_path = line.split(' ')[0]
        image_path_split_list = image_path.split('/')
        source_image_path = source_path + '/' + image_path_split_list[1] + '/' + image_path_split_list[2]
        source_test_category = source_test_path + '/' + image_path_split_list[1]
        if not os.path.isdir(source_test_category):
            os.makedirs(source_test_category)
        source_test_image_path = source_test_category + '/' + image_path_split_list[2]
        print('copy image from %s -> %s' % (source_image_path, source_test_image_path))
        shutil.copyfile(source_image_path, source_test_image_path)
        line = txt_file.readline()
