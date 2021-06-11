# -*- coding: UTF-8 -*-
'''
@author: mengting gu
@contact: 1065504814@qq.com
@time: 2020/11/2 上午11:47
@file: widerValFile.py
@desc:
'''
import os
import argparse

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--dataset_folder', default=r'E:\pytorch\Retinaface\data\widerface\WIDER_val\images/', type=str, help='dataset path')
args = parser.parse_args()

if __name__ == '__main__':
    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "label.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    for i, img_name in enumerate(test_dataset):
        print("line i :{}".format(i))
        if img_name.endswith('.jpg'):
            print("     img_name :{}".format(img_name))
            f = open(args.dataset_folder[:-7] + 'wider_val.txt', 'a')
            f.write(img_name + '\n')
    f.close()