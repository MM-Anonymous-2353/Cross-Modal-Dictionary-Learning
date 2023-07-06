import os.path
import glob
import random
import cv2
from PIL import Image

from prepare_dataset.depth_fill import depth_completion


def gray16_convert_color_cityscapes():
    # depth_path = '/test/datasets/cityscapes_fine_and_coarse/disparity/depth_fill'
    # color_depth_path = '/test/datasets/cityscapes_fine_and_coarse/disparity/depth_fill_color'
    depth_path = '/test/datasets/cityscapes_fine_and_coarse/disparity'
    color_depth_path = '/test/datasets/cityscapes_fine_and_coarse/disparity/depth_color'
    if not os.path.exists(color_depth_path):
        os.makedirs(color_depth_path)
    for data_type in os.listdir(depth_path):
        if 'test' in data_type or 'val' in data_type or 'train' in data_type:
            for city in os.listdir(os.path.join(depth_path, data_type)):
                for image_name in os.listdir(os.path.join(depth_path, data_type, city)):
                    uint16_img = cv2.imread(os.path.join(depth_path, data_type, city, image_name), -1)
                    uint16_img -= uint16_img.min()
                    uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
                    uint16_img *= 255
                    uint16_img = 255 - uint16_img

                    im_color = cv2.applyColorMap(cv2.convertScaleAbs(uint16_img, alpha=1), cv2.COLORMAP_JET)
                    im = Image.fromarray(im_color)
                    if not os.path.exists(os.path.join(color_depth_path, data_type, city)):
                        os.makedirs(os.path.join(color_depth_path, data_type, city))
                    im.save(os.path.join(color_depth_path, data_type, city, image_name))

def gray16_convert_color_sunrgbd():
    depth_dir_path = ['/test/datasets/sunrgbd/train_depth.txt', '/test/datasets/sunrgbd/test_depth.txt']
    for i in range(len(depth_dir_path)):
        with open(depth_dir_path[i], 'r') as file:
            depth_dir_list = file.read().splitlines()
            # print(os.path.dirname(depth_dir_path[i]))
            for j in range(len(depth_dir_list)):
                uint16_img = cv2.imread(os.path.join(os.path.dirname(depth_dir_path[i]), depth_dir_list[j]), -1) # 在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取，否则默认的读取类型为8位。
                uint16_img -= uint16_img.min()
                uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
                uint16_img *= 255
                uint16_img = 255 - uint16_img

                im_color = cv2.applyColorMap(cv2.convertScaleAbs(uint16_img, alpha=1), cv2.COLORMAP_JET)
                im = Image.fromarray(im_color)
                # print(os.path.basename(depth_dir_list[j]))
                depth_fill_color_name = os.path.join(os.path.dirname(depth_dir_path[i]), depth_dir_list[j]).split('.png')[0] + '_depth_fill_color.png'
                im.save(depth_fill_color_name)
                
def gray16_convert_color_scannet():
    folder_path = '/test/datasets/scannet/tasks/scannet_frames_25k'
    for scene in os.listdir(folder_path):
        if not os.path.exists(os.path.join(folder_path, scene, 'depth_fill_color')):
            os.makedirs(os.path.join(folder_path, scene, 'depth_fill_color'))
        for image_name in os.listdir(os.path.join(folder_path, scene, 'depth_fill')):
            img_path = os.path.join(folder_path, scene, 'depth_fill', image_name)
            uint16_img = cv2.imread(img_path, -1) # 在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取，否则默认的读取类型为8位。
            uint16_img -= uint16_img.min()
            uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
            uint16_img *= 255
            uint16_img = 255 - uint16_img

            im_color = cv2.applyColorMap(cv2.convertScaleAbs(uint16_img, alpha=1), cv2.COLORMAP_JET)
            im = Image.fromarray(im_color)
            im.save(os.path.join(folder_path, scene, 'depth_fill_color', image_name))

def gray16_convert_color_nyudv2():
    folder_path = '/test/datasets/nyudv2'
    depth_dir_path = '/test/datasets/nyudv2/depth_fill'
    if not os.path.exists(os.path.join(folder_path, 'depth_fill_color')):
        os.makedirs(os.path.join(folder_path, 'depth_fill_color'))
    for image_name in os.listdir(depth_dir_path):
        img_path = os.path.join(depth_dir_path, image_name)
        uint16_img = cv2.imread(img_path, -1) # 在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取，否则默认的读取类型为8位。
        uint16_img -= uint16_img.min()
        uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
        uint16_img *= 255
        uint16_img = 255 - uint16_img

        im_color = cv2.applyColorMap(cv2.convertScaleAbs(uint16_img, alpha=1), cv2.COLORMAP_JET)
        im = Image.fromarray(im_color)
        im.save(os.path.join(folder_path, 'depth_fill_color', image_name))

def gray8_convert_color_nyudv2():
    folder_path = '/test/datasets/nyudv2'
    depth_dir_path = '/test/datasets/nyudv2/nyu_depths'
    if not os.path.exists(os.path.join(folder_path, 'depth_fill_color_uint8')):
        os.makedirs(os.path.join(folder_path, 'depth_fill_color_uint8'))
    for image_name in os.listdir(depth_dir_path):
        img_path = os.path.join(depth_dir_path, image_name)
        uint8_img = cv2.imread(img_path) # 在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取，否则默认的读取类型为8位。

        im_color = cv2.applyColorMap(cv2.convertScaleAbs(uint8_img, alpha=1), cv2.COLORMAP_JET)
        im = Image.fromarray(im_color)
        im.save(os.path.join(folder_path, 'depth_fill_color_uint8', image_name))
 
def depth_fill_cityscapes():
    depth_path = '/test/datasets/cityscapes_fine_and_coarse/disparity'
    if not os.path.exists(depth_path):
        os.mkdir(depth_path)
    output_depth = '/test/datasets/cityscapes_fine_and_coarse/disparity/depth_fill'
    for data_type in os.listdir(depth_path):
            if 'test' in data_type or 'train' in data_type or 'val' in data_type:
                city_dir = os.path.join(depth_path, data_type)
                for city in os.listdir(city_dir):
                    data_path = os.path.join(city_dir, city)
                    outputs_dir = os.path.join(output_depth, data_type, city)
                    depth_completion.DepthFill(data_path, outputs_dir)

def depth_fill_scannet():
    folder_path = '/test/datasets/scannet/tasks/scannet_frames_25k'
    for scene in os.listdir(folder_path):
        if not os.path.exists(os.path.join(folder_path, scene, 'depth_fill')):
            os.makedirs(os.path.join(folder_path, scene, 'depth_fill'))
        data_path = os.path.join(folder_path, scene, 'depth')
        outputs_dir = os.path.join(folder_path, scene, 'depth_fill')
        depth_completion.DepthFill(data_path, outputs_dir)
    
def depth_fill_nyudv2():
    folder_path = '/test/datasets/nyudv2'
    if not os.path.exists(os.path.join(folder_path, 'depth_fill')):
        os.makedirs(os.path.join(folder_path, 'depth_fill'))
    data_path = os.path.join(folder_path, 'depth')
    outputs_dir = os.path.join(folder_path, 'depth_fill')
    depth_completion.DepthFill(data_path, outputs_dir)


def split_cityscapes_fine():
    data_root = '/test/datasets/cityscapes_fine_and_coarse'
    if not os.path.exists(os.path.join(data_root, 'ImageSets_fine')):
        os.mkdir(os.path.join(data_root, 'ImageSets_fine'))
    model1_path = os.path.join(data_root, 'leftImg8bit')
    model2_path = os.path.join(data_root, 'disparity', 'depth_fill_color')
    gt_path = os.path.join(data_root, 'gtFine')
    data_type_set = ['train', 'val']
    imageset_path = os.path.join(data_root, 'ImageSets_fine')
    for data_type in data_type_set:
        for city in os.listdir(os.path.join(model1_path, data_type)):
            model1_img = [os.path.basename(image_path) for image_path in glob.glob(
                os.path.join(model1_path, data_type, city, '*.png'))]
            model2_img = [os.path.basename(image_path) for image_path in glob.glob(
                os.path.join(model2_path, data_type, city, '*.png'))]
            gt_img = [os.path.basename(image_path) for image_path in glob.glob(
                os.path.join(gt_path, data_type, city, '*_labelIds.png'))]
            model1_img.sort()
            model2_img.sort()
            gt_img.sort()
            for model1_name, model2_name, gt_name in zip(model1_img, model2_img, gt_img):
                if 'train' in data_type:
                    with open(os.path.join(imageset_path, 'train.txt'), 'a') as file:
                        file.write('{}'.format(os.path.join(model1_path, data_type, city, model1_name)) + ' ' + '{}'.format(os.path.join(model2_path, data_type, city, model2_name)) + ' ' + '{}'.format(os.path.join(gt_path, data_type, city, gt_name)) + '\n')
                else:
                    with open(os.path.join(imageset_path, data_type+'.txt'), 'a') as file:
                        file.write('{}'.format(os.path.join(model1_path, data_type, city, model1_name)) + ' ' + '{}'.format(os.path.join(model2_path, data_type, city, model2_name)) + ' ' + '{}'.format(os.path.join(gt_path, data_type, city, gt_name)) + '\n')


        # with open(os.path.join(imageset_path, data_type+'.txt'), 'w') as file:
        #     model1_img = [os.path.basename(image_path) for image_path in glob.glob(
        #         os.path.join(model1_path, data_type, '*.png'))]
        #     model2_img = [os.path.basename(image_path) for image_path in glob.glob(
        #         os.path.join(model2_path, data_type, '*.png'))]
        #     gt_img = [os.path.basename(image_path) for image_path in glob.glob(
        #         os.path.join(gt_path, data_type, '*.png'))]
        #     model1_img.sort()
        #     model2_img.sort()
        #     gt_img.sort()

        #     for model1_name, model2_name, gt_name in zip(model1_img, model2_img, gt_img):
        #         file.write('{}'.format(os.path.join(model1_path, data_type, model1_name)) + ' ' + '{}'.format(os.path.join(model2_path, data_type, model2_name)) + ' ' + '{}'.format(os.path.join(gt_path, data_type, gt_name)) + '\n')

def split_cityscapes_coarse():
    data_root = '/test/datasets/cityscapes_fine_and_coarse'
    # os.mkdir(os.path.join(data_root, 'ImageSets'))
    if not os.path.exists(os.path.join(data_root, 'ImageSets')):
        os.makedirs(os.path.join(data_root, 'ImageSets'))
    model1_path = os.path.join(data_root, 'leftImg8bit')
    model2_path = os.path.join(data_root, 'disparity', 'depth_fill_color')
    gt_path = os.path.join(data_root, 'gtCoarse')
    data_type_set = ['train', 'train_extra', 'val']
    imageset_path = os.path.join(data_root, 'ImageSets')
    for data_type in data_type_set:
        for city in os.listdir(os.path.join(model1_path, data_type)):
            model1_img = [os.path.basename(image_path) for image_path in glob.glob(
                os.path.join(model1_path, data_type, city, '*.png'))]
            model2_img = [os.path.basename(image_path) for image_path in glob.glob(
                os.path.join(model2_path, data_type, city, '*.png'))]
            gt_img = [os.path.basename(image_path) for image_path in glob.glob(
                os.path.join(gt_path, data_type, city, '*_labelIds.png'))]
            model1_img.sort()
            model2_img.sort()
            gt_img.sort()
            for model1_name, model2_name, gt_name in zip(model1_img, model2_img, gt_img):
                if 'train' in data_type:
                    with open(os.path.join(imageset_path, 'train.txt'), 'a') as file:
                        file.write('{}'.format(os.path.join(model1_path, data_type, city, model1_name)) + ' ' + '{}'.format(os.path.join(model2_path, data_type, city, model2_name)) + ' ' + '{}'.format(os.path.join(gt_path, data_type, city, gt_name)) + '\n')
                else:
                    with open(os.path.join(imageset_path, data_type+'.txt'), 'a') as file:
                        file.write('{}'.format(os.path.join(model1_path, data_type, city, model1_name)) + ' ' + '{}'.format(os.path.join(model2_path, data_type, city, model2_name)) + ' ' + '{}'.format(os.path.join(gt_path, data_type, city, gt_name)) + '\n')

def split_cityscapes_coarse_depth_with_noise():
    data_root = '/test/datasets/cityscapes_fine_and_coarse'
    # os.mkdir(os.path.join(data_root, 'ImageSets'))
    if not os.path.exists(os.path.join(data_root, 'ImageSets_depth_with_noise')):
        os.makedirs(os.path.join(data_root, 'ImageSets_depth_with_noise'))
    model1_path = os.path.join(data_root, 'leftImg8bit')
    model2_path = os.path.join(data_root, 'disparity', 'depth_color')
    gt_path = os.path.join(data_root, 'gtCoarse')
    data_type_set = ['train', 'train_extra', 'val']
    imageset_path = os.path.join(data_root, 'ImageSets_depth_with_noise')
    for data_type in data_type_set:
        for city in os.listdir(os.path.join(model1_path, data_type)):
            model1_img = [os.path.basename(image_path) for image_path in glob.glob(
                os.path.join(model1_path, data_type, city, '*.png'))]
            model2_img = [os.path.basename(image_path) for image_path in glob.glob(
                os.path.join(model2_path, data_type, city, '*.png'))]
            gt_img = [os.path.basename(image_path) for image_path in glob.glob(
                os.path.join(gt_path, data_type, city, '*_labelIds.png'))]
            model1_img.sort()
            model2_img.sort()
            gt_img.sort()
            for model1_name, model2_name, gt_name in zip(model1_img, model2_img, gt_img):
                if 'train' in data_type:
                    with open(os.path.join(imageset_path, 'train.txt'), 'a') as file:
                        file.write('{}'.format(os.path.join(model1_path, data_type, city, model1_name)) + ' ' + '{}'.format(os.path.join(model2_path, data_type, city, model2_name)) + ' ' + '{}'.format(os.path.join(gt_path, data_type, city, gt_name)) + '\n')
                else:
                    with open(os.path.join(imageset_path, data_type+'.txt'), 'a') as file:
                        file.write('{}'.format(os.path.join(model1_path, data_type, city, model1_name)) + ' ' + '{}'.format(os.path.join(model2_path, data_type, city, model2_name)) + ' ' + '{}'.format(os.path.join(gt_path, data_type, city, gt_name)) + '\n')




def split_sunrgbd():
    data_root = '/test/datasets/sunrgbd'
    path_dir_txt_file = {'train': ['train_rgb.txt', 'train_depth.txt', 'train_label.txt'],
                         'test': ['test_rgb.txt', 'test_depth.txt', 'test_label.txt']}
    if not os.path.exists(os.path.join(data_root, 'ImageSets')):
        os.makedirs(os.path.join(data_root, 'ImageSets'))
    for key in path_dir_txt_file.keys():
        with open(os.path.join(data_root, 'ImageSets', key + '.txt'), 'a') as data_type_file:
            for modality_txt_file in path_dir_txt_file[key]:
                with open(os.path.join(data_root, modality_txt_file), 'r') as file:
                    if 'rgb' in modality_txt_file:
                        rgb_list = file.read().splitlines()
                    elif 'depth' in modality_txt_file:
                        depth_list = file.read().splitlines()
                    elif 'label' in modality_txt_file:
                        label_list = file.read().splitlines()
            for rgb_img_path, depth_img_path, label_img_path in zip(rgb_list, depth_list, label_list):
                data_type_file.write('{}'.format(os.path.join(data_root, rgb_img_path)) + ' ' + '{}'.format(os.path.join(data_root, depth_img_path.split('.png')[0]+'_depth_fill_color.png')) + ' ' + '{}'.format(os.path.join(data_root, label_img_path)) + '\n')

def split_scannet():
    data_root = '/test/datasets/scannet/tasks'
    if not os.path.exists(os.path.join(data_root, 'ImageSets')):
        os.makedirs(os.path.join(data_root, 'ImageSets'))
    data_type_set = ['train', 'val']
    imageset_path = os.path.join(data_root, 'ImageSets')
    for data_type in data_type_set:
        with open(os.path.join(data_root, 'scannetv2_' + data_type + '.txt'), 'r') as file:
            for scene_name in file.readlines():
                scene = scene_name.split('\n')[0]
                modal1_img = [image_path for image_path in glob.glob(
                    os.path.join(data_root, 'scannet_frames_25k', scene, 'color', '*.jpg'))]
                modal2_img = [image_path for image_path in glob.glob(
                    os.path.join(data_root, 'scannet_frames_25k', scene, 'depth_fill_color', '*.png'))]
                gt_img = [image_path for image_path in glob.glob(
                    os.path.join(data_root, 'scannet_frames_25k', scene, 'label', '*.png'))]
                modal1_img.sort()
                modal2_img.sort()
                gt_img.sort()
                for modal1_name, modal2_name, gt_name in zip(modal1_img, modal2_img, gt_img):
                    with open(os.path.join(imageset_path, data_type + '.txt'), 'a') as split_file:
                        split_file.write(modal1_name + ' ' + modal2_name + ' ' + gt_name + '\n')


def split_nyudv2():
    data_root = '/test/datasets/nyudv21'
    if not os.path.exists(os.path.join(data_root, 'ImageSets')):
        os.makedirs(os.path.join(data_root, 'ImageSets'))
    data_type_set = ['train', 'test']
    imageset_path = os.path.join(data_root, 'ImageSets')
    for data_type in data_type_set:
        modal1_img = []
        modal2_img = []
        gt_img = []
        with open(os.path.join(data_root, data_type + '.txt'), 'r') as file:
            for img_name in file.readlines():
                img_name = img_name.strip()
                modal1_img.append(os.path.join(data_root, 'nyu_images', img_name + '.jpg'))
                modal2_img.append(os.path.join(data_root, 'depth_fill_color_uint8', img_name + '.png'))
                gt_img.append(os.path.join(data_root, 'nyu_labels', img_name + '.png'))
            for modal1_name, modal2_name, gt_name in zip(modal1_img, modal2_img, gt_img):
                with open(os.path.join(imageset_path, data_type + '.txt'), 'a') as split_file:
                    split_file.write(modal1_name + ' ' + modal2_name + ' ' + gt_name + '\n')
    


if __name__ == '__main__':
    ## cityscapes
    depth_fill_cityscapes()
    gray16_convert_color_cityscapes()
    # split_cityscapes_fine()
    # split_cityscapes_coarse()
    split_cityscapes_coarse_depth_with_noise()

    ## sunrgbd
    os.system("python ./prepare_dataset/sunrgbd.py /test/datasets/sunrgbd")
    gray16_convert_color_sunrgbd()
    split_sunrgbd()

    # scannet
    # os.system("python ./prepare_dataset/scannet.py --out_dir /test/datasets/scannet --preprocessed_frames")
    # 需要自行解压缩下载得到的数据集，以及scannetv2_train.txt, scannetv2_train.txt两个文件，并放入相应文件夹下 ../scannet/tasks
    # https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/scannetv2_train.txt
    # https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/scannetv2_val.txt
    # depth_fill_scannet()
    # gray16_convert_color_scannet()
    # split_scannet()


    depth_fill_nyudv2()
    gray16_convert_color_nyudv2()
    # gray8_convert_color_nyudv2()
    split_nyudv2()






