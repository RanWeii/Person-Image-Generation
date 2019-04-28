# coding:utf8
import os
from PIL import Image
from torch.utils import data
import torch
import numpy as np
import shutil
import cv2
from torchvision import transforms as T
from torch.utils.data import DataLoader

def splitdir(path):
    pathlist = os.listdir(path)

    dir_path = list()
    fil_path = list()

    for p in pathlist:
        p = os.path.join(path, p)

        if os.path.isdir(p):
            dir_path.append(p)

        else:
            fil_path.append(p)
    return dir_path, fil_path



transform = T.Compose([
    T.Resize(192), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    # T.CenterCrop(192), # 从图片中间切出224*224的图片
    T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize((.5,.5,.5),(.5,.5,.5)) # 标准化至[-1, 1]，规定均值和标准差
])

class Wrapcloth(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        '''
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        '''
        self.test = test
        dirs = [os.path.join(root, dir) for dir in os.listdir(root)]

        # train:    traindata/0/1_target.jpg        traindata/0/target.jpg                  traindata/0/mask.jpg
        # train:    traindata/0/1_target/pose.mat   traindata/0/1_target/segment_vis.png

        if self.test:
            dirs = sorted(dirs, key=lambda x: int(x))
        else:
            dirs = sorted(dirs)


        # shuffle dirs
        # np.random.seed(100)
        # dirs = np.random.permutation(dirs)

        # imgs2dict
        i=0
        for dir in dirs:
            # 新建train目录
            new_path='/home/villion/color_segment/{}'.format(i)
            if not os.path.exists(new_path):
                os.mkdir(new_path)

            # 获取相应img文件路径
            files_dir,files=splitdir(dir)
            mask=[mask_path for mask_path in files if 'mask' in mask_path][0]

            targets=[target_path for target_path in files if 'target' in target_path]
            target_cloth=[target_path for target_path in targets if '_' not in target_path][0]
            target_img = [target_path for target_path in targets if '_' in target_path][0]

            condition_path = [target_path for target_path in files_dir if 'target' in target_path][0]
            pose=os.path.join(condition_path,'pose.mat')
            segment=os.path.join(condition_path,'segment_vis.png')
            # mask_img=os.path.join(condition_path,'segment.png')



            # 获取pose.mat中关键点的坐标
            # pose_data=sio.loadmat(pose)
            # subset=pose_data['subset']
            # candidate=pose_data['candidate']
            # point_color = (221, 119, 0)  # BGR
            # for point in candidate:
            #         x=int(point[0]*192/762)
            #         y=int(point[1]*256/1000)
            #         cv2.circle(img, (x,y), 2, point_color, 0)
            #         test_path=os.path.join('/home/villion','test.jpg')
            #         cv2.imwrite(test_path, img)


            #根据分割图像获取cloth的分割图
            img=cv2.imread(target_img)
            seg=cv2.imread(segment)

            #蓝色外套
            lower_red = np.array([215, 115, 0])
            upper_red = np.array([235, 125, 0])
            mask = cv2.inRange(seg, lower_red, upper_red)
            output = cv2.bitwise_and(img, img, mask=mask)
            seg_path = os.path.join(new_path,'segment.jpg')
            cv2.imwrite(seg_path,output)

            # 橙色内衣
            max=np.max(output)
            if max==0:
                lower_orange = np.array([0, 70,210])
                upper_orange = np.array([0, 100,255])
                seg1 = cv2.imread(segment)
                mask = cv2.inRange(seg1, lower_orange, upper_orange)
                output = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite(seg_path, output)

            condition_img=cv2.imread(target_img)
            condition_img=condition_img-output
            condition_img_path=os.path.join(new_path,'condition.jpg')
            cv2.imwrite(condition_img_path,condition_img)

            target_img_path=os.path.join(new_path,'target_img.jpg')
            target_cloth_path=os.path.join(new_path,'target_cloth.jpg')
            target_segment_path=os.path.join(new_path,'target_segment.jpg')
            shutil.copyfile(target_img,target_img_path)
            shutil.copyfile(target_cloth, target_cloth_path)
            shutil.copyfile(segment,target_segment_path)

            # condition_img_path
            # target_cloth_path
            # target_segment_path

            # =======>>target_img_path
            dict={'target_img':target_img_path, 'condition_img':condition_img_path, 'target_cloth':target_cloth_path,   'target_seg':target_segment_path,   'seg_path':seg_path}
            dirs[i]=dict
            i=i+1
            print(i)

        self.dirs=dirs
        self.transforms = transforms



    def __getitem__(self, index):
        '''
        一次返回一组图片的数据
        '''
        dir_path = self.dirs[index]


        condition_img=Image.open(dir_path['condition_img']).convert('RGB')
        condition_cloth=Image.open(dir_path['target_cloth']).convert('RGB')
        condition_seg=Image.open(dir_path['target_seg']).convert('RGB')
        target=Image.open(dir_path['target_img']).convert('RGB')


        if self.transforms:
            condition_img = self.transforms(condition_img)
            condition_cloth=self.transforms(condition_cloth)
            condition_seg=self.transforms(condition_seg)
            target=self.transforms(target)

        # condition_img=torch.from_numpy(condition_img)
        # condition_cloth=torch.from_numpy(condition_cloth)
        # condition_seg=torch.from_numpy(condition_seg)
        # target=torch.from_numpy(target)

        condition=torch.cat([condition_img,condition_cloth,condition_seg],0)

        return condition, target

    def __len__(self):
        return len(self.dirs)




if __name__=='__main__':
    # Wrapcloth('/home/villion/zhengna/traindata')
    dataset=Wrapcloth('/home/villion/reid/wrapcloth/data',transforms=transform)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)
    dataiter = iter(dataloader)
    imgs, targetimgs = next(dataiter)
    print(imgs.size(),targetimgs.size())






