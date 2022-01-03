import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2
import open3d as o3d

train_num_per_obj = 800
eval_num_per_obj = 200

train_obj_num = 3
test_obj_num = 3

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}
        self.list_pcd = []
        self.list_complete_pcd = []
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine
        self.obj_list = []

        # if self.mode == 'train':
        #     with open(f"{self.root}/train_list_sort.txt", 'r') as f:
        #         for _ in range(train_obj_num):
        #             self.obj_list.append(int(f.readline().strip()))
        # else:
        #     with open(f"{self.root}/test_list_sort.txt", 'r') as f:
        #         for _ in range(test_obj_num):
        #             self.obj_list.append(int(f.readline().strip()))
        with open(f"{self.root}/train_list_sort.txt", 'r') as f:
                for _ in range(train_obj_num):
                    self.obj_list.append(int(f.readline().strip()))
        
        if self.mode == 'train':
            image_idx = []
            for i in range(train_num_per_obj // 5):
                image_idx.append(5 * i)
                image_idx.append(5 * i + 1)
                image_idx.append(5 * i + 2)
                image_idx.append(5 * i + 3)
        elif self.mode == 'eval':
            image_idx = []
            # for i in range(train_num_per_obj // 5):
            #     image_idx.append(5 * i)
            #     image_idx.append(5 * i + 4)
            # use new data
            for i in range(train_num_per_obj, train_num_per_obj + eval_num_per_obj):
                image_idx.append(i)
        else:
            image_idx = [i * 5 + 4 for i in range(train_num_per_obj // 5)]


        item_count = 0
        # train: 80, test: 20
        for item in self.obj_list:
            poses = {}
            for idx in image_idx:
                item_count += 1
                # if self.mode == 'test' and item_count % 10 != 0:
                #     continue

                self.list_rgb.append(f'{self.root}/train/exr/{item}/clr/{idx}.png')
                self.list_depth.append(f'{self.root}/train/depth/{item}/{idx}.png')
                # self.list_label.append(f'{self.root}/train/exr/{item}/mask/{idx}.png')
                
                self.list_obj.append(item)
                self.list_rank.append(idx)

                idx_gt = np.loadtxt(f'{self.root}/train/pose/{item}/{idx}.txt')
                poses[idx] = {'cam_R_m2c': idx_gt[:3,:3].reshape(9).tolist(), 'cam_t_m2c': idx_gt[:3,3].reshape(3).tolist()}
                # self.list_pcd.append(f'{self.root}/train/pcd/{item}/{idx}.pcd')
                # self.list_complete_pcd.append(f'{self.root}/complete/construction/{item}.pcd')

            self.meta[item] = poses
            self.pt[item] = np.asarray(o3d.io.read_point_cloud(f'{self.root}/complete/construction/{item}.pcd').points) 
            
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.cam_cx = 320
        self.cam_cy = 240
        self.cam_fx = 320
        self.cam_fy = 320

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 1000
        self.num_pt_mesh_small = 1000
        # self.symmetry_obj_idx = self.obj_list.copy()
        self.symmetry_obj_idx = []

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index]))
        # label = np.array(Image.open(self.list_label[index])) 
        label = depth.copy() # grayscale, 480*640*1
        np.place(label, label > 0, 255)
        if self.mode != 'eval':
            label = np.dstack([label] * 3)
        obj = self.list_obj[index]
        rank = self.list_rank[index] # rank represents idx in the init function above      

        # if obj == 2:
        #     for i in range(0, len(self.meta[obj][rank])):
        #         if self.meta[obj][rank][i]['obj_id'] == 2:
        #             meta = self.meta[obj][rank][i]
        #             break
        # else:
        meta = self.meta[obj][rank]

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3]

        # img, depth = random_occlusion(img, depth, self.list_rgb[index], self.list_depth[index])

        img = np.transpose(img, (2, 0, 1))
        img_masked = img

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        
        mask = mask_label * mask_depth

        # if self.mode == 'eval':
        #     rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
        # else:
        #     rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])

        # current do not have obj_bb info
        rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))

        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        #p_img = np.transpose(img_masked, (1, 2, 0))
        #scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c'])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        # @TODO: really small size of this choose. Need to check the bbox
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        if self.add_noise:
            cloud = np.add(cloud, add_t)

        #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
        #for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()


        # # model_points = self.pt[obj] / 1000.0
        # pcd = o3d.io.read_point_cloud(self.list_pcd[index]) 
        # target = np.asarray(pcd.points) 
        # dellist = [j for j in range(0, len(target))]
        # dellist = random.sample(dellist, len(target) - self.num_pt_mesh_small)
        # target = np.delete(target, dellist, axis=0) # @check pcd is rotated with x axis
        # target = np.dot(target, np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]))

        # model_points = target.copy() # model_points should be the pcd in its own coordinate
        # model_points = np.add(model_points, -target_t)
        # model_points = np.dot(model_points, target_r)
        # if self.add_noise:
        #     target = np.add(target, add_t)

        # use complete pcd
        model_points = self.pt[obj]
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)
        # model_points = np.dot(model_points, np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]))

        target = np.add(model_points, -target_t)
        target = np.dot(target, np.dot(target_r, np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])))
        if self.add_noise:
            target = np.add(target, add_t)


        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.obj_list.index(obj)])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

def random_occlusion(img, dep, pth_rgb, pth_dep):
    # img: 480 640 3
    # dep: 480 640
    rand_x = np.random.randint(200, 280)
    rand_y = np.random.randint(260, 380)
    if rand_x > 240:
        dx = 1
    else:
        dx = -1

    if rand_y > 320:
        dy = 1
    else:
        dy = -1
    
    # dx = np.random.randint(0, 2) * 2 - 1
    # dy = np.random.randint(0, 2) * 2 - 1
    for x in range(rand_x, 240 + 220 * dx, dx):
        for y in range(rand_y, 320 + 300 * dy, dy):
            img[x][y] = np.array([0, 0, 0])
            dep[x][y] = 0
    cv2.imwrite(pth_rgb, img)
    new_depth_img = o3d.geometry.Image(np.uint16(dep))
    o3d.io.write_image(pth_dep, new_depth_img)
    return img, dep
