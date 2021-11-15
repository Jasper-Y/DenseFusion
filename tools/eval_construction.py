import _init_paths
import argparse
import os
import random
import time
import numpy as np
import yaml
import copy
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.construction.dataset import PoseDataset as PoseDataset_construction
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

test_icp = True
num_objects = 3
objlist = [0, 1, 2]
num_points = 2000
iteration = 10
bs = 1
dataset_config_dir = 'datasets/construction/dataset_config'
output_result_dir = 'experiments/eval_result/construction'
knn = KNearestNeighbor(1)

estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()

testdataset = PoseDataset_construction('eval', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

# diameter = []
# meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
# meta = yaml.load(meta_file)
# for obj in objlist:
#     diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1) # in meter
# print(diameter)
diameter = [0.07] * num_objects

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

import ipdb
ipdb.set_trace()
from PIL import Image
import time
import cv2


def project_3d_2d(p3d, intrinsic_matrix=np.array([[320, 0., 320],[0., 320, 240],[0., 0., 1.]])):
    p2d = np.dot(p3d * 1000, intrinsic_matrix.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d

# img: 2d points with rgb, 480 x 640 x 3
def draw_p2ds(img, p2ds, color=(0, 255, 0)):
    r = 1
    h, w = img.shape[0], img.shape[1]
    for pt_2d in p2ds:
        pt_2d[0] = np.clip(pt_2d[0], 0, w)
        pt_2d[1] = np.clip(pt_2d[1], 0, h)
        img = cv2.circle(
            img, (pt_2d[0], pt_2d[1]), r, color, -1
        )
    return img # next, save img as .png

last_item = 0
last_item_id = 0
t_start = time.time()
for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx = data
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    # points: torch.Size([1, 2000, 3])
    # choose: torch.Size([1, 1, 2000])
    # img: torch.Size([1, 3, 80, 80])
    # target: torch.Size([1, 500, 3])
    # model_points: torch.Size([1, 500, 3])
    # idx: torch.Size([1, 1])
    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                     Variable(choose).cuda(), \
                                                     Variable(img).cuda(), \
                                                     Variable(target).cuda(), \
                                                     Variable(model_points).cuda(), \
                                                     Variable(idx).cuda()

    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    for ite in range(0, iteration):
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t
        
        new_points = torch.bmm((points - T), R).contiguous()
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)
        my_mat_2[0:3, 3] = my_t_2

        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final
        my_t = my_t_final

    # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

    model_points = model_points[0].cpu().detach().numpy()
    my_r = quaternion_matrix(my_r)[:3, :3]
    pred = np.dot(model_points, my_r.T) + my_t
    target = target[0].cpu().detach().numpy()

    if test_icp:
        time_start = time.time()
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(model_points)
        # source_pcd = o3d.io.read_point_cloud(f'datasets/construction/Construction_data/complete/construction/{idx[0]}.pcd')
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        # init_pose = np.eye(4)
        # init_pose[1][1] = -1
        # init_pose[2][2] = -1
        # init_pose[2][3] = 1
        init_pose = np.concatenate((my_r, np.array([my_t]).T), axis=1)
        init_pose = np.concatenate((init_pose, np.array([[0, 0, 0, 1]])), axis=0) 
        reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, 10, init_pose, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=4000))
        pred_icp = np.dot(model_points, reg_p2p.transformation[:3, :3]) + reg_p2p.transformation[:3,3]
        tmp_img = np.array(Image.open(testdataset.list_rgb[i]))
        tmp_img = tmp_img[:,:,:3].copy()

        pts_2d = project_3d_2d(pred_icp)
        new_img = draw_p2ds(tmp_img, pts_2d)
        cv2.imwrite(f"eval_vis/{idx[0].item()}/{i - last_item_id}_icp.jpg", new_img)

        # @TODO: only asymetric here
        dis = np.mean(np.linalg.norm(pred_icp - target, axis=1))

        # if dis < diameter[idx[0].item()]:
        #     print('Item {0} No.{1} Pass using ICP! Distance: {2}'.format(idx[0].item(), i - last_item_id, dis))
        #     fw.write('Item {0} No.{1} Pass using ICP! Distance: {2}\n'.format(idx[0].item(), i - last_item_id, dis))
        # else:
        #     print('Item {0} No.{1} NOT Pass using ICP! Distance: {2}'.format(idx[0].item(), i - last_item_id, dis))
        #     fw.write('Item {0} No.{1} NOT Pass using ICP! Distance: {2}\n'.format(idx[0].item(), i - last_item_id, dis))
        # print(f"Inference using ICP cost {time.time() - time_start} s")

    # pts = points.cpu().numpy().astype("float32")[0]
    # trans_pts = np.dot(pts, my_r.T) + my_t[:3]

    tmp_img = np.array(Image.open(testdataset.list_rgb[i])) # need to change tmp_img[0][0][0] and tmp_img[0][0][2]
    # tmp_img = np.flip(tmp_img, 2).copy()
    tmp_img = tmp_img[:,:,:3].copy()

    pts_2d = project_3d_2d(pred)
    new_img = draw_p2ds(tmp_img, pts_2d)
    
    if idx[0].item() != last_item:
        last_item = idx[0].item()
        last_item_id = i
    cv2.imwrite(f"eval_vis/{idx[0].item()}/{i - last_item_id}.jpg", new_img)

    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    if dis < diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        print('Item {0} No.{1} Pass! Distance: {2}'.format(idx[0].item(), i - last_item_id, dis))
        fw.write('Item {0} No.{1} Pass! Distance: {2}\n'.format(idx[0].item(), i - last_item_id, dis))
    else:
        print('Item {0} No.{1} NOT Pass! Distance: {2}'.format(idx[0].item(), i - last_item_id, dis))
        fw.write('Item {0} No.{1} NOT Pass! Distance: {2}\n'.format(idx[0].item(), i - last_item_id, dis))
    num_count[idx[0].item()] += 1

t_end = time.time()

for i in range(num_objects):
    if num_count[i] <= 0:
        continue
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.write(f'Evarage run time: {(t_end - t_start) / sum(num_count)}')
fw.close()
