import sys
import numpy as np
import cv2
sys.path.insert(0, './pnpransac')
from pnpransac import pnpransac

def scale_K(K, rescale_factor):
    K = K * rescale_factor
    K[2, 2] = 1.0
    return K

def get_pose_err(pose_gt, pose_est):
    transl_err = np.linalg.norm(pose_gt[0:3,3]-pose_est[0:3,3])
    rot_err = pose_est[0:3,0:3].T.dot(pose_gt[0:3,0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return transl_err, rot_err[0]

rot_err_list = []
transl_err_list = []


DIR = "/SCM_DIR/" # generated

PRE_NAME = "fake"
SIZE_H = 480
SIZE_W = 640

K = np.array([[525,0,320],
    [0 , 525, 240],
    [0 ,0,1]])
K = scale_K(K,SIZE_H/480)
#print(K)
for i in range(2000):
    coord = np.load(DIR + PRE_NAME+str(i).zfill(6)+".npy").squeeze()
    #print(coord.shape)
    coord = coord.transpose(1,2,0)
    tcw = np.load("/tcw_"+str(i).zfill(6)+".npy")

    #print(coord.shape)
    #print(tcw)
    pose_solver = pnpransac(K[0,0], K[1,1],
                K[0,2], K[1,2])
    x = np.linspace(0, SIZE_W-1, SIZE_W)
    y = np.linspace(0, SIZE_H-1, SIZE_H)
    xx, yy = np.meshgrid(x, y)
    pcoord = np.concatenate((np.expand_dims(xx,axis=2),
    np.expand_dims(yy,axis=2)), axis=2)
    #print(pcoord)
    coord = np.ascontiguousarray(coord)
    pcoord = np.ascontiguousarray(pcoord)
    rot, transl = pose_solver.RANSAC_loop(np.reshape(pcoord,
                    (-1,2)).astype(np.float64), np.reshape(coord,
                    (-1,3)).astype(np.float64), 256)
    #print(rot)
    #print(transl)

    pose_gt = tcw #pose.data.numpy()[0,:,:]

    #pose_gt[0:3,0:3] = pose_gt[0:3,0:3].T #memo
    #pose_gt[0:3,3] = -np.dot(pose_gt[0:3,0:3], pose_gt[0:3,3]) #memo


    pose_est = np.eye(4)
    pose_est[0:3,0:3] = cv2.Rodrigues(rot)[0]#.T
    pose_est[0:3,3] = transl#-np.dot(pose_est[0:3,0:3], transl)
    #print(pose_gt)
    #print(pose_est)

    transl_err, rot_err = get_pose_err(pose_gt, pose_est)

    rot_err_list.append(rot_err)
    transl_err_list.append(transl_err)

    print('{}th Pose error: {}m, {}\u00b0'.format(i,transl_err, rot_err))
    if(i%50==0):
        results = np.array([transl_err_list, rot_err_list]).T
        print('Median pose error: {}m, {}\u00b0'.format(np.median(results[:,0]),
                np.median(results[:,1])))

results = np.array([transl_err_list, rot_err_list]).T
print('Median pose error: {}m, {}\u00b0'.format(np.median(results[:,0]),
        np.median(results[:,1])))
