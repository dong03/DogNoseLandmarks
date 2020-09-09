import pdb
import torch
import numpy as np
import time
from tools.utils import Progbar,AverageMeter
from matplotlib import pyplot as plt
from scipy.integrate import simps


def predict_set(nets, dataloader, runtime_params):
    run_type = runtime_params['run_type']
    #net = net.eval()
    progbar = Progbar(len(dataloader.dataset), stateful_metrics=['run-type'])
    batch_time = AverageMeter()
    names = []
    pred_landmarks = np.array([])
    gt_landmarks = np.array([])
    with torch.no_grad():
        for i, (landmarks, imgs, img_paths) in enumerate(dataloader):
            s_time = time.time()
            imgs = imgs.cuda()

            names.extend(img_paths)

            net = nets[0]
            if 'half' in runtime_params.values():
                output = net(imgs.half())
            else:
                output = net(imgs)
            output = output.cpu().numpy()
            pred_landmarks = np.concatenate((pred_landmarks,output),axis=0)
            gt_landmarks = np.concatenate((gt_landmarks,landmarks.data.numpy()),axis=0)
            progbar.add(imgs.size(0), values=[('run-type', run_type)])  # ,('batch_time', batch_time.val)])
            batch_time.update(time.time() - s_time)
            if runtime_params['debug'] and i:
                break
    pred_landmarks = pred_landmarks.reshape((-1,28,2))
    gt_landmarks = gt_landmarks.reshape((-1,28,2))
    assert gt_landmarks.shape == pred_landmarks.shape
    return gt_landmarks, gt_landmarks, names



def dist(gtLandmark, dist_type='centers', left_pt=0, right_pt=8, num_eye_pts=8):
    if dist_type=='centers':
        normDist = np.linalg.norm(np.mean(gtLandmark[left_pt:left_pt+num_eye_pts], axis=0) -
                                  np.mean(gtLandmark[right_pt:right_pt+num_eye_pts], axis=0))
    elif dist_type=='corners':
        normDist = np.linalg.norm(gtLandmark[left_pt] - gtLandmark[right_pt+num_eye_pts/2])
    elif dist_type=='diagonal':
        height, width = np.max(gtLandmark, axis=0) - np.min(gtLandmark, axis=0)
        normDist = np.sqrt(width**2 + height**2)
    return normDist

def landmark_error(gtLandmarks, predict_Landmarks, dist_type='centers', show_results=False, verbose=False):
    norm_errors = []
    errors = []
    for i in range(len(gtLandmarks)):
        norm_dist = dist(gtLandmarks[i], dist_type=dist_type)
        error = np.mean(np.sqrt(np.sum((gtLandmarks[i] - predict_Landmarks[i])**2, axis=1)))
        norm_error = error/norm_dist
        errors.append(error)
        norm_errors.append(norm_error)
        if verbose:
            print('{0}: {1}'.format(i, error))

    if verbose:
        print("Image idxs sorted by error")
        print(np.argsort(errors))
    avg_error = np.mean(errors)
    avg_norm_error = np.mean(norm_errors)
    print("Average error: {0}".format(avg_error))
    print("Average norm error: {0}".format(avg_norm_error))
    return norm_errors, errors

def auc_error(errors, failure_threshold=0.03, step=0.0001, save_path='', showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failure_threshold+step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]
    auc = simps(ced, x=xAxis) / failure_threshold
    failure_rate = 1. - ced[-1]
    print("AUC @ {0}: {1}".format(failure_threshold, auc))
    print("Failure rate: {0}".format(failure_rate))

    if showCurve:
        plt.plot(xAxis, ced)
        plt.savefig(save_path)

    return auc, failure_rate

def evaluate(gt_landmarks, landmarks,th,save_path):
    gt_landmarks = gt_landmarks.permute((1,0,2)).cpu().numpy()
    landmarks = landmarks.permute((1, 0, 2)).cpu().numpy()
    norm_errors, errors = landmark_error(gt_landmarks,landmarks)
    auc, failure_rate = auc_error(errors,th,save_path=save_path)
    return {'auc':auc,'failure_rate':failure_rate,"errors":errors}

