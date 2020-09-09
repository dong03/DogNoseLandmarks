import argparse
import os
import re
import pdb
import json
from torch.utils.data import DataLoader
import torch
import cv2
from models.classifier import DeepFakeClassifier
from torch.backends import cudnn
from dataset import DeepFakeClassifierDataset, collate_function
from tools.utils import read_annotations
from tools.eval import predict_set, evaluate
from tools.transforms import create_val_transforms, create_train_transforms
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='predict_img')
    parser.add_argument('--models', nargs='+', required=True, help="checkpoint files")
    parser.add_argument('--test_dir', nargs='+', required=True, help="path to directory with videos")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--submit', default=0, type=int)
    args = parser.parse_args()
    return args


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    modelss = []
    model_paths = [model for model in args.models]
    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
        print("loading state dict {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        model.eval()
        model.cuda()
        del checkpoint
        modelss.append([model.half()])
    print("load models finish")

    test_dirs = [each for each in args.test_dir]
    data_name = [os.path.basename(each).split('.')[0] for each in test_dirs]

    for each in list(zip(test_dirs, data_name)):
        save_dir = os.path.split(each[0])[0]
        print("begin to pred %s" % each[1])
        annotations = read_annotations(each[0])
        # test_samples = [x.strip() for x in open(args.test_dir).readlines() if x.strip()]
        # annotations = [(x,0) for x in test_samples]

        test_set = DeepFakeClassifierDataset(
            annotations=annotations,
            mode="val",
            balance=False,
            transforms=create_val_transforms(380))
            #create_train_transforms(380))#
        test_loader = DataLoader(
            dataset=test_set,
            num_workers=8,
            batch_size=128,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_function
        )
        probs_avg = []

        for (index, models) in enumerate(modelss):
            probs, gt_labels, names = predict_set(models,test_loader,{'run_type':'test','debug':0,'data_type':'half'})
            if not args.submit:
                probs = probs.reshape(-1)
                fw = open("/data/dongchengbo/VisualSearch/all_results/img/%s.txt"%each[1],'w')
                #fw = open('%s/%s_%s.txt'%(save_dir,os.path.basename(model_paths[0]),each[1]), 'w')
                fw.write('\n'.join(['{} {} {}'.format(names[i], 1.0*(probs>0.5)[i], probs[i]) for i in range(len(names))]))
                fw.close()
                print("result save at %s" % ("/data/dongchengbo/VisualSearch/all_results/img/%s.txt"%each[1]))
                #print("result save at: %s" % ('%s/%s_%s.txt'%(save_dir,os.path.basename(model_paths[index]),each[1])))
                gt_labels = gt_labels.reshape(-1)
                metrix = evaluate(gt_labels, probs > 0.5, probs)
                print("model: %s\ndata: %s"%(os.path.basename(model_paths[index]),each[1]))
                print(metrix)
                print('\n')
                probs_avg.append(probs)
            else:
                probs = probs.reshape(-1)
                probs_avg.append(probs)
                results = {}
                label = ['real','fake']
                for i in range(len(names)):
                    results[os.path.split(names[i])[-1]] = label[probs[i]>0.5]
                with open('%s/%s_%s.json'%(save_dir,os.path.basename(model_paths[index]),each[1]), "w", encoding='utf8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

        probs_avg = np.array(probs_avg)
        probs_avg = np.mean(probs_avg,axis=0)
        # if args.submit:
        #     results = {}
        #     for i in range(len(names)):
        #         results[os.path.split(names[i])[-1]] = label[probs[i] > 0.5]
        #     with open('%s/merge_%s.json'%(save_dir,os.path.basename(each[1])), "w", encoding='utf8') as f:
        #         json.dump(results, f, ensure_ascii=False, indent=4)
        # else:
        #     # fw = open('%s/merge_%s.txt'%(save_dir,os.path.basename(each[1])), 'w')
        #     # fw.write(
        #     #     '\n'.join(['{} {} {}'.format(names[i], 1.0 * (probs > 0.5)[i], probs[i]) for i in range(len(names))]))
        #     # fw.close()
        #     # print("merge result save at: %s"%('%s/merge_%s.txt'%(save_dir,os.path.basename(each[1]))))
        #
        #     metrix = evaluate(gt_labels, probs_avg > 0.5, probs_avg)
        #     print("model: merge\ndata: %s"%(each[1]))
        #     print(metrix)
        #     pdb.set_trace()
