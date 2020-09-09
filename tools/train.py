from torch.autograd import Variable
from tools.utils import AverageMeter, Progbar
from tools.eval import predict_set, evaluate
import torch
import json
import pdb
import time
from apex import amp

def validate(args, data_val, bce_best, model, model_dir, current_epoch, summary_writer,conf):
    print("Test phase")
    if args.debug:
        pdb.set_trace()
    model = model.eval()
    landmarks, gt_landmarks, names = predict_set([model],data_val,{'run_type':'val','debug':args.debug})
    matrix = evaluate(gt_landmarks, landmarks, th=0.1,save_path = model_dir + "/epoch_%d.png"%current_epoch)

    auc = matrix['auc']

    if args.local_rank == 0:
        summary_writer.add_scalar('val/bce', float(auc), global_step=current_epoch)
        if auc < bce_best:
            print("Epoch {} improved from {} to {}".format(current_epoch, bce_best, auc))
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce_best': auc,
            }, model_dir + "/model_best_dice.pth.tar")
            bce_best = auc

        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'bce_best': bce_best,
        }, model_dir + '/model_last.pth.tar')

        print("Epoch: {} bce: {}, bce_best: {}".format(current_epoch, auc, bce_best))
        print(matrix)
    return bce_best


def train_epoch(current_epoch, loss_function, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                local_rank, debug):
    #存储平均值
    progbar = Progbar(len(train_data_loader.dataset), stateful_metrics=['epoch', 'config','lr'])
    batch_time = AverageMeter()
    end = time.time()
    losses = AverageMeter()
    max_iters = conf['optimizer']['schedule']['params']['max_iter']
    print("training epoch {}".format(current_epoch))
    model.train()

    for i, (landmarks, imgs, img_path) in enumerate(train_data_loader):
        numm = imgs.shape[0]
        optimizer.zero_grad()
        imgs = imgs.reshape((-1,imgs.size(-3),imgs.size(-2), imgs.size(-1)))
        imgs = Variable(imgs, requires_grad=True).cuda()


        landmarks = landmarks.cuda().float()
        output = model(imgs)

        loss = loss_function(output,landmarks)


        losses.update(loss.item(), imgs.size(0))
        summary_writer.add_scalar('train/loss', loss.item(), global_step=i + current_epoch * max_iters)
        summary_writer.add_scalar('train/lr',float(scheduler.get_lr()[-1]), global_step=i + current_epoch * max_iters)

        if conf['fp16']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
        optimizer.step()
        torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()

        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
            if (i == max_iters - 1) or debug:
                break
        progbar.add(numm, values=[('epoch', current_epoch),
                                          ('loss', losses.avg),
                                          ("lr",float(scheduler.get_lr()[-1]))
                                  ])


    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    if local_rank == 0:
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        #summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)

if __name__ == '__main__':
    pass