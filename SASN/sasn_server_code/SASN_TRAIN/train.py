import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.util import confusion_matrix, getScores, tensor2labelim, tensor2im, print_current_losses, save_images
import numpy as np
import random
import torch
import cv2
from tensorboardX import SummaryWriter
import os




if __name__ == '__main__':
    
    train_opt = TrainOptions().parse()
    stop_epoch = train_opt.early_stop
    save_begin = train_opt.save_begin
    # np.random.seed(train_opt.seed)
    # random.seed(train_opt.seed)
    # torch.manual_seed(train_opt.seed)
    # torch.cuda.manual_seed(train_opt.seed)



    train_opt.prefetch_factor = 4
    train_data_loader = CreateDataLoader(train_opt)
    train_dataset = train_data_loader.load_data()
    train_dataset_size = len(train_data_loader)
    print('#training images = %d' % train_dataset_size)

    valid_opt = TrainOptions().parse()
    valid_opt.phase = 'val'
    valid_opt.batch_size = 4
    valid_opt.num_threads = 2
    valid_opt.serial_batches = True
    valid_opt.isTrain = False
    valid_opt.prefetch_factor = 2
    valid_data_loader = CreateDataLoader(valid_opt)
    valid_dataset = valid_data_loader.load_data()
    valid_dataset_size = len(valid_data_loader)
    print('#validation images = %d' % valid_dataset_size)

    valid2_opt = TrainOptions().parse()
    valid2_opt.phase = 'val2'
    valid2_opt.batch_size = 4
    valid2_opt.num_threads = 10
    valid2_opt.serial_batches = True
    valid2_opt.isTrain = False
    valid2_opt.prefetch_factor = 2
    valid2_data_loader = CreateDataLoader(valid2_opt)
    valid2_dataset = valid2_data_loader.load_data()
    valid2_dataset_size = len(valid2_data_loader)
    print('#validation images = %d' % valid2_dataset_size)

    writer = SummaryWriter()

    palet_file = './datasets/palette.txt'
    impalette = list(np.genfromtxt(palet_file,dtype=np.uint8).reshape(3*256))

    model = create_model(train_opt, train_dataset.dataset)
    model.setup(train_opt)
    total_steps = 0
    F_score_max = 0
    F_score_max = 0
    imax_epoch = 0
    name = train_opt.make_name

    real_dir = os.path.join(model.save_dir, name)
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)


    for epoch in range(train_opt.epoch_count, train_opt.nepoch + 1):
        ### Training on the training set ###
        model.train()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        train_loss_iter = []
        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()
            if total_steps % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % train_opt.print_freq == 0:
                losses = model.get_current_losses()
                train_loss_iter.append(losses["segmentation"])
                t = (time.time() - iter_start_time) / train_opt.batch_size
                print_current_losses(epoch, epoch_iter, losses, t, t_data)

            iter_data_time = time.time()

        mean_loss = np.mean(train_loss_iter)
        # One average training loss value in tensorboard in one epoch
        writer.add_scalar('train/mean_loss', mean_loss, epoch)

        if epoch % 5 == 0:
            tempDict = model.get_current_visuals()
            rgb = tensor2im(tempDict['rgb_image'])
            label = tensor2labelim(tempDict['label'], impalette)
            output = tensor2labelim(tempDict['output'], impalette)
            image_numpy = np.concatenate((rgb, label, output), axis=1)
            image_numpy = image_numpy.astype(np.float64) / 255
            writer.add_image('Epoch' + str(epoch), image_numpy, dataformats='HWC')  # show training images in tensorboard

        print('End of epoch %d / %d \t Time Taken: %d sec' %   (epoch, train_opt.nepoch, time.time() - epoch_start_time))
        writer.add_scalar('train/lr', model.optimizers[0].param_groups[0]['lr'], epoch)
        model.update_learning_rate()

        ### Evaluation on the validation set ###
        model.eval()
        valid_loss_iter = []
        epoch_iter = 0
        conf_mat = np.zeros((valid_dataset.dataset.num_labels, valid_dataset.dataset.num_labels), dtype=float)
        with torch.no_grad():
            for i, data in enumerate(valid_dataset):
                model.set_input(data)
                model.forward()
                model.get_loss()
                gt = model.label.cpu().int().numpy()
                _, pred = torch.max(model.output.data.cpu(), 1)
                pred = pred.float().detach().int().numpy()
                epoch_iter += gt.shape[0]

                conf_mat += confusion_matrix(gt, pred, valid_dataset.dataset.num_labels)
                valid_loss_iter.append(model.loss_segmentation)
                print('epoch {0:} valid, iters: {1:}/{2:} '.format(epoch, epoch_iter, len(valid_dataset)), end='\r')

        avg_valid_loss = torch.mean(torch.stack(valid_loss_iter))
        globalacc, pre, recall, F_score, iou = getScores(conf_mat)

        # Record performance on the validation set
        writer.add_scalar('valid/loss', avg_valid_loss, epoch)
        # writer.add_scalar('valid/global_acc', globalacc, epoch)
        # writer.add_scalar('valid/pre', pre, epoch)
        # writer.add_scalar('valid/recall', recall, epoch)
        writer.add_scalar('valid/F_score', F_score, epoch)
        writer.add_scalar('valid/iou', iou, epoch)



        if epoch % train_opt.save_period == 0:
            model.save_networks(name+str(epoch), foldername=name)

        # Save the best model according to the F-score, and record corresponding epoch number in tensorboard
        if F_score > F_score_max:
            imax_epoch = epoch
            if epoch > save_begin:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
                model.save_networks(name, foldername=name)
            F_score_max = F_score
            writer.add_text('best model', str(epoch))
            writer.add_text('best/valid/loss', str(avg_valid_loss))
            writer.add_text('best/valid/global_acc', str(globalacc))
            writer.add_text('best/valid/pre', str(pre))
            writer.add_text('best/valid/recall', str(recall))
            writer.add_text('best/valid/F_score', str(F_score))
            writer.add_text('best/valid/iou', str(iou))
            
        if epoch - imax_epoch > stop_epoch and epoch > 50:
            break
    
    writer.close()