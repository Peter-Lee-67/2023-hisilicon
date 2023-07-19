import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.util import confusion_matrix, getScores, save_images, tensor2im
import torch
import numpy as np
import cv2


def get_wrong_and_save(img, pred, gt, image_name, save):
    k = (gt==1)
    m = (pred==1)
    pic = img.squeeze(0)
    pred_and_right = (k & m)
    pred_but_wrong = (k & (~ m))
    unpred_but_right = ((~ k) & m)
    pic[0][pred_but_wrong[0]] = 1 # blue
    pic[2][unpred_but_right[0]] = 1 # red
    pic[1][pred_and_right[0]] = 1 # green
    pic = (np.transpose(pic, (1, 2, 0)))* 255.0
    image_name = image_name[0]
    cv2.imwrite(os.path.join(save, image_name[:-10]+'road_'+image_name[-10:]), 
        cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))



if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False
    save = './visual/'+str(opt.epoch)+opt.epoch_load
    if opt.visual is True:
        if not os.path.exists(save):
            os.makedirs(save)

    save_dir = os.path.join(opt.results_dir, opt.name, opt.phase + '_' + opt.epoch + opt.epoch_load)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt, dataset.dataset)
    model.setup(opt)
    model.eval()

    test_loss_iter = []
    epoch_iter = 0
    conf_mat = np.zeros((dataset.dataset.num_labels, dataset.dataset.num_labels), dtype=float)
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.forward()
            model.get_loss()
            epoch_iter += opt.batch_size
            gt = model.label.cpu().int().numpy()
            rgb = model.rgb_image.cpu().numpy()
            _, pred = torch.max(model.output.data.cpu(), 1)
            pred = pred.float().detach().int().numpy()
            if opt.visual is True:
                get_wrong_and_save(rgb, pred, gt, model.get_image_names(), save)
            save_images(save_dir, model.get_current_visuals(), model.get_image_names(), model.get_image_oriSize(), opt.prob_map)

            # Resize images to the original size for evaluation
            image_size = model.get_image_oriSize()
            oriSize = (image_size[0].item(), image_size[1].item())
            gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            conf_mat += confusion_matrix(gt, pred, dataset.dataset.num_labels)
            # print(conf_mat)
            test_loss_iter.append(model.loss_segmentation)
            print('Epoch {0:}, iters: {1:}/{2:}, loss: {3:.3f} '.format(opt.epoch,
                                                                        epoch_iter,
                                                                        len(dataset) * opt.batch_size,
                                                                        test_loss_iter[-1]), end='\r')
        # print(conf_mat)
        avg_test_loss = torch.mean(torch.stack(test_loss_iter))
        print ('Epoch {0:} test loss: {1:.3f} '.format(opt.epoch, avg_test_loss))
        globalacc, pre, recall, F_score, iou = getScores(conf_mat)
        print ('Epoch {0:} glob acc : {1:.3f}, pre : {2:.3f}, recall : {3:.3f}, F_score : {4:.3f}, IoU : {5:.3f}'.format(opt.epoch, globalacc, pre, recall, F_score, iou))
