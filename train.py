import os
import sys
import yaml
import argparse
import logging
import math
import importlib
import datetime
import random
import munch
import time
import torch
import torch.optim as optim
import warnings
import shutil
import subprocess
import wandb
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

from dataset import MeshDataset, DPC_Dataset
from utils.train_utils import *

def train():
    #######################################
    # wandb init() 初始化
    wandb.init(
        project="Point2SSM",  # 项目名称
        entity="jerryhu0209-technical-university-of-munich", # Team名称
        name="train_process_visualize",  # 实验名称
        config=args  # 超参数
    )
    #######################################

    logging.info(str(args))
    metrics = ['cd_p', 'cd_t']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    if args.model_name == 'dpc':
        dataset = DPC_Dataset(args, 'train')
        scale_factor = dataset.get_scale_factor()
        dataset_test = DPC_Dataset(args, 'val', scale_factor=scale_factor, ref_path=args.ref_path)
    else:
        dataset = MeshDataset(args, 'train')
        print(f"数据集大小: {len(dataset)}")
        scale_factor = dataset.get_scale_factor()
        dataset_test = MeshDataset(args, 'val', scale_factor=scale_factor)
        print(f"数据集大小: {len(dataset_test)}")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    # breakpoint()
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))
    # breakpoint()

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = args.device
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args)
    net.to(device)
    if hasattr(model_module, 'weights_init'):
        net.apply(model_module.weights_init)

    lr = args.lr
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    epochs_since_best_cd_t = 0
    for epoch in range(args.start_epoch, args.nepoch):
        start_time = time.time()
        torch.cuda.empty_cache()
        train_loss_meter.reset()
        net.train()

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            if args.model_name[:3] == 'dpc':
                pc, ref, gt, names = data
                pc, ref, gt = pc.to(device), ref.to(device), gt.to(device)
                source, target = pc.contiguous(), ref.contiguous()
                out, loss = net(source, target, gt)
            else:
                pc, gt, names = data
                pc, gt = pc.to(device), gt.to(device)
                inputs = pc.contiguous()
                if args.model_name == 'cpae':
                    out, loss = net(inputs, gt, epoch=epoch)
                else:
                    out, loss = net(inputs, gt)


            train_loss_meter.update(loss.mean().item())
            loss.backward()
            optimizer.step()
            #################################################
            # wandb train loss
            wandb.log({
                "train_loss": loss.mean().item(),
                "epoch": epoch,
                "batch": i,
                "lr": lr
            })
            #################################################

            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, loss: %f lr: %f' %
                             (epoch, i, len(dataset) / args.batch_size, args.loss, loss.mean().item(), lr) + ' time: ' + str(time.time()-start_time)[:4] + ' track: ' + str(epochs_since_best_cd_t) )

        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            best_cd_t = val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, device)
            if args.early_stop:
                if best_cd_t:
                    epochs_since_best_cd_t = 0
                else:
                    if epoch > args.early_stop_start:
                        epochs_since_best_cd_t += 1
                if epochs_since_best_cd_t > args.early_stop_patience:
                    print("Early stopping epoch:", epoch)
                    break

    best_cd_t = val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, device)

    args['best_model_path'] = log_dir+'/best_cd_p_network.pth'
    args['scale_factor'] = scale_factor
    return


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses, device):
    best_cd_t = False
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.eval()

    with torch.no_grad():
        ######################################################
        random_pc_sample = None  # 用于存储随机样本
        random_recon1 = None  # 用于存储该样本的网络输出
        random_name1 = None
        ######################################################
        for i, data in enumerate(dataloader_test):
            if args.model_name[:3] == 'dpc':
                pc, ref, gt, names = data
                pc, ref, gt = pc.to(device), ref.to(device), gt.to(device)
                source, target = pc.contiguous(), ref.contiguous()
                result_dict = net(source, target, gt, is_training=False)
            else:
                pc, gt, names = data
                pc, gt = pc.to(device), gt.to(device)
                inputs = pc.contiguous() 
                result_dict = net(inputs, gt, is_training=False)

            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())

                if random_pc_sample is None:
                    random_numbers = random.sample(range(0, 5 + 1), 2)
                    random_gt_sample = gt[random_numbers[0]].cpu().numpy()  # 原始groundtruth点云
                    random_pc_sample = pc[random_numbers[0]].cpu().numpy()  # 原始parcial点云
                    random_recon1 = result_dict["recon"][random_numbers[0]].cpu().numpy()  # 生成的点云
                    random_name1 = names[random_numbers[0]]  # 记录名称
                    random_recon2 = result_dict["recon"][random_numbers[1]].cpu().numpy()  # 生成的点云
                    random_name2 = names[random_numbers[1]]  # 记录名称
        #######################################################
        # wandb val loss
        wandb.log({
            "val_cd_p": val_loss_meters["cd_p"].avg,
            "val_cd_t": val_loss_meters["cd_t"].avg,
            "epoch": curr_epoch_num
        })
        #######################################################
        #######################################################

        # ✅ 如果有选取的随机样本，则进行可视化
        if random_pc_sample is not None:
            fig = plt.figure(figsize=(24, 6))

            # 原始gt点云
            ax0 = fig.add_subplot(141, projection='3d')
            color0 = normalize_coordinates_to_rgb(random_gt_sample)
            ax0.scatter(random_gt_sample[:, 0], random_gt_sample[:, 1], random_gt_sample[:, 2], c=color0, marker='o', s=100)
            ax0.set_title(f'Original: {random_name1}')
            ax0.set_xlabel('X')
            ax0.set_ylabel('Y')
            ax0.set_zlabel('Z')
            x_min, x_max = np.min(random_gt_sample[:, 0]), np.max(random_gt_sample[:, 0])
            y_min, y_max = np.min(random_gt_sample[:, 1]), np.max(random_gt_sample[:, 1])
            z_min, z_max = np.min(random_gt_sample[:, 2]), np.max(random_gt_sample[:, 2])

            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

            mid_x = (x_max + x_min) / 2.0
            mid_y = (y_max + y_min) / 2.0
            mid_z = (z_max + z_min) / 2.0

            ax0.set_xlim(mid_x - max_range, mid_x + max_range)
            ax0.set_ylim(mid_y - max_range, mid_y + max_range)
            ax0.set_zlim(mid_z - max_range, mid_z + max_range)
            ax0.set_box_aspect([1, 1, 1])

            # 原始pc点云
            ax1 = fig.add_subplot(142, projection='3d')
            color1 = normalize_coordinates_to_rgb(random_pc_sample)
            ax1.scatter(random_pc_sample[:, 0], random_pc_sample[:, 1], random_pc_sample[:, 2], c=color1, marker='o', s=100)
            ax1.set_title(f'Original: {random_name1}')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            x_min, x_max = np.min(random_pc_sample[:, 0]), np.max(random_pc_sample[:, 0])
            y_min, y_max = np.min(random_pc_sample[:, 1]), np.max(random_pc_sample[:, 1])
            z_min, z_max = np.min(random_pc_sample[:, 2]), np.max(random_pc_sample[:, 2])

            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

            mid_x = (x_max + x_min) / 2.0
            mid_y = (y_max + y_min) / 2.0
            mid_z = (z_max + z_min) / 2.0

            ax1.set_xlim(mid_x - max_range, mid_x + max_range)
            ax1.set_ylim(mid_y - max_range, mid_y + max_range)
            ax1.set_zlim(mid_z - max_range, mid_z + max_range)
            ax1.set_box_aspect([1, 1, 1])

            # 生成点云
            ax2 = fig.add_subplot(143, projection='3d')
            color2 = normalize_coordinates_to_rgb(random_recon1)
            ax2.scatter(random_recon1[:, 0], random_recon1[:, 1], random_recon1[:, 2], c=color2, marker='o', s=100)
            ax2.set_title(f'Reconstructed: {random_name1}')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            x_min, x_max = np.min(random_recon1[:, 0]), np.max(random_recon1[:, 0])
            y_min, y_max = np.min(random_recon1[:, 1]), np.max(random_recon1[:, 1])
            z_min, z_max = np.min(random_recon1[:, 2]), np.max(random_recon1[:, 2])

            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

            mid_x = (x_max + x_min) / 2.0
            mid_y = (y_max + y_min) / 2.0
            mid_z = (z_max + z_min) / 2.0

            ax2.set_xlim(mid_x - max_range, mid_x + max_range)
            ax2.set_ylim(mid_y - max_range, mid_y + max_range)
            ax2.set_zlim(mid_z - max_range, mid_z + max_range)
            ax2.set_box_aspect([1, 1, 1])

            # 生成点云
            ax3 = fig.add_subplot(144, projection='3d')
            ax3.scatter(random_recon2[:, 0], random_recon2[:, 1], random_recon2[:, 2], c=color2, marker='o', s=100)
            ax3.set_title(f'Reconstructed: {random_name2}')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            x_min, x_max = np.min(random_recon2[:, 0]), np.max(random_recon2[:, 0])
            y_min, y_max = np.min(random_recon2[:, 1]), np.max(random_recon2[:, 1])
            z_min, z_max = np.min(random_recon2[:, 2]), np.max(random_recon2[:, 2])

            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

            mid_x = (x_max + x_min) / 2.0
            mid_y = (y_max + y_min) / 2.0
            mid_z = (z_max + z_min) / 2.0

            ax3.set_xlim(mid_x - max_range, mid_x + max_range)
            ax3.set_ylim(mid_y - max_range, mid_y + max_range)
            ax3.set_zlim(mid_z - max_range, mid_z + max_range)
            ax3.set_box_aspect([1, 1, 1])

            plt.tight_layout()

            # ✅ 记录到 wandb
            wandb.log({"Validation Point Clouds": wandb.Image(fig)})

            # 关闭 Matplotlib 避免内存泄漏
            plt.close(fig)
        #######################################################
        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
                if loss_type == 'cd_t': # or loss_type =='kld': #TODO
                    best_cd_t = True
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)
    return best_cd_t

def normalize_coordinates_to_rgb(data):
    # 归一化处理，保证每个点的坐标都在 [0, 1] 范围内
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized = (data - min_vals) / (max_vals - min_vals)
    # 将归一化后的数据映射为 RGB 值
    return normalized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    
    print_time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        if 'encoder' in args:
            exp_name = args.model_name+'_'+args.encoder
        else:
            exp_name = args.model_name
        exp_name += '_'+print_time.replace(':',"-")
        log_dir = os.path.join(args.work_dir, args.dataset, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    print(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    

    train()

    # Update yaml in log dir
    with open(os.path.join(log_dir, os.path.basename(config_path)), 'w') as f:
        yaml.dump(args, f)
    print(os.path.join(log_dir, os.path.basename(config_path)))

    # Test
    subprocess.call(['python', 'test.py', '-c', os.path.join(log_dir, os.path.basename(config_path))])



