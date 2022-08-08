
import argparse
import os
import random
import yaml
import time
import logging
import pprint

import scipy.stats as stats
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import grad
from torch.cuda.amp import GradScaler, autocast
from easydict import EasyDict

from data.train_both_sketch_aug_black import CreateDataLoader as train_loader
from utils import create_logger, save_checkpoint, load_state, get_scheduler, AverageMeter
from models.standard import *

import copy

parser = argparse.ArgumentParser(description='PyTorch Colorization Training')

parser.add_argument('--config', default='experiments/origin/config.yaml')
# parser.add_argument('--resume', default='experiments/origin/ckpt.pth.tar', type=str, help='path to checkpoint')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')

def get_z_random(batch_size, nz, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, nz)
    return z.detach().to(config.device)

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = criterionBCE(logits, targets)
    return loss

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = config.batch_size
    grad_dout = torch.autograd.grad(
        outputs=scalerD.scale(d_out.sum()), inputs= x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    with autocast(enabled=ampD):
        inv_scale = 1./scalerD.get_scale()
        grad_dout = grad_dout * inv_scale 
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def encode(netE, sketch, pose):
    inp = torch.cat([sketch,pose],1)
    mu, logvar = netE(inp)
    std = logvar.mul(0.5).exp_()
    eps = get_z_random(std.size(0), std.size(1))
    z = eps.mul(std).add_(mu)
    return z, mu, logvar


def main():
    global args, config, X, ampD, criterionBCE, scalerD
    args = parser.parse_args()
    print(args)

    with open(args.config) as f:
        config = EasyDict(yaml.safe_load(f))

    config.save_path = os.path.dirname(args.config)
    gpu_id = 0


    ####### regular set up
    assert torch.cuda.is_available()
    device = torch.device("cuda:"+str(gpu_id))
    config.device = device

    # random seed setup
    print("Random Seed: ", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True

    ####### regular set up end


    netG = torch.nn.DataParallel(Generator(img_size=config.image_size, style_dim=config.nz*4), device_ids =[gpu_id])
    netD = torch.nn.DataParallel(NetD(img_size=config.image_size), device_ids =[gpu_id])
    netE = torch.nn.DataParallel(StyleEncoder(img_size=config.image_size, style_dim=config.nz*4), device_ids =[gpu_id])
    mapping_network = torch.nn.DataParallel(MappingNetwork(latent_dim=config.nz, style_dim=config.nz*4), device_ids =[gpu_id])
    netF = torch.nn.DataParallel(NetF(), device_ids =[gpu_id])

    criterion_MSE = nn.MSELoss()
    criterion_MSE2 = nn.MSELoss()
    criterionL1 = torch.nn.L1Loss()
    criterionBCE = torch.nn.BCEWithLogitsLoss()

    fixed_sketch = torch.tensor(0, device=device).float()
    # fixed_sketch2 = torch.tensor(0, device=device).float()
    fixed_pose_ori = torch.tensor(0, device=device).float()
    fixed_pose_target = torch.tensor(0, device=device).float()
    # fixed_keypoint_ori = torch.tensor(0, device=device).float()
    # fixed_keypoint_target = torch.tensor(0, device=device).float()

    ####################
    netD = netD.to(device)
    netG = netG.to(device)
    netE = netE.to(device)
    netF = netF.to(device)
    mapping_network = mapping_network.to(device)
    criterion_MSE = criterion_MSE.to(device)
    criterion_MSE2 = criterion_MSE2.to(device)
    criterionL1 = criterionL1.to(device)
    criterionBCE = criterionBCE.to(device)

    # ema
    netG_ema = copy.deepcopy(netG)
    mapping_network_ema = copy.deepcopy(mapping_network)
    netE_ema = copy.deepcopy(netE)

    netE.apply(he_init)
    mapping_network.apply(he_init)
    netG.apply(he_init)
    # netD.apply(he_init)

    # setup optimizer
    ampG=True
    ampD=True
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr_scheduler.base_lr, betas=(0, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr_scheduler.base_lr, betas=(0, 0.99))
    optimizerE = optim.Adam(netE.parameters(), lr=config.lr_scheduler.base_lr, betas=(0, 0.99))
    optimizerMN = optim.Adam(mapping_network.parameters(), lr=config.lr_scheduler.base_lr*0.01, betas=(0, 0.99))

    scalerG = torch.cuda.amp.GradScaler(enabled=ampG)
    scalerD = torch.cuda.amp.GradScaler(enabled=ampD)
    last_iter = -1

    if args.resume:
        last_iter = load_state(args.resume, netG, netD, optimizerG, optimizerD, scalerG, scalerD)

    config.lr_scheduler['last_iter'] = last_iter

    config.lr_scheduler['optimizer'] = optimizerG
    lr_schedulerG = get_scheduler(config.lr_scheduler)
    config.lr_scheduler['optimizer'] = optimizerD
    lr_schedulerD = get_scheduler(config.lr_scheduler)
    config.lr_scheduler['optimizer'] = optimizerE
    lr_schedulerE = get_scheduler(config.lr_scheduler)
    config.lr_scheduler['optimizer'] = optimizerMN
    lr_schedulerMN = get_scheduler(config.lr_scheduler)

    tb_logger = SummaryWriter(config.save_path + '/events')
    logger = create_logger('global_logger', config.save_path + '/log.txt')
    logger.info(f'args: {pprint.pformat(args)}')
    logger.info(f'config: {pprint.pformat(config)}')

    batch_time = AverageMeter(config.print_freq)
    data_time = AverageMeter(config.print_freq)
    flag = 1
    i = 0
    curr_iter = last_iter + 1

    dataloader = train_loader(config)
    data_iter = iter(dataloader)

    end = time.time()

    lambda_kl = 0
    lambda_ds = 1 - curr_iter/ config.ds_iter

    while i < len(dataloader):
        lr_schedulerG.step(curr_iter)
        lr_schedulerD.step(curr_iter)
        lr_schedulerE.step(curr_iter)
        lr_schedulerMN.step(curr_iter)
        current_lr = lr_schedulerG.get_lr()[0]
        ############################
        # (1) Update D network
        ###########################
        data_end = time.time()
        sketch, _, pose_ori, _,_,_ = data_iter.next()
        data_time.update(time.time() - data_end)

        # if (curr_iter+1 > config.kl_start_iter):
        #     lambda_kl += (1 / config.kl_iter)
        lambda_ds -= (1 / config.ds_iter)

            
        # get real images
        i += 1
        sketch, pose_ori= sketch.to(device), pose_ori.to(device)

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netG.parameters():
            p.requires_grad = False  # to avoid computation ft_params
        for p in netE.parameters():
            p.requires_grad = False  # to avoid computation ft_params
        for p in mapping_network.parameters():
            p.requires_grad = False  # to avoid computation ft_params


        with autocast(enabled=ampD):

            netD.zero_grad()
            with torch.no_grad():
                # get encoded z
                z_encoded = netE(sketch, pose_ori)
                fake_sketch_encoded = netG(pose_ori, z_encoded)

            sketch.requires_grad_()
            D_fake_encoded  = netD(fake_sketch_encoded)
            D_real = netD(sketch)

            loss_real = adv_loss(D_real, 1)
            loss_fake_encoded = adv_loss(D_fake_encoded, 0)

        loss_reg = r1_reg(D_real, sketch)

        #Optimise
        combine_lossD = loss_real + loss_fake_encoded + loss_reg 
        scalerD.scale(combine_lossD).backward()
        scalerD.step(optimizerD)
        scalerD.update()

        # get real images
        sketch, _, pose_ori, _,_,_ = data_iter.next()
        sketch, pose_ori= sketch.to(device), pose_ori.to(device)
        i += 1

        with autocast(enabled=ampD):
            netD.zero_grad()
            with torch.no_grad():
                # get random z
                z_random = get_z_random(config.batch_size, config.nz)
                z_mn = mapping_network(z_random)
                fake_sketch_random = netG(pose_ori, z_mn)

            sketch.requires_grad_()
            D_fake_random  = netD(fake_sketch_random)
            D_real = netD(sketch)

            loss_real = adv_loss(D_real, 1)
            loss_fake_random = adv_loss(D_fake_random, 0)

        loss_reg = r1_reg(D_real, sketch)

        #Optimise
        combine_lossD = loss_real + loss_fake_random + loss_reg 
        scalerD.scale(combine_lossD).backward()
        scalerD.step(optimizerD)
        scalerD.update()

        ############################
        # (2) Update G network
        ############################

        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        for p in netG.parameters():
            p.requires_grad = True
        for p in netE.parameters():
            p.requires_grad = True  # to avoid computation ft_params
        for p in mapping_network.parameters():
            p.requires_grad = True  # to avoid computation ft_params

        netG.zero_grad()
        netE.zero_grad()
        mapping_network.zero_grad()

        sketch, _, pose_ori, pose_target, _, _ = data_iter.next()
        sketch, pose_ori, pose_target  = sketch.to(device), pose_ori.to(device), pose_target.to(device)
        i += 1

        if flag:  # fix samples
            tb_logger.add_image('sketch imgs', vutils.make_grid(sketch.mul(0.5).add(0.5), nrow=config.batch_size))
            tb_logger.add_image('pose_ori imgs', vutils.make_grid(pose_ori.mul(0.5).add(0.5), nrow=config.batch_size))
            tb_logger.add_image('pose_target', vutils.make_grid(pose_target.mul(0.5).add(0.5), nrow=config.batch_size))

            fixed_sketch.resize_as_(sketch).copy_(sketch)
            # fixed_sketch2.resize_as_(sketch2).copy_(sketch2)
            fixed_pose_ori.resize_as_(pose_ori).copy_(pose_ori)
            fixed_pose_target.resize_as_(pose_target).copy_(pose_target)
            # fixed_keypoint_ori.resize_as_(keypoint_ori).copy_(keypoint_ori)
            # fixed_keypoint_target.resize_as_(keypoint_target).copy_(keypoint_target)
            
            flag -= 1

        freeze = 0
        if curr_iter > freeze:
            with torch.cuda.amp.autocast(enabled=ampG):

                z_random = get_z_random(config.batch_size, config.nz)
                z_mn = mapping_network(z_random)
                fake_sketch_random = netG(pose_ori, z_mn)

                G_adv_loss_random = netD(fake_sketch_random)
                G_loss_adv_random = adv_loss(G_adv_loss_random, 1) 

                # style reconstruction loss
                z_pred = netE(fake_sketch_random, pose_ori)
                loss_sty_random = torch.mean(torch.abs(z_pred - z_mn))

                #5 perceptual reconstruct
                with torch.no_grad():
                    feat_real_random = netF(sketch)

                feat_fake_random = netF(fake_sketch_random)
                perp_loss_random = criterion_MSE(feat_real_random, feat_fake_random)

                z_random2 = get_z_random(config.batch_size, config.nz)
                z_mn2 = mapping_network(z_random2)
                with torch.no_grad():
                    fake_sketch_random2 = netG(pose_ori, z_mn2)
                    feat_fake_random2 = netF(fake_sketch_random2)
                loss_ds_random = torch.mean(torch.abs(feat_fake_random - feat_fake_random2)) * lambda_ds

                # KL
                z_encoded = netE(sketch, pose_ori)

                if lambda_kl > 0:
                    loss_kl  = torch.mean(torch.pow(z_encoded,2)) + torch.mean(torch.pow(z_mn,2)) * lambda_kl 
                    combine_lossG = G_loss_adv_random  + perp_loss_random + loss_sty_random - loss_ds_random + loss_kl
                else:
                    combine_lossG = G_loss_adv_random  + perp_loss_random + loss_sty_random - loss_ds_random 

                meandiff_mn = torch.mean(torch.abs(z_mn2 - z_mn))
            
            scalerG.scale(combine_lossG).backward()
            scalerG.step(optimizerG)
            scalerG.step(optimizerE)
            scalerG.step(optimizerMN)
            scalerG.update()

            sketch_e, sketch_e2, pose_ori_e, pose_ori_e2, _, _ = data_iter.next()
            sketch_e, sketch_e2, pose_ori_e, pose_ori_e2  = sketch_e.to(device), sketch_e2.to(device), pose_ori_e.to(device), pose_ori_e2.to(device)
            i += 1
            with torch.cuda.amp.autocast(enabled=ampG):

                netG.zero_grad()
                # netE.zero_grad()
                z_encoded = netE(sketch_e, pose_ori_e)
                fake_sketch_encoded = netG(pose_ori_e, z_encoded)

                G_adv_loss_encoded = netD(fake_sketch_encoded)
                G_loss_adv_encoded = adv_loss(G_adv_loss_encoded, 1) 

                # style reconstruction loss
                z_pred = netE(fake_sketch_encoded, pose_ori_e)
                loss_sty_encoded = torch.mean(torch.abs(z_pred - z_encoded))

                #5 perceptual reconstruct
                with torch.no_grad():
                    feat_real_encoded = netF(sketch_e)

                feat_fake_encoded = netF(fake_sketch_encoded)
                perp_loss_encoded = criterion_MSE(feat_real_encoded, feat_fake_encoded)

                z_encoded2 = netE(sketch_e2, pose_ori_e2)

                with torch.no_grad():
                    fake_sketch_encoded2 = netG(pose_ori_e, z_encoded2)
                    feat_fake_encoded2 = netF(fake_sketch_encoded2)
                loss_ds_encoded = torch.mean(torch.abs(feat_fake_encoded - feat_fake_encoded2)) * lambda_ds

                if lambda_kl > 0:
                    loss_kl  = torch.mean(torch.pow(z_encoded,2)) + torch.mean(torch.pow(z_encoded2,2))  * lambda_kl 
                    combine_lossG = G_loss_adv_encoded  + perp_loss_encoded + loss_sty_encoded - loss_ds_encoded + loss_kl

                else:
                    combine_lossG = G_loss_adv_encoded  + perp_loss_encoded + loss_sty_encoded - loss_ds_encoded 

                meandiff_e = torch.mean(torch.abs(z_encoded2 - z_encoded))
            
            scalerG.scale(combine_lossG).backward()
            scalerG.step(optimizerG)
            # scalerG.step(optimizerE)

            scalerG.update()

        batch_time.update(time.time() - end)

        moving_average(netG, netG_ema, beta=0.999)
        moving_average(mapping_network, mapping_network_ema, beta=0.999)
        moving_average(netE, netE_ema, beta=0.999)

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        curr_iter += 1


        if curr_iter > freeze and curr_iter % config.print_freq == 0 and curr_iter > 0:
            tb_logger.add_scalar('errD_real', loss_real.item(), curr_iter)
            tb_logger.add_scalar('errD_fake_encode', loss_fake_encoded.item(), curr_iter)
            tb_logger.add_scalar('errD_fake_random', loss_fake_random.item(), curr_iter)
            tb_logger.add_scalar('errG_encoded', G_loss_adv_encoded.item(), curr_iter)
            tb_logger.add_scalar('errG_random', G_loss_adv_random.item(), curr_iter)
            
            tb_logger.add_scalar('r1', loss_reg.item(), curr_iter)
            tb_logger.add_scalar('lr', current_lr, curr_iter)

            tb_logger.add_scalar('G_perp_loss_encoded', perp_loss_encoded.item(), curr_iter)
            tb_logger.add_scalar('G_perp_loss_random', perp_loss_random.item(), curr_iter)
            tb_logger.add_scalar('sty_encoded', loss_sty_encoded.item(), curr_iter)
            tb_logger.add_scalar('sty_random', loss_sty_random.item(), curr_iter)

            tb_logger.add_scalar('loss_ds_random', loss_ds_random.item(), curr_iter)
            tb_logger.add_scalar('loss_ds_encoded', loss_ds_encoded.item(), curr_iter)

            tb_logger.add_scalar('meandiff_mn', meandiff_mn.item(), curr_iter)
            tb_logger.add_scalar('meandiff_e', meandiff_e.item(), curr_iter)

            # tb_logger.add_scalar('scalerD', scalerD.get_scale(), curr_iter)
            # tb_logger.add_scalar('scalerG', scalerG.get_scale(), curr_iter)

            if lambda_kl > 0:
                tb_logger.add_scalar('loss_kl', loss_kl.item(), curr_iter)

            logger.info(f'Iter: [{curr_iter}/{len(dataloader)//(config.diters+1)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'errG_e {G_loss_adv_encoded.item():.4f}\t'
                        f'errG_r {G_loss_adv_random.item():.4f}\t'
                        f'errD {loss_real.item():.4f}\t'
                        f'LR {current_lr:.6f} \t'
                        )

        if curr_iter > freeze and  curr_iter % config.print_img_freq == 0 and curr_iter > 0:
            with torch.no_grad():

                # without ema
                tb_logger.add_image('current batch_sketch (encoded)',
                    vutils.make_grid(torch.cat([sketch_e, fake_sketch_encoded],0).detach().mul(0.5).add(0.5),  nrow=config.batch_size //2),
                    curr_iter)

                tb_logger.add_image('current batch_sketch (random)',
                    vutils.make_grid(torch.cat([sketch, fake_sketch_random],0).detach().mul(0.5).add(0.5),  nrow=config.batch_size //2),
                    curr_iter)

                ### Use ema instead
                with torch.no_grad():
                    z_encoded = netE_ema(sketch_e, pose_ori_e)
                    fake_sketch_encoded = netG_ema(pose_ori_e, z_encoded)
                    z_mn = mapping_network_ema(z_random)
                    fake_sketch_random = netG_ema(pose_ori, z_mn)

                tb_logger.add_image('current batch_sketch ema (encoded)',
                    vutils.make_grid(torch.cat([sketch_e, fake_sketch_encoded],0).detach().mul(0.5).add(0.5),  nrow=config.batch_size //2),
                    curr_iter)

                tb_logger.add_image('current batch_sketch ema (random)',
                    vutils.make_grid(torch.cat([sketch, fake_sketch_random],0).detach().mul(0.5).add(0.5),  nrow=config.batch_size //2),
                    curr_iter)


                ### With fixed input
                with autocast():
                    z_encoded = netE_ema(fixed_sketch, fixed_pose_ori)
                    fake1  = netG_ema(fixed_pose_ori, z_encoded)
                    fake4  = netG_ema(fixed_pose_target, z_encoded)
                    
                    z_random = get_z_random(config.batch_size, config.nz)
                    z_mn = mapping_network_ema(z_random)
                    
                    fake2  = netG_ema(fixed_pose_ori, z_mn)
                    fake3  = netG_ema(fixed_pose_target, z_mn)

                    fake5  = netG_ema(fixed_pose_ori[:1].repeat(config.batch_size,1,1,1), z_encoded)
                    fake6  = netG_ema(fixed_pose_ori[:1].repeat(config.batch_size,1,1,1), z_mn)

                grid = vutils.make_grid(torch.cat([fixed_sketch, fake1, fake2, fake4, fake3],0).detach().mul(0.5).add(0.5), nrow=config.batch_size)
                vutils.save_image(torch.cat([fixed_sketch, fake1, fake2, fake4, fake3],0), 'clean_tensorboard_images/' + str(curr_iter) + '.png')


                tb_logger.add_image('a_sketch (fix)',
                                    grid,
                                    curr_iter)

                tb_logger.add_image('a_sketch (fix_z)',
                                    vutils.make_grid(fake5.detach().mul(0.5).add(0.5), nrow=config.batch_size),
                                    curr_iter)

                tb_logger.add_image('a_sketch (random_z)',
                                    vutils.make_grid(fake6.detach().mul(0.5).add(0.5), nrow=config.batch_size),
                                    curr_iter)

                target1 = z_encoded[0:1].float()
                target2 = z_mn[0:1].float()
                scale = [0.125,0.25,0.375,0.5,0.625,0.75,0.875]
                z121 = torch.lerp(target1,target2, scale[0])
                z122 = torch.lerp(target1,target2, scale[1])
                z123 = torch.lerp(target1,target2, scale[2])
                z124 = torch.lerp(target1,target2, scale[3])
                z125 = torch.lerp(target1,target2, scale[4])
                z126 = torch.lerp(target1,target2, scale[5])
                z127 = torch.lerp(target1,target2, scale[6])

                with autocast():
                    fake = netG_ema(fixed_pose_ori[0:1].repeat(len(scale)+2,1,1,1), torch.cat([target1,z121,z122,z123,z124,z125,z126,z127,target2],0),)
                    fake2 = netG_ema(fixed_pose_ori[-1:].repeat(len(scale)+2,1,1,1), torch.cat([target1,z121,z122,z123,z124,z125,z126,z127,target2],0),)

                tb_logger.add_image('imgs interpolation',
                        vutils.make_grid(torch.cat([fake, fake2],0).detach().mul(0.5).add(0.5),  nrow=len(scale)+2),
                        curr_iter)
                # tb_logger.add_image('a_pose (fix)',
                #     vutils.make_grid(pred_pose.detach().mul(0.5).add(0.5),  nrow=config.batch_size),
                #     curr_iter)



        if curr_iter % config.val_freq == 0 or curr_iter % config.lr_scheduler.lr_steps[0] == 0:
            save_checkpoint({
                    'step': curr_iter - 1,
                    'state_dictG': netG_ema.state_dict(),
                    'state_dictD': netD.state_dict(),
                    'state_dictE': netE_ema.state_dict(),
                    'state_dictMN': mapping_network_ema.state_dict(),


                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'optimizerE': optimizerE.state_dict(),
                    'optimizerMN': optimizerMN.state_dict(),
                    'scalerG': scalerG.state_dict(),
                    'scalerD': scalerD.state_dict(),


            }, False, config.save_path + '/ckpt')


            if(curr_iter == config.lr_scheduler.lr_steps[0]):
                save_checkpoint({
                    'step': curr_iter - 1,
                    'state_dictG': netG_ema.state_dict(),
                    'state_dictD': netD.state_dict(),
                    'state_dictE': netE_ema.state_dict(),
                    'state_dictMN': mapping_network_ema.state_dict(),

                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'optimizerE': optimizerE.state_dict(),
                    'optimizerMN': optimizerMN.state_dict(),
                    'scalerG': scalerG.state_dict(),
                    'scalerD': scalerD.state_dict(),

                }, False, config.save_path + '/ckpt_milestone')
  
        end = time.time()

if __name__ == '__main__':
    main()
