# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Entry point for testing AttGAN network."""

import argparse
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np


import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F


from data import CelebA
import attacks

# AttGAN配置
from AttGAN.attgan import AttGAN
from AttGAN.data import check_attribute_conflict
from AttGAN.helpers import Progressbar
from AttGAN.utils import find_model
import AttGAN.attacks as attgan_attack # Attack of AttGan

# stargan配置
from stargan.solver import Solver

# attentiongan配置
from AttentionGAN.AttentionGAN_v1_multi.solver import Solver as AttentionGANSolver



class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
        
        
    return args_attack


args_attack = parse()
print(args_attack)
os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
print("实验目录已创建")
os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
print("实验配置已保存")

# 初始化攻击器
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, feat=None, args=args_attack.attacks)
    return pgd_attack

# 初始化攻击数据
def init_attack_data(args_attack, attgan_args):
    test_dataset = CelebA(args_attack.global_settings.data_path, args_attack.global_settings.attr_path, args_attack.global_settings.img_size, 'test', attgan_args.attrs,args_attack.stargan.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=args_attack.global_settings.batch_size, num_workers=0,
        shuffle=False, drop_last=False
    )
    if args_attack.global_settings.num_test is None:
        print('Testing images:', len(test_dataset))
    else:
        print('Testing images:', min(len(test_dataset), args_attack.global_settings.num_test))
    return test_dataloader

# 初始化测试数据
def init_inference_data(args_attack, attgan_args):
    test_dataset = CelebA(args_attack.global_settings.data_path, args_attack.global_settings.attr_path, args_attack.global_settings.img_size, 'test', attgan_args.attrs,args_attack.stargan.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, num_workers=0,
        shuffle=False, drop_last=False
    )
    if args_attack.global_settings.num_test is None:
        print('Testing images:', len(test_dataset))
    else:
        print('Testing images:', min(len(test_dataset), args_attack.global_settings.num_test))
    return test_dataloader

# 初始化attGAN
def init_attGAN(args_attack):
    with open(join('./AttGAN/output', args_attack.AttGAN.attgan_experiment_name, 'setting.txt'), 'r') as f:
        args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    args.test_int = args_attack.AttGAN.attgan_test_int
    args.num_test = args_attack.global_settings.num_test
    args.gpu = args_attack.global_settings.gpu
    args.load_epoch = args_attack.AttGAN.attgan_load_epoch
    args.multi_gpu = args_attack.AttGAN.attgan_multi_gpu
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)
    attgan.load(find_model(join('./AttGAN/output', args.experiment_name, 'checkpoint'), args.load_epoch))
    attgan.eval()
    return attgan, args

# 初始化stargan
def init_stargan(args_attack, test_dataloader):
    return Solver(celeba_loader=test_dataloader, rafd_loader=None, config=args_attack.stargan)

# 初始化attentiongan
def init_attentiongan(args_attack, test_dataloader):
    return AttentionGANSolver(celeba_loader=test_dataloader, rafd_loader=None, config=args_attack.AttentionGAN)
    
# 初始化FaderNetworks
def init_FaderNetworks(args_attack):
    ae = load_ae_model(args_attack.FaderNetworks)
    return ae


# 初始化攻击器
pgd_attack = init_Attack(args_attack)

# 初始化待攻击模型
attgan, attgan_args = init_attGAN(args_attack)
test_dataloader = init_attack_data(args_attack, attgan_args)
solver = init_stargan(args_attack, test_dataloader)
solver.restore_model(solver.test_iters)
attentiongan_solver = init_attentiongan(args_attack, test_dataloader)
attentiongan_solver.restore_model(attentiongan_solver.test_iters)
fader_ae = init_FaderNetworks(args_attack)
fader_test_data = load_test_data(args_attack.FaderNetworks, fader_ae)
print("初始化完毕")



# 载入保存的扰动
universal_perturbation = torch.load(args_attack.global_settings.universal_perturbation_path)
print('载入通用扰动成功')
pgd_attack.up = universal_perturbation
print(pgd_attack.up.shape)
test_dataloader = init_inference_data(args_attack, attgan_args)



# 载入电影人脸数据

tf = transforms.Compose([
            # transforms.CenterCrop(170),
            transforms.Resize(args_attack.global_settings.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])



# stargan inference and evaluating
l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
n_dist, n_samples = 0, 0
for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
    # print(idx)
    if idx!=291:
        continue
    img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
    att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
    att_a = att_a.type(torch.float)
    x_noattack_list, x_fake_list = solver.test_universal_model_level(idx, img_a, c_org, pgd_attack.up*3, args_attack.stargan)
    for j in range(len(x_fake_list)):
        gen_noattack = x_noattack_list[j]
        gen = x_fake_list[j]
        l1_error += F.l1_loss(gen, gen_noattack)
        l2_error += F.mse_loss(gen, gen_noattack)
        l0_error += (gen - gen_noattack).norm(0)
        min_dist += (gen - gen_noattack).norm(float('-inf'))
        if F.mse_loss(gen, gen_noattack) > 0.05:
            n_dist += 1
        n_samples += 1
    
    ############# 保存图片做指标评测 #############
    # 保存原图
    out_file = './stargan_{}.jpg'.format(idx + 182638)
    vutils.save_image(img_a.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.))
    for j in range(len(x_fake_list)):
        # 保存原图生成图片
        gen_noattack = x_noattack_list[j]
        out_file = './stargan_{}_{}.jpg'.format(idx + 182638, j)
        vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, range=(-1., 1.))
        # 保存对抗样本生成图片
        gen = x_fake_list[j]
        out_file = './stargan_{}_{}.jpg'.format(idx + 182638, j)
        vutils.save_image(gen, out_file, nrow=1, normalize=True, range=(-1., 1.))
