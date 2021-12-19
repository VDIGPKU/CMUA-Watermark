import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm
from PIL import Image


import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision import transforms

from AttGAN.data import check_attribute_conflict



from data import CelebA
import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models


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


# init attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return pgd_attack

if __name__ == "__main__":
    args_attack = parse()
    print(args_attack)
    os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
    print("experiment dir is created")
    os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
    print("experiment config is saved")


    pgd_attack = init_Attack(args_attack)

    # load the trained CMUA-Watermark
    if args_attack.global_settings.universal_perturbation_path:
        pgd_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)

    # Init the attacked models
    attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F_, T, G, E, reference, gen_models = prepare()
    print("finished init the attacked models")

    tf = transforms.Compose([
            # transforms.CenterCrop(170),
            transforms.Resize(args_attack.global_settings.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    image = Image.open(sys.argv[1])
    img = image.convert("RGB")
    img = tf(img).unsqueeze(0)

    # AttGAN inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        img_a = img.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)   
        att_b_list = [att_a]
        for i in range(attgan_args.n_attrs):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
            att_b_list.append(tmp)
        samples = [img_a, img_a+pgd_attack.up]
        noattack_list = []
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
            with torch.no_grad():
                gen = attgan.G(img_a+pgd_attack.up, att_b_)
                gen_noattack = attgan.G(img_a, att_b_)
            samples.append(gen)
            noattack_list.append(gen_noattack)
            l1_error += F.l1_loss(gen, gen_noattack)
            l2_error += F.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if F.mse_loss(gen, gen_noattack) > 0.05:
                n_dist += 1
            n_samples += 1
        
        ############# 保存图片做指标评测 #############
        # 保存原图
        out_file = './demo_results/AttGAN_original.jpg'
        vutils.save_image(
            img_a.cpu(), out_file,
            nrow=1, normalize=True, range=(-1., 1.)
        )
        for j in range(len(samples)-2):
            # 保存对抗样本生成的图片
            out_file = './demo_results/AttGAN_advgen_{}.jpg'.format(j)
            vutils.save_image(samples[j+2], out_file, nrow=1, normalize=True, range=(-1., 1.))
            # 保存原图生成的图片
            out_file = './demo_results/AttGAN_gen_{}.jpg'.format(j)
            vutils.save_image(noattack_list[j], out_file, nrow=1, normalize=True, range=(-1., 1.))
        
        break
        
    print('AttGAN {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    # stargan inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        img_a = img.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        x_noattack_list, x_fake_list = solver.test_universal_model_level(idx, img_a, c_org, pgd_attack.up, args_attack.stargan)
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
        out_file = './demo_results/stargan_original.jpg'
        vutils.save_image(img_a.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.))
        for j in range(len(x_fake_list)):
            # 保存原图生成图片
            gen_noattack = x_noattack_list[j]
            out_file = './demo_results/stargan_gen_{}.jpg'.format(j)
            vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, range=(-1., 1.))
            # 保存对抗样本生成图片
            gen = x_fake_list[j]
            out_file = './demo_results/stargan_advgen_{}.jpg'.format(j)
            vutils.save_image(gen, out_file, nrow=1, normalize=True, range=(-1., 1.))
        break
    print('stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    # AttentionGAN inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        img_a = img.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        x_noattack_list, x_fake_list = attentiongan_solver.test_universal_model_level(idx, img_a, c_org, pgd_attack.up, args_attack.AttentionGAN)
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
        out_file = './demo_results/attentiongan_original.jpg'
        vutils.save_image(img_a.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.))
        for j in range(len(x_fake_list)):
            # 保存原图生成图片
            gen_noattack = x_noattack_list[j]
            out_file = './demo_results/attentiongan_gen_{}.jpg'.format(j)
            vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, range=(-1., 1.))
            # 保存对抗样本生成图片
            gen = x_fake_list[j]
            out_file = './demo_results/attentiongan_advgen_{}.jpg'.format(j)
            vutils.save_image(gen, out_file, nrow=1, normalize=True, range=(-1., 1.))
        break
    print('attentiongan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        img_a = img.cuda() if args_attack.global_settings.gpu else img_a
        
        with torch.no_grad():
            # clean
            c = E(img_a)
            c_trg = c
            s_trg = F_(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen_noattack = G(c_trg)
            # adv
            c = E(img_a + pgd_attack.up)
            c_trg = c
            s_trg = F_(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen = G(c_trg)
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1

            ############# 保存图片做指标评测 #############
            # 保存原图
            out_file = './demo_results/HiSD_original.jpg'
            vutils.save_image(img_a.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.))
            
            out_file = './demo_results/HiSD_gen.jpg'
            vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, range=(-1., 1.))
            # 保存对抗样本生成图片
            gen = x_fake_list[j]
            out_file = './demo_results/HiSD_advgen.jpg'
            vutils.save_image(gen, out_file, nrow=1, normalize=True, range=(-1., 1.))
        break
    print('HiDF {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

