import argparse
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm


import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F


from data import CelebA
import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models

from AttGAN.data import check_attribute_conflict

import nni


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




# Init the attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return pgd_attack


def search():
    args_attack = parse()
    tuner_params = nni.get_next_parameter()
    args_attack.attacks.star_factor = float(tuner_params['star_factor'])
    args_attack.attacks.aggan_factor = float(tuner_params['aggan_factor'])
    args_attack.attacks.att_factor = float(tuner_params['att_factor'])
    args_attack.attacks.HiSD_factor = float(tuner_params['HiSD_factor'])
    print(args_attack)
    os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
    print("experiment dir is created")
    os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
    print("experiment config is saved")

    # Init the attacker
    pgd_attack = init_Attack(args_attack)
    # Init the attacked models
    attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
    print("finished init the attacked models, only attack 2 epochs")

    # attacking models
    for i in range(1):
        for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
            if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
                break
            img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
            att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
            att_a = att_a.type(torch.float)

            # attack stargan
            solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)

            # attack attentiongan
            attentiongan_solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)

            # attack AttGAN
            att_b_list = [att_a]
            for i in range(attgan_args.n_attrs):
                tmp = att_a.clone()
                tmp[:, i] = 1 - tmp[:, i]
                tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
                att_b_list.append(tmp)

            for i, att_b in enumerate(att_b_list):
                att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
                if i > 0:
                    att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
                with torch.no_grad():
                    gen_noattack = attgan.G(img_a, att_b_)
                x_adv, perturb = pgd_attack.universal_perturb_attgan(img_a, att_b_, gen_noattack, attgan)

            # attack HiDF
            with torch.no_grad():
                c = E(img_a)
                c_trg = c
                s_trg = F(reference, 1)
                c_trg = T(c_trg, s_trg, 1)
                x_trg = G(c_trg)
                mask = abs(x_trg - img_a)
                mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
                mask[mask>0.5] = 1
                mask[mask<0.5] = 0
                vutils.save_image(((x_trg + 1)/ 2).data, './adv_output_ori.jpg', padding=0)
            pgd_attack.universal_perturb_HiSD(img_a.cuda(), transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)
            torch.save(pgd_attack.up, args_attack.global_settings.universal_perturbation_path)
            print('save the CMUA-Watermark')


    print('The size of CMUA-Watermark: ', pgd_attack.up.shape)
    HiDF_prop_dist, stargan_prop_dist, attgan_prop_dist, aggan_prop_dist = evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, pgd_attack)
    nni.report_intermediate_result(HiDF_prop_dist)
    nni.report_intermediate_result(stargan_prop_dist)
    nni.report_intermediate_result(attgan_prop_dist)
    nni.report_intermediate_result(aggan_prop_dist)
    nni.report_final_result(float((stargan_prop_dist+aggan_prop_dist+attgan_prop_dist+HiDF_prop_dist)/4))

if __name__=="__main__":
    search()