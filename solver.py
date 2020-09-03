import torch
import numpy as np
import sys
import os
import time
from datetime import datetime, timedelta
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model import Generator, Discriminator
from data_utils import get_data_loader
from data_utils import PickleDataset
from utils import *
from functools import reduce
from collections import defaultdict

class Solver(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)

        # args store other information
        self.args = args
        print(self.args)

        # logger to use tensorboard
        self.logger = Logger(self.args.logdir)

        # get dataloader
        self.get_data_loaders()

        # init the model with config
        self.build_model()
        self.save_config()

        if args.load_model:
            self.load_model()

    def save_model(self, iteration):
        # save model and discriminator and their optimizer
        torch.save(self.G.state_dict(), os.path.join(self.args.store_model_path, f'G_{iteration}.ckpt'))
        torch.save(self.D.state_dict(), os.path.join(self.args.store_model_path, f'D_{iteration}.ckpt'))

    def save_config(self):
        with open(f'{self.args.store_model_path}' + '/config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}' + '/args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def load_model(self):
        print(f'Load model from {self.args.load_model_path}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}.ckpt'))
        self.opt.load_state_dict(torch.load(f'{self.args.load_model_path}.opt'))
        return

    def get_data_loaders(self):
        data_dir = self.args.data_dir
        self.train_dataset = PickleDataset(os.path.join(data_dir, f'{self.args.train_set}.pkl'), 
                os.path.join(data_dir, self.args.train_index_file), 
                segment_size=self.config['data_loader']['segment_size'])
        self.train_loader = get_data_loader(self.train_dataset,
                frame_size=self.config['data_loader']['frame_size'],
                batch_size=self.config['data_loader']['batch_size'], 
                shuffle=self.config['data_loader']['shuffle'], 
                num_workers=4, drop_last=False)
        self.train_iter = infinite_iter(self.train_loader)
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        # self.model = cc(AE(self.config))
        # print(self.config)
        self.G = cc(Generator(self.config))
        self.D = cc(Discriminator(self.config))
        self.G.load_base_generator()
        print(self.G)
        print(self.D)
        optimizer = self.config['optimizer']
        self.opt_G = torch.optim.Adam(self.G.parameters(),
                lr=optimizer['lr'], betas=(optimizer['beta1'], optimizer['beta2']), 
                amsgrad=optimizer['amsgrad'], weight_decay=optimizer['weight_decay'])

        self.opt_D = torch.optim.Adam(self.D.parameters(),
                lr=optimizer['lr'], betas=(optimizer['beta1'], optimizer['beta2']),
                amsgrad=optimizer['amsgrad'], weight_decay=optimizer['weight_decay'])
        print(self.opt_G)
        return

    def ae_step(self, data, lambda_kl):
        x = cc(data)
        mu, log_sigma, emb, dec = self.model(x)
        criterion = nn.L1Loss()
        loss_rec = criterion(dec, x)
        loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
        loss = self.config['lambda']['lambda_rec'] * loss_rec + \
                lambda_kl * loss_kl
        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                max_norm=self.config['optimizer']['grad_norm'])
        self.opt.step()
        meta = {'loss_rec': loss_rec.item(),
                'loss_kl': loss_kl.item(),
                'grad_norm': grad_norm}
        return meta

    def train(self, n_iterations):
        print('Start training......')
        start_time = datetime.now()
        loss = {}
        for iteration in range(n_iterations):

            # Prepare data
            x_real = next(self.train_iter)
            x_real = cc(x_real)
            rand_idx = torch.randperm(x_real.size(0))
            x_trg = x_real[rand_idx]

            # Train discriminator
            for _ in range(5):
              self.reset_grad()
              x_fake = self.G(x_real, x_trg)

              # Compute loss
              out_r = self.D(x_real)
              out_f = self.D(x_fake)
              # print(out_r[:5])
              # print(out_f[:5])

              d_loss_t = F.binary_cross_entropy_with_logits(input=out_f, target=torch.zeros_like(out_r, dtype=torch.float)) + \
                        F.binary_cross_entropy_with_logits(input=out_r, target=torch.ones_like(out_r, dtype=torch.float))

              # Compute loss for gradient penalty.
              alpha = cc(torch.rand(x_real.size(0), 1, 1))
              x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
              out_src = self.D(x_hat)
              d_loss_gp = self.gradient_penalty(out_src, x_hat)
              
              # print(d_loss_t, d_loss_gp)
              d_loss = d_loss_t

              self.reset_grad()
              d_loss.backward()
              grad_norm = torch.nn.utils.clip_grad_norm_(self.D.parameters(),
                      max_norm=self.config['optimizer']['grad_norm'])
              self.opt_D.step()

              # loss['D/d_loss_t'] = d_loss_t.item()
              # loss['D/loss_cls'] = d_loss_cls.item()
              # loss['D/D_gp'] = d_loss_gp.item()
              loss['D/D_loss'] = d_loss.item()

            if iteration >= 0:
                # Train generator
                for _ in range(self.config['n_critics']):

                    # Original-to-target domain.
                    x_fake = self.G(x_real, x_trg)
                    g_out_src = self.D(x_fake)
                    g_loss_fake = F.binary_cross_entropy_with_logits(input=g_out_src,
                                                                     target=torch.ones_like(g_out_src, dtype=torch.float))
                    # Target-to-original domain. (Cycle Loss)
                    x_reconst = self.G(x_fake, x_real)
                    g_loss_rec = F.l1_loss(x_reconst, x_real)

                    # Original-to-Original domain. (Identity Loss)
                    x_fake_iden = self.G(x_real, x_real)
                    id_loss = F.l1_loss(x_fake_iden, x_real)

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.config['lambda']['lambda_cycle'] * g_loss_rec + \
                             self.config['lambda']['lambda_id'] * id_loss

                    self.reset_grad()
                    g_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(),
                            max_norm=self.config['optimizer']['grad_norm'])
                    self.opt_G.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_id'] = id_loss.item()
                    loss['G/g_loss'] = g_loss.item()

                """
                if iteration >= self.config['annealing_iters']:
                    lambda_kl = self.config['lambda']['lambda_kl']
                else:
                    lambda_kl = self.config['lambda']['lambda_kl'] * (iteration + 1) / self.config['annealing_iters'] 
                data = next(self.train_iter)
                meta = self.ae_step(data, lambda_kl)
                """

            # Training Info
            if (iteration + 1) % self.args.log_step == 0:
                et = datetime.now() - start_time
                et = str(et)[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, iteration + 1, n_iterations)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            """
            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/ae_train', meta, iteration)
            loss_rec = meta['loss_rec']
            loss_kl = meta['loss_kl']

            print(f'AE:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.2f}, '
                    f'loss_kl={loss_kl:.2f}, lambda={lambda_kl:.1e}     ', end='\r')
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()
            """

            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()

        return

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = cc(torch.ones(y.size()))
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.opt_G.zero_grad()
        self.opt_G.zero_grad()
