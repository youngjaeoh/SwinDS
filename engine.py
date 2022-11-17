import os
from collections import OrderedDict

import torch
import torch.nn as nn
import wandb
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam
from torch.optim import lr_scheduler

from models.swinir import SwinIR as net


class engine():
    def __init__(self, opt):
        super(engine, self).__init__()

        self.opt = opt  # opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']  # training or not
        self.schedulers = []  # schedulers
        self.opt_train = self.opt['train']  # training option
        self.netG = self.define_network(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = self.define_network(opt).to(self.device).eval()

    def define_network(self, opt):
        opt_net = opt['netG']
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])

        return netG

    def model_to_device(self, network):
        """
        Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            network (nn.Module)
        """
        network = network.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', True)
            use_static_graph = self.opt.get('use_static_graph', False)
            network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()],
                                              find_unused_parameters=find_unused_parameters)
            if use_static_graph:
                print('Using static graph. Make sure that "unused parameters" will not change during training loop.')
                network._set_static_graph()
        else:
            network = DataParallel(network)
        return network

    def init_train(self):
        self.load()  # load models
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']

        if load_path_G is not None:
            print('Loading models for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading models for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'],
                                  param_key='params_ema')
            else:
                print('Copying models for E ...')
                self.update_E(0)
            self.netE.eval()

    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old), (key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer_(load_path_optimizerG, self.G_optimizer)

    def load_optimizer_(self, load_path, optimizer):
        optimizer.load_state_dict(
            torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    def define_scheduler(self):

        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)

        self.G_optimizer.step()

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        ########### wandb loss
        wandb.log({"loss": G_loss.item()})

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def netG_forward(self):
        self.E = self.netG(self.L)

    def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1 - decay)

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def current_log(self):
        return self.log_dict

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict
