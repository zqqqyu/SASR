import os
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util
import math
from data import df2k, benchmark
from torch.utils.data import Dataset,DataLoader
from importlib import import_module
import imageio
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

class Trainer():
    def __init__(self, args, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        #self.loader_train = loader.loader_train
        #self.loader_test = loader.loader_test
        self.model = my_model
        self.model_E = torch.nn.DataParallel(self.model.get_model().E, range(self.args.n_GPUs))
        self.loss = my_loss
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        #self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=500)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        train_dataset = df2k.DF2K(self.args)
        self.loader_train = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.n_threads, pin_memory=False)

        if args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100','DIV2K','REALSR']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test, train=False)
        else:
            module_test = import_module('data.' + args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        #test_dataset = benchmark.Benchmark(args, name=args.data_test, train=False)
        self.loader_test = DataLoader(dataset=testset, batch_size=1, shuffle=False)

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        #epoch = 200
        # lr stepwise
        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            #if epoch ==101:
           #     lr = 1e-4
           #     for param_group in self.optimizer.param_groups:
           #         param_group['lr'] = lr
           # lr = self.scheduler.get_last_lr()

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        #self.ckp.write_log('[Epoch {}]\tLearning rate: {}'.format(epoch, lr))
        self.loss.start_log()
        self.model.train()

        degrade = util.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

        timer = utility.timer()
        losses_contrast, losses_sr = utility.AverageMeter(), utility.AverageMeter()

        for batch, (hr, _) in enumerate(self.loader_train):
            hr = hr.cuda()                              # b, n, c, h, w
            lr, b_kernels = degrade(hr)                 # bn, c, h, w

            self.optimizer.zero_grad()

            timer.tic()
            # forward
            ## train degradation encoder
            if epoch <= self.args.epochs_encoder:
                _, output, target, fead = self.model_E(im_q=lr[:,0,...], im_k=lr[:,1,...])
               # _, output, target = self.model_E(im_q=lr[:,0,...], im_k=lr[:,1,...])
                loss_constrast = self.contrast_loss(output, target)
                loss = loss_constrast

                losses_contrast.update(loss_constrast.item())
                
                
            ## train the whole network
            else:
              #  for k,v in self.model.named_parameters():
              #      if 'E1' in k:
               #           v.requires_grad = False
               #     if 'E2' in k:
               #           v.requires_grad = False
                #    if 'E3' in k:
                 #         v.requires_grad = False
            
                #for k,v in self.model.named_parameters():
                #    print('{}: {}'.format(k, v.requires_grad))
                sr, output, target = self.model(lr)
                loss_SR = self.loss(sr, hr[:,0,...])
                loss_constrast = self.contrast_loss(output, target)
                loss = loss_constrast + loss_SR
                #loss = loss_SR

                losses_sr.update(loss_SR.item())
                losses_contrast.update(loss_constrast.item())

            # backward
            loss.backward()
            self.optimizer.step()
            timer.hold()

            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_contrast.avg,
                            timer.release()
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [SR loss:{:.3f} | contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_sr.avg, losses_contrast.avg,
                            timer.release(),
                        ))

        self.loss.end_log(len(self.loader_train))

        # save model
       # target = self.model.get_model()
       # model_dict = target.state_dict()
       # keys = list(model_dict.keys())
       # for key in keys:
       #     if 'E.encoder_k' in key or 'queue' in key:
       #         del model_dict[key]
        #if epoch % 50 == 0:
        if epoch % 10 == 0:
        
          target = self.model.get_model()
          model_dict = target.state_dict()
          keys = list(model_dict.keys())
          for key in keys:
              if 'E.encoder_k' in key or 'queue' in key:
                  del model_dict[key]
          torch.save(
              model_dict,
              os.path.join(self.ckp.dir, 'model_m_1.1_v_0.01', 'model_{}.pt'.format(epoch))
          ) 

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                degrade = util.SRMDPreprocessing(
                    self.scale[0],
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=self.args.sig,
                    lambda_1=self.args.lambda_1,
                    lambda_2=self.args.lambda_2,
                    theta=self.args.theta,
                    noise=self.args.noise
                )

                for idx_img, (hr, filename) in enumerate(self.loader_test):
                    hr = hr.cuda()                      # b, 1, c, h, w
                    hr = self.crop_border(hr, scale)
                    lr, _ = degrade(hr, random=False)   # b, 1, c, h, w
                    #print(filename)
                    #lr = imageio.imread('/home/zhangqianyu/SASR/benchmark/DIV2K/LR/' + filename[0] + '.png')
                    #lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
                    #lr = torch.from_numpy(lr).float().cuda().unsqueeze(0).unsqueeze(0)
                    #lr = self.crop_border(lr, scale)
                    
                    hr = hr[:, 0, ...]                  # b, c, h, w
                    #print(hr.shape,lr.shape)
                    # inference
                    timer_test.tic()
                    sr = self.model(lr[:, 0, ...])
                   # b, c, h, w = sr.size()
                   # hr = hr[:, :, :int(h), :int(w)]
                    
                    timer_test.hold()

                    sr = utility.quantize(sr, self.args.rgb_range)
                    hr = utility.quantize(hr, self.args.rgb_range)

                    # metrics
                    eval_psnr += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_ssim += utility.calc_ssim(
                        sr, hr, scale,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    
                    self.args.save_results =False 
                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        #save_list = [lr[:, 0, ...]]
                        filename = filename[0]
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                        '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))


    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()
        img_hr = img_hr[:, :, :, :int(h//16*16), :int(w//16*16)]
       # b, n, c, h, w = img_hr.size()
        #img_hr = img_hr[:, :, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr

