import os.path as osp
import torch
import pytorch_warmup as warmup
from tqdm import tqdm
import os

# tools:
from tools.timer import Timer
from utils.logger import get_logger, get_work_dir
from utils.model_saving import save_model
from termcolor import cprint

# config file:
from config.culane import env, workflow
from config.culane import epochs as max_epochs, totol_steps

# build the instance:
from models.model_builder import net
from dataset.dataloader_builder import train_loader, test_loader
from optimizer.optim import get_optimizer
from optimizer.scheduler import get_scheduler
from evaluator.evaluator_builder import culane_eval
from loss.trainer_builder import trainer, pretrainer


class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pid = -1       # pid buffer
        
        # get the work_dir and the logger, the dir will be automatically created
        self.work_dir = get_work_dir(cfg)    # cfg.work_dir: 'work_dir/culane' + '/{{now()}}'
        self.log_path = osp.join(self.work_dir, cfg.exp_name+'.log')
        self.logger = get_logger(cfg.log_name, self.log_path)       # init the log

        # get the model:
        self.net = net
        
        # single GPU or muti-GPUs
        if cfg.parallel == True:
            self.net = torch.nn.parallel.DataParallel(self.net,device_ids=[2])
        # load state dict or not 
        if hasattr(cfg, 'resume_from') and cfg.resume_from is not None and os.path.exists(cfg.resume_from):
            self.resume()
        # to cuda:
        self.net = self.net.cuda()
        
        # get the DataLoader:
        self.train_loader = train_loader
        self.test_loader = test_loader

        # get the optimizer:
        self.optim = get_optimizer(self.net)
        self.scheduler = get_scheduler(self.optim)
        self.warmup_scheduler = warmup.LinearWarmup(self.optim, warmup_period=5000)

        # get the culane evaluator:
        self.evaluator = culane_eval
        self.evaluator.out_dir = os.path.join(self.work_dir, 'lines')

        # set the metric to 0:
        self.metric = 0.

        # get the state(pretraining, train, val, ...):
        self.workflow = workflow
        self.state = workflow[0][0]
        
        # get the trainer with loss function on cuda:
        # self.trainer = trainer
        self.trainer = trainer
        self.pretrainer = pretrainer
    
    def before_run(self):
        
        # get the pid:
        pid = os.getpid()
        self.pid = pid
        self.logger.info("program pid: " + str(pid))
        cprint("program pid: " + str(pid), on_color="on_red")

        # record the config:
        self.logger.info("config: " + str(self.cfg))
        cprint("config: " + str(self.cfg), color="green")

        # record the work dir:
        self.logger.info("work_dir: " + self.work_dir)
        cprint("work_dir: " + self.work_dir, color="green")
        
        # record the evaluator out dir:
        self.logger.info("evaluato out_dir: " + self.evaluator.out_dir)
        cprint("evaluato out_dir: " + self.evaluator.out_dir, color="green")     
        
        # record the model structure: 
        self.logger.info("model structure: \n" + self.net.__str__())
        cprint("model structure:\n" + self.net.__str__(), color="yellow")
        
        # record the optimizer:
        self.logger.info("model optimizer: \n" + self.optim.__str__())
        cprint("model optimizer: \n" + self.optim.__str__(), color="blue")

        # record the loss fuction:
        self.logger.info("loss function: \n" + str(self.trainer))
        cprint("loss function: \n" + str(self.trainer), color="grey")
        
        # print the log path on screen:
        cprint("log path: " + self.log_path, on_color="on_red",attrs=['bold'])
        
        # start:
        cprint("\n\nstart running ...\n", color="green")
        self.logger.info("\nrunning workflow:\n" + str(self.workflow))
        cprint("\nrunning workflow:\n" + str(self.workflow), color="green")
    
    def get_dataloader(self):
        if self.state != 'val':
            # mode in  train, pretraining, finetuning, ...:
            return self.train_loader
        else:
            # mode in val:
            return self.test_loader
    
    def run(self):
        # to record the config
        self.before_run()
        curr_epoch = 0   
        for flow in self.workflow:
            state, epochs = flow
            self.set_state(state)                    # set the model state
            epoch_runner = getattr(self, state)      # self.train or self.val
            for _ in range(epochs):                 
                epoch_runner(self.get_dataloader(), curr_epoch=curr_epoch)
                if state != 'val': curr_epoch += 1

    # train for a epoch
    @Timer(log_name=env['log_name'])
    def train(self, data_loader, **kwargs):
        # print epoch: 
        curr_epoch = kwargs['curr_epoch']
        cprint("current epoch %d/%d" % (curr_epoch, max_epochs), on_color="on_red")
        
        # for batch_idx, batch_data in enumerate(tqdm(data_loader, ascii=False)):
        for batch_idx, batch_data in enumerate(data_loader):
            # loss_dict: {'exist_loss': l1, 'seg_loss': l2, 'loss': loss}
            img, label, exist = batch_data
            img, label, exist = img.cuda(), label.cuda(), exist.cuda()
            loss_dict = self.trainer.forward(self.net, img, label, exist)   # forward
            # backward:
            self.after_train_iter(loss_dict=loss_dict, curr_epoch=curr_epoch, steps=batch_idx)
        
        # save the model:
        if curr_epoch == 5:
            save_model(self.net, self.optim, self.scheduler, self.work_dir, curr_epoch)

    # validate
    @Timer(log_name=env['log_name'])
    def val(self, data_loader, **kwargs):
        with torch.no_grad():
            # batch_data: (img.cuda(), path) path like: /driver_100_30frame/05251517_0433.MP4/00000.jpg
            # for _, batch_data in enumerate(tqdm(data_loader, ascii=False)):
            for _, batch_data in enumerate(data_loader):
                img, path = batch_data
                img = img.cuda()
                output = self.net(img)    # output: {'seg', 'exist'}
                self.evaluator.evaluate(output=output, batch_path=path
                , with_heat_map=self.cfg.with_heat_map)
                
        self.after_val(curr_epoch=kwargs['curr_epoch'])
    
    def after_val(self, curr_epoch):
        metric = self.evaluator.summarize()
        if not metric:
            return
        self.logger.info("metric: " + str(metric))
        if metric >= self.metric:
            self.metric = metric
            save_model(self.net, self.optim, self.scheduler, self.work_dir, curr_epoch)
        return 

    def after_train_iter(self, loss_dict, **kwargs):
        curr_epoch = kwargs['curr_epoch']
        steps = kwargs['steps']
        self.optim.zero_grad()
        loss_dict['loss'].backward()
        self.optim.step()
        self.scheduler.step()
        self.warmup_scheduler.dampen()
        if steps % 500 == 0:
            if 'heat_map_loss' in loss_dict:
                # pretraining:
                self.logger.info(
                    "pid %d epoch: %d/%d\tsteps:%d/%d \tseg_loss: %.3f\theat_map_loss: %.3f" \
                    % (self.pid, curr_epoch+1, max_epochs, steps, totol_steps
                    ,  loss_dict['seg_loss'], loss_dict['heat_map_loss']))
            else:
                self.logger.info(
                    "pid: %d epoch: %d/%d\tsteps:%d/%d \tseg: %.3f\texist_loss: %.3f" \
                    % (self.pid, curr_epoch+1, max_epochs, steps, totol_steps
                    , loss_dict['seg_loss'], loss_dict['exist_loss']))
        return 
        
    def resume(self):
        model_path = self.cfg.resume_from
        self.net.load_state_dict(torch.load(model_path)['net'], strict=True)
        self.logger.info("load model from path: " + model_path)
        cprint("load model from path: " + model_path, on_color="on_red")
        return 
    
    def fintuning(self):
        model_path  ='/disk/zhangyunzhi/py/resa/work_dirs/CULane/20220429NO-RESA/ckpt/11.pth'
        self.net.load_state_dict(torch.load(model_path)['net'], strict=True)
        print("load from model path: " + model_path)
        runner  = getattr(self, 'val')
        runner(self.data_loaders[1], curr_epoch=5)

    def set_state(self, state):
        assert state in ['train', 'val', 'finetuning', 'pretraining']
        if self.cfg.parallel == True:
            self.net.module.set_state(state)
        else:
            self.net.set_state(state)
        self.state = state
        self.logger.info("curr workflow: " + self.state)
        if state=='val':
            self.net.eval()
            cprint("start validating ...", on_color="on_red")    
        else:
            self.net.train()

    # train for a epoch
    @Timer(log_name=env['log_name'])
    def pretraining(self, data_loader, **kwargs):
        # print epoch: 
        curr_epoch = kwargs['curr_epoch']
        cprint("current epoch %d/%d" % (curr_epoch, max_epochs), on_color="on_red")
        
        # for batch_idx, batch_data in enumerate(tqdm(data_loader, ascii=False)):
        for batch_idx, batch_data in enumerate(data_loader):
            # loss_dict: {'exist_loss': l1, 'pool_loss':l2, 'loss': loss}
            img, label, exist = batch_data
            img, label, exist = img.cuda(), label.cuda(), exist.cuda()
            loss_dict = self.pretrainer.forward(self.net, img, label, exist)   # forward
            # backward:
            self.after_train_iter(loss_dict=loss_dict, curr_epoch=curr_epoch, steps=batch_idx)
        
        # save the model:
        if curr_epoch == 5:
            save_model(self.net, self.optim, self.scheduler, self.work_dir, curr_epoch)
    