import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm
import wandb

class SignLang_trainer:
    def __init__(self, config, model):
        self.task = config.task
        self.train_id = config.train_id
        self.EPOCHS = config.epochs
        self.lr = config.lr
        self.base_path = config.base_path
        self.use_wandb = config.use_wandb
        self.save_freq = config.save_freq
        self.resume = config.resume
        self.device = config.device
        self.batch_size = config.batch_size
        self.num_gloss = config.num_gloss

        self.model = model

        self.logger = defaultdict(list)
    def save_model(self, file_name, epoch):
        ckpt = {
            'epoch':epoch,
            'model':self.model.state_dict(),
            'optimizer':self.optim.state_dict()
                }
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(ckpt, file_name)
        print(f'save checkpoint to {file_name}')

    def load_model(self, file_name, use_optim=True):
        ckpt = torch.load(file_name)
        self.model.load_state_dict(ckpt['model'])
        if use_optim:
            self.optim.load_state_dict(ckpt['optimizer'])
        print(f'load model from {file_name}')
        return ckpt['epoch']

    def train(self,train_loader, val_loader=None):

        ##### init Training #####
        self.optim = optim.Adam(self.model.parameters(),
                                lr=self.lr,
                                betas=(0.9,0.999))
        start_epoch = 1
        
        if self.resume is None:
            self.model.init_weights()
        else:
            start_epoch = self.load_model(self.resume)
                
        self.model.to(self.device)
        ##### Training loop #####
        for e in range(start_epoch, self.EPOCHS+1):
            ### Training ###
            losses, train_metric = self.train_one_epoch(train_loader)

            # log training result
            self.logger['epoch'].append(e)
            self.logger['learning_rate'].append(self.optim.state_dict()['param_groups'][0]['lr'])
            for k, v in losses.items():
                self.logger['train_'+k].append(v.item())
            for k, v in train_metric.items():
                self.logger['train_'+k].append(v.item())

            ### Validation ###
            if val_loader is not None:
                val_loss, val_metric = self.test(val_loader)
                # log validation result
                for k, v in val_loss.items():
                    self.logger['val_'+k].append(v)
                for k, v in val_metric.items():
                    self.logger['val_'+k].append(v)
            
            print(f'Epoch: {e:04d}', end='')
            for k,v in self.logger.items():
                vv = v[-1]
                if type(vv)==int and k!='epoch':
                    print(f' | {k}:{vv:03d}', end='')
                else:
                    print(f' | {k}:{vv:00.4f}', end='')
            print()

            wandb.log({k+'_':float(v[-1]) for i, (k,v) in enumerate(self.logger.items()) if i>1})
            # save model
            if e%self.save_freq == 0:
                file_name = os.path.join(self.base_path, 'checkpoint',self.task,str(self.train_id),f'epoch_{e}.pt')
                self.save_model(file_name, e)
            file_name = os.path.join(self.base_path, 'checkpoint',self.task,str(self.train_id),f'epoch_last.pt')
            self.save_model(file_name, e)

        return self.model, self.logger
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        losses = defaultdict(list)
        metrics = defaultdict(float)
        for iter, (pose_batch, gloss_batch) in enumerate(tqdm(train_loader),1):
            pose_batch = pose_batch[:,:,:,:2].reshape(-1,pose_batch.shape[1],pose_batch.shape[2]*pose_batch.shape[3]).to(self.device)
            gloss_batch = gloss_batch.to(self.device)

            pred_gloss_seq = self.model(pose_batch, gloss_batch)
            loss_CE = F.cross_entropy(pred_gloss_seq.reshape(-1,self.num_gloss), gloss_batch.reshape(-1,self.num_gloss).argmax(1), ignore_index=0)
            
            loss = loss_CE

            # update weights
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # losses['loss_CE'].append(loss_CE.item())
            losses['loss'].append(loss.item())

            # calc metric
            with torch.no_grad():

                gloss_mask = (gloss_batch.argmax(2)>2).sum(0)
                if gloss_mask.min() ==0:
                    gloss_mask[gloss_mask==0]=1
                metrics['acc'] += torch.sum(((pred_gloss_seq.argmax(2) ==gloss_batch.argmax(2))*(gloss_batch.argmax(2)!=0)).sum(0)/(gloss_batch.argmax(2)!=0).sum(0)).item()
                metrics['g_acc'] += torch.sum(((pred_gloss_seq.argmax(2) ==gloss_batch.argmax(2))*(gloss_batch.argmax(2)>2)).sum(0)/gloss_mask).item()
        
        metrics['acc']/=train_loader.dataset.__len__()
        metrics['g_acc']/=train_loader.dataset.__len__()

        for k,v in losses.items():
            losses[k] = np.mean(v)

        for k,v in metrics.items():
            metrics[k] = np.mean(v)

        return losses, metrics
    
    @torch.no_grad()
    def test(self,test_loader, resume=None, verbose=0):
        if resume is not None:
            epoch = self.load_model(resume,use_optim=False)
        self.model.eval()
        losses = defaultdict(list)
        metrics = defaultdict(float)
        for iter, (pose_batch, gloss_batch) in enumerate(tqdm(test_loader),1):
            pose_batch = pose_batch[:,:,:,:2].reshape(-1,pose_batch.shape[1],pose_batch.shape[2]*pose_batch.shape[3]).to(self.device)
            gloss_batch = gloss_batch.to(self.device)

            pred_gloss_seq = self.model(pose_batch, gloss_batch)
            loss_CE = F.cross_entropy(pred_gloss_seq.reshape(-1,self.num_gloss), gloss_batch.reshape(-1,self.num_gloss).argmax(1), ignore_index=0)            
            
            loss = loss_CE

            # losses['loss_CE'].append(loss_CE.item())
            losses['loss'].append(loss.item())

            # calc metric
            gloss_mask = (gloss_batch.argmax(2)>2).sum(0)
            if gloss_mask.min() ==0:
                gloss_mask[gloss_mask==0]=1
            metrics['acc'] += torch.sum(((pred_gloss_seq.argmax(2) ==gloss_batch.argmax(2))*(gloss_batch.argmax(2)!=0)).sum(0)/(gloss_batch.argmax(2)!=0).sum(0)).item()
            metrics['g_acc'] += torch.sum(((pred_gloss_seq.argmax(2) ==gloss_batch.argmax(2))*(gloss_batch.argmax(2)>2)).sum(0)/gloss_mask).item()
        
        metrics['acc']/=train_loader.dataset.__len__()
        metrics['g_acc']/=train_loader.dataset.__len__()
        losses = {k:np.mean(v) for k,v in losses.items()}
        if verbose:
            print(f'Epoch: {epoch:04d}', end='')
            for k,v in losses.items():
                if type(v)==float:
                    print(f' | {k}:{v:00.4f}', end='')
            for k,v in metrics.items():
                if type(v)==float:
                    print(f' | {k}:{v:00.4f}', end='')
            print()

        return losses, metrics

        

if __name__ == '__main__':

    class CONFIG:
        task = 'pose2gloss'
        group = 'seq2seq(LSTM)'
        description = 'seq2seq(LSTM)-small gloss'
        train_id = 1
        epochs = 1000
        lr = 0.002
        base_path = './'
        use_wandb = True
        save_freq = 50
        resume = None
        # resume = '/home/horang1804/Dolmaggu/coda-modeling/Pose2Gloss/checkpoint/pose2gloss/1/epoch_last.pt'
        phase = 'train'
        device = 'cuda:0'
        batch_size = 64
        num_worker = 6
        num_gloss = 427


    from signlang_dataloader import AIHUB_SIGNLANG_DATASET, collate_fn
    from torch.utils.data import DataLoader
    train_dataset = AIHUB_SIGNLANG_DATASET(base_path='/home/horang1804/HDD1/dataset/aihub_sign',
                                           phase='train',
                                           modality=['pose','gloss'],
                                           gloss_type=['SEN','WORD'])
    
    train_loader = DataLoader(train_dataset,CONFIG.batch_size,shuffle=True, collate_fn=collate_fn)
    valid_dataset = AIHUB_SIGNLANG_DATASET(base_path='/home/horang1804/HDD1/dataset/aihub_sign',
                                           phase='valid',
                                           modality=['pose','gloss'],
                                           gloss_type=['SEN','WORD'])
    
    valid_loader = DataLoader(valid_dataset,CONFIG.batch_size,shuffle=True, collate_fn=collate_fn)
    pose_batch, gloss_batch = next(iter(train_loader))

    from p2g_model import Encoder, Decoder, Seq2Seq
    enc = Encoder(input_dim=(25+21+21)*2, 
                  emb_dim=16, 
                  hid_dim=512, 
                  n_layers=2, 
                  dropout=0.5)
    dec = Decoder(output_dim=CONFIG.num_gloss,
                  emb_dim=256,
                  hid_dim=512,
                  n_layers=2,
                  dropout=0.5)
    model = Seq2Seq(enc,dec,CONFIG.device).to(CONFIG.device)

    train_loader = DataLoader(train_dataset,CONFIG.batch_size,shuffle=True,num_workers=CONFIG.num_worker,collate_fn=collate_fn)
    gloss_batch, pose_batch = next(iter(train_loader))



    trainer = SignLang_trainer(CONFIG, model)
    
    if CONFIG.phase =='train':

        ##### init wandb #####
        if CONFIG.use_wandb:
            wandb.init(project=f'Dolmaenggu-{CONFIG.task}',
                       group = CONFIG.group,
                       name = 'hyper param',
                       notes=CONFIG.description,
                       config = {k:v for k,v in CONFIG.__dict__.items() if k[:2]!='__'})
        trainer.train(train_loader,valid_loader)
    else:
        import glob
        ckpt_file = glob.glob(os.path.join('./checkpoint/pose2gloss/1/*'))
        for cf in ckpt_file:
            trainer.test(valid_loader, resume=cf, verbose=1)

    print('end')