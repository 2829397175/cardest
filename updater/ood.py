
import imp

from matplotlib import axis
from colorlog import root

from numpy import insert
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import DEVICE
import common
import copy
import time
import warnings
import flash,transformer,made
import pandas as pd
from datasets import LoadDmv, UpdateDmv,DF_to_CSVtable_Dmv


class OOD_dection():
    def __init__(self) :
        self.old_distribution=None
        self.transfer_set=[]
        
    def offline_sample(self):
        pass
    
    def online_sample(self):
        pass
    
    def detect_sample(self,sample):
        
        pass
    
    def update_model_info(self,
                     model,
                    upto=None,
                    verbose=False,
                    log_every=10,
                    return_losses=False,
                    seed=0
                    ):
        self.model = model
        self.upto = upto 
        self.verbose=verbose
        self.log_every=log_every
        self.return_losses=return_losses
        self.seed=seed

        
    def get_model_opt(self,model=None):
        model = self.model if model ==None else model
        
        if isinstance(model, (transformer.Transformer)):
            opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        else:
            opt = torch.optim.Adam(list(model.parameters()), 2e-4)
            
        if isinstance(model, (flash.FLASHTransformer)):
            opt = torch.optim.AdamW(list(model.parameters()),lr=7e-4)
        return opt
            

    
    def RunEpoch(self,
                split,
                dataset,
                opt=None,
                batch_size=256,
                epoch_num=None,
                table_bits=0,
                model=None,
                reduce_loss=True,
                **kargs
                ):
        f_model = self.model if model is None else model
        torch.set_grad_enabled(split == 'train')
        f_model.train() if split == 'train' else f_model.eval()
        losses = []

        loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=(split == 'train'))

        # How many orderings to run for the same batch?
        nsamples = 1
        if hasattr(f_model, 'orderings'):
            nsamples = len(f_model.orderings)

        for step, xb in enumerate(loader):
            if split == 'train':
                assert opt is not None," the optimizer of training process must be specified"
                base_lr = 8e-4
                for param_group in opt.param_groups:
                    if kargs['constant_lr']:
                        lr = kargs['constant_lr']
                    elif kargs['warmups']: # Transformer, FLASH
                        t = kargs['warmups']
                        d_model = f_model.embed_dim
                        global_steps = len(loader) * epoch_num + step + 1
                        lr = (d_model**-0.5) * min(
                            (global_steps**-.5), global_steps * (t**-1.5))
                    else:
                        lr = 1e-2

                    param_group['lr'] = lr

            if self.upto and step >= self.upto:
                break

            xb = xb.to(DEVICE).to(torch.float32)

            # Forward pass, potentially through several orderings.
            xbhat = None
            model_logits = []
            num_orders_to_forward = 1
            if split == 'test' and nsamples > 1:
                # At test, we want to test the 'true' nll under all orderings.
                num_orders_to_forward = nsamples

            for i in range(num_orders_to_forward):
                if hasattr(f_model, 'update_masks'):
                    # We want to update_masks even for first ever batch.
                    f_model.update_masks()

                model_out = f_model(xb)
                model_logits.append(model_out)
                if xbhat is None:
                    xbhat = torch.zeros_like(model_out)
                xbhat += model_out

            if xbhat.shape == xb.shape:
                # if mean:
                #     xb = (xb * std) + mean
                loss = F.binary_cross_entropy_with_logits(
                    xbhat, xb, size_average=False) / xbhat.size()[0]
                # ??? what's this for
            else:
                if f_model.input_bins is None:
                    # NOTE: we have to view() it in this order due to the mask
                    # construction within MADE.  The masks there on the output unit
                    # determine which unit sees what input vars.
                    xbhat = xbhat.view(-1, f_model.nout // f_model.nin, f_model.nin)
                    # Equivalent to:
                    loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                            .sum(-1)
                            
                            
                else:
                    if num_orders_to_forward == 1:
                        loss = f_model.nll(xbhat, xb)
                    else:
                        # Average across orderings & then across minibatch.
                        #
                        #   p(x) = 1/N sum_i p_i(x)
                        #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                        #             = log(1/N) + logsumexp ( log p_i(x) )
                        #             = log(1/N) + logsumexp ( - nll_i (x) )
                        #
                        # Used only at test time.
                        logps = []  # [batch size, num orders]
                        assert len(model_logits) == num_orders_to_forward, len(
                            model_logits)
                        for logits in model_logits:
                            # Note the minus.
                            logps.append(-f_model.nll(logits, xb))
                        logps = torch.stack(logps, dim=1)
                        logps = logps.logsumexp(dim=1) + torch.log(
                            torch.tensor(1.0 / nsamples, device=logps.device))
                        loss = (-logps)
            if (reduce_loss):
                loss=loss.mean()
                losses.append(loss.item())
            else:
                losses.append(loss.cpu().numpy())

            if step % self.log_every == 0:
                if split == 'train':
                    print(
                        'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                        .format(epoch_num, step, split,
                                loss.item() / np.log(2) - table_bits,
                                loss.item() / np.log(2), table_bits, lr))
                else:
                    if (not reduce_loss):
                        mean_loss = loss.mean().item()
                    else:
                        mean_loss=loss.item()
                    print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                        format(epoch_num, step, split, mean_loss,
                                mean_loss / np.log(2)))

            if split == 'train':
                opt.zero_grad()
                loss.backward()
                opt.step()

            if self.verbose:
                print('%s epoch average loss: %f' % (split, np.mean(losses)))
        
        if (not reduce_loss):
            return losses
        return np.mean(losses)
    

class OOD_DDup_detection(OOD_dection):
    def __init__(self,
                 old_sample_num,
                 old_sample_size,
                 new_sample_size,
                 update_step_threshold,
                 queue_size=200
                 ) :
        super().__init__()
        self.hem_size=queue_size
        self.hem_queue=None
        self.old_sample_num=old_sample_num
        self.old_sample_size=old_sample_size
        self.new_sample_size=new_sample_size
        self.update_step_threshold=update_step_threshold
        self.update_step=0
        
    def offline_sample(self,old_data:common.TableDataset):
        assert old_data is not None
        old_data_size=old_data.size()
        loss_queue=[]
        bs=256 if 256>self.old_sample_size else self.old_sample_size
        for i in range(self.old_sample_num):
            random_ids=np.random.randint(0,old_data_size,size=self.old_sample_size)
            sampled_tuples=old_data[random_ids]
            
            old_loss=self.RunEpoch(
                split="test",
                dataset=sampled_tuples,
                batch_size=bs,
                reduce_loss=False              
                )
            
            assert len(old_loss)==1,"test index error"
            
            sub_loss_queue=np.stack([old_loss[0],random_ids],axis=1)
            
            loss_queue.append(sub_loss_queue)

        loss_queue=np.asarray(loss_queue)        
        loss_queue=np.concatenate(loss_queue,axis=0)
        self.old_distribution={
            "mean":np.mean(loss_queue[:,0]),
            "std":np.std(loss_queue[:,0]),
        }
        

        self.hem_queue=np.array(sorted(loss_queue, key=lambda x: x[0], reverse=True)[:self.hem_size])
    
        
        
            
    def online_sample(self,new_data,old_data=None):
        import scipy.stats as st
        if self.old_distribution is None:
            warnings.warn("doing bootstrap online, this can be slow.")
            self.offline_sample(old_data)
            
        new_data_size=new_data.size()
        random_ids=np.random.randint(0,new_data_size,size=self.new_sample_size)
        sampled_tuples=new_data[random_ids]
        try:
            loss=self.RunEpoch(
            split="test",
            dataset=sampled_tuples,
            batch_size=256,              
            )
        except:
            return True
        if (self.old_sample_num<50):
        
            t_interval=st.t.interval(alpha=0.95,
                df=self.old_sample_num,
                loc=self.old_distribution['mean'], 
                scale=self.old_distribution['std'])
        else:
            t_interval=st.norm.interval(alpha=0.95,
            loc=self.old_distribution['mean'],
            scale=self.old_distribution['std'])
            
        # true: ood, needs upgrade

        return loss>t_interval[1] or loss<t_interval[0]
    
    
    
    

        
    def distribution_shift(self,
                        insert_table:common.CsvTable,
                        old_table:common.CsvTable):

        for idx,insert_col in enumerate(insert_table.columns):
            for value in insert_col.DistinctValue():
                if value not in old_table.columns[idx].DistinctValue():
                    return True
        return False
    
    
    def ReportModel(self,model, blacklist=None):
        ps = []
        for name, p in model.named_parameters():
            if blacklist is None or blacklist not in name:
                ps.append(np.prod(p.size()))
        num_params = sum(ps)
        mb = num_params * 4 / 1024 / 1024
        print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
        print(model)
        return mb
    
    
    
    # def run_ester(self,):
    #     pass # query_train
    
    def distillation(self,
                     train_data,
                     distill_data,
                     new_model,
                     batch_size=256,
                     epochs=10,
                     alpha=0.5,
                     beta=0.5,
                     val_data=None,
                     val_bits=None,
                     save_model=False,
                     save_log=False,
                     **kargs):

        df_train=pd.DataFrame()
        losses=[]
        
        from .early_stopping import EarlyStopping
        import os
        from time import time
        # from pytorchtools import EarlyStopping
        if (save_model or save_log):
            mb = self.ReportModel(self.model)
            seed = self.seed
            root_path= 'models/retrain/'if alpha==1 else 'models/distilled/'
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            PATH = os.path.join(root_path,'/{}-{:.1f}MB-{}-seed{}-alpha{}-beta{}'.format(
                train_data.name, mb, self.model.name(),seed,alpha,beta))
            early_stopping = EarlyStopping(PATH,2)
        opt = self.get_model_opt(new_model)
        alpha = torch.nn.Parameter(torch.tensor(alpha),requires_grad=True)
        beta = torch.nn.Parameter(torch.tensor(beta),requires_grad=True)
        
        val_losses=0
        epoch=0
        
        
        for i in range(epochs):
            epoch+=1
            start_time=time()
            mean_epoch_train_loss, new_model = self.runepoch_distillation(
                                            train_data,
                                            distill_data,
                                            new_model,
                                            opt,
                                            batch_size,
                                            epoch,
                                            alpha=alpha,
                                            beta=beta,
                                            **kargs                                        
                                            )

            val_losses = self.RunEpoch('test',
                    dataset=val_data,
                    return_losses=True,
                    table_bits=val_bits,
                    model=new_model)
            val_nats = np.mean(val_losses)
            val_bits_loss = val_nats / np.log(2)
            val_bits_gap = val_bits_loss-val_bits
            val_losses = np.abs(val_bits_gap) # entropy gap
            
            mean_epoch_train_loss=np.append(mean_epoch_train_loss,val_losses)
            
            losses.append(mean_epoch_train_loss)
            early_stopping(val_losses, new_model,epoch_num=epoch)
            end_time=time()
            time_training=end_time-start_time
            print(f"epoch {i}: training time {time_training}")
            if early_stopping.early_stop:
                print("Early stopping;")
                break #跳出迭代，结束训练
            
        if (save_model or save_log):
            path=early_stopping.save_checkpoint(val_losses,new_model,epoch,save_model)
        
        losses=np.array(losses)
        df_train['loss']=losses[:,0]
        df_train['model_loss']=losses[:,1]
        df_train['d_loss']=losses[:,2]
        df_train['val_loss']=losses[:,3]
        if (save_log):
            if alpha==1:
                df_train.to_csv(os.path.join('distill_log/retrain/',path[15:-3]+".csv"))
            else:
                df_train.to_csv(os.path.join('distill_log/distilled/',path[17:-3]+".csv"))
        self.model = new_model
        return df_train
    
    # def distillation_loss(self,new_model_out,old_model_out):
    #     d_loss=nn.CrossEntropyLoss()
    
    

    
    def runepoch_distillation(self,
                     train_data,
                     distill_data,
                     new_model,
                     opt,
                     batch_size=256,
                     epoch_num=None,
                     alpha=0.5,
                     beta=0.5,
                     **kargs):
        torch.set_grad_enabled(True)
        new_model.cuda()
        new_model.train() 
        self.model.eval()
        
        
        losses = []
        loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size,
                                            shuffle=True)
        
        distill_loader = torch.utils.data.DataLoader(distill_data,
                                    batch_size=batch_size,
                                    shuffle=True)

        # How many orderings to run for the same batch?
        nsamples = 1
        if hasattr(self.model, 'orderings'):
            nsamples = len(self.model.orderings)



        for step, xb in enumerate(loader):

            assert opt is not None," the optimizer of training process must be specified"
            # base_lr = 8e-4
            # for param_group in opt.param_groups:
            #     if kargs['constant_lr']:
            #         lr = kargs['constant_lr']
            #     elif kargs['warmups']: # Transformer, FLASH
            #         t = kargs['warmups']
            #         d_model = self.model.embed_dim
            #         global_steps = len(loader) * epoch_num + step + 1
            #         lr = (d_model**-0.5) * min(
            #             (global_steps**-.5), global_steps * (t**-1.5))
            #     else:
            #         lr = 1e-2

            #     param_group['lr'] = lr

            if self.upto and step >= self.upto:
                break
            
            d_loss=[]
            for step_d,xb_d in enumerate(distill_loader):
                new_xbhat = None
                new_model_logits = []
                num_orders_to_forward = 1
                xb_d = xb_d.to(DEVICE).to(torch.float32)
                
                for i in range(num_orders_to_forward):

                    new_model_dout = new_model(xb_d)
                    old_model_dout=self.model(xb_d)
                    # d_loss = F.cross_entropy(F.softmax(new_model_out,dim=2),F.softmax(old_model_out,dim=2))
                    d_loss_epoch=F.mse_loss(new_model_dout,old_model_dout,reduction='none').mean()
                    new_model_logits.append(new_model_dout)
                    if new_xbhat is None:
                        new_xbhat = torch.zeros_like(new_model_dout)
                    new_xbhat += new_model_dout
                    
                if num_orders_to_forward == 1:
                    model_loss_epoch = new_model.nll(new_xbhat, xb_d).mean()
                
                loss_distillution=beta*d_loss_epoch+model_loss_epoch*(1-beta)
                
                opt.zero_grad()
                loss_distillution.backward()
                opt.step()
                
                d_loss.append(loss_distillution.item())
                
            
            d_loss=np.mean(d_loss)
                
            
            # new_model forward: model loss
            xb = xb.to(DEVICE).to(torch.float32)

            # Forward pass, potentially through several orderings.
            new_xbhat = None
            new_model_logits = []
            num_orders_to_forward = 1

            for i in range(num_orders_to_forward):
                if hasattr(new_model, 'update_masks'):
                    # We want to update_masks even for first ever batch.
                    new_model.update_masks()

                new_model_out = new_model(xb)
                # d_loss = F.cross_entropy(F.softmax(new_model_out,dim=2),F.softmax(old_model_out,dim=2))
                new_model_logits.append(new_model_out)
                if new_xbhat is None:
                    new_xbhat = torch.zeros_like(new_model_out)
                new_xbhat += new_model_out

            if new_xbhat.shape == xb.shape:
                # if mean:
                #     xb = (xb * std) + mean
                model_loss = F.binary_cross_entropy_with_logits(
                    new_xbhat, xb, size_average=False) / new_xbhat.size()[0]
            else:
                if new_model.input_bins is None:
                    # NOTE: we have to view() it in this order due to the mask
                    # construction within MADE.  The masks there on the output unit
                    # determine which unit sees what input vars.
                    new_xbhat = new_xbhat.view(-1, new_model.nout // new_model.nin, new_model.nin)
                    # Equivalent to:
                    model_loss = F.cross_entropy(new_xbhat, xb.long(), reduction='none') \
                            .sum(-1).mean()
                else:
                    if num_orders_to_forward == 1:
                        model_loss = new_model.nll(new_xbhat, xb).mean()
                    else:
                        # Average across orderings & then across minibatch.
                        #
                        #   p(x) = 1/N sum_i p_i(x)
                        #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                        #             = log(1/N) + logsumexp ( log p_i(x) )
                        #             = log(1/N) + logsumexp ( - nll_i (x) )
                        #
                        # Used only at test time.
                        logps = []  # [batch size, num orders]
                        assert len(new_model_logits) == num_orders_to_forward, len(
                            new_model_logits)
                        for logits in new_model_logits:
                            # Note the minus.
                            logps.append(-new_model.nll(logits, xb))
                        logps = torch.stack(logps, dim=1)
                        logps = logps.logsumexp(dim=1) + torch.log(
                            torch.tensor(1.0 / nsamples, device=logps.device))
                        model_loss = (-logps).mean()
                        

                        
            loss = alpha*(model_loss)+(1 - alpha)*d_loss
            losses.append([loss.item(),model_loss.item(),d_loss.item()])

            if step % self.log_every == 0:
                print('Epoch {} Iter {}, train loss {:.3f}, distilll loss {:.3f}, model loss {:.3f}) '
                    .format(epoch_num, step, 
                            loss,d_loss,model_loss))

           
            opt.zero_grad()
            loss.backward()
            opt.step()

            if self.verbose:
                print('train epoch average loss: %f' % np.mean(losses))

        
        losses=np.array(losses)
        return np.mean(losses,axis=0), new_model
            
        
    
    def Entropy(self,name, data, bases=None):
        import scipy.stats
        s = 'Entropy of {}:'.format(name)
        ret = []
        for base in bases:
            assert base == 2 or base == 'e' or base is None
            e = scipy.stats.entropy(data, base=base if base != 'e' else None)
            ret.append(e)
            unit = 'nats' if (base == 'e' or base is None) else 'bits'
            s += ' {:.4f} {}'.format(e, unit)
        print(s)
        return ret
    
    
    def search_param(self,
                       new_table,
                       new_tablebits,
                       train_data,
                       new_data,
                       epochs=10,
                       start_lossw=0.5
                       ):
        nin = len(new_table.columns)
        input_bins = [c.DistributionSize() for c in new_table.columns]
        new_model = copy.deepcopy(self.model)
        # new_model.update_structure(nin,input_bins)
        
        df_distill=self.distillation(train_data,
                            new_model,
                            constant_lr=None,
                            warmups=0,
                            epochs=epochs,
                            table_bits=new_tablebits,
                            start_lossw=start_lossw)
        
        all_losses = self.RunEpoch('test',
                            dataset=new_data,
                            return_losses=True,
                            table_bits=new_tablebits)
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        return model_bits,df_distill
        

    def data_augmentation(self,
                          train_table:common.CsvTable,
                          old_table:common.CsvTable,
                          sample_rate_hem=0.6,
                          sample_rate=0.6
                          ):
        sample_size=int(train_table.data.shape[0] *sample_rate)
        old_data_size=old_table.data.shape[0]
        sample_size = sample_size if sample_size<old_data_size else old_data_size
        
        hem_idx=self.hem_queue[:sample_size,1]
        hem_size=self.hem_queue.shape[0]
        indexs=[int(hem_size*0.05*sample_rate_hem),int(hem_size*0.2*sample_rate_hem),
                int(hem_size*0.5*sample_rate_hem),int(hem_size*0.75*sample_rate_hem)]
        id_start=[0,int(hem_size*0.05),int(hem_size*0.2),
                int(hem_size*0.5)]
        
        hem_idx=np.concatenate((hem_idx[0:indexs[0]],
                                hem_idx[id_start[1]:indexs[1]],
                                hem_idx[id_start[2]:indexs[2]],
                                hem_idx[id_start[3]:indexs[3]])
                               )

        sample_idx=np.random.randint(0,old_data_size,sample_size)
        sample_idx=np.concatenate((hem_idx,sample_idx),axis=0)
        
        df_sampled=old_table.data.iloc[sample_idx,:]
        
        
        augment_table = DF_to_CSVtable_Dmv(df_sampled)
        return augment_table

        
        
    def get_distill_val(self,train_table,old_table,val_size=5000):
        old_data_size=old_table.data.shape[0]
        val_size=old_data_size if val_size>old_data_size else val_size
        randidx=np.random.randint(0,int(old_data_size),int(val_size))

        df_sampled=old_table.data.iloc[randidx,:]
        randidx_train=np.random.randint(0,int(train_table.data.shape[0]),int(val_size))
        df_new_samples=train_table.data.iloc[randidx_train,:]
        
        df_val = pd.concat([df_sampled,df_new_samples])
        df_val = df_val.sample(frac=1.0) 
        
        from datasets import DF_to_CSVtable_Dmv
        return DF_to_CSVtable_Dmv(df_sampled)
    
    
    
    
    
    
    def online_update_db(self,
                       old_table:common.CsvTable,
                        insert_table:common.CsvTable=None,
                       new_table:common.CsvTable=None, 
                       epochs=10,
                       save_model=False,
                       save_log=False,
                       args=None):
    # new_table :replace old table
    # insert table :insert table+old table = new
    # when new table is set,insert table is useless 
        
        if (new_table is not None):
            new_data = common.TableDataset(new_table,old_table)
            train_table = new_table
            train_data = new_data
        elif (insert_table is not None):
            new_table = UpdateDmv(insert_table,old_table)
            new_data = common.TableDataset(new_table,old_table)
            insert_data = common.TableDataset(insert_table,old_table)
            train_table = insert_table
            train_data = insert_data
        else:
            assert False, "no update data set."
        old_data = common.TableDataset(old_table)
        
        ood = self.online_sample(train_data,old_data) 
        
        new_tablebits= self.Entropy(
        new_table,
        new_table.data.fillna(value=0).groupby([c.name for c in new_table.columns
                                        ]).size(), [2])[0]
        
        
        ## data augmentation

        val_table=self.get_distill_val(train_table,old_table)
        val_data=common.TableDataset(val_table,old_table)
        
        
        val_bits=self.Entropy(
            val_table,
            val_table.data.fillna(value=0).groupby([c.name for c in val_table.columns
                                            ]).size(), [2])[0]
        
        print(f"OOD state:{ood}")
        
        if (ood and not args.finetune and not args.retrain): # false for finetune experiment
            new_model = copy.deepcopy(self.model)            
            # new_model.update_structure(nin,input_bins)
            
            ## data augmentation
            sample_rate=5e3/train_table.data.shape[0]
            sample_rate=sample_rate if sample_rate<=1 else 1
            distill_table=self.data_augmentation(train_table,old_table,sample_rate_hem=1.0,sample_rate=sample_rate)
            
            distill_data=common.TableDataset(distill_table,old_table,mixup=True)
            train_data=common.TableDataset(train_table,old_table,mixup=True)  

            val_table=self.get_distill_val(train_table,old_table)
            val_data=common.TableDataset(val_table,old_table)
            
            
            val_bits=self.Entropy(
                val_table,
                val_table.data.fillna(value=0).groupby([c.name for c in val_table.columns
                                                ]).size(), [2])[0]
            
            
            df_distill = self.distillation(train_data,
                                        distill_data,
                                        new_model,
                                        epochs=epochs,
                                        alpha=args.alpha,
                                        beta=0.2,
                                        val_data=val_data,
                                        val_bits=val_bits,
                                        save_model=save_model,
                                        save_log=save_log,
                                        constant_lr=None,
                                        warmups=0,)

            assert self.model == new_model
            
        else:   
            from .early_stopping import EarlyStopping
            import os
            df_distill=pd.DataFrame()
            mb = self.ReportModel(self.model)
            seed = self.seed
        
            root_dir='models/finetune/' if args.finetune else 'models/retrain/'
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            PATH =os.path.join(root_dir,'{}-{:.1f}MB-{}-seed{}.pt'.format(
            train_data.name, mb, self.model.name(),seed))
            early_stopping = EarlyStopping(PATH,2)
            train_losses=[]
            train_start=time.time()
            opt=self.get_model_opt()
            
            # freeze weights:
            if (args.finetune):
                for name, para in self.model.named_parameters():
                    # 除最后的全连接层外，其他权重全部冻结
                    if "norm" not in name:
                        para.requires_grad_(False)

            for epoch in range(epochs):
    
                mean_epoch_train_loss = self.RunEpoch(
                                                'train',
                                                train_data,
                                                opt=opt,
                                                batch_size=1024,
                                                epoch_num=epoch,
                                                constant_lr=None,
                                                warmups=0,
                                                table_bits=new_tablebits,
                                                
                                                )

                if epoch % 1 == 0:
                    print('epoch {} train loss {:.4f} nats / {:.4f} bits'.format(
                        epoch, mean_epoch_train_loss,
                        mean_epoch_train_loss / np.log(2)))
                    since_start = time.time() - train_start
                    print('time since start: {:.1f} secs'.format(since_start))

                train_losses.append(mean_epoch_train_loss)
                val_losses = self.RunEpoch('test',
                    dataset=val_data,
                    return_losses=True,
                    table_bits=val_bits)
                val_nats = np.mean(val_losses)
                val_bits_loss = val_nats / np.log(2)
                val_bits_gap = val_bits_loss-val_bits
                val_losses = np.abs(val_bits_gap) # entropy gap

                early_stopping(val_losses, self.model,epoch_num=epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break #跳出迭代，结束训练
            df_distill['loss']=train_losses
            path=early_stopping.save_checkpoint(val_losses,self.model,epoch,save_model=save_model)
            root_dir='distill_log/finetune/' if args.finetune else 'distill_log/retrain/'
            if (save_log):
                df_distill.to_csv(os.path.join(root_dir,path[16:-3]+".csv"))
            
        # print('Training done; evaluating likelihood on full data:')  
        # all_losses = self.RunEpoch('test',
        #                     dataset=new_data,
        #                     return_losses=True,
        #                     table_bits=new_tablebits)
        # model_nats = np.mean(all_losses)
        # model_bits = model_nats / np.log(2)
        # self.model.model_bits = model_bits
        
        # print("model_bits:{}".format(model_bits),"table_bits:{}".format(new_tablebits))
        return new_table, self.model, ood, df_distill
                
    def get_model(self):
        return self.model
            


        
   