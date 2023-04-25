
import imp

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
            self.opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        else:
            self.opt = torch.optim.Adam(list(model.parameters()), 2e-4)
            
        if isinstance(model, (flash.FLASHTransformer)):
            self.opt = torch.optim.AdamW(list(model.parameters()),lr=7e-4)
            

    
    def RunEpoch(self,
                split,
                dataset,
                batch_size=256,
                epoch_num=None,
                table_bits=0,
                **kargs
                ):
        self.get_model_opt()
        torch.set_grad_enabled(split == 'train')
        self.model.train() if split == 'train' else self.model.eval()
        losses = []

        loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=(split == 'train'))

        # How many orderings to run for the same batch?
        nsamples = 1
        if hasattr(self.model, 'orderings'):
            nsamples = len(self.model.orderings)

        for step, xb in enumerate(loader):
            if split == 'train':
                assert self.opt is not None," the optimizer of training process must be specified"
                base_lr = 8e-4
                for param_group in self.opt.param_groups:
                    if kargs['constant_lr']:
                        lr = kargs['constant_lr']
                    elif kargs['warmups']: # Transformer, FLASH
                        t = kargs['warmups']
                        d_model = self.model.embed_dim
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
                if hasattr(self.model, 'update_masks'):
                    # We want to update_masks even for first ever batch.
                    self.model.update_masks()

                model_out = self.model(xb)
                model_logits.append(model_out)
                if xbhat is None:
                    xbhat = torch.zeros_like(model_out)
                xbhat += model_out

            if xbhat.shape == xb.shape:
                # if mean:
                #     xb = (xb * std) + mean
                loss = F.binary_cross_entropy_with_logits(
                    xbhat, xb, size_average=False) / xbhat.size()[0]
            else:
                if self.model.input_bins is None:
                    # NOTE: we have to view() it in this order due to the mask
                    # construction within MADE.  The masks there on the output unit
                    # determine which unit sees what input vars.
                    xbhat = xbhat.view(-1, self.model.nout // self.model.nin, self.model.nin)
                    # Equivalent to:
                    loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                            .sum(-1).mean()
                else:
                    if num_orders_to_forward == 1:
                        loss = self.model.nll(xbhat, xb).mean()
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
                            logps.append(-self.model.nll(logits, xb))
                        logps = torch.stack(logps, dim=1)
                        logps = logps.logsumexp(dim=1) + torch.log(
                            torch.tensor(1.0 / nsamples, device=logps.device))
                        loss = (-logps).mean()

            losses.append(loss.item())

            if step % self.log_every == 0:
                if split == 'train':
                    print(
                        'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                        .format(epoch_num, step, split,
                                loss.item() / np.log(2) - table_bits,
                                loss.item() / np.log(2), table_bits, lr))
                else:
                    print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                        format(epoch_num, step, split, loss.item(),
                                loss.item() / np.log(2)))

            if split == 'train':
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if self.verbose:
                print('%s epoch average loss: %f' % (split, np.mean(losses)))
        if self.return_losses:
            return losses
        
        
        return np.mean(losses)
    

class OOD_DDup_detection(OOD_dection):
    def __init__(self,
                 old_sample_num,
                 old_sample_size,
                 new_sample_size,
                 update_step_threshold) :
        super().__init__()
        self.old_sample_num=old_sample_num
        self.old_sample_size=old_sample_size
        self.new_sample_size=new_sample_size
        self.update_step_threshold=update_step_threshold
        self.update_step=0
        
    def offline_sample(self,old_data:common.TableDataset):
        assert old_data is not None
        old_data_size=old_data.size()
        old_losses=[]
        for i in range(self.old_sample_num):
            random_ids=np.random.randint(0,old_data_size,size=self.old_sample_size)
            sampled_tuples=old_data[random_ids]
            old_loss=self.RunEpoch(
                split="test",
                dataset=sampled_tuples,
                batch_size=256,              
                )
            old_losses.append(old_loss)

        self.old_distribution={
            "mean":np.mean(old_losses),
            "std":np.std(old_losses)
        }
            
    def online_sample(self,new_data,old_data=None):
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
        
        # true: ood, needs upgrade
        return np.abs(loss-self.old_distribution['mean'])>2*self.old_distribution['std']
    
    

        
    def distribution_shift(self,
                        insert_table:common.CsvTable,
                        old_table:common.CsvTable):

        for idx,insert_col in enumerate(insert_table.columns):
            for value in insert_col.DistinctValue():
                if value not in old_table.columns[idx].DistinctValue():
                    return True
        return False
    
    
    # def gridsearch(self,insert_data,new_model,new_data,new_tablebits,**kargs):
    #     new_model_copy=copy.deepcopy(new_model)
    #     old_model_copy=copy.deepcopy(self.model)
    #     epochs = [10,15,20]
    #     start_lossw=np.arange(0,1,0.1)
    #     import pandas as pd
    #     from time import time
    #     df_params=pd.DataFrame()
    #     model_bits_min=10000
    #     best_param={}
    #     data_types=['new_data']
    #     for data_type in data_types:
    #         for epoch_nums in epochs:
    #             for start_lossw_ in start_lossw:
    #                 print("training starts for param",{'start_lossw':start_lossw_,
    #                                               'epoch':epoch_nums,
    #                                               'data_type':data_type})
    #                 new_model = new_model_copy
    #                 self.model = old_model_copy
    #                 start_time=time()
    #                 data_train= new_data
    #                 for epoch in range(epoch_nums):
    #                     mean_epoch_train_loss, new_model = self.runepoch_distillation(
    #                                             data_train,
    #                                             new_model,
    #                                             batch_size=256,
    #                                             epoch_num=epoch,                                
    #                                             start_lossw=start_lossw_,
    #                                             table_bits=new_tablebits,
    #                                             **kargs                                        
    #                                             )
    #                 end_time=time()
    #                 self.model = new_model
    #                 all_losses = self.RunEpoch('test',
    #                             dataset=new_data,
    #                             return_losses=True,
    #                             table_bits=new_tablebits)
    #                 model_nats = np.mean(all_losses)
    #                 model_bits = model_nats / np.log(2)
    #                 if (model_bits<model_bits_min):
    #                     model_bits_min=model_bits
    #                     best_param={'epoch':epoch_nums,
    #                                 'start_lossw':start_lossw_,
    #                                 'data_type':data_type,
    #                                 'err':model_bits,
    #                                 'time_avg_epoch':(end_time-start_time)/epoch_nums}
    #                 print(f"training end. model_bits_err:{model_bits} time_avg_epoch:{(end_time-start_time)/epoch_nums}\n\n")
    #                 df_params = df_params.append({'start_lossw':start_lossw_,
    #                                               'epoch':epoch_nums,
    #                                               'data_type':data_type,
    #                                             'err':model_bits,
    #                                             'time_avg_epoch':(end_time-start_time)/epoch_nums},ignore_index=True)
    #     df_params.to_csv("/home/jixy/naru/params/params_gridsearch_newdata.csv")      
    #     print(best_param)
    
    def distillation(self,
                     insert_data,
                     new_model,
                     batch_size=256,
                     epochs=10,
                     start_lossw=0.5,
                     table_bits=0,
                     **kargs):
        for epoch in range(epochs):

            mean_epoch_train_loss, new_model = self.runepoch_distillation(
                                            insert_data,
                                            new_model,
                                            batch_size,
                                            epoch,
                                            start_lossw,
                                            table_bits,
                                            **kargs                                        
                                            )
        self.model = new_model
    
    
    # def distillation_loss(self,new_model_out,old_model_out):
    #     d_loss=nn.CrossEntropyLoss()
    
    def runepoch_distillation(self,
                     insert_data,
                     new_model,
                     batch_size=256,
                     epoch_num=None,
                     start_lossw=0.5,
                     table_bits=0,
                     **kargs):
        
        torch.set_grad_enabled(True)
        self.get_model_opt(new_model)
        new_model.train() 
        self.model.eval()
        loss_weight = torch.nn.Parameter(torch.tensor(start_lossw),requires_grad=True)
        
        losses = []
        loader = torch.utils.data.DataLoader(insert_data,
                                            batch_size=batch_size,
                                            shuffle=True)

        # How many orderings to run for the same batch?
        nsamples = 1
        if hasattr(self.model, 'orderings'):
            nsamples = len(self.model.orderings)

        for step, xb in enumerate(loader):

            assert self.opt is not None," the optimizer of training process must be specified"
            base_lr = 8e-4
            for param_group in self.opt.param_groups:
                if kargs['constant_lr']:
                    lr = kargs['constant_lr']
                elif kargs['warmups']: # Transformer, FLASH
                    t = kargs['warmups']
                    d_model = self.model.embed_dim
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
            new_xbhat = None
            new_model_logits = []
            model_logits=[]
            num_orders_to_forward = 1

            for i in range(num_orders_to_forward):
                if hasattr(self.model, 'update_masks'):
                    # We want to update_masks even for first ever batch.
                    self.model.update_masks()

                old_model_out = self.model(xb)
                new_model_out = new_model(xb)
                # d_loss = F.cross_entropy(F.softmax(new_model_out,dim=2),F.softmax(old_model_out,dim=2))
                d_loss = new_model.distillation_loss(new_model_out,old_model_out,self.model).mean()
                new_model_logits.append(old_model_out)
                if new_xbhat is None:
                    new_xbhat = torch.zeros_like(new_model_out)
                new_xbhat += new_model_out

            if new_xbhat.shape == xb.shape:
                # if mean:
                #     xb = (xb * std) + mean
                model_loss = F.binary_cross_entropy_with_logits(
                    new_xbhat, xb, size_average=False) / new_xbhat.size()[0]
            else:
                if self.model.input_bins is None:
                    # NOTE: we have to view() it in this order due to the mask
                    # construction within MADE.  The masks there on the output unit
                    # determine which unit sees what input vars.
                    new_xbhat = new_xbhat.view(-1, self.model.nout // self.model.nin, self.model.nin)
                    # Equivalent to:
                    model_loss = F.cross_entropy(new_xbhat, xb.long(), reduction='none') \
                            .sum(-1).mean()
                else:
                    if num_orders_to_forward == 1:
                        model_loss = self.model.nll(new_xbhat, xb).mean()
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
                            logps.append(-self.model.nll(logits, xb))
                        logps = torch.stack(logps, dim=1)
                        logps = logps.logsumexp(dim=1) + torch.log(
                            torch.tensor(1.0 / nsamples, device=logps.device))
                        model_loss = (-logps).mean()
                        
            loss = loss_weight*(model_loss)+(1 - loss_weight)*d_loss
            losses.append(loss.item())

            if step % self.log_every == 0:
                print('Epoch {} Iter {}, train entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num, step, 
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, lr))

           
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.verbose:
                print('train epoch average loss: %f' % np.mean(losses))
        if self.return_losses:
            return losses
        

        
        return np.mean(losses), new_model
            
        
    
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
                     insert_table:common.CsvTable,
                       old_table:common.CsvTable,
                       new_table,
                       new_tablebits,
                       train_data,
                       new_data,
                       epochs=10,
                       start_lossw=0.5
                       ):
        nin = len(new_table.columns)
        input_bins = [c.DistributionSize() for c in new_table.columns]
        self.model.update_structure(nin,input_bins)
        new_model = copy.deepcopy(self.model)
        self.distillation(train_data,
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
        return model_bits
        
    
    def online_update_db(self,
                       insert_table:common.CsvTable,
                       old_table:common.CsvTable,
                       epochs=10):

        
        from datasets import UpdateDmv
        new_table = UpdateDmv(insert_table,old_table)

    
        new_data = common.TableDataset(new_table)
        old_data = common.TableDataset(old_table)
        insert_data = common.TableDataset(insert_table)
        ood = self.online_sample(new_data,old_data) 
        
        new_tablebits= self.Entropy(
        new_table,
        new_table.data.fillna(value=0).groupby([c.name for c in new_table.columns
                                        ]).size(), [2])[0]
        
        
        if (ood):
            nin = len(new_table.columns)
            input_bins = [c.DistributionSize() for c in new_table.columns]
            self.model.update_structure(nin,input_bins)
            new_model = copy.deepcopy(self.model)
            self.distillation(insert_data,
                              new_model,
                              constant_lr=None,
                                warmups=0,
                                epochs=epochs,
                                table_bits=new_tablebits,
                                start_lossw=0.5)
            # self.gridsearch(insert_data,new_model,new_data,new_tablebits,constant_lr=None,warmups=0)
            
            # self.distillation(insert_data,
            #         new_model,
            #         constant_lr=None,
            #         warmups=0,
            #         epochs=epochs,
            #         table_bits=new_tablebits,
            #         start_lossw=np.array(1.0,dtype=np.float))
            
        else:   
            train_losses=[]
            train_start=time.time()
            for epoch in range(epochs):
    
                mean_epoch_train_loss = self.RunEpoch(
                                                'train',
                                                insert_data,
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
                
        print('Training done; evaluating likelihood on full data:')  
        all_losses = self.RunEpoch('test',
                            dataset=new_data,
                            return_losses=True,
                            table_bits=new_tablebits)
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        self.model.model_bits = model_bits
        return new_table, self.model
                
    def get_model(self):
        return self.model
            


        
   