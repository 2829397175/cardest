from ctypes import Structure
from sklearn import model_selection
from sqlalchemy import false
import made
import transformer
from construct_model import MakeFlash,MakeMade,MakeTransformer,DEVICE
import common
import re
import torch
from updater import OOD_DDup_detection
import flash
import numpy as np
import time
import os
import pandas as pd


class DB_cardest():
    def __init__(self,
                 model_path,
                 table,
                 updater=OOD_DDup_detection,
                 args=None):
        self.updater = updater(old_sample_num=2000,
                             old_sample_size=30,
                             new_sample_size=30,
                             update_step_threshold=5,
                             queue_size=2000)
        self.model = self.load_model(model_path,table,args)
        
        
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
    

               
             
    def LoadOracleCardinalities(self,args):
        ORACLE_CARD_FILES = {
        'dmv': 'datasets/dmv-2000queries-oracle-cards-seed1234.csv'
    }
        path = ORACLE_CARD_FILES.get(args.dataset, None)
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            assert len(df) == 2000, len(df)
            return df.values.reshape(-1)
        return None
               
             
    def SaveEstimators(self,path, estimators, return_df=False,args=None):
        # name, query_dur_ms, errs, est_cards, true_cards
        results = pd.DataFrame()
        for est in estimators:
            data = {
                'est': [est.name] * len(est.errs),
                'err': est.errs,
                'est_card': est.est_cards,
                'true_card': est.true_cards,
                'query_dur_ms': est.query_dur_ms,
            }
            results = results.append(pd.DataFrame(data))
        if return_df:
            return results
        results.to_csv(path, index=False)
               
    def run_ester(self, table, args=None,save_err=False,train_data=None):
        import estimators as estimators_lib
        estimators=[estimators_lib.ProgressiveSampling(self.model,
                                            table,
                                            args.psample,
                                            device=DEVICE,
                                            shortcircuit=args.column_masking)]
        for est in estimators:
            est.name=str(est)
        
        oracle_est = estimators_lib.Oracle(table)
        oracle_cards = self.LoadOracleCardinalities(args)
        cols_to_train = table.columns
        self.model.eval()
        self.RunN(
            table,
            cols_to_train,
            estimators,
            rng=np.random.RandomState(1234),
            num=args.num_queries,
            log_every=1,
            num_filters=None,
            oracle_cards=oracle_cards,
            oracle_est=oracle_est)
        
        if (save_err):
            root_dir="distill_results/"
            if (args.finetune):
                root_dir+="finetune"
            elif (args.retrain):
                root_dir+="retrain"
            else:
                root_dir+="distill"
            
            model_path='{}-{}-mixupTrue-alpha{}-beta{}-train_data{}-'.format(table.name,
                                                                             self.model.name(),
                                                                             args.alpha,
                                                                             args.beta,
                                                                             train_data)
            err_path=os.path.join(root_dir,"results_{}.csv".format(model_path))
            if os.path.exists(err_path):
                err_path=err_path[:-4]+"_num{}.csv".format(args.num_queries)    
            # SaveEstimators(args.err_csv, estimators)
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
            self.SaveEstimators(err_path, estimators)
            print('...Done, result:', err_path)
    
    def SampleTupleThenRandom(self,
                              all_cols,
                            num_filters,
                                rng,
                                table,

                                return_col_idx=False):
        s = table.data.iloc[rng.randint(0, table.cardinality)]
        vals = s.values

        # DMV data_preprocessing
        vals[6] = vals[6].to_datetime64()

        idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
        cols = np.take(all_cols, idxs)

        # If dom size >= 10, okay to place a range filter.
        # Otherwise, low domain size columns should be queried with equality.
        ops = rng.choice(['<=', '>=', '='], size=num_filters)
        ops_all_eqs = ['='] * num_filters
        sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
        ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

        if num_filters == len(all_cols):
            if return_col_idx:
                return np.arange(len(all_cols)), ops, vals
            return all_cols, ops, vals

        vals = vals[idxs]
        if return_col_idx:
            return idxs, ops, vals

        return cols, ops, vals
    
    def GenerateQuery(self,all_cols, rng, table, return_col_idx=False):
        """Generate a random query."""
        num_filters = rng.randint(5, 12)
        cols, ops, vals = self.SampleTupleThenRandom(all_cols,
                                                num_filters,
                                                rng,
                                                table,
                                                return_col_idx=return_col_idx)
        return cols, ops, vals
    
    def ErrorMetric(self,est_card, card):
        if card == 0 and est_card != 0:
            return est_card
        if card != 0 and est_card == 0:
            return card
        if card == 0 and est_card == 0:
            return 1.0
        return max(est_card / card, card / est_card)
    
    
    def Query(self,
              estimators,
          do_print=True,
          oracle_card=None,
          query=None,
          table=None,
          oracle_est=None):
        assert query is not None
        cols, ops, vals = query

        ### Actually estimate the query.

        def pprint(*args, **kwargs):
            if do_print:
                print(*args, **kwargs)

        # Actual.
        card = oracle_est.Query(cols, ops,
                                vals) if oracle_card is None else oracle_card
        if card == 0:
            return

        pprint('Q(', end='')
        for c, o, v in zip(cols, ops, vals):
            pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
        pprint('): ', end='')

        pprint('\n  actual {} ({:.3f}%) '.format(card,
                                                card / table.cardinality * 100),
            end='')

        for est in estimators:
            est_card = est.Query(cols, ops, vals)
            err = self.ErrorMetric(est_card, card)
            est.AddError(err, est_card, card)
            pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
        pprint()
    
    def RunN(
        self,
        table,
         cols,
         estimators,
         rng=None,
         num=20,
         log_every=50,
         num_filters=11,
         oracle_cards=None,
         oracle_est=None):
        if rng is None:
            rng = np.random.RandomState(1234)

        last_time = None
        for i in range(num):
            do_print = False
            if i % log_every == 0:
                if last_time is not None:
                    print('{:.1f} queries/sec'.format(log_every /
                                                    (time.time() - last_time)))
                do_print = True
                print('Query {}:'.format(i), end=' ')
                last_time = time.time()
            query = self.GenerateQuery(cols, rng, table)
            self.Query(estimators,
                do_print,
                oracle_card=oracle_cards[i]
                if oracle_cards is not None and i < len(oracle_cards) else None,
                query=query,
                table=table,
                oracle_est=oracle_est)

            max_err = self.ReportEsts(estimators)
            print(f"max_err: {max_err}")
        return False
    
    def ReportEsts(self,estimators):
        v = -1
        for est in estimators:
            print(est.name, 'max', np.max(est.errs), '99th',
                np.quantile(est.errs, 0.99), '95th', np.quantile(est.errs, 0.95),
                'median', np.quantile(est.errs, 0.5))
            v = max(v, np.max(est.errs))
        return v
    
    
        
    def load_model(self,
                   model_path,
                   table,
                   args):

        order = None
        if args.order is None:
            z = re.match('.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+).*.pt',
                    model_path)
        else:
            z = re.match(
                '.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+)-order.*.pt', model_path)
        assert z
        # model_bits = float(z.group(1))
        # data_bits = float(z.group(2))
        # bits_gap = model_bits - data_bits
        seed = int(z.group(3))
        if args.order is not None:
            order = list(args.order)
            

        if args.heads > 0:
            model = MakeTransformer(cols_to_train=table.columns,
                            fixed_ordering=order,
                            use_flash_attn=args.use_flash_attn,
                            seed=seed,
                            args=args)
            
        elif args.FLASH:
            model = MakeFlash(cols_to_train=table.columns,
                            fixed_ordering=order,
                            seed=seed,
                            args=args)
        else:
            if args.dataset in ['dmv-tiny', 'dmv']:
                model = MakeMade(
                    scale=args.fc_hiddens,
                    cols_to_train=table.columns,
                    seed=seed,
                    fixed_ordering=order,
                )
            else:
                assert False, args.dataset
        
        
        self.ReportModel(model)
        print('Loading ckpt:', model_path)
        model.load_state_dict(torch.load(model_path))
        
            
        # table_bits = self.Entropy(
        #         table,
        #         table.data.fillna(value=0).groupby([c.name for c in table.columns
        #                                         ]).size(), [2])[0]
        
        self.updater.update_model_info(model,
                                        log_every=200,
                                        seed=seed)
        return model
    

        
    
        
        
            
    def update(self,
               old_table:common.CsvTable,
                insert_table:common.CsvTable=None,
                new_table:common.CsvTable=None, 
               update_epoches=20,
               save_model = False,
               save_log=False,
               args=None):
            # remain to be modified!!!!
    # new_table :replace old table
    # insert table :insert table+old table = new
    # when new table is set,insert table is useless 
        new_table, self.model, ood, df_distill = self.updater.online_update_db(old_table,
                                                            insert_table=insert_table, 
                                                            new_table=new_table,
                                                            epochs=update_epoches,
                                                            save_model=save_model,
                                                            save_log=save_log,
                                                            args=args)

        

    
        
        # if(save_model):
        #     if (ood):
                
        #         PATH = 'models/distilled/{}-{:.1f}MB-model{:.3f}-{}-{}epochs-seed{}.pt'.format(
        #         new_table.name, mb, self.model.model_bits, self.model.name(),
        #         update_epoches, seed)
        #         if (df_distill is not None):
        #             df_distill.to_csv("distill_log/"+PATH[17:-3]+".csv")
        #     else:
        #         PATH = 'models/finetune/{}-{:.1f}MB-model{:.3f}-{}-{}epochs-seed{}.pt'.format(
        #         new_table.name, mb, self.model.model_bits, self.model.name(),
        #         update_epoches, seed)
        #     os.makedirs(os.path.dirname(PATH), exist_ok=True)
        #     torch.save(self.model.state_dict(), PATH)
        #     print('Saved to:')
        #     print(PATH)
        
        return new_table,self.model,ood,df_distill
        # self.model = self.updater.get_model()
        
        
        

        
if __name__=="__main__":
    import datasets
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--inference-opts',
                        action='store_true',
                        help='Tracing optimization for better latency.')

    parser.add_argument('--num-queries', type=int, default=20, help='# queries.')
    parser.add_argument('--dataset', type=str, default='dmv-tiny', help='Dataset.')
    parser.add_argument('--err-csv',
                        type=str,
                        default='results/results.csv',
                        help='Save result csv to what path?')
    parser.add_argument('--glob',
                        type=str,
                        help='Checkpoints to glob under models/.')
    parser.add_argument('--blacklist',
                        type=str,
                        help='Remove some globbed checkpoint files.')
    parser.add_argument('--psample',
                        type=int,
                        default=2000,
                        help='# of progressive samples to use per query.')
    parser.add_argument(
        '--column-masking',
        action='store_true',
        help='Turn on wildcard skipping.  Requires checkpoints be trained with '\
        'column masking.')
    parser.add_argument('--order',
                        nargs='+',
                        type=int,
                        help='Use a specific order?')

    # MADE.
    parser.add_argument('--fc-hiddens',
                        type=int,
                        default=128,
                        help='Hidden units in FC.')
    parser.add_argument('--layers', type=int, default=4, help='# layers in FC.')
    parser.add_argument('--residual', action='store_true', help='ResMade?')
    parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
    parser.add_argument(
        '--inv-order',
        action='store_true',
        help='Set this flag iff using MADE and specifying --order. Flag --order'\
        'lists natural indices, e.g., [0 2 1] means variable 2 appears second.'\
        'MADE, however, is implemented to take in an argument the inverse '\
        'semantics (element i indicates the position of variable i).  Transformer'\
        ' does not have this issue and thus should not have this flag on.')
    parser.add_argument(
        '--input-encoding',
        type=str,
        default='binary',
        help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
    parser.add_argument(
        '--output-encoding',
        type=str,
        default='one_hot',
        help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
        'then input encoding should be set to embed as well.')

    # Transformer.
    parser.add_argument(
        '--heads',
        type=int,
        default=0,
        help='Transformer: num heads.  A non-zero value turns on Transformer'\
        ' (otherwise MADE/ResMADE).'
    )
    parser.add_argument('--use_flash_attn',
                        action='store_true',
                        help='use use_flash_attn in Transform?')
    parser.add_argument('--blocks',
                        type=int,
                        default=2,
                        help='Transformer: num blocks.')
    parser.add_argument('--dmodel',
                        type=int,
                        default=32,
                        help='Transformer: d_model.')
    parser.add_argument('--dff', type=int, default=128, help='Transformer: d_ff.')
    parser.add_argument('--transformer-act',
                        type=str,
                        default='gelu',
                        help='Transformer activation.')


    # flash
    parser.add_argument('--FLASH',
                        action='store_true',
                        help='use FLASH?')
    parser.add_argument('--flash_dim',
                        type=int,
                        default=256,
                        help='the dim of flash.')
    parser.add_argument('--group_size',
                    type=int,
                    default=128,
                    help='the dim of group attention.')

    # Estimators to enable.
    parser.add_argument('--run-sampling',
                        action='store_true',
                        help='Run a materialized sampler?')
    parser.add_argument('--run-maxdiff',
                        action='store_true',
                        help='Run the MaxDiff histogram?')
    parser.add_argument('--run-bn',
                        action='store_true',
                        help='Run Bayes nets? If enabled, run BN only.')

    # Bayes nets.
    parser.add_argument('--bn-samples',
                        type=int,
                        default=200,
                        help='# samples for each BN inference.')
    parser.add_argument('--bn-root',
                        type=int,
                        default=0,
                        help='Root variable index for chow liu tree.')
    # Maxdiff
    parser.add_argument(
        '--maxdiff-limit',
        type=int,
        default=30000,
        help='Maximum number of partitions of the Maxdiff histogram.')

    # backbone_distill    
    parser.add_argument(
    '--backbone',
    type=str,
    default="/home/jixy/naru/models/pretrained/DMV/DMV_ori_p100",
    help='Backbone of distill.')
    
    
    # experiment setup
    parser.add_argument(
    '--alpha',
    type=float,
    default=0.8,
    help='loss parameter.')
    
    parser.add_argument(
    '--beta',
    type=float,
    default=0.2,
    help='distill loss parameter.')
    
    parser.add_argument(
    '--finetune',
    action='store_true',
    help='Run finetune experiment.')
        
    parser.add_argument(
    '--retrain',
    action='store_true',
    help='Run retrain experiment.')
    
    parser.add_argument(
    '--insert_dataset',
    type=str,
    default="insert_dmv_2000.csv",
    help='file name of insert data.')
    
    args = parser.parse_args()
    model_path= args.backbone
    table=datasets.LoadDmv()

    # db_est=DB_cardest(model_path,table,args=args)
    # # db_est.run_ester(table,args)
    
    # # update_db
    # table_insert=datasets.LoadDmv("insert_dmv_2000.csv",name='2000dmv')
    # table_insert=datasets.LoadDmv("insert_dmv_2000.csv",name='2000dmv')
    match = re.search( r'insert_dmv_(\d+).csv', args.insert_dataset, re.M|re.I)
    num_insert=match.group(1)
    table_insert=datasets.LoadDmv(args.insert_dataset,name=f'{num_insert}dmv')
    # table_insert=datasets.LoadDmv('chop_dmv.csv')
    # new_table=table_insert
    
    # from datasets import UpdateDmv
    # new_table = UpdateDmv(table_insert,table)


    # new_data = common.TableDataset(table_insert)
    # old_data = common.TableDataset(table)
    # insert_data = common.TableDataset(table_insert)
    

    
    def gridsearch(db_est:DB_cardest,insert_data):     
        epochs = [20]
        start_lossw=np.arange(0.75,0.9,0.05)
        import pandas as pd
        from time import time
        df_params=pd.DataFrame()
        model_bits_min=10000
        best_param={}
        data_types=['insert_data']
        for data_type in data_types:
            for epoch_nums in epochs:
                for start_lossw_ in start_lossw:
                    db_est_temp=copy.deepcopy(db_est)
                    print("training starts for param",{'alpha':start_lossw_,
                                                  'epoch':epoch_nums,
                                                  'data_type':data_type})

                    start_time=time()
                    model_bits,df_distill=db_est_temp.updater.search_param(
                                                           new_table=new_table,
                                                           new_tablebits=3.322,
                                                           train_data=in_data,
                                                           new_data=new_data,
                                                           epochs=epoch_nums,
                                                           start_lossw=start_lossw_)
                    end_time=time()

                    if (model_bits<model_bits_min):
                        model_bits_min=model_bits
                        best_param={'epoch':epoch_nums,
                                    'start_lossw':start_lossw_,
                                    'data_type':data_type,
                                    'err':model_bits,
                                    'time_avg_epoch':(end_time-start_time)/epoch_nums}
                    print(f"training end. model_bits_err:{model_bits} time_avg_epoch:{(end_time-start_time)/epoch_nums}\n\n")
                    df_params = df_params.append({'start_lossw':start_lossw_,
                                                  'epoch':epoch_nums,
                                                  'data_type':data_type,
                                                'err':model_bits,
                                                'time_avg_epoch':(end_time-start_time)/epoch_nums},ignore_index=True)
        df_params.to_csv("/home/jixy/naru/params/params_gridsearch_transformer_mse.csv")      
        print(best_param)
        
    import copy
    db_est_ori=DB_cardest(model_path,table,args=args)
    new_table,model, ood, df_distill= db_est_ori.update(old_table=table,insert_table=table_insert,save_model=False,save_log=True,update_epoches=20,args=args)
    # db_est_ori.run_ester(new_table,args,save_err=True)
    db_est_ori.run_ester(table,args,save_err=True,train_data=new_table.name)
    # db_est_ori.run_ester(table_insert,args,save_err=True)
    #gridsearch(db_est_ori)
    
    