import torch
import torch.nn as nn
from torchvision import transforms as A
from final_jaad_dataloder import DataSet
from models.ped_graph23 import pedMondel 
import time
import copy

from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm, tgrange
import os
import pandas as pd
#from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


def seed_everything(seed):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def data_loader(args):
              
    transform = A.Compose([ 
        A.ToTensor(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
       ])

    te_data = DataSet(
        path=args.data_path, 
        jaad_path=args.jaad_path, 
        frame=True, 
        vel=True,
        seg_map=args.seg,
        h3d=args.H3D,
        balance=args.balance,
        bh=args.bh,
        t23=args.balance,
        transforms=transform, 
        pcpa=args.pcpa,
        forecast=args.forecast,
        last2=args.last2
        )
    
    te = DataLoader(te_data, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
    
    return te


class musterModel(nn.Module):
    def __init__(self, args):
        super(musterModel, self).__init__()

        self.model = pedMondel(args.frames, vel=args.velocity, seg=args.seg, h3d=args.H3D, n_clss=3)
        ckpt = torch.load(args.ckpt, map_location=args.device)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(args.device)
        self.model.eval()
    
    def forward(self, x, f, v):
        with torch.no_grad():
            cx = self.model(x, f, v).softmax(1)
        return cx

def prepare_input(resolution):

    x = torch.FloatTensor(1, 4, 62, 19).cuda()
    f = torch.FloatTensor(1, 4, 192, 64).cuda()
    v = torch.FloatTensor(1, 2, 62).cuda()
    return dict(x = x, f=f, v=v)

class statsClass:
    def __init__(self, pcpa_pred, g_pcpca_pred, fussi_pred, pedgraph_pred, y) -> None:
        
        self.pcpa = pcpa_pred
        self.pcpa_g = g_pcpca_pred
        self.fussi_pred = fussi_pred
        self.pedgraph_pred = pedgraph_pred
        self.one_hot_y = np.eye(3)[y.reshape(-1).astype(np.int32)]
        y[y==2]=0
        self.y = y
        # initialize dict
        self.results = {
        accuracy_score.__name__: [0.0, 0.0, 0.0, 0.0, ],  
        }
        self.weather_dict = {
            accuracy_score.__name__: [0.0, 0.0, 0.0, 0.0, ],
        }
        self.models_k = ['PCPA', 'Glob_PCPA', 'FUSSI', 'PedGraph+', ]
        self.results = pd.DataFrame(self.results, 
                    index=self.models_k)
        
        self.weather_keys = ['cloudy', 'clear', 'rain', 'cloud', 'snow']
        self.weather_dict = pd.DataFrame(columns=self.weather_keys, index=self.models_k) 
        
    def stats(self, fn, rn, mult=1):

        pcpa = np.round(self.pcpa) if rn else self.pcpa
        pcpca_g = np.round(self.pcpa_g) if rn else self.pcpa_g
        fussi_pred = np.round(self.fussi_pred) if rn else self.fussi_pred
        pedgraph_pred = np.round(self.pedgraph_pred) if rn else self.pedgraph_pred

        self.results.at[self.models_k[0], fn.__name__] = fn(self.y, pcpa) * mult
        self.results.at[self.models_k[1], fn.__name__] = fn(self.y, pcpca_g) * mult
        self.results.at[self.models_k[2], fn.__name__] = fn(self.y, fussi_pred) * mult
        self.results.at[self.models_k[3], fn.__name__] = fn(self.y, pedgraph_pred) * mult
    
    def weather(self, weather_PedG, weather_pcpa, weather_Gpcpa, weather_fussi, fn, mult=100):
        
        w_list = [weather_pcpa, weather_Gpcpa, weather_fussi, weather_PedG]
        for dictNae , dict_w in zip(self.models_k, w_list):
            for k in dict_w.keys():
                if len(dict_w[k][1]) == 0:
                    continue
                res_ = fn(np.asarray(dict_w[k][1]), np.round(np.asarray(dict_w[k][0])))        
                self.weather_dict.at[dictNae, k] = res_ * mult
    

def main(args):

    seed_everything(args.seed)
    try:
        m_feat = args.ckpt.split('/')[-2].split('-')[2]
    except IndexError:
        m_feat = 'None'
    args.frames =    True if 'I' in m_feat else False
    args.velocity =  True if 'V' in m_feat else False
    args.seg =       True if 'S' in m_feat else False
    args.forecast =  True if 'F' in m_feat else False
    args.time_crop = True if 'T' in m_feat else False
    args.H3D = False if args.ckpt.split('/')[-2].split('-')[-1] == 'h2d' else True
    

    weather_PedG = {'cloudy': [[], []], 
                    'clear': [[], []],
                    'rain': [[], []],
                    'cloud': [[], []],
                    'snow':[[], []]}
    weather_pcpa = copy.deepcopy(weather_PedG)
    weather_Gpcpa = copy.deepcopy(weather_PedG)
    weather_fussi = copy.deepcopy(weather_PedG)
    
    model = musterModel(args)
    model.eval()
    model.half()
    data = data_loader(args)  
    str_t = torch.cuda.Event(enable_timing=True)
    end_t = torch.cuda.Event(enable_timing=True)

    timimg = []
    
    ys = np.zeros([len(data), 1])

    pedgraph_pred_all = np.zeros([len(data), 3])
    pedgraph_pred = np.zeros([len(data), 1])
    pcpa_pred = np.zeros([len(data), 1])
    fussi_pred = np.zeros([len(data), 1])
    g_pcpca_pred = np.zeros([len(data), 1])

    #if args.frames and args.velocity and args.seg and args.H3D:
    #    in_cn = 4 if args.seg == True else 3
    #    flops, params = get_model_complexity_info(
    #        model, input_res=((1, 4, 62, 19), (1, in_cn, 192, 64), (1, 2, 62)), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=False)
    #    print(*['-']*30)
    #    print('Pedestrian Graph +, Number of Parameters: ', params, '@', flops)
    #    print(*['-']*30)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data, desc='Testing samples')):

            x = batch[0].float().to(args.device)
            y = batch[1].long().to(args.device)
            f = batch[2].float().to(args.device) if args.frames else None
            v = batch[3].float().to(args.device) if args.velocity else None
            weather_ = batch[-2]
            models_data = batch[-1]
            
            str_t.record()
            if args.last2:
                x = x[:, :, -(args.lastn+30):] if args.forecast else x[:, :, -args.lastn:]
                pred = model(x.contiguous().half(), f.half(), v.half())
            else:
                pred = model(x.half(), f.half(), v.half())
            
            end_t.record()

            torch.cuda.synchronize()
            timimg.append(str_t.elapsed_time(end_t))
            
            ys[i] = int(y.item())
            pedgraph_pred_all[i] = pred.float().detach().cpu().numpy()

            if args.argmax:
                prov = pred[:, pred.argmax(1)].cpu().numpy()
                prov = 1 - prov if pred.argmax(1) != 1 else prov
                pedgraph_pred[i] = prov
            else:

                pred[:, 0] = min(1, pred[:, 0] + pred[:, 2])
                pred[:, 1] = max(0, pred[:, 1] - pred[:, 2])
                pedgraph_pred[i] = pred[:, 1].item() if pred.argmax(1) == 1 else 1 - pred[:, 0].item()
            
            pcpa_pred[i] = models_data['pcpa']['result'].cpu().numpy()
            fussi_pred[i] = models_data['fussi']['result'].cpu().numpy()
            g_pcpca_pred[i] = models_data['g_pcpca']['result'].cpu().numpy()
            
            y[y==2] = 0
            assert y.item() == models_data['pcpa']['label'].item() == models_data['fussi']['label'].item() == models_data['g_pcpca']['label'].item(), 'labels sanity check'

            weather_PedG[weather_[0]][0].append(pedgraph_pred[i].item())
            weather_PedG[weather_[0]][1].append(y.item())
            
            weather_pcpa[weather_[0]][0].append(pcpa_pred[i].item())
            weather_pcpa[weather_[0]][1].append(y.item())
            
            weather_Gpcpa[weather_[0]][0].append(g_pcpca_pred[i].item())
            weather_Gpcpa[weather_[0]][1].append(y.item())
            
            weather_fussi[weather_[0]][0].append(fussi_pred[i].item())
            weather_fussi[weather_[0]][1].append(y.item())
            
    y = ys.copy()
    y[y==2] = 0
    pedgraph_pred = np.clip(pedgraph_pred, 0, 1)
    stats_fn = statsClass(pcpa_pred, g_pcpca_pred, fussi_pred, pedgraph_pred, ys)
    
    stats_fn.stats(accuracy_score, True, 100)
    stats_fn.stats(f1_score, True, 100)
    stats_fn.stats(precision_score, True, 100)
    stats_fn.stats(recall_score, True, 100)
    stats_fn.stats(roc_auc_score, False, 100)
    stats_fn.stats(average_precision_score, False, 100)
    
    print(*['-']*30)
    print(f'balance data: {args.balance}, bh: {args.bh}, last2: {args.last2}, Model: ' + args.ckpt.split('/')[-2], )
    print(*['-']*30)
    print('Models statistics')
    print(stats_fn.results)
    
    no_xing = 100 * np.mean((pedgraph_pred_all[:, 0]>0.5) == stats_fn.one_hot_y[:, 0])
    xing = 100 * np.mean((pedgraph_pred_all[:, 1]>0.5) == stats_fn.one_hot_y[:, 1])
    irr = 100 * np.mean((pedgraph_pred_all[:, 2]>0.5) == stats_fn.one_hot_y[:, 2])
    
    print(f'No crossing: {no_xing:.3f}')
    print(f'Crossing: {xing:.3f}')
    print(f'Irrelevant {irr:.3f}')
    
    # stats_fn.weather(weather_PedG, weather_pcpa, weather_Gpcpa, weather_fussi, accuracy_score)
    stats_fn.weather(weather_PedG, weather_pcpa, weather_Gpcpa, weather_fussi, f1_score)
    # stats_fn.weather(weather_PedG, weather_pcpa, weather_Gpcpa, weather_fussi, precision_score)
    # stats_fn.weather(weather_PedG, weather_pcpa, weather_Gpcpa, weather_fussi, recall_score)
    print(*['-']*30)
    print('Weather statistics')
    print(stats_fn.weather_dict)
    
    print(*['-']*30)
    print(f'Average run time for Pedestrian Graph +: {np.mean(timimg):.3f} ms')
    print('finish')     


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Pedestrian prediction crosing")
    parser.add_argument('--ckpt', type=str, default="./weigths/jaad-23-IVFT/best.pth", help="Path to model weigths")
    parser.add_argument('--device', type=str, default='cuda:0', help="GPU")
    parser.add_argument('--data_path', type=str, default='./data/JAAD', help='Path to the train and test data')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training and test")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the dataloader")
    parser.add_argument('--frames', type=bool, default=False, help='avtivate the use of raw frames')
    parser.add_argument('--velocity', type=bool, default=False, help='activate the use of the odb and gps velocity')
    parser.add_argument('--seg', type=bool, default=False, help='Use the segmentation map')
    parser.add_argument('--forcast', type=bool, default=False, help='Use the human pose forcasting data')
    parser.add_argument('--H3D', type=bool, default=True, help='Use 3D human keypoints')
    parser.add_argument('--jaad_path', type=str, default='./JAAD')
    parser.add_argument('--bh', type=str, default='all', help='all or bh, if use all samples or only samples with behaevior labers')
    parser.add_argument('--balance', type=bool, default=False, help='Balnce or not the data set (over sampling)')
    parser.add_argument('--pcpa', type=str, default='./data/test_results', help='path with results for pcpa and other models')
    parser.add_argument('--last2', type=bool, default=False, help='Use the last 2 frames on PCPA, G_PCPA and FUSSI (stored data, no inference)')
    parser.add_argument('--lastn', type=int, default=2, help='Use the last n frame for inference in Pedestrian Graph +')
    parser.add_argument('--seed', type=int, default=42, help='Initialization of the random seed')
    parser.add_argument('--argmax', type=bool, default=False, help='Use argemax funtion, if false use maping')
    args = parser.parse_args()
    main(args)
