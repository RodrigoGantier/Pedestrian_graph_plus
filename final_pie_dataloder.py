import torch
import torch.utils.data as data
import os
import pickle5 as pk
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from torchvision import transforms as A

from tqdm import tqdm, trange
from pie_data import PIE


class DataSet(data.Dataset):
    def __init__(self, path, pie_path, frame, vel, balance=True, bh='all', t23=False, transforms=None, seg_map=False, h3d=True, pcpa=None, forecast=True, last2=False, time_crop=False):
        
        np.random.seed(42)
        self.time_crop = time_crop
        self.forecast = forecast
        self.last2 = last2
        self.h3d = h3d # bool if true 3D human key points are avalable otherwise 2D is going to be used
        self.bh = bh
        self.seg = seg_map
        self.t23 = t23
        self.pcpa = os.getcwd() / Path(pcpa)
        self.transforms = transforms
        self.frame = frame
        self.vel= vel
        self.balance = balance
        self.data_set = 'test'
        self.maxw_var = 9
        self.maxh_var = 6
        self.maxd_var = 2
        self.input_size = int(32 * 1)
        
        nsamples = [1624, 879, 611]
        balance_data = [max(nsamples) / s for s in nsamples]
        
        if bh != 'all':
            balance_data[2] = 0
        elif t23:
            balance_data = [1, (nsamples[0] + nsamples[2])/nsamples[1], 1]

        self.data_path = os.getcwd() / Path(path) / 'data'
        self.imgs_path = os.getcwd() / Path(path) / 'imgs'
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]
        self.pie_path = pie_path
        
        imdb = PIE(data_path=self.pie_path)
        params = {'data_split_type': 'default',}
        self.vid_ids, _ = imdb._get_data_ids(self.data_set, params)
        
        filt_list =  lambda x: not 'r' in x.split('.')[0].split('_')[-1] 
        ped_ids = list(filter(filt_list, self.data_list))

        filt_list =  lambda x: x.split('_')[0] in self.vid_ids
        ped_ids = list(filter(filt_list, self.data_list))

        pcpa_ = self.load_3part()
        self.models_data = {}

        for k_id in pcpa_['ped_id'].keys():
            set_n, vid_n, frm, ped_id = k_id.split('-')
            pcpa_key =  set_n + '_' + vid_n + '_pid_' + ped_id + '_fr_' + frm
            
            if pcpa_key + '.pkl' in ped_ids:
                self.models_data[pcpa_key] = [pcpa_['result'][k_id], pcpa_['labels'][k_id]]
        list_k = list(self.models_data.keys())

        filt_list =  lambda x: x.split('.')[0] in list_k
        ped_ids = list(filter(filt_list, ped_ids))

        self.ped_data = {}

        for ped_id in tqdm(ped_ids, desc=f'loading {self.data_set} data in memory'):

            ped_path = self.data_path.joinpath(ped_id).as_posix()
            loaded_data = self.load_data(ped_path)
            img_file = str(self.imgs_path / loaded_data['crop_img'].stem) + '.pkl'
            loaded_data['crop_img'] = self.load_data(img_file)

            if loaded_data['irr'] == 1 and bh != 'all': 
                continue
            
            if balance:
                if loaded_data['irr'] == 1:         # irrelevant
                    self.repet_data(balance_data[2], loaded_data, ped_id)
                elif loaded_data['crossing'] == 0:  # no crossing
                    self.repet_data(balance_data[0], loaded_data, ped_id)
                elif loaded_data['crossing'] == 1:  # crossing
                    self.repet_data(balance_data[1], loaded_data, ped_id)
            else:
                self.ped_data[ped_id.split('.')[0]] = loaded_data
        
        self.ped_ids = list(self.ped_data.keys())
        self.data_len = len(self.ped_ids)

    def load_3part(self, ):

        pcpa = {}
        if self.last2:
            pcpa['result'] = self.load_data(self.pcpa / f'test_results/pie/pcpa_preds_pie_all_last2.pkl')
            pcpa['labels'] = self.load_data(self.pcpa / f'test_results/pie/pcpa_labels_pie_all_last2.pkl')
            pcpa['ped_id'] = self.load_data(self.pcpa / f'test_results/pie/pcpa_ped_ids_pie_all_last2.pkl')
        else:
            pcpa['result'] = self.load_data(self.pcpa / f'test_results/pie/pcpa_preds_pie_all.pkl')
            pcpa['labels'] = self.load_data(self.pcpa / f'test_results/pie/pcpa_labels_pie_all.pkl')
            pcpa['ped_id'] = self.load_data(self.pcpa / f'test_results/pie/pcpa_ped_ids_pie_all.pkl')

        return pcpa


    def repet_data(self, n_rep, data, ped_id):
        ped_id = ped_id.split('.')[0]
        
        if self.data_set == 'train' or self.data_set == 'val' or self.t23:
            prov = n_rep % 1  
            n_rep = int(n_rep) if prov == 0 else int(n_rep) + np.random.choice(2, 1, p=[1 - prov, prov])[0]
        else:
            n_rep = int(n_rep)
        
        for i in range(int(n_rep)):
            self.ped_data[ped_id + f'-r{i}'] = data
    
    def load_data(self, data_path):

        with open(data_path, 'rb') as fid:
            database = pk.load(fid, encoding='bytes')
        return database
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, item):
        
        ped_id = self.ped_ids[item]
        pcpa_data = self.models_data[ped_id.split('-')[0]]
        ped_data = self.ped_data[ped_id]
        w, h = ped_data['w'], ped_data['h']
        
        if self.forecast:
            ped_data['kps'][-30:] = ped_data['kps_forcast']
            kp = ped_data['kps']
        else:
            kp = ped_data['kps'][:-30]
            
        # key points data augmentation
        if self.data_set == 'train':
            kp[..., 0] = np.clip(kp[..., 0] + np.random.randint(self.maxw_var, size=kp[..., 0].shape), 0, w)
            kp[..., 1] = np.clip(kp[..., 1] + np.random.randint(self.maxh_var, size=kp[..., 1].shape), 0, w)
            kp[..., 2] = np.clip(kp[..., 2] + np.random.randint(self.maxd_var, size=kp[..., 2].shape), 0, w)
        # normalize key points data
        kp[..., 0] /= w
        kp[..., 1] /= h
        kp[..., 2] /= 80
        kp = torch.from_numpy(kp.transpose(2, 0, 1)).float().contiguous()

        seg_map = torch.from_numpy(ped_data['crop_img'][:1]).float() 
        seg_map = (seg_map - 78.26) / 45.12
        img = ped_data['crop_img'][1:]
        img = self.transforms(img.transpose(1, 2, 0)).contiguous()
        if self.seg:
            img = torch.cat([seg_map, img], 0)
        vel_obd = np.asarray(ped_data['obd_speed']).reshape(1, -1) / 120.0 # normalize
        vel_gps = np.asarray(ped_data['gps_speed']).reshape(1, -1) / 120.0 # normalize
        vel = torch.from_numpy(np.concatenate([vel_gps, vel_obd], 0)).float().contiguous()
        if not self.forecast:
            vel = vel[:, :-30]
            
        # 0 for no crosing, 1 for crossing, 2 for irrelevant
        if ped_data['irr']:  
            bh = torch.from_numpy(np.ones(1).reshape([1])) * 2
        else:       
            bh = torch.from_numpy(ped_data['crossing'].reshape([1])).float()
        
        if not self.h3d:
            kp = kp[[0, 1, 3], ].clone()
            
        if self.frame and not self.vel:
            return kp, bh, img, pcpa_data
        elif self.frame and self.vel:
            return kp, bh, img, vel, pcpa_data
        else:
            return kp, bh, pcpa_data

def main():
    
    data_path = './data/pose_forcasting/PIE'
    pie_path = '/home/rodrigo/data/PIE'
    pcpa = './data/new2'

    transform = A.Compose([
        A.ToTensor(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    
    tr_data = DataSet(path=data_path,  pie_path=pie_path, balance=True, frame=True, vel=True, seg_map=True, bh='all', t23=False, transforms=transform, pcpa=pcpa, last2=True)
    iter_ = trange(len(tr_data))
    cx = np.zeros([len(tr_data), 3])

    for i in iter_:
        x, y, f, v, pcpca = tr_data.__getitem__(i)
        
        # y = np.clip(y - 1, 0, 1)
        # y[y==2] = 0
        cx[i, y.long().item()] = 1
            
    print(f'No Crosing: {cx.sum(0)[0]} Crosing: {cx.sum(0)[1]}, Irrelevant: {cx.sum(0)[2]} ')
    print('finish')

if __name__ == "__main__":
    main()
