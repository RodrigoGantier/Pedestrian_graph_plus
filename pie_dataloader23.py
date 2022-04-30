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
    def __init__(self, path, pie_path, data_set, frame, vel, balance=False, bh='all', t23=False, transforms=None, seg_map=True, h3d=True, forecast=False, time_crop=False):
        
        np.random.seed(42)
        self.time_crop = time_crop
        self.forecast = forecast
        self.h3d = h3d # bool if true 3D human key points are avalable otherwise 2D is going to be used
        self.bh = bh
        self.seg = seg_map
        self.t23 = t23
        self.transforms = transforms
        self.frame = frame
        self.vel= vel
        self.balance = balance
        self.data_set = data_set
        self.maxw_var = 9
        self.maxh_var = 6
        self.maxd_var = 2
        self.input_size = int(32 * 1)
        if data_set == 'train':
            nsamples = [9974, 5956, 7867]
        elif data_set == 'test':
            nsamples = [9921, 5346, 3700]
        elif data_set == 'val':
            nsamples = [3404, 1369, 1813]

        balance_data = [max(nsamples) / s for s in nsamples]
        if data_set == 'test':
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
        self.vid_ids, _ = imdb._get_data_ids(data_set, params)
        
        filt_list =  lambda x: x.split('_')[0] in self.vid_ids
        ped_ids = list(filter(filt_list, self.data_list))

        self.ped_data = {}
        ped_ids = ped_ids[:1000]

        for ped_id in tqdm(ped_ids, desc=f'loading {data_set} data in memory'):

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
            return kp, bh, img
        elif self.frame and self.vel:
            return kp, bh, img, vel
        else:
            return kp, bh

def main():
    
    data_path = './data/PIE'
    pie_path = './PIE'

    transform = A.Compose(
        [
        A.ToTensor(), 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    
    tr_data = DataSet(path=data_path,  pie_path=pie_path, data_set='train', balance=False, frame=True, vel=True, bh='all', h3d=True, t23=False, transforms=transform)
    iter_ = trange(len(tr_data))
    cx = np.zeros([len(tr_data), 3])
    fs = np.zeros([len(tr_data), 192, 64])

    for i in iter_:
        x, y, f, v = tr_data.__getitem__(i)
        
        # y = np.clip(y - 1, 0, 1)
        # y[y==2] = 0
        fs[i] = f[0]
        cx[i, y.long().item()] = 1
            
    print(f'No Crosing: {cx.sum(0)[0]} Crosing: {cx.sum(0)[1]}, Irrelevant: {cx.sum(0)[2]} ')
    print('finish')

if __name__ == "__main__":
    main()
