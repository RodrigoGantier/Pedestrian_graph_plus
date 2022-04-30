import torch
import torch.utils.data as data
import os
import pickle5 as pk
from pathlib import Path
import numpy as np
from torchvision import transforms as A

from tqdm import tqdm, trange
from jaad_data import JAAD


class DataSet(data.Dataset):
    def __init__(self, path, jaad_path, data_set, frame, vel, balance=True, bh='all', t23=False, transforms=None, seg_map=True, h3d=True, pcpa=None, forcast=True):
        
        np.random.seed(1)
        self.forcast = forcast
        self.h3d = h3d # bool if true 3D human key points are avalable otherwise 2D is going to be used
        self.t23 = t23
        self.seg = seg_map
        if not pcpa is None:
            self.pcpa = Path(pcpa)
        self.transforms = transforms
        self.frame = frame
        self.vel= vel
        self.balance = balance
        self.data_set = data_set
        self.maxw_var = 9
        self.maxh_var = 6
        self.maxd_var = 2
        
        if data_set == 'train':
            nsamples = [1025, 4778, 17582]
        elif data_set == 'test':
            nsamples = [1871, 3204, 13037]
        elif data_set == 'val':
            nsamples = [176, 454, 2772]
        
        balance_data = [max(nsamples) / s for s in nsamples]
        if t23:
            balance_data = [1, (nsamples[0] + nsamples[2])/nsamples[1], 1]

        self.data_path = os.getcwd() / Path(path) / 'data'
        self.imgs_path = os.getcwd() / Path(path) / 'imgs'
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]
        self.jaad_path = jaad_path
        
        imdb = JAAD(data_path=self.jaad_path)
        self.vid_ids = imdb._get_video_ids_split(data_set)
        
        filt_list =  lambda x: "_".join(x.split('_')[:2]) in self.vid_ids
        ped_ids = list(filter(filt_list, self.data_list))

        if bh != 'all':
            filt_list =  lambda x: 'b' in x 
            ped_ids = list(filter(filt_list, ped_ids))

        if data_set == 'test' and not pcpa is None:

            pcpa_res = self.load_data(self.pcpa / 'test_res_jaad.pkl')
            pcpa_lab = self.load_data(self.pcpa / 'test_labels_jaad.pkl')
            pcpa_ids = self.load_data(self.pcpa / 'ped_ids_jaad.pkl')
            self.pcpa_data = {}

            for id_num in range(len(pcpa_ids)):
                vid_n, frm, ped_id = pcpa_ids[id_num].split('-')
                pcpa_key =  vid_n + '_pid_' + ped_id + '_fr_' + frm
                
                if pcpa_key + '.pkl' in ped_ids:
                    self.pcpa_data[pcpa_key] = [pcpa_res[id_num], pcpa_lab[id_num]]
            list_k = list(self.pcpa_data.keys())

            filt_list =  lambda x: x.split('.')[0] in list_k
            ped_ids = list(filter(filt_list, ped_ids))

        self.ped_data = {}

        for ped_id in tqdm(ped_ids, desc=f'loading {data_set} data in memory'):
            
            ped_path = self.data_path.joinpath(ped_id).as_posix()
            loaded_data = self.load_data(ped_path)
            img_file = self.imgs_path.joinpath(loaded_data['crop_img'].stem + '.pkl').as_posix()
            loaded_data['crop_img'] = self.load_data(img_file)

            if balance: # b stands for behavior
                if 'b' not in ped_id:               # irrelevant
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

        for i in range(n_rep):
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

        if self.forcast:
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
        img = ped_data['crop_img'][1:].transpose(1, 2, 0).copy()
        img = self.transforms(img).contiguous()
        if self.seg:
            img = torch.cat([seg_map, img], 0)
        vel = torch.from_numpy(np.tile(ped_data['vehicle_act'], [1, 2]).transpose(1, 0)).float().contiguous()
        vel = vel[:, :-30] # assert only 32 samples 

        # 0 for no crossing,  1 for crossing, 2 for irrelevant
        idx = -2 if self.balance else -1 
        if 'b' not in ped_id.split('-')[idx]: # if is irrelevant
            bh = torch.from_numpy(np.ones(1).reshape([1])) * 2 
            # bh = torch.from_numpy(np.ones(1).reshape([1])) * 0 
        else:                               # if is crosing or not
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

    data_path = './data/JAAD'
    jaad_path = './JAAD'

    transform = A.Compose(
        [
        A.ToTensor(), 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    
    tr_data = DataSet(path=data_path,  jaad_path=jaad_path, data_set='train', frame=True, vel=True, balance=True, bh='all', h3d=False, transforms=transform)
    iter_ = tqdm(range(len(tr_data)))
    labels = np.zeros([len(tr_data), 3])
    fs = np.zeros([len(tr_data), 192, 64])

    for i in iter_:
        x, y, f,v = tr_data.__getitem__(i)
        fs[i] = f[0]
        labels[i, y.long().item()] = 1
            
    print(f'No Crossing: {int(labels.sum(0)[0])}, Crossing: {int(labels.sum(0)[1])}, Irrenlevants : {int(labels.sum(0)[2])} ')
    print('finish')

if __name__ == "__main__":
    main()
