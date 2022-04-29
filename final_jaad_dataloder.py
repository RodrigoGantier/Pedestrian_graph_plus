import torch
import torch.utils.data as data
import os
import pickle5 as pk
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from torchvision import transforms as A
import torch.nn.functional as F

from tqdm import tqdm, trange
from jaad_data import JAAD


class DataSet(data.Dataset):
    def __init__(self, path, jaad_path, frame, vel, balance=True, bh='all', t23=False, transforms=None, seg_map=False, h3d=True, pcpa=None, forecast=True, last2=False, time_crop=False):

        np.random.seed(42)
        self.time_crop = time_crop
        self.forecast = forecast
        self.last2 = last2
        self.h3d = h3d # bool if true 3D human keypoints data is enable otherwise 2D is only used
        self.bh = bh
        self.t23 = t23
        self.seg = seg_map
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
        
        nsamples = [1871, 3204, 13037]
        balance_data = [max(nsamples) / s for s in nsamples]
        if t23:
            balance_data = [1, (nsamples[0] + nsamples[2])/nsamples[1], 1]

        self.data_path = os.getcwd() / Path(path) / 'data'
        self.imgs_path = os.getcwd() / Path(path) / 'imgs'
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]
        self.jaad_path = jaad_path
        
        imdb = JAAD(data_path=self.jaad_path)
        self.vid_ids = imdb._get_video_ids_split(self.data_set)
        
        filt_list =  lambda x: "_".join(x.split('_')[:2]) in self.vid_ids
        ped_ids = list(filter(filt_list, self.data_list))

        if bh != 'all':
            filt_list =  lambda x: 'b' in x 
            ped_ids = list(filter(filt_list, ped_ids))

        pcpa_, dense_, fussi_ = self.load_3part()
        
        self.models_data = {}

        for k_id in pcpa_['ped_id'].keys():
            vid_n = int(k_id.split('_')[1])
            vid_n = f'video_{vid_n:04}' 
            ped_id = k_id.split('fr')[0]
            frm = k_id.split('fr')[1]

            pcpa_key =  vid_n + '_pid_' + ped_id + '_fr' + frm
            try:
                dense_res = dense_['result'][k_id]
                dense_lab = dense_['labels'][k_id]
                fussi_res = fussi_['result'][k_id]
                fussi_lab = fussi_['labels'][k_id]
                pcpa_res = pcpa_['result'][k_id]
                pcpa_lab = pcpa_['labels'][k_id]
                assert dense_lab == fussi_lab == pcpa_lab
            except KeyError:
                continue

            pcpa_dict = {'result': pcpa_res, 'label': pcpa_lab}
            dense_dict = {'result': dense_res, 'label': dense_lab}
            fussi_dict = {'result': fussi_res, 'label': fussi_lab}
            if pcpa_key + '.pkl' in ped_ids:
                self.models_data[pcpa_key] = [pcpa_dict, dense_dict, fussi_dict]
        list_k = list(self.models_data.keys())

        filt_list =  lambda x: x.split('.')[0] in list_k
        ped_ids = list(filter(filt_list, ped_ids))

        

        self.ped_data = {}
        for ped_id in tqdm(ped_ids, desc=f'loading {self.data_set} data in memory'):
            
            ped_path = self.data_path.joinpath(ped_id).as_posix()
            loaded_data = self.load_data(ped_path)
            img_file = str(self.imgs_path / loaded_data['crop_img'].stem) + '.pkl'
            loaded_data['crop_img'] = self.load_data(img_file)

            if balance:
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
    
    def load_3part(self, ):
        pcpa, fussi, g_pcpca = {}, {}, {}
        mod = 'bh' if self.bh == 'bh' else 'all'
        
        if self.last2:
            pcpa['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_preds_jaad_{mod}_last2.pkl')
            pcpa['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_labels_jaad_{mod}_last2.pkl')
            pcpa['ped_id'] = self.load_data(self.pcpa.__str__() + f'/test_results/jaad/pcpa_ped_ids_jaad_{mod}_last2.pkl')
            
            fussi['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_preds_jaad_last2.pkl')
            fussi['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_labels_jaad_last2.pkl')
            fussi['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_ped_ids_jaad_last2.pkl')
            
            g_pcpca['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpa_preds_jaad_{mod}_last2.pkl')
            g_pcpca['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpa_labels_jaad_{mod}_last2.pkl')
            g_pcpca['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpa_ped_ids_jaad_{mod}_last2.pkl')
        
        else:
            pcpa['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_preds_jaad_{mod}.pkl')
            pcpa['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_labels_jaad_{mod}.pkl')
            pcpa['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_ped_ids_jaad_{mod}.pkl')

            fussi['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_preds_jaad.pkl')
            fussi['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_labels_jaad.pkl')
            fussi['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_ped_ids_jaad.pkl')

            g_pcpca['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpca_preds_jaad_{mod}.pkl')
            g_pcpca['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpca_labels_jaad_{mod}.pkl')
            g_pcpca['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpca_ped_ids_jaad_{mod}.pkl')
        
        return pcpa, fussi, g_pcpca

    def repet_data(self, n_rep, data, ped_id):
        ped_id = ped_id.split('.')[0]

        if self.data_set == 'train' or self.data_set == 'val' or self.t23:
            prov = n_rep % 1  
            n_rep = int(n_rep) if prov == 0 else int(n_rep) + np.random.choice(2, 1, p=[1 - prov, prov])[0]
            # n_rep = int(n_rep * 2)
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
        pcpa_dict, dense_dict, g_pcpca_dict = self.models_data[ped_id.split('.')[0].split('-')[0]]
        models_data = {'pcpa': pcpa_dict, 'fussi': dense_dict, 'g_pcpca': g_pcpca_dict}
        ped_data = self.ped_data[ped_id]

        weather_ = ped_data['weather'] 
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

        vel = torch.from_numpy(np.tile(ped_data['vehicle_act'], [1, 2]).transpose(1, 0)).float().contiguous()
        if not self.forecast:
            vel = vel[:, :-30]
            
        if np.random.randint(10) >= 9 and self.time_crop:
            crop_size = np.random.randint(2, 21)
            kp = kp[:crop_size]
            vel = vel[:crop_size]
        
        # 0 for no crossing,  1 for crossing, 2 for irrelevant
        idx = -2 if self.balance else -1 
        if 'b' not in ped_id.split('-')[idx]: # if is irrelevant
            bh = torch.from_numpy(np.ones(1).reshape([1])) * 2 
        else:                               # if is crosing or not
            bh = torch.from_numpy(ped_data['crossing'].reshape([1])).float()
        
        if not self.h3d:
            kp = kp[[0, 1, 3], ].clone()
            
        if self.frame and not self.vel:
            return kp, bh, img, weather_, models_data
        elif self.frame and self.vel:
            return kp, bh, img, vel, weather_, models_data
        else:
            return kp, bh, weather_, models_data

def main():
    
    data_path = './data/JAAD'
    jaad_path = './JAAD'
    pcpa = './data/test_results'

    transform = A.Compose([
        A.ToTensor(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    
    tr_data = DataSet(path=data_path,  jaad_path=jaad_path, frame=True, vel=True, balance= False, bh='all', transforms=transform, pcpa=pcpa)
    iter_ = tqdm(range(len(tr_data)))
    labels = np.zeros([len(tr_data), 3])

    for i in iter_:
        x, y, f, v, pcpa = tr_data.__getitem__(i)
        
        labels[i, y.long().item()] = 1
            
    print(f'No Crossing: {int(labels.sum(0)[0])}, Crossing: {int(labels.sum(0)[1])}, Irrenlevants : {int(labels.sum(0)[2])} ')
    print('finish')

if __name__ == "__main__":
    main()
