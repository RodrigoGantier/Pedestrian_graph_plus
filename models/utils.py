""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np


class Graph():
    """ by defauld uses spatial mode"""
    def __init__(self, nodes=19):

        self.num_node = nodes
        self.self_link = [(i, i) for i in range(self.num_node)]
        
        self.inward_ori_index = [(0, 1), (0, 2), (1, 3), (2, 4),  
                    (17, 5), (17, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                    (18, 11), (18, 12),  
                    (11, 13), (12, 14), (13, 15), (14, 16),
                    (0, 17), (17, 18)]
        if nodes == 22:
            self.inward_ori_index.extend([(16, 19),(15, 20), (18, 21)])
        self.inward = [(i, j) for (i, j) in self.inward_ori_index]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        
        self.A = self.get_spatial_graph()
    
    def get_spatial_graph(self):
        
        I = self.edge2mat(self.self_link)
        In = self.normalize_digraph(self.edge2mat(self.inward))
        Out = self.normalize_digraph(self.edge2mat(self.outward))
        A = np.stack((I, In, Out))
        return A

    def edge2mat(self, link):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in link:
            A[j, i] = 1
        return A

    def normalize_digraph(self, A):  # 除以每列的和
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    


def get_data(dataset, data_path,cutout_length, validation):
    """ Get JAAD or PIE dataset """
    
    if data_path.split('_')[0] == 'jaad':
        from jaad_dataloader2 import DataSet
    elif data_path.split('_')[0] == 'pie':
        # from pie_dataloader2 import DataSet
        from pie_bert_loader import DataSet
    
    trn_data = DataSet(data_path, image_set='train')
    
    # the shape is NCHW 
    batch = trn_data.__getitem__(0)
    _, _, ped_keypoits, behavios = tuple(batch[k] for k in batch.keys())
    input_channels = ped_keypoits.shape[0]
    input_size = trn_data.input_size
    n_classes = behavios.shape[-1] 
    

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        ret.append(DataSet(data_path, image_set='test'))

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

def accuracy_mul(logits, target, thr=0.5):
    pred = torch.sigmoid(logits)                                                                                                         
    pred = pred.ge(thr).float()
    accu = torch.mean(target.eq(pred).float()) 
    return accu

def accuracy_cross_(logits, target):
    pred = logits.softmax(1).argmax(1)
    pred = torch.clamp_max(pred, 1)
    y = torch.clamp_max(target.argmax(1), 1)
    accu = torch.mean(y.eq(pred).float())
    return accu

def accuracy_cross_2(logits, target):
    pred = logits.softmax(1).argmax(1)
    pred[pred==3] = 0
    pred = torch.clamp_max(pred, 1)
    y = target.argmax(1)
    y[y==3] = 0
    y = torch.clamp_max(y, 1)
    accu = torch.mean(y.eq(pred).float())
    return accu

def accuracy_cross(logits, target):
    pred = torch.softmax(logits, 1).argmax(1) 
    # pred = logits.argmax(1)
    accu = torch.mean(target.argmax(1).eq(pred).float())
    return accu

def accuracy_position(logits, target, p):
    pred = torch.sigmoid(logits)
    y = target[:, p:p + 1]  
    pred = pred[:, p:p + 1].ge(0.5).float()
    accu = torch.mean(pred.eq(y).float()) 
    return accu

def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


class lr_schedule:
    def __init__(self, max_v, min_v, len_v, cicle=1, invert=False):
        
        self.max_v = max_v
        self.min_v = min_v
        self.len_v = len_v
        self.cicle = cicle
        self.invert = invert

        self.lr_sch = self.init_sch()
        self.current_step = 0
    
    def init_sch(self):

        flen = self.len_v
        len_v = self.len_v
        len_v /= self.cicle

        lr_sch0 = np.cos(np.arange(np.pi, 2*np.pi, np.pi / (len_v * 0.1)))
        lr_sch1 = np.cos(np.arange(0, np.pi, np.pi / (len_v * 0.9)))

        lr_sch0 = (lr_sch0 + 1) / 2
        lr_sch0 = (self.max_v - self.min_v) * lr_sch0
        lr_sch0 += self.min_v

        lr_sch1 = (lr_sch1 + 1) / 2
        lr_sch1 = lr_sch1 * self.max_v

        lr_sch = np.concatenate([lr_sch0, lr_sch1])

        if self.invert:
            lr_sch = np.cos(lr_sch)

        lr_sch = np.tile(lr_sch, [self.cicle])

        return lr_sch[:flen]

    def step(self):
        try:        
            lr = self.lr_sch[self.current_step]
            self.current_step += 1
            return lr
        except IndexError:
            return self.lr_sch[-1]

    def reset(self):
        self.len_v -= self.current_step
        self.lr_sch = self.init_sch()
        self.current_step = 0 
    
    def lr(self):
        return self.lr_sch[self.current_step]
