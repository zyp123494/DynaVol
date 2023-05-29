import torch
import torch.nn.functional as F
from skimage import measure
import numpy as np
from sklearn.cluster import DBSCAN
import skimage

import dgl
import networkx
from tqdm import tqdm

#density thresh 1e-2, dx_thresh 0.08, rgb= 0.06 by default , rgb = 0.3 for metal
def connected_components(binary_mask,dx,rgb, dx_thresh= 0.08, rgb_thresh = 0.06):
    #binary_mask[H,W,D]
    #dx [T, 3, H, W,D]
    #rgb[3,H,W,D]
    #This algorithm is not so sensitive to hyperparameters, and usually a value between the mean and median is a good choice.
    num_nodes = np.sum(binary_mask > 0)
    print(rgb.shape)
    g = dgl.graph([],num_nodes = num_nodes)
    
    idx2num = (np.zeros(binary_mask.shape) -1).astype(np.int32)
    idx2num[binary_mask == 1] = np.arange(num_nodes)
    
    obj_x, obj_y, obj_z = np.nonzero(binary_mask)
    srcs = []
    tgts = []
    dx_dists = []
    rgb_dists = []
    
    for i in tqdm(range(obj_x.shape[0])):
        x, y, z = obj_x[i], obj_y[i], obj_z[i]
        src = idx2num[x,y,z]
        dx_src = dx[:,:,x,y,z]
        rgb_src = rgb[:, x, y, z]
        for u in range(x-1,x+1):
            for v in range(y-1, y+1):
                for k in range(z-1,z+1):
                    tgt = idx2num[u,v,k]
                    if tgt ==-1 or tgt==src:
                        continue
                    dx_tgt = dx[:,:,u,v,k]  #[T,3]

                    dx_dist = np.sqrt(((dx_tgt - dx_src)**2).sum(1)).max()
                    dx_dists.append(dx_dist)

                    rgb_tgt = rgb[:, u, v, k]
                    rgb_dist = np.sqrt(((rgb_src - rgb_tgt)**2).sum())

                    rgb_dists.append(rgb_dist)

                    if dx_dist < dx_thresh and rgb_dist< rgb_thresh:
                        srcs.append(src)
                        tgts.append(tgt)
              
 
    g = dgl.add_edges(g,srcs,tgts)
    
    ngx = dgl.to_networkx(g)
    ngx = ngx.to_undirected()
   
    labels_ = np.zeros([num_nodes],dtype = np.int32)
    compnts = networkx.connected_components(ngx)
    for (label, node_list) in enumerate(compnts):
        for node in node_list:
            labels_[node] = label + 1
    
    labels = np.zeros(binary_mask.shape,dtype = np.int32)
    labels[binary_mask > 0] = labels_
    print(max(dx_dists),min(dx_dists), sum(dx_dists)/len(dx_dists))

    dx_dists.sort()
    mid = len(dx_dists) // 2
    res = (dx_dists[mid] + dx_dists[~mid]) / 2
    print(res)

    print(max(rgb_dists),min(rgb_dists), sum(rgb_dists)/len(rgb_dists))

    rgb_dists.sort()
    mid = len(rgb_dists) // 2
    res = (rgb_dists[mid] +rgb_dists[~mid]) / 2
    print(res)

    return labels

#connected_components

def post_process(density, act_shift,num_slots,dx,rgb,thresh = 1e-3,method = 'cc',hyper=False, grad = None):
   
    assert density.shape[1] == 1
    density = density[0,0]  #[X,Y,Z]
    density = F.softplus(density + act_shift,True)
    density = density.detach().cpu().numpy()
    torch.set_default_tensor_type(torch.FloatTensor)
   
    binary_mask = (density >= thresh)
    if method == 'cc':
       
        if hyper:
            
            labels = connected_components(binary_mask,dx,rgb,dx_thresh = 0.004,rgb_thresh = 0.15)   #printer_tune      
            
        else:
            labels = connected_components(binary_mask,dx,rgb)
        
    elif method =='dbscan':
        binary_mask_flatten  = binary_mask.reshape([-1])
        labels = np.zeros(binary_mask_flatten.shape)
        obj_idx = np.where(binary_mask==1)
        obj_idx = [x[:,None] for x in obj_idx]
        obj_idx = np.concatenate(obj_idx,axis = -1)
        clustering = DBSCAN(eps=3).fit(obj_idx)
        labels[binary_mask_flatten == 1] = clustering.labels_ - clustering.labels_.min() + 1
        print(labels.min())
        assert labels.min() == 0

        #make labels continuous
        unique_labels = np.unique(labels)
        new_label = np.zeros(labels.shape)
        for (idx, i ) in enumerate(unique_labels):
            new_label[labels == i] = idx
        labels = new_label.copy().astype(np.int64).reshape(binary_mask.shape)

    else:
        raise NotImplementedError
    new_density = np.zeros([1, num_slots, *density.shape])

    #process background
    bg = np.where(labels == 0 )
    base_density = (density[bg[0], bg[1], bg[2]] / num_slots)[:, None].repeat(num_slots,axis = 1)
    base_density[base_density>1e-4] = 1e-4
    new_density[0,:,bg[0], bg[1], bg[2]] = base_density

    size = np.zeros([labels.max()])  
    print(labels.max())
    
    if hyper:
        for i in range(1, labels.max() + 1):
            size[i-1] =(binary_mask[labels == i] *  (grad[labels == i]**2)).sum()
    else:
        for i in range(1, labels.max() + 1):
            size[i-1] =(binary_mask[None,labels == i] *  (1 -  rgb[:,labels == i])).sum()
    sort_idx = np.argsort(size)[::-1]

    

    count =0
    for i in range(sort_idx.shape[0]):
        new_density[0,count, labels == (sort_idx[i] + 1)] = density[labels == (sort_idx[i] + 1)]
        count += 1
        count = min(num_slots-1, count)


    new_density = torch.from_numpy(new_density)        
    masks = new_density / (new_density.sum(1,keepdim = True) + 1e-8)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    return masks
   


