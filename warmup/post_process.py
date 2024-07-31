import torch
import torch.nn.functional as F
from skimage import measure
import numpy as np
from sklearn.cluster import DBSCAN
import skimage
from scipy import spatial
import networkx
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import pdb


def connected_components(binary_mask,pos,dx,rgb, dino=None, pos_thresh = None, rgb_thresh = None, traj_thresh = None, use_dbscan=False):
    #binary_mask[H,W,D]
    #dx [T, 3, H, W,D]
    #rgb[3,H,W,D]
    #pos[3,h,w,d]
    #dino[H,W,D]
    rgb = (rgb * 255).astype(np.uint8)
    num_nodes = np.sum(binary_mask > 0)
    print(rgb.shape, dx.shape,pos.shape)
    H,W,D = dino.shape
  

    idx2num = (np.zeros(binary_mask.shape) -1).astype(np.int32)
    idx2num[binary_mask == 1] = np.arange(num_nodes)

    
    
    srcs = []
    tgts = []
    unique_labels = np.unique(dino)
    for label in unique_labels:
        if not np.logical_and(binary_mask, dino == label).any():
            continue

        if use_dbscan:
            rgb_label = np.zeros([H,W,D])-1
            clustering = DBSCAN(eps=rgb_thresh, min_samples=5).fit(rgb[:,np.logical_and(binary_mask, dino == label)].transpose(1,0).astype(np.float32))
            rgb_label[np.logical_and(binary_mask, dino == label)] = clustering.labels_


        obj_x, obj_y, obj_z = np.nonzero(np.logical_and(binary_mask, dino == label))
        pos_obj = pos[:, obj_x, obj_y, obj_z].transpose(1,0)  #[N,3]
        print(pos_obj.shape)

        kdtree = cKDTree(pos_obj)
        for i in range(pos_obj.shape[0]):
            # 查询每个点在给定距离内的近邻
            indices = np.array(kdtree.query_ball_point(pos_obj[i], pos_thresh))  
            valid_indices = indices[indices != i]
          

            # 过滤掉自环
            
            if valid_indices.size > 0:
              
                rgb_dist = np.sqrt(((rgb[:, obj_x[i], obj_y[i], obj_z[i]][:,None] - 
                            rgb[:, obj_x[valid_indices], obj_y[valid_indices], obj_z[valid_indices]])**2).sum(0))
                rgb_dist = rgb_dist.reshape([-1])
               

                traj_cur = pos[:,obj_x[i], obj_y[i],obj_z[i]][None,...] + dx[:,:, obj_x[i], obj_y[i], obj_z[i]]
                traj_neighbor = pos[:, obj_x[valid_indices], obj_y[valid_indices], obj_z[valid_indices]][None,...] +  dx[:, :,obj_x[valid_indices], obj_y[valid_indices], obj_z[valid_indices]]



                dx_dist = np.sqrt(((traj_cur[...,None] - 
                            traj_neighbor)**2).sum(1))
                dx_dist = dx_dist.max(0)
               

             
                if use_dbscan:
                    valid_indices = valid_indices[np.logical_and(rgb_label[obj_x[i],obj_y[i],obj_z[i]] == rgb_label[obj_x[valid_indices],obj_y[valid_indices],obj_z[valid_indices]], dx_dist < traj_thresh)]
                else:
                    valid_indices = valid_indices[np.logical_and(rgb_dist < rgb_thresh, dx_dist < traj_thresh)]
                
                

                
            
                u = idx2num[obj_x[i], obj_y[i], obj_z[i]]
                vs = idx2num[obj_x[valid_indices], obj_y[valid_indices], obj_z[valid_indices]]
 
                srcs.extend([u] * vs.size)
                tgts.extend(vs.tolist())
    
        
    
    g = networkx.Graph()

    
    g.add_edges_from(zip(srcs,tgts))
    g.add_edges_from(zip(list(range(num_nodes)),list(range(num_nodes))))

 
    labels_ = np.zeros([num_nodes],dtype = np.int32)
    compnts = networkx.connected_components(g)
    for (label, node_list) in enumerate(compnts):
        for node in node_list:
            labels_[node] = label + 1
    
    labels = np.zeros(binary_mask.shape,dtype = np.int32)
    labels[binary_mask > 0] = labels_


    return labels

#connected_components

def post_process(density, act_shift,num_slots,pos, dx,rgb,split_val_nonprune, thresh = 1e-3,grad = None, dino_label = None,voxel_size = None, dataset_type = 'dnerf', cluster_args = None):
   
    assert density.shape[1] == 1
    density = density[0,0]  #[X,Y,Z]
    density = F.softplus(density + act_shift)
    density = density.detach().cpu().numpy()


    if dino_label is not None:
        dino_label = dino_label.copy()
        dino_hard_label = dino_label.argmax(-1)  #[x,y,z]
    else:
        dino_hard_label = np.zeros_like(density)

        
    torch.set_default_tensor_type(torch.FloatTensor)
    binary_mask = np.logical_and(density >= thresh, grad > split_val_nonprune)


    if dataset_type =='dnerf':
        pos_thresh = 2*voxel_size
        rgb_thresh = 8  #8 by default 15 for metal&3realcmpx
        traj_thresh = 5*voxel_size
        use_dbscan =False   #True for 3fall3still else False
    else:
        pos_thresh = cluster_args.pos_thresh *voxel_size
        rgb_thresh = cluster_args.rgb_thresh
        traj_thresh = cluster_args.traj_thresh * voxel_size
        use_dbscan = cluster_args.use_dbscan

  
  
    labels= connected_components(binary_mask,pos, dx,rgb, dino = dino_hard_label, pos_thresh = pos_thresh, rgb_thresh = rgb_thresh, traj_thresh = traj_thresh, use_dbscan = use_dbscan)
    new_density = np.zeros([1, num_slots, *density.shape])

    #process background
    bg = np.where(labels == 0 )
    base_density = (density[bg[0], bg[1], bg[2]] / num_slots)[:, None].repeat(num_slots,axis = 1)
    base_density[base_density>1e-4] = 1e-4
    new_density[0,:,bg[0], bg[1], bg[2]] = base_density

    size = np.zeros([labels.max()])  
    print(labels.max())
    
    for i in range(1, labels.max() + 1):
        
        size[i-1] =(binary_mask[labels == i] *  (grad[labels == i]**2)).sum()#/  np.sum(labels == i)
        if np.sum(labels == i) < 10:
            size[i-1] = -1000
            

        
            
    print(np.sum(size>=0))

   
    sort_idx = np.argsort(size)[::-1] + 1  # 直接调整为标签值

    # 重新标记labels，将最大的size对应的标签设置为1，第二大的为2，以此类推
    # 直接创建新标签数组，而不在循环中修改labels
    new_labels = np.zeros_like(labels)  # 初始化为0

    # 构建一个从旧标签到新标签的映射
    for new_label, old_label in enumerate(sort_idx, start=1):
        new_labels[labels == old_label] = new_label
    labels = new_labels.copy()
 
    
    
    #nearest interpolation for each dino label
    for i in np.unique(dino_hard_label):
        labels_nearest = np.ones_like(labels)
        labels_copy = labels.copy()
        labels_nearest[np.logical_and(dino_hard_label==i, labels_copy <= num_slots)] = 0

        #与1距离最近的0的indices,为1做插值
        distance ,indices = ndimage.distance_transform_edt(labels_nearest,return_indices = True)
        x,y,z = indices[:,...]

        print(distance.max())

  
        labels_copy = labels_copy[x,y,z]
        
        labels[np.logical_and( dino_hard_label==i, distance < 150)] = labels_copy[np.logical_and( dino_hard_label==i, distance < 150)]


    #in case that some dino label does not have nearest labels
    labels_nearest = labels.copy()
    labels_copy = labels.copy()
    labels_nearest[labels_nearest>num_slots] = 0
    labels_nearest = 1 - (labels_nearest >0)
    _,indices = ndimage.distance_transform_edt(labels_nearest,return_indices = True)
    x,y,z = indices[:,...]
    labels_copy = labels_copy[x,y,z]
    labels[labels>-1] = labels_copy[labels > -1]

   
    print(np.unique(labels))
       

    masks = torch.from_numpy(labels-1).long()
    masks = F.one_hot(masks).permute(3,0,1,2).unsqueeze(0)
    print(masks.shape)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    #label smoothing
    masks = masks * 0.9 + 0.1 * torch.ones_like(masks) / num_slots
    return masks
   


