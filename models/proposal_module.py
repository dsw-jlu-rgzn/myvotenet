# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils
from gmm_conv import GMMConv
def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:,:,5:5+num_heading_bin]
    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.predict_center = torch.nn.Conv1d(128,3,1)#+self.num_class
        self.predict_sem = torch.nn.Conv1d(128, self.num_class, 1)
        self.relation_fc_1 = torch.nn.Conv1d(128,128,1)
        self.relation_fc_2 = torch.nn.Conv1d(128,128,1)
        self.gaussian = GMMConv(128, 128, dim=3, kernel_size=25)
        self.sg_conv_1 = torch.nn.Conv1d(128,256,1)
        self.sg_conv_2 = torch.nn.Conv1d(256,128,1)
        self.conv3_5 = torch.nn.Conv1d(256,128,1)
        self.bn3 = torch.nn.BatchNorm1d(128)


    def _region_classification(self, features, base_xyz):
        residual_center = self.predict_center(features)# (batch_size, 128, 256)
        cls_score = self.predict_sem(features) # (b, c, 256)
        residual_center = residual_center.transpose(2, 1)  # (batch_size, 256, ..)
        cls_score = cls_score.transpose(2, 1)
        center = base_xyz + residual_center  # (batch_size, num_proposal, 3)
        cls_prob = F.softmax(cls_score,dim=2)
        return cls_prob, center
    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        device = features.device
        batch_size = features.shape[0]
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))#b*128*256
        cls_prob, center_pred = self._region_classification(net, xyz)#b, 256, 18  | b, 256, 3
        z = self.relation_fc_1(net)#8*128*256
        z = F.relu(self.relation_fc_2(z))
        z = z.transpose(2,1)#8*256*128
        eps = torch.bmm(z, z.t())
        _, indices = torch.topk(eps, k=16, dim=1)# 16, 256
        cls_w = self.predict_sem.weight.unsqueeze(0),squeeze(-1).repeat(batch_size, 1, 1)
        represent = torch.bmm(cls_prob, cls_w)
        cls_pred = torch.max(cls_prob,2)[1]
        relation = torch.empty(batch_size, 2, 16*256, dtype=torch.long).to(device)
        relation[:, 0] = torch.Tensor(list(range(256)) * 16).unsqueeze(0).repeat(batch_size,1)
        relation[:, 1] = indices.view(batch_size, -1)
        # coord_i, coord_j = torch.zeros(batch_size, 16*256, 3), torch.zeros(batch_size, 16*256, 3)
        # coord_i = center_pred[relation[:,0]]
        f = torch.zeros_like(z)
        for batch_id in range(batch_size):
            center_ = center_pred[batch_id]
            relation_ = relation[batch_id]
            # coord_i[batch_id] = center_[relation_[0]]
            # coord_j[batch_id] = center_[relation_[1]]
            coord_i = center_[relation_[0]]
            coord_j = center_[relation_[1]]
            d = torch.sqrt((coord_i[:, 0] - coord_j[:, 0]) ** 2 + (coord_i[:, 1] - coord_j[:, 1]) ** 2 + (coord_i[:, 2] - coord_j[:, 2]) ** 2)
            theta_y = torch.atan2((coord_j[:, 1] - coord_i[:, 1]), (coord_j[:, 0] - coord_i[:, 0]))
            theta_z = torch.atan2((coord_j[:, 2] - coord_i[:, 2]), (coord_j[:, 0] - coord_i[:, 0]))
            U = torch.stack([d, theta_y, theta_z], dim=1).to(device)
            f[batch_id] = self.gaussian(represent[batch_id], relation[batch_id], U)
        f2 = F.relu(self.sg_conv_1(f.transpose(2,1)))
        h = F.relu(self.sg_conv_2(f2))
        new_net = torch.cat([net, h],dim=1)
        new_net =  F.relu(self.bn3(self.conv3_5(new_net)))
        new_net = self.conv3(new_net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(new_net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
