
import torch
import torch.nn as nn
#import numpy as np
from models.pointnet import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer, MLP_Stacks
from models.m_dconv import SurfaceConstrainedUp
from pugcn_lib.feature_extractor import InceptionFeatureExtractor
from copy import deepcopy
# pugcn_feature_info = channels = 64
# k = 16
# dilations = [1, 2, 4]
# n_idgcn_blocks = 3
# n_dgcn_blocks = 3
# use_radius_graph = False
# idgcn_kwargs = {'use_bottleneck': True, 'use_pooling': True, 'use_residual': True, 'conv': 'edge', 'pool_type': 'mean', 'dynamic': False, 'hierarchical': False}



# class FeatureExtractor(nn.Module):
#     def __init__(self, out_dim=1024):
#         '''
#         Extract information from partial point cloud
#         '''
#         super(FeatureExtractor, self).__init__()
#         self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [32, 64], group_all=False, if_bn=False, if_idx=True)
#         self.transformer_1 = Transformer(64, dim=64)
#         self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 64, [128, 256], group_all=False, if_bn=False, if_idx=True)
#         self.transformer_2 = Transformer(256, dim=64)
#         #self.sa_module_3 = PointNet_SA_Module_KNN(128, 16, 256, [256, 384], group_all=False, if_bn=False, if_idx=True)
#         #self.transformer_3 = Transformer(384, dim=64)
#         self.sa_module_4 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

#     def forward(self, point_cloud):
#         """
#         Args:
#              point_cloud: b, 3, n

#         Returns:
#             l3_points: (B, out_dim, 1)
#         """
#         l0_xyz = point_cloud
#         l0_points = point_cloud

#         l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
#         l1_points = self.transformer_1(l1_points, l1_xyz)
#         l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)
#         l2_points = self.transformer_2(l2_points, l2_xyz)
#         #l3_xyz, l3_points, idx3 = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 32), (B, 384, 32)
#         #l3_points = self.transformer_3(l3_points, l3_xyz)
#         l4_xyz, l4_points = self.sa_module_4(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)
#         return l4_points, l2_xyz, l2_points


class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024):
        '''
        Extract information from partial point cloud
        '''
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [32, 64], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(64, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 64, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_4 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

        channels = 64
        k = 16
        dilations = [1, 2, 4]
        n_idgcn_blocks = 3
        n_dgcn_blocks = 3
        use_radius_graph = False
        idgcn_kwargs = {'use_bottleneck': True, 'use_pooling': True, 'use_residual': True, 'conv': 'edge', 'pool_type': 'mean', 'dynamic': False, 'hierarchical': False}

        self.feature_extractor_pugcn = InceptionFeatureExtractor(
                                channels=channels,
                                k=k,
                                dilations=dilations,
                                n_idgcn_blocks=n_idgcn_blocks,
                                n_dgcn_blocks=n_dgcn_blocks,
                                use_radius_graph=use_radius_graph,
                                **idgcn_kwargs,
        )

    def forward(self, pcd_bnc, mode=None):
        """
        Args:
             pcd_bnc: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        if mode=='x':
            x_min = pcd_bnc[:, :, 0].min()
            pcd_bnc[:, :, 0] = x_min
        if mode=='y':
            y_min = pcd_bnc[:, :, 1].min()
            pcd_bnc[:, :, 1] = y_min
        if mode=='z':
            z_min = pcd_bnc[:, :, 2].min()
            pcd_bnc[:, :, 2] = z_min
        l1_feature = []
        for idx in range(pcd_bnc.shape[0]):
            x = pcd_bnc[idx]
            pos = x
            x, _ = self.feature_extractor_pugcn(
                x, pos=pos, batch=None, return_index=True
            )
            x = x[None, ...]
            l1_feature.append(x)
        l1_points = torch.cat(l1_feature, dim=0).permute(0, 2, 1).contiguous()
        l1_xyz = pcd_bnc.permute(0, 2, 1).contiguous()
        l1_points = self.transformer_1(l1_points, l1_xyz) # [16, 64, 2048]
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l4_xyz, l4_points = self.sa_module_4(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)
        return l4_points, l2_xyz, l2_points



class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=128):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=256, out_dim=256)
        self.mlp_2 = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1)) # (b, 256, 128)
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion,x3




class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        #self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])

        #SurfaceConstrainedUp(self,upscale,manifold_mlp)
        self.deconv1 = SurfaceConstrainedUp(1, MLP_Stacks([512+3+128,512,256,64]))
        self.deconv2 = SurfaceConstrainedUp(4, MLP_Stacks([512+3+64+128,512,256,64]))
        #self.deconv3 = SurfaceConstrainedUp(8, MLP_Stacks([512+3+64+128,512,128,64]))


    def forward(self, global_shape_fea, pcd,return_P0=False):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
#        seeds, seeds_fea = self.decoder_coarse(global_shape_fea)
#        #pcd = self.decoder_coarse(global_shape_fea).permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
#        p0 = fps_subsample( partial, self.num_p0)
#        pcd = fps_subsample(torch.cat([seeds.permute(0, 2, 1), p0], 1), self.num_p0)
        pcd = pcd.permute(0, 2, 1).contiguous()
        #print(pcd.size())
        feat_1 = self.mlp_1(pcd)
        #min_d = min_dist(pcd)
#        min_v,min_i = torch.min(min_d,1) # (B, N)
#        #r = torch.max(min_v)
#        r = torch.mean(min_v,1) #(B, )
#        r = 0.25

        p1,f1,mf1,rot_1,z1 = self.deconv1(global_shape_fea, pcd,feat_1,None)
        #min_d1 = min_dist(p1)
        p2,f2,mf2,rot_2,z2 = self.deconv2(global_shape_fea, p1,f1, mf1)
        #min_d2 = min_dist(p2)
        #p3,f3,mf3,rot_3,z3 = self.deconv3(global_shape_fea, p2,f2, mf2)

        pcd = pcd.permute(0, 2, 1).contiguous()
        p1=p1.permute(0, 2, 1).contiguous()
        p2=p2.permute(0, 2, 1).contiguous()
        #p3=p3.permute(0, 2, 1).contiguous()

        z = torch.cat([z1,z2],-1) # N+4*N+16*N by 1

        return pcd,p1,p2,rot_2,z


class Upsample_Net(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, rad=1, up_factors=None):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_pc: int
            num_p0: int
            radius: searching radius
            up_factors: list of int
        """
        super(Upsample_Net, self).__init__()

        self.feat_extractor = FeatureExtractor(out_dim=dim_feat)
        self.feat_extractor_x = FeatureExtractor(out_dim=dim_feat)
        self.feat_extractor_y = FeatureExtractor(out_dim=dim_feat)
        self.feat_extractor_z = FeatureExtractor(out_dim=dim_feat)
        self.decoder = Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0)
        self.fuse = MLP_CONV(2048, [2048, 1024, 1024, 512, 512])
    def forward(self, point_cloud, return_P0=False):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        pcd_bnc = point_cloud
        pcd_bnc_x = deepcopy(point_cloud)
        pcd_bnc_y = deepcopy(point_cloud)
        pcd_bnc_z = deepcopy(point_cloud)
        # import pdb;pdb.set_trace()
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        shape_code, parent_pos, parent_fea = self.feat_extractor(pcd_bnc)
        shape_code_y, _, _ = self.feat_extractor_y(pcd_bnc_x, "y")
        shape_code_x, _, _ = self.feat_extractor_x(pcd_bnc_y, "x")
        shape_code_z, _, _ = self.feat_extractor_z(pcd_bnc_z, "z")
        shape_code = torch.cat((shape_code, shape_code_x, shape_code_y, shape_code_z), 1)
        shape_code = self.fuse(shape_code)
        pcd,p1,p2,rot3, z = self.decoder(shape_code, pcd_bnc)
        return pcd,p1,p2,z,rot3,shape_code


    def _load_state_dict(self, params_dict):

        for k, v in params_dict.items():
            k = k.replace("module.","")
            if "feat_extractor" in k:
                if "feature_extractor_pugcn" in k:
                    continue
                k_x = k.replace("feat_extractor", "feat_extractor_x")
                k_y = k.replace("feat_extractor", "feat_extractor_y")
                k_z = k.replace("feat_extractor", "feat_extractor_z")
                self.state_dict()[k].copy_(v)
                self.state_dict()[k_x].copy_(v)
                self.state_dict()[k_y].copy_(v)
                self.state_dict()[k_z].copy_(v)
            elif "feature_extractor_pugcn" in k:
                ...
            else:
                if k in self.state_dict().keys():
                    self.state_dict()[k].copy_(v)
                else:
                    print("===> Miss key:", k)
