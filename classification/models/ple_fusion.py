import torch
import torch.nn as nn
import torchmetrics
from einops import rearrange

from utils.metrics import feature_loss
from models import _3d_resnet

def Transformer(dim=512, depth=4, heads=4, mlp_dim=3072, dropout=0.1):
    transformer = nn.ModuleList([
        nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout,
                                   batch_first=True)
        for _ in range(depth)
    ])
    return nn.Sequential(*transformer)


def ResNet18(in_channels, mid_channels):
    return _3d_resnet.resnet18(in_channels=in_channels, mid_channels=mid_channels)


class PleTrans(nn.Module):
    def __init__(self, in_channels, embed_dim, num_phase, num_depth, num_classes, drop_rate=0.):
        super(PleTrans, self).__init__()
        self.specific_mask = None
        self.num_phase = num_phase
        self.in_channels = in_channels
        self.predict = None
        self.res_feature = None
        self.sne_feature = None

        self.emb_dim = embed_dim

        self.res_model = ResNet18(1, embed_dim)

        self.trans_model = Transformer(embed_dim * 4, num_depth, num_depth, num_depth * embed_dim, drop_rate)

        self.out = nn.Linear(embed_dim * 4 * (num_phase + 1), num_classes)

    def forward(self, x):
        # input [batchsize, phase, spatial, h, w]
        b, p, s, h, w = x.shape
        self.res_feature = self.res_model(x.reshape(b * p, 1, s, h, w)).reshape(b, p, -1)
        share_feature = torch.mean(self.res_feature[..., :self.emb_dim * 4], dim=1, keepdim=True)
        specific_feature = self.res_feature[..., self.emb_dim * 4:]
        trans_feature = self.trans_model(torch.cat([share_feature, specific_feature], 1))
        self.sne_feature = trans_feature.view([b, -1])
        self.predict = self.out(self.sne_feature)
        return self.predict

    def get_loss(self, gt, loss_fn=nn.CrossEntropyLoss(), tau=0.):
        # calculate losses
        cls_losses = loss_fn(self.predict, gt)

        share_feature_losses = feature_loss(self.res_feature[..., :self.emb_dim * 4])

        specific_gn = CcLoss(int(gt.shape[0]))
        specific_feature_losses = specific_gn(self.res_feature[..., self.emb_dim * 4:], tau)

        total_losses = cls_losses + specific_feature_losses + share_feature_losses
        return total_losses


class CcLoss(nn.Module):
    def __init__(self, outputs_):
        super(CcLoss, self).__init__()

        self.pro_loss = nn.MSELoss()
        self.metric = torchmetrics.PearsonCorrCoef(outputs_).cuda()

    def forward(self, features, tau):
        bs, phase, dims = features.shape
        # calculate cos-sim matrix [b, p, p]
        feature_norm = features / torch.norm(features, dim=-1, keepdim=True)
        prototype_features = torch.zeros_like(features)

        for i in range(bs):
            similarity = torch.mm(feature_norm[i], feature_norm[i].T)  # 矩阵乘法
            # Utilize tau binarization
            similarity = torch.where(similarity > tau, 1., 0.)
            # Update features and gain prototype feature
            prototype_features[i] = torch.mm(similarity, features[i]) / torch.sum(similarity, dim=1, keepdim=True)

        # gain prototype gt feature
        prototype_gt = torch.mean(prototype_features, dim=1, keepdim=True)

        # Calculate correlations with prototype feature
        feature_reshape = rearrange(prototype_features, 'b p d -> (d p) b')
        gt_reshape = rearrange(prototype_gt.repeat(1, phase, 1), 'b p d -> (d p) b')
        output_corr = 0.5 * (self.metric(feature_reshape, gt_reshape) + 1)
        
        # align features with prototype_features
        output_single = self.pro_loss(prototype_features, features)
        losses = output_single + output_corr.mean()
        return losses

