import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils_boundary import PointNetSetAbstraction,PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 12 + 3, [32, 32, 64], False) 
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)

        # Decoder Seg
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        # Decoder boundary
        self.fp4_boundary = PointNetFeaturePropagation(768, [256, 256])
        self.fp3_boundary = PointNetFeaturePropagation(384, [256, 256])
        self.fp2_boundary = PointNetFeaturePropagation(320, [256, 128])
        self.fp1_boundary = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1_boundary = nn.Conv1d(128, 128, 1)
        self.bn1_boundary = nn.BatchNorm1d(128)
        self.drop1_boundary = nn.Dropout(0.5)
        self.conv2_boundary = nn.Conv1d(128, 2, 1)

        # Decoder offset
        self.fp4_dir = PointNetFeaturePropagation(768, [256, 256])
        self.fp3_dir = PointNetFeaturePropagation(384, [256, 256])
        self.fp2_dir = PointNetFeaturePropagation(320, [256, 128])
        self.fp1_dir = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1_dir = nn.Conv1d(128, 128, 1)
        self.bn1_dir = nn.BatchNorm1d(128)
        self.drop1_dir = nn.Dropout(0.5)
        self.conv2_dir = nn.Conv1d(128, 3, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        # Encoding
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Copy the latent feature
        l4_boundaries = l4_points.clone()
        l4_dirs = l4_points.clone()

        # Decoding boundary
        l3_boundaries = self.fp4_boundary(l3_xyz, l4_xyz, l3_points, l4_boundaries)
        l2_boundaries = self.fp3_boundary(l2_xyz, l3_xyz, l2_points, l3_boundaries)
        l1_boundaries = self.fp2_boundary(l1_xyz, l2_xyz, l1_points, l2_boundaries)
        l0_boundaries = self.fp1_boundary(l0_xyz, l1_xyz, None, l1_boundaries)

        x_boundary = self.drop1_boundary(F.relu(self.bn1_boundary(self.conv1_boundary(l0_boundaries))))
        x_boundary = self.conv2_boundary(x_boundary)
        x_boundary = F.log_softmax(x_boundary, dim=1)
        x_boundary = x_boundary.permute(0, 2, 1)

        # Decoding directions
        l3_dirs = self.fp4_dir(l3_xyz, l4_xyz, l3_points, l4_dirs)
        l2_dirs = self.fp3_dir(l2_xyz, l3_xyz, l2_points, l3_dirs)
        l1_dirs = self.fp2_dir(l1_xyz, l2_xyz, l1_points, l2_dirs)
        l0_dirs = self.fp1_dir(l0_xyz, l1_xyz, None, l1_dirs)

        x_dirs = self.drop1_dir(F.relu(self.bn1_dir(self.conv1_dir(l0_dirs))))
        x_dirs = self.conv2_dir(x_dirs)
        x_dirs = F.log_softmax(x_dirs, dim=1)
        x_dirs = x_dirs.permute(0, 2, 1)

        # Decoding semantics
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points, x_boundary, x_dirs)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x, x_boundary, x_dirs, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, bpred, dpred, target, btarget, dtarget, trans_feat, weight, bweights):
        # Segmentation loss
        seg_loss = F.nll_loss(pred, target)
        # Boundary loss
        boundary_loss = F.nll_loss(bpred, btarget, weight=bweights)
        # Direction loss
        dir_loss = F.mse_loss(dpred, dtarget)

        # Lambdas
        lambda1 = 1.0
        lambda2 = 3.0
        lambda3 = 0.3
        total_loss = lambda1*seg_loss + lambda2*boundary_loss + lambda3*dir_loss

        return total_loss, seg_loss, boundary_loss, dir_loss


if __name__ == '__main__':
    import torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))
