from models.blocks import *
import numpy as np


def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """
    def __init__(self, config, lbl_values, ign_lbls, bl_values):
        super(KPFCNN, self).__init__()
        # Initializing
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.bC = len(bl_values)
        self.r = r
        self.up_i = []

        # Encoding layers
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []
        for block_i, block in enumerate(config.architecture):
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
            if 'upsample' in block:
                break
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        # Decoding initialization
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []
        self.decoder_blocks_boundary = nn.ModuleList()
        self.decoder_blocks_dir = nn.ModuleList()

        # Decoding layers
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break
        for block_i, block in enumerate(config.architecture[start_i:]):
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)
            # 3 decoders for 3 tasks
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))
            self.decoder_blocks_boundary.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))
            self.decoder_blocks_dir.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))
            in_dim = out_dim
            if 'upsample' in block:
                self.up_i += [block_i]
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        # 3 heads for 3 tasks
        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)
        self.head_mlp_boundary = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax_boundary = UnaryBlock(config.first_features_dim, self.bC, False, 0)
        self.head_mlp_dir = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax_dir = UnaryBlock(config.first_features_dim, 3, False, 0)

        # losses
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
        self.boundary_labels = bl_values
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        boundary_w = torch.tensor([0.4, 0.6])
        self.criterion_boundary = torch.nn.CrossEntropyLoss(weight=boundary_w, ignore_index=-1)
        self.criterion_dir = nn.MSELoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0.0
        self.reg_loss = 0.0
        self.l1 = nn.L1Loss()
        self.boutput_loss = 0.0
        self.dir_loss = 0.0

        return

    def forward(self, batch, config):
        x = batch.features.clone().detach()
        skip_x = []
        skip_xb = []
        # encoding
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
                skip_xb.append(x)
            x = block_op(x, batch)
        x_boundary = x.clone()
        x_dir = x.clone()
        # decoding of boundaries and directions
        for block_i, block_op in enumerate(self.decoder_blocks_boundary):
            if block_i in self.decoder_concats:
                skip_features = skip_xb.pop()
                x_boundary = torch.cat([x_boundary, skip_features], dim=1)
                x_dir = torch.cat([x_dir, skip_features], dim=1)
            x_boundary = block_op(x_boundary, batch)
            block_dir = self.decoder_blocks_dir[block_i]
            x_dir = block_dir(x_dir, batch)

        # Prediction outputs of boundaries and directions
        x_boundary = self.head_mlp_boundary(x_boundary, batch)
        x_boundary = self.head_softmax_boundary(x_boundary, batch)
        x_dir = self.head_mlp_dir(x_dir, batch)
        x_dir = self.head_softmax_dir(x_dir, batch)

        # Decoding of semantic segmentation
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                skip_features = skip_x.pop()
                x = torch.cat([x, skip_features], dim=1)
            # boundary guided propagation for feature upsampling
            if block_i in self.up_i:
                x = block_op(x, batch, x_boundary, x_dir)
            else:
                x = block_op(x, batch)

        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x, x_boundary, x_dir

    def loss(self, outputs, boutputs, doutputs, labels, boundaries, dirs):
        # Semantic segmentation loss
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = (target.unsqueeze(0)).type(torch.long)
        self.output_loss = self.criterion(outputs, target)

        # Boundary loss
        btargets = - torch.ones_like(boundaries)
        for i, c in enumerate(self.boundary_labels):
            btargets[boundaries == c] = i
        btargets = (btargets.unsqueeze(0)).type(torch.long)
        boutputs = torch.transpose(boutputs, 0, 1)
        boutputs = boutputs.unsqueeze(0)
        self.boutput_loss = self.criterion_boundary(boutputs, btargets)

        # direction loss
        self.dir_loss = self.criterion_dir(doutputs, dirs)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combine losses
        lambda1 = 3.0
        lambda2 = 0.3
        return self.output_loss + lambda1*self.boutput_loss + lambda2*self.dir_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i
        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total

    def baccuracy(self, boutputs, boundaries):
        btargets = - torch.ones_like(boundaries)
        for i, c in enumerate(self.boundary_labels):
            btargets[boundaries == c] = i
        predicted = torch.argmax(boutputs.data, dim=1)
        total = btargets.size(0)
        correct = (predicted == btargets).sum().item()

        return correct / total