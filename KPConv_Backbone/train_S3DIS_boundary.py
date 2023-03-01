import signal
import os
import sys
from datasets.S3DIS_boundary import *
from torch.utils.data import DataLoader
from utils.config import Config
from utils.trainer_boundary import ModelTrainer
from models.architectures_boundary import KPFCNN


class S3DISConfig(Config):
    # Initialize dataset
    dataset = 'S3DIS'
    num_classes = None
    dataset_task = ''
    input_threads = 1

    # Define architecture. We use rigid KP-Conv as the backbone network
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    # Input batch data specification
    in_radius = 1.5
    first_subsampling_dl = 0.05

    # Kernel details
    num_kernel_points = 15
    conv_radius = 2.5
    deform_radius = 6.0
    KP_extent = 1.2
    KP_influence = 'linear'
    aggregation_mode = 'sum'

    # Input feature dimension and first-layer feature dimension
    first_features_dim = 128
    in_features_dim = 8

    # Network details
    modulated = False
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0
    deform_lr_factor = 0.1
    repulse_extent = 1.2

    # Training parameters
    max_epoch = 500
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0
    batch_num = 6
    epoch_steps = 300
    validation_size = 50
    checkpoint_gap = 500

    # Data augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Saving configuration
    saving = True
    saving_path = None


if __name__ == '__main__':
    # GPU
    GPU_ID = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Default: training from scratch
    previous_training_path = ''
    chkp_idx = None
    if previous_training_path:
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)
    else:
        chosen_chkp = None

    # Loading data
    print()
    print('Data Preparation')
    print('****************')
    config = S3DISConfig()
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]
    training_dataset = S3DISDataset(config, set='training', use_potentials=True)
    test_dataset = S3DISDataset(config, set='validation', use_potentials=True)
    training_sampler = S3DISSampler(training_dataset)
    test_sampler = S3DISSampler(test_dataset)
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=S3DISCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=S3DISCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # Model setup
    print('\nModel Preparation')
    print('*****************')
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels, training_dataset.boundary_values)
    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Training setup
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))
    print('\nStart training')
    print('**************')
    trainer.train(net, training_loader, test_loader, config)

    # Quit training
    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
