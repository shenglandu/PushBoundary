import os
from datasets.S3DIS_boundary import *
from torch.utils.data import DataLoader
from utils.config import Config
from utils.tester import ModelTester
from models.architectures_boundary import KPFCNN


if __name__ == '__main__':
    # specifying the chosen model directory
    chosen_log = 'results/Log_ours'
    chkp_idx = None
    set = 'validation'

    # GPU
    GPU_ID = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Default: find the last model
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initializing configuration class
    config = Config()
    config.load(chosen_log)
    config.validation_size = 200
    config.input_threads = 1

    # Loading data
    print()
    print('Data Preparation')
    print('****************')
    test_dataset = S3DISDataset(config, set=set, use_potentials=True)
    test_sampler = S3DISSampler(test_dataset)
    collate_fn = S3DISCollate
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    # Model setup
    print('\nModel Preparation')
    print('*****************')
    t1 = time.time()
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels, test_dataset.boundary_values)
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    # Testing
    print('\nStart test')
    print('**********\n')
    tester.cloud_segmentation_test(net, test_loader, config)
