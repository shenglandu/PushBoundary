import torch
import numpy as np
from os import makedirs
from os.path import exists, join
import time
from utils.ply import write_ply
from utils.metrics import IoU_from_confusions, fast_confusion


class ModelTester:
    def __init__(self, net, chkp_path=None, on_gpu=True):
        # Device
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)
        # Model loading
        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    def cloud_segmentation_test(self, net, test_loader, config, num_votes=50):
        """
        Test method for s3dis segmentation models
        """
        # Testing parameters
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Initializing
        nc_model = config.num_classes
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]
        self.test_bprobs = [np.zeros((l.shape[0], 2))
                                      for l in test_loader.dataset.input_boundaries]
        self.test_dpreds = [np.zeros((l.shape[0], 3))
                                      for l in test_loader.dataset.input_boundaries]

        # Saving
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
        else:
            test_path = None

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                 for labels in test_loader.dataset.validation_labels])
                    i += 1
        else:
            val_proportions = None

        # Timing
        test_epoch = 0
        last_min = -0.5
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Testing
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):
                t = t[-1:]
                t += [time.time()]
                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))
                if 'cuda' in self.device.type:
                    batch.to(self.device)
                # Forward pass
                outputs, boutputs, doutputs = net(batch, config)
                t += [time.time()]
                # Getting predictions
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                stacked_bprobs = softmax(boutputs).cpu().detach().numpy()
                stacked_dir = doutputs.cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)
                # Getting predictions and labels per instance
                i0 = 0
                for b_i, length in enumerate(lengths):
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]
                    bprobs = stacked_bprobs[i0:i0 + length]
                    dpreds = stacked_dir[i0:i0 + length]
                    if 0 < test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]
                        bprobs = bprobs[mask]
                        dpreds = dpreds[mask]
                    # Updating current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    self.test_bprobs[c_i][inds] = test_smooth * self.test_bprobs[c_i][inds] + (1 - test_smooth) * bprobs
                    self.test_dpreds[c_i][inds] = test_smooth * self.test_dpreds[c_i][inds] + (1 - test_smooth) * dpreds
                    i0 += length
                # Displaying every 10 seconds
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))
                if (t[-1] - last_display) > 10.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # Updating minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            # Saving predicted cloud
            if last_min + 1 < new_min:
                last_min += 1
                if test_loader.dataset.set == 'validation':
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):
                        probs = np.array(self.test_probs[i], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                        targets = test_loader.dataset.input_labels[i]
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]
                    # Obtaining IoUs
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')
                # IoU on the whole cloud
                if int(np.ceil(new_min)) % 10 == 0:
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    proj_probs = []
                    proj_bprobs = []
                    proj_dpreds = []
                    for i, file_path in enumerate(test_loader.dataset.files):
                        print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)
                        print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                        print(test_loader.dataset.test_proj[i][:5])
                        # Reprojecting probs on the evaluations points
                        probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
                        proj_probs += [probs]
                        bprobs = self.test_bprobs[i][test_loader.dataset.test_proj[i], :]
                        proj_bprobs += [bprobs]
                        dpreds = self.test_dpreds[i][test_loader.dataset.test_proj[i], :]
                        proj_dpreds += [dpreds]
                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))
                    # Voting results
                    if test_loader.dataset.set == 'validation':
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i, file_path in enumerate(test_loader.dataset.files):
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)
                            targets = test_loader.dataset.validation_labels[i]
                            Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]
                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))
                        C = np.sum(np.stack(Confs), axis=0)
                        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)
                        oa = 0
                        for i in range(len(C)):
                            oa += C[i][i]
                        oa = oa * 1.0 / np.sum(C)
                        print('OA is {:.1f}\n'.format(100*oa))
                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')
                    # Saving
                    print('Saving clouds')
                    t1 = time.time()
                    for i, file_path in enumerate(test_loader.dataset.files):
                        points = test_loader.dataset.load_evaluation_points(file_path)
                        preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)
                        bpreds = test_loader.dataset.boundary_values[np.argmax(proj_bprobs[i], axis=1)].astype(np.int32)
                        bpred_probs = proj_bprobs[i][:, 1].astype(np.float32)
                        dpreds = proj_dpreds[i].astype(np.float32)
                        cloud_name = file_path.split('/')[-1]
                        test_name = join(test_path, 'predictions', cloud_name)
                        write_ply(test_name,
                                  [points, preds, bpreds, bpred_probs, dpreds],
                                  ['x', 'y', 'z', 'preds', 'is_boundary_preds', 'is_boundary_probs', 'dx', 'dy', 'dz'])
                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

            test_epoch += 1
            if last_min > num_votes:
                break

        return

    def sensat_segmentation_test(self, net, test_loader, config, num_votes=100):
        """
        Todo: Test method for SensatUrban
        """
        return 0