import torch
import numpy as np
from os import makedirs, remove
from os.path import exists, join
import time
from utils.ply import write_ply
from utils.metrics import IoU_from_confusions, fast_confusion


class ModelTrainer:
    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """
        # Initializing
        self.epoch = 0
        self.step = 0
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

        # Choosing device
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        # Training from previous models
        if chkp_path is not None:
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """
        # Initializing
        if config.saving:
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps out_loss boundary_loss reg_loss dir_loss train_accuracy boundary_accuracy time\n')
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        # Timing
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Training
        for epoch in range(config.max_epoch):
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)
            self.step = 0
            for batch in training_loader:
                # Initializing
                if config.saving and not exists(PID_file):
                    continue
                t = t[-1:]
                t += [time.time()]
                if 'cuda' in self.device.type:
                    batch.to(self.device)
                # Forward pass
                self.optimizer.zero_grad()
                outputs, boutputs, doutputs = net(batch, config)
                loss = net.loss(outputs, boutputs, doutputs, batch.labels, batch.boundaries,batch.dirs)
                acc = net.accuracy(outputs, batch.labels)
                bacc = net.baccuracy(boutputs, batch.boundaries)
                t += [time.time()]
                # Backward
                loss.backward()
                if config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                torch.cuda.synchronize(self.device)
                # Average timing
                t += [time.time()]
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))
                # Console display (only one per 10 seconds)
                if (t[-1] - last_display) > 10.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L={:.3f} acc={:3.0f}% bacc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                    print(message.format(self.epoch, self.step,
                                         loss.item(),
                                         100 * acc,
                                         100 * bacc,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         1000 * mean_dt[2]))
                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                        file.write(message.format(self.epoch,
                                                  self.step,
                                                  net.output_loss,
                                                  net.boutput_loss,
                                                  net.reg_loss,
                                                  net.dir_loss,
                                                  acc,
                                                  bacc,
                                                  t[-1] - t0))
                self.step += 1

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate and epoch
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]
            self.epoch += 1

            # Saving
            if config.saving:
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            # Validation
            net.eval()
            self.cloud_segmentation_validation(net, val_loader, config)
            net.train()

        print('Finished Training')
        return

    def cloud_segmentation_validation(self, net, val_loader, config):
        """
        Validation method for cloud segmentation models
        """
        # Initializing
        t0 = time.time()
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)
        if val_loader.dataset.validation_split not in val_loader.dataset.all_splits:
            return
        nc_tot = val_loader.dataset.num_classes
        nc_model = config.num_classes
        nc_boundary = val_loader.dataset.num_boundaries
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model))
                                     for l in val_loader.dataset.input_labels]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in val_loader.dataset.label_values:
                if label_value not in val_loader.dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in val_loader.dataset.validation_labels])
                    i += 1
            self.validation_bprobs = [np.zeros((l.shape[0], nc_boundary))
                                      for l in val_loader.dataset.input_boundaries]
            self.validation_dpreds = [np.zeros((l.shape[0], 3))
                                      for l in val_loader.dataset.input_boundaries]
        predictions = []
        targets = []
        bpredictions = []
        btargets = []
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        t1 = time.time()

        # Validating
        for i, batch in enumerate(val_loader):
            # Initializing
            t = t[-1:]
            t += [time.time()]
            if 'cuda' in self.device.type:
                batch.to(self.device)
            to = time.time()
            # Forward pass
            outputs, boutputs, doutputs = net(batch, config)
            # Obtaining the prediction outputs
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            labels = batch.labels.cpu().numpy()
            stacked_bprobs = softmax(boutputs).cpu().detach().numpy()
            boundaries = batch.boundaries.cpu().numpy()
            stacked_dir = doutputs.cpu().detach().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            in_inds = batch.input_inds.cpu().numpy()
            cloud_inds = batch.cloud_inds.cpu().numpy()
            torch.cuda.synchronize(self.device)
            # Getting predictions and labels per instance
            i0 = 0
            for b_i, length in enumerate(lengths):
                target = labels[i0:i0 + length]
                probs = stacked_probs[i0:i0 + length]
                inds = in_inds[i0:i0 + length]
                c_i = cloud_inds[b_i]
                btarget = boundaries[i0:i0 + length]
                bprobs = stacked_bprobs[i0:i0 + length]
                dpreds = stacked_dir[i0:i0 + length]
                # Updating current probs in whole cloud
                self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                   + (1 - val_smooth) * probs
                self.validation_bprobs[c_i][inds] = val_smooth * self.validation_bprobs[c_i][inds] \
                                                    + (1 - val_smooth) * bprobs
                self.validation_dpreds[c_i][inds] = val_smooth * self.validation_dpreds[c_i][inds] \
                                                    + (1 - val_smooth) * dpreds
                # Stacking all prediction for this epoch
                predictions.append(probs)
                targets.append(target)
                bpredictions.append(bprobs)
                btargets.append(btarget)
                i0 += length
            # Displaying per 10 seconds
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))
            if (t[-1] - last_display) > 10.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # IoUs for semantic predictions
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)
        C = np.sum(Confs, axis=0).astype(np.float32)
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
        IoUs = IoU_from_confusions(C)

        # IoUs for boundary predictions
        bConfs = np.zeros((len(bpredictions), nc_boundary, nc_boundary), dtype=np.int32)
        for i, (bprobs, truth) in enumerate(zip(bpredictions, btargets)):
            bpreds = val_loader.dataset.boundary_values[np.argmax(bprobs, axis=1)]
            bConfs[i, :, :] = fast_confusion(truth, bpreds, val_loader.dataset.boundary_values).astype(np.int32)
        bC = np.sum(bConfs, axis=0).astype(np.float32)
        bIoUs = IoU_from_confusions(bC)

        # Saving
        if config.saving:
            # Saving semantic prediction scores
            test_file = join(config.saving_path, 'val_IoUs.txt')
            line = ''
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line = line + '\n'
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

        # Printing validated mIoU and boundary mIOU
        mIoU = 100 * np.mean(IoUs)
        bIoU = 100 * np.mean(bIoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))
        print('mean boundary IoU = {:.1f}%'.format(bIoU))

        # Saving validated model if the gap is met
        if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0:
            val_path = join(config.saving_path, 'val_preds_{:d}'.format(self.epoch + 1))
            if not exists(val_path):
                makedirs(val_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):
                # getting predictions
                points = val_loader.dataset.load_evaluation_points(file_path)
                sub_probs = self.validation_probs[i]
                sub_bprobs = self.validation_bprobs[i]
                sub_dpreds = self.validation_dpreds[i]
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)
                sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]
                sub_bpreds = val_loader.dataset.boundary_values[np.argmax(sub_bprobs, axis=1).astype(np.int32)]
                # projecting predictions to the whole points
                preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)
                bpreds = (sub_bpreds[val_loader.dataset.test_proj[i]]).astype(np.int32)
                dpreds = (sub_dpreds[val_loader.dataset.test_proj[i]]).astype(np.float32)
                # saving as .ply
                cloud_name = file_path.split('/')[-1]
                val_name = join(val_path, cloud_name)
                obj_labels = val_loader.dataset.validation_labels[i].astype(np.int32)
                write_ply(val_name,
                          [points, preds, bpreds, dpreds, obj_labels],
                          ['x', 'y', 'z', 'preds', 'is_boundary_preds', 'dx', 'dy', 'dz', 'class'])

        return
