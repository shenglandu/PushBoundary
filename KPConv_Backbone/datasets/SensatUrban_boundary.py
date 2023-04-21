import time
import numpy as np
import pickle
import torch
from multiprocessing import Lock
import glob
from os.path import exists, join
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.ply import read_ply
from sklearn.neighbors import KDTree
from datasets.common import grid_subsampling
from utils.config import bcolors


class SensatDataset(PointCloudDataset):
    """Class to handle S3DIS dataset."""

    def __init__(self, config, set='training', use_potentials=True, load_data=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'Sensat')
        # Initializing classes and boundary categories
        self.label_to_names = {0: 'Ground',
                               1: 'High Vegetation',
                               2: 'Buildings',
                               3: 'Walls',
                               4: 'Bridge',
                               5: 'Parking',
                               6: 'Rail',
                               7: 'traffic Roads',
                               8: 'Street Furniture',
                               9: 'Cars',
                               10: 'Footpath',
                               11: 'Bikes',
                               12: 'Water'}
        self.boundary_to_names = {0: 'interior',
                                  1: 'boundary'}
        self.init_labels()
        self.init_boudanries()
        self.ignored_labels = np.array([])

        # Data setup
        self.path = '/data_sensat'
        self.dataset_task = 'cloud_segmentation'
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task
        self.config = config
        self.set = set
        if use_potentials is False:
            use_potentials = True
        self.use_potentials = use_potentials
        self.train_path = 'complete_ply'
        ply_path = join(self.path, self.train_path)
        self.all_files = np.sort(glob.glob(join(ply_path, '*.ply')))
        self.val_file_name = ['birmingham_block_1',
                              'birmingham_block_5',
                              'cambridge_block_10',
                              'cambridge_block_7']
        self.test_file_name = ['birmingham_block_2',
                               'birmingham_block_8',
                               'cambridge_block_15',
                               'cambridge_block_22',
                               'cambridge_block_16',
                               'cambridge_block_27']
        self.cloud_names = []
        self.all_splits = []
        self.validation_split = []
        self.test_split = []
        for i, file_path in enumerate(self.all_files):
            cloud_name = file_path.split('/')[-1][:-4]
            self.cloud_names.append(cloud_name)
            self.all_splits.append(i)
            if cloud_name in self.val_file_name:
                self.validation_split.append(i)
            if cloud_name in self.test_file_name:
                self.test_split.append(i)
        if self.set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        elif self.set in ['validation', 'test', 'ERF']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for Sensat data: ', self.set)
        if not load_data:
            return
        self.files = []
        for i, f in enumerate(self.cloud_names):
            if self.set == 'training':
                if not (self.all_splits[i] in self.validation_split or self.all_splits[i] in self.test_split):
                    self.files += [join(ply_path, f + '.ply')]
            elif self.set == 'validation':
                if self.all_splits[i] in self.validation_split:
                    self.files += [join(ply_path, f + '.ply')]
            elif self.set == 'test':
                if self.all_splits[i] in self.test_split:
                    self.files += [join(ply_path, f + '.ply')]
            else:
                raise ValueError('Unknown set for Sensat Training: ', self.set)
        if self.set == 'training':
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if not (self.all_splits[i] in self.validation_split or self.all_splits[i] in self.test_split)]
        elif self.set == 'validation':
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] in self.validation_split]
        elif self.set == 'test':
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] in self.test_split]
        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Container setup
        self.input_trees = []
        self.input_colors = []
        self.input_boundaries = []
        self.input_dirs = []
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        # Start loading
        self.load_subsampled_clouds()

        # Initializing batch configuration
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]
            # Share potential memory
            self.argmin_potentials = torch.from_numpy(np.array(self.argmin_potentials, dtype=np.int64))
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()
            self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            N = config.epoch_steps * config.batch_num
            self.epoch_inds = torch.from_numpy(np.zeros((2, N), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = Lock()
        return

    def init_boudanries(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_boundaries = len(self.boundary_to_names)
        self.boundary_values = np.sort([k for k, v in self.boundary_to_names.items()])
        self.boundary_names = [self.boundary_to_names[k] for k in self.boundary_values]
        self.boundary_to_idx = {l: i for i, l in enumerate(self.boundary_values)}
        self.name_to_boundary = {v: k for k, v in self.boundary_to_names.items()}

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """
        return self.potential_item(batch_i)

    def potential_item(self, batch_i, debug_workers=False):
        # Initializing
        t = [time.time()]
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0
        b_list = []
        dir_list = []

        info = get_worker_info()
        if info is not None:
            wid = info.id
        else:
            wid = None

        while True:
            t += [time.time()]
            if debug_workers:
                message = ''
                for wi in range(info.num_workers):
                    if wi == wid:
                        message += ' {:}X{:} '.format(bcolors.FAIL, bcolors.ENDC)
                    elif self.worker_waiting[wi] == 0:
                        message += '   '
                    elif self.worker_waiting[wi] == 1:
                        message += ' | '
                    elif self.worker_waiting[wi] == 2:
                        message += ' o '
                print(message)
                self.worker_waiting[wid] = 0
            with self.worker_lock:
                if debug_workers:
                    message = ''
                    for wi in range(info.num_workers):
                        if wi == wid:
                            message += ' {:}v{:} '.format(bcolors.OKGREEN, bcolors.ENDC)
                        elif self.worker_waiting[wi] == 0:
                            message += '   '
                        elif self.worker_waiting[wi] == 1:
                            message += ' | '
                        elif self.worker_waiting[wi] == 2:
                            message += ' o '
                    print(message)
                    self.worker_waiting[wid] = 1
                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])
                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)
                # Center point of input region
                center_point = pot_points[point_ind, :].reshape(1, -1)
                # Add a small noise to center point
                if self.set != 'ERF':
                    center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)
                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)
                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]
                # Update potentials (Tukey weights)
                if self.set != 'ERF':
                    tukeys = np.square(1 - d2s / np.square(self.config.in_radius))
                    tukeys[d2s > np.square(self.config.in_radius)] = 0
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]
            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)
            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            t += [time.time()]
            # Number collected
            n = input_inds.shape[0]
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_boundaries = np.zeros(input_points.shape[0])
                input_labels = np.zeros(input_points.shape[0])
                input_dirs = np.zeros((input_points.shape[0], 3))
            else:
                input_boundaries = self.input_boundaries[cloud_ind][input_inds]
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_dirs = self.input_dirs[cloud_ind][input_inds]

            t += [time.time()]
            # Augmentation
            input_points, scale, R = self.augmentation_transform(input_points)
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0
            input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            # Stack batch
            t += [time.time()]
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]
            batch_n += n
            b_list += [input_boundaries]
            dir_list += [input_dirs]

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        # concatenate batch
        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)
        boundaries = np.concatenate(b_list, axis=0)
        dirs = np.concatenate(dir_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        stacked_features = np.hstack((stacked_features, features))

        # Get the whole input list
        t += [time.time()]
        input_list = self.segmentation_boundary_inputs(stacked_points,
                                                       stacked_features,
                                                       boundaries,
                                                       dirs,
                                                       labels,
                                                       stack_lengths,
                                                       self.config.in_features_dim)
        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        if debug_workers:
            message = ''
            for wi in range(info.num_workers):
                if wi == wid:
                    message += ' {:}0{:} '.format(bcolors.OKBLUE, bcolors.ENDC)
                elif self.worker_waiting[wi] == 0:
                    message += '   '
                elif self.worker_waiting[wi] == 1:
                    message += ' | '
                elif self.worker_waiting[wi] == 2:
                    message += ' o '
            print(message)
            self.worker_waiting[wid] = 2

        t += [time.time()]
        return input_list

    def load_subsampled_clouds(self):
        # Initializing
        dl = self.config.first_subsampling_dl

        # Create path for files
        if self.set == 'test':
            tree_path = join(self.path, 'input_{:.3f}_test'.format(dl))
        else:
            tree_path = join(self.path, 'input_{:.3f}_train'.format(dl))

        for i, file_path in enumerate(self.files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if self.set in ['training', 'validation']:
                    sub_labels = data['class']
                    sub_boundaries = data['is_boundary'].astype(np.int32)
                    sub_dirs = np.vstack((data['dx'], data['dy'], data['dz'])).T
                else:
                    sub_labels = np.zeros(data.shape[0])
                    sub_boundaries = np.zeros(data.shape[0])
                    sub_dirs = np.zeros((data.shape[0], 3))
                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPlease use the preprocessed subsampled point clouds')

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]
            self.input_boundaries += [sub_boundaries]
            self.input_dirs += [sub_dirs]

            size = sub_colors.shape[0] * 4 * 14
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        # Potential preparation: only for validation and test set
        if self.use_potentials:
            print('\nPreparing potentials')
            t0 = time.time()
            pot_dl = self.config.in_radius / 10
            cloud_ind = 0
            for i, file_path in enumerate(self.files):
                cloud_name = self.cloud_names[i]
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))
                if exists(coarse_KDTree_file):
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)
                else:
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)
                    search_tree = KDTree(coarse_points, leaf_size=50)
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)
                self.pot_trees += [search_tree]
                cloud_ind += 1
            print('Done in {:.1f}s'.format(time.time() - t0))

        # Re-projecting index preparation: only for validation and test set
        self.num_clouds = len(self.input_trees)
        if self.set in ['validation', 'test']:
            print('\nPreparing reprojection indices for testing')
            for i, file_path in enumerate(self.files):
                t0 = time.time()
                cloud_name = self.cloud_names[i]
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        if self.set == 'validation':
                            proj_inds, labels = pickle.load(f)
                        else:
                            proj_inds = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    proj_inds = np.squeeze(idxs).astype(np.int32)
                    if self.set == 'validation':
                        labels = data['class']
                        with open(proj_file, 'wb') as f:
                            pickle.dump([proj_inds, labels], f)
                        self.validation_labels += [labels]
                    else:
                        with open(proj_file, 'wb') as f:
                            pickle.dump([proj_inds], f)
                self.test_proj += [proj_inds]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))
        print()
        return

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """
        data = read_ply(file_path)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        return points


class SensatSampler(Sampler):
    """Sampler for Sensat"""

    def __init__(self, dataset: SensatDataset):
        Sampler.__init__(self, dataset)
        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """
        if not self.dataset.use_potentials:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int32)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.N * self.dataset.config.batch_num
            random_pick_n = int(np.ceil(num_centers / (self.dataset.num_clouds * self.dataset.config.num_classes)))

            # Choose random points of each class for each cloud
            for cloud_ind, cloud_labels in enumerate(self.dataset.input_labels):
                epoch_indices = np.empty((0,), dtype=np.int32)
                for label_ind, label in enumerate(self.dataset.label_values):
                    if label not in self.dataset.ignored_labels:
                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        if len(label_indices) <= random_pick_n:
                            epoch_indices = np.hstack((epoch_indices, label_indices))
                        elif len(label_indices) < 50 * random_pick_n:
                            new_randoms = np.random.choice(label_indices, size=random_pick_n, replace=False)
                            epoch_indices = np.hstack((epoch_indices, new_randoms.astype(np.int32)))
                        else:
                            rand_inds = []
                            while len(rand_inds) < random_pick_n:
                                rand_inds = np.unique(np.random.choice(label_indices, size=5 * random_pick_n, replace=True))
                            epoch_indices = np.hstack((epoch_indices, rand_inds[:random_pick_n].astype(np.int32)))

                # Stack those indices with the cloud index
                epoch_indices = np.vstack((np.full(epoch_indices.shape, cloud_ind, dtype=np.int32), epoch_indices))

                # Update the global indice container
                all_epoch_inds = np.hstack((all_epoch_inds, epoch_indices))

            # Random permutation of the indices
            random_order = np.random.permutation(all_epoch_inds.shape[1])
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)

            # Update epoch inds
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds[:, :num_centers])

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def fast_calib(self):
        """
        This method calibrates the batch sizes while ensuring the potentials are well initialized. Indeed on a dataset
        like Semantic3D, before potential have been updated over the dataset, there are cahnces that all the dense area
        are picked in the begining and in the end, we will have very large batch of small point clouds
        :return:
        """

        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.dataset.config.batch_num

        # Calibration parameters
        low_pass_T = 10
        Kp = 100.0
        finer = False
        breaking = False

        # Convergence parameters
        smooth_errors = []
        converge_threshold = 0.1

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(2)

        for epoch in range(10):
            for i, test in enumerate(self):

                # New time
                t = t[-1:]
                t += [time.time()]

                # batch length
                b = len(test)

                # Update estim_b (low pass filter)
                estim_b += (b - estim_b) / low_pass_T

                # Estimate error (noisy)
                error = target_b - b

                # Save smooth errors for convergene check
                smooth_errors.append(target_b - estim_b)
                if len(smooth_errors) > 10:
                    smooth_errors = smooth_errors[1:]

                # Update batch limit with P controller
                self.dataset.batch_limit += Kp * error

                # finer low pass filter when closing in
                if not finer and np.abs(estim_b - target_b) < 1:
                    low_pass_T = 100
                    finer = True

                # Convergence
                if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                    breaking = True
                    break

                # Average timing
                t += [time.time()]
                mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d},  //  {:.1f}ms {:.1f}ms'
                    print(message.format(i,
                                         estim_b,
                                         int(self.dataset.batch_limit),
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1]))

            if breaking:
                break

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """
        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()
        redo = force_redo

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.use_potentials:
            sampler_method = 'potentials'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                               self.dataset.config.in_radius,
                                               self.dataset.config.first_subsampling_dl,
                                               self.dataset.config.batch_num)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:
            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            # Calibration
            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.cloud_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.dataset.batch_limit)))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            if self.dataset.use_potentials:
                sampler_method = 'potentials'
            else:
                sampler_method = 'random'
            key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                                   self.dataset.config.in_radius,
                                                   self.dataset.config.first_subsampling_dl,
                                                   self.dataset.config.batch_num)
            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class SensatCustomBatch:
    """Custom batch definition with memory pinning for S3DIS"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]
        L = 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.boundaries = torch.from_numpy(input_list[ind])
        ind += 1
        self.dirs = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.boundaries = self.boundaries.pin_memory()
        self.dirs = self.dirs.pin_memory()
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.boundaries = self.boundaries.to(device)
        self.dirs = self.dirs.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])

                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def SensatCollate(batch_data):
    return SensatCustomBatch(batch_data)
