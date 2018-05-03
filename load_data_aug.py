import numpy as np
import h5py
import glob
import math
import os


# -- slice processing utils
def gen_slice_idx(data, resolution, Z_MIN, Z_MAX, axis=2):
    indices = np.zeros((data.shape[0], data.shape[2]))
    for n in range(data.shape[0]):
        indices[n] = gen_slice_idx_routine(data[n], resolution, Z_MIN, Z_MAX, axis)
    #
    return indices


def gen_slice_idx_routine(data, resolution, Z_MIN, Z_MAX, axis):
    if axis == 2:
        z_min, z_max = Z_MIN, Z_MAX
    else:
        z_min, z_max = data[:, :, axis].min(), data[:, :, axis].max()

    # gap = (z_max - z_min + 0.001) / numSlices
    gap = resolution
    indices = np.ones((data.shape[1], 1)) * float('inf')
    for i in range(data.shape[1]):
        z = data[0, i, axis]
        idx = int((z - z_min) / gap)
        indices[i, 0] = idx
    return indices[:, 0]


# -- utils for loading data, from Pointnet
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


# -- load data here
# - dataset setting, update when neccessay
block_size = 1.0
stride = 0.5
area_name = 'Area_5'
DATA_DIR = './data/'

TRAIN_DIR = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_{}_{}m_{}s_train/'.format(area_name, block_size, stride))
TEST_DIR = os.path.join(DATA_DIR,
                        'indoor3d_sem_seg_hdf5_data_{}_{}m_{}s_test/'.format(area_name, block_size, block_size))

print("loading raw data...")
train_files = glob.glob(TRAIN_DIR + '*.h5')
np.random.shuffle(train_files)
test_files = glob.glob(TEST_DIR + '*.h5')
np.random.shuffle(test_files)

assert len(train_files) != 0, "dataset not processed correctly"
assert len(test_files) != 0, "dataset not processed correctly"

test_index = 0
train_index = 0
def load_data_batch(train_flag = True):
    global test_index
    global train_index
    global is_finished
    if train_flag:
        filename = train_files[train_index]
        is_finished = train_index == len(train_files) - 1
        train_index += 1
        print("Train_index")
        print(str(train_index) + " / " + str(len(train_files)))
    else:
        filename = test_files[test_index]
        is_finished = test_index == len(test_files) - 1
        test_index += 1
        print("Test_index")
        print(str(test_index) + " / " + str(len(test_files)))
    data, label = loadDataFile(filename)
    return data, label, is_finished

def reset():
    global is_finished
    global test_index
    global train_index
    test_index = 0
    train_index = 0
    is_finished = False

temp_data = np.zeros((30, 1000, 4096, 9))
skip = 0
for i in range(15):
    filename = test_files[i + skip]
    data_batch, _ = loadDataFile(filename)
    while data_batch.shape[0] != 1000:
        skip += 1
        filename = test_files[i + skip]
        data_batch, _ = loadDataFile(filename)
    temp_data[i] = data_batch

skip = 0
for i in range(15):
    filename = train_files[i + skip]
    data_batch, _ = loadDataFile(filename)
    while data_batch.shape[0] != 1000:
        skip += 1
        filename = test_files[i + skip]
        data_batch, _ = loadDataFile(filename)
    temp_data[i + 15] = data_batch


is_finished = False
Z_MIN, Z_MAX = temp_data[:, :, :, 2].min(), temp_data[:, :, :, 2].max()
temp_data = []

def iterate_data(batchsize, resolution, train_flag=True, require_ori_data=False, block_size=1.0):
    global is_finished
    while not is_finished:
        data_all, label_all, is_finished = load_data_batch(train_flag)

        if train_flag:
            indices = list(range(data_all.shape[0]))
            np.random.shuffle(indices)
        else:
            indices = range(data_all.shape[0])

        file_size = data_all.shape[0]
        num_batches = int(math.floor(file_size / float(batchsize)))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            excerpt = indices[start_idx:start_idx + batchsize]

            inputs = data_all[excerpt].astype('float32')

            if require_ori_data:
                ori_inputs = inputs.copy()

            for b in range(inputs.shape[0]):
                minx = min(inputs[b, :, 0])
                miny = min(inputs[b, :, 1])
                inputs[b, :, 0] -= (minx + block_size / 2)
                inputs[b, :, 1] -= (miny + block_size / 2)

            inputs = np.expand_dims(inputs, 3).astype('float32')
            inputs = inputs.transpose(0, 3, 1, 2)

            seg_target = label_all[excerpt].astype('int64')  # num_batch, num_points

            if len(resolution) == 1:
                resolution_x = resolution_y = resolution_z = resolution
            else:
                resolution_x, resolution_y, resolution_z = resolution

            x_slices_indices = gen_slice_idx(inputs, resolution_x, Z_MIN, Z_MAX, 0).astype('int32')
            y_slices_indices = gen_slice_idx(inputs, resolution_y, Z_MIN, Z_MAX, 1).astype('int32')
            z_slices_indices = gen_slice_idx(inputs, resolution_z, Z_MIN, Z_MAX, 2).astype('int32')

            if not require_ori_data:
                yield inputs, x_slices_indices, y_slices_indices, z_slices_indices, seg_target
            else:
                yield inputs, x_slices_indices, y_slices_indices, z_slices_indices, seg_target, ori_inputs

    print("Finished")