import glob
import utils
import sys
import numpy as np
import tensorflow as tf
import msl_model
import keras
from keras.callbacks import ModelCheckpoint


full_images = {}
ratios = {}
r_dict = {}
data_path = ''

def read_ind_files(path):
    """
    reads the indices of the images for a given file
    """
    fin = open(path, 'r')
    links = []
    for line in fin:
        links.append(data_path + line.strip())
    return links

def read_indices(repository_path):
    """
    it goes through train/test/validation files
    """
    path = repository_path + '/code/data_split/'
    tr = path + 'train_ids.txt'
    v = path + 'valid_ids.txt'
    t = path + 'test_ids.txt'
    tr_links = read_ind_files(tr)
    v_links = read_ind_files(v)
    t_links = read_ind_files(t)
    return tr_links, v_links, t_links


def read_images(links, size):
    """
    for a given list of paths, it reads the image,
    prepares it for input and it calculates the target value
    """
    dim = 203
    inp = np.zeros((size, dim, dim, 3))
    m_out = np.zeros((size, 3))
    q_out = np.zeros((size, 9))
    r_out = np.zeros((size, 17))
    count = 0
    for link in links[:size]:
        res_img = utils.load_image(link, dim)
        inp[count] = res_img
        cat = link.strip().split('/')[-2][-2:]
        for i in range(9):
            q_out[count][i] = ratios[cat][str(i)]
#            print out[count]
        r_out[count][r_dict[cat]] = 1.0
        if cat[1] == 'Y' or cat[0] == 'X':
            if cat[1] == 'Y':
                ratio_val = 0.0
            else:
                ratio_val = 1.0
        else:
            ratio_val = float(cat[0]) / (float(cat[1]) + float(cat[0]))
        if ratio_val < 0.5:
            m_out[count][0] = 1.0
        if ratio_val == 0.5:
            m_out[count][1] = 1.0
        if ratio_val > 0.5:
            m_out[count][2] = 1.0
        count += 1
        if count % 100 == 0:
            print count
    return inp, m_out


def create_ratio_dict(ratios):
    count = 0
    r = sorted(ratios.keys())
    print r
    for i in range(len(r)):
        r_dict[r[i]] = count
        count += 1

if __name__ == '__main__':
    """
    it reads the parameters,
    initializes the hyperparameters,
    preprocesses the input,
    trains the model
    """
    repository_path = sys.argv[1]
    data_path = sys.argv[2]
    tr_inds, v_inds, t_inds = read_indices(repository_path)
    ratios = utils.read_qprobs(repository_path)
    tr_size = 11900
    v_size = 1700
    create_ratio_dict(ratios)
    tr_inp, tr_m_out = read_images(tr_inds, tr_size)
    v_inp, v_m_out = read_images(v_inds, v_size)
    nb_epochs = 100
    b_size = 85
    m = msl_model.MslInc()
    model = m.build()
    filepath = "best_model/weight.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
    callback_list = [checkpoint]
    hist = model.fit(tr_inp, tr_m_out, batch_size = b_size, epochs = nb_epochs, validation_data = (v_inp, v_m_out), callbacks = callback_list)
