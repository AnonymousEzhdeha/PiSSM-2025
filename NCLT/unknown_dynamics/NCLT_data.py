

import sys, os
sys.path.append('../')
#os.chdir('data path')
import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
import pickle
import math

dates = ['2012-01-22']
path_gps = "./dataset/gps.csv"
path_gps_rtk = "./dataset/gps_rtk.csv"
path_gps_rtk_err = "./dataset/gps_rtk_err.csv"
path_gt = "./dataset/groundtruth_%s.csv"
compact_path = "./dataset/nclt_%s.pickle"

class NCLT(data.Dataset):
    def __init__(self, date, partition='train', ratio=1.0):
        self.partition = partition
        self.ratio = ratio
        if not os.path.exists(compact_path % date):
            print("Loading NCLT dataset ...")
            self.gps, self.gps_rtk, self.gps_rtk_err, self.gt = self.__load_data(date)
            self.__process_data()
            self.dump(compact_path % date, [self.gps, self.gps_rtk, self.gps_rtk_err, self.gt])

        else:
            [self.gps, self.gps_rtk, self.gps_rtk_err, self.gt] = self.load(compact_path % date)

        if self.partition == 'train':
            indexes = [1, 3]
        elif self.partition == 'val':
            indexes = [0, 2]
        elif self.partition == 'test':
            indexes = [4, 5, 6]
        else:
            raise Exception('Wrong partition')


        self.gps = [self.gps[i].astype(np.float32) for i in indexes]
        self.gps_rtk = [self.gps_rtk[i].astype(np.float32) for i in indexes]
        self.gt = [self.gt[i].astype(np.float32) for i in indexes]

        self.cut_data()


        print("NCLT %s loaded: %d samples " % (partition, sum([x.shape[0] for x in self.gps_rtk])))

        self.operators_b = [self.__buildoperators_sparse(self.gps[i].shape[0]) for i in range(len(self.gps))]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (state, meas) where target is index of the target class.
        """
        x0, P0 = self.__pos2x0(self.gps_rtk[index][0, 1:].astype(np.float32))
        return self.gt[index][:, 0], self.gt[index][:, 1:], self.gps_rtk[index][:, 1:], x0, P0, self.operators_b[index]

    def cut_data(self):
        self.gps = [self.cut_array(e, self.ratio) for e in self.gps]
        self.gps_rtk = [self.cut_array(e, self.ratio) for e in self.gps_rtk]
        self.gt = [self.cut_array(e, self.ratio) for e in self.gt]

    def __pos2x0(self, pos):
        x0 = np.zeros(4).astype(np.float32)
        x0[0] = pos[0]
        x0[2] = pos[1]
        P0 = np.eye(4)*1
        return x0, P0

    def dump(self, path, object):
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            return pickle.load(f)

    def __len__(self):
        return len(self.gt)

    def total_len(self):
        total = 0
        for arr in self.gt:
            total += arr.shape[0]
        return total

    # def _generate_sample(self, seed):
    #     np.random.seed(seed)

    #     if self.acceleration:
    #         return simulate_system(create_model_parameters_a, K=self.K, x0=self.x0)
    #     else:
    #         return simulate_system(create_model_parameters_v, K=self.K, x0=self.x0)

    def __buildoperators_sparse_old(self, nn=20):
        # Identity
        i = torch.LongTensor([[i, i] for i in range(nn)])
        v = torch.FloatTensor([1 for i in range(nn)])
        I = torch.sparse.FloatTensor(i.t(), v)

        #Message right
        i = torch.LongTensor([[i, i+1] for i in range(nn-1)] + [[nn-1, nn-1]])
        v = torch.FloatTensor([1 for i in range(nn-1)] + [0])
        mr = torch.sparse.FloatTensor(i.t(), v)

        #Message left
        i = torch.LongTensor([[0, nn-1]] + [[i+1, i] for i in range(nn-1)])
        v = torch.FloatTensor([0] + [1 for i in range(nn-1)])
        ml = torch.sparse.FloatTensor(i.t(), v)

        return [I, mr, ml]

    def __buildoperators_sparse(self, nn=20):
        # Message right to left
        m_left_r = []
        m_left_c = []

        m_right_r = []
        m_right_c = []

        m_up_r = []
        m_up_c = []

        for i in range(nn - 1):
            m_left_r.append(i)
            m_left_c.append((i + 1))

            m_right_r.append(i + 1)
            m_right_c.append((i))

        for i in range(nn):
            m_up_r.append(i)
            m_up_c.append(i + nn)

        m_left = [torch.LongTensor(m_left_r), torch.LongTensor(m_left_c)]
        m_right = [torch.LongTensor(m_right_r), torch.LongTensor(m_right_c)]
        m_up = [torch.LongTensor(m_up_r), torch.LongTensor(m_up_c)]

        return {"m_left": m_left, "m_right": m_right, "m_up": m_up}

    def __load_gps(self, path, date):
        df = pd.read_csv(path )
        df = df.iloc[:, [0, 3, 4]]
        return df.values

    def __load_gps_err(self, date):
        df = pd.read_csv(path_gps )
        df = df.iloc[:, 6]
        return df.values

    def __load_gt(self, date):
        df = pd.read_csv(path_gt % date)
        gt = df.iloc[:, [0, 2, 1]].values
        gt_err = df.iloc[:, [5, 4]].values
        return gt, gt_err

    def __load_gps_rtk_err(self, date):
        df = pd.read_csv(path_gps_rtk_err )
        return df.values

    def __compute_gps_err(self, gps, gt):
        return np.mean(np.square(gps - gt), axis=1)

    def __load_data(self, date):
        "We use the timestamp of gps_rtk which has the lowest frequency 1 Hz"
        gps = self.__load_gps(path_gps, date)
        gps_rtk = self.__load_gps(path_gps_rtk, date)
        gps_rtk_err = self.__load_gps_rtk_err(date)
        gt, _ = self.__load_gt(date)

        self.lat0 = gps_rtk[0, 1]
        self.lng0 = gps_rtk[0, 2]
        self.bias = [gt[0, 1], gt[0, 2]]

        gps_rtk_dec = self.__decompose(gps_rtk, date)
        gps_rtk_err_dec = self.__decompose(gps_rtk_err, date)

        gps_ar = []
        gt_ar = []
        gps_rtk_ar, gps_rtk_err_ar = [], []

        for gps_rtk_i, gps_rtk_err_i in zip(gps_rtk_dec, gps_rtk_err_dec):
            idxs = self.__filer_freq(gps_rtk_i[:, 0], f=1.)
            gps_rtk_ar.append(gps_rtk_i[idxs, :])
            gps_rtk_err_ar.append(gps_rtk_err_i[idxs, :])


            #Matching with GT
            idxs_gt = self.__match_tt(gps_rtk_ar[-1][:, 0], gt[:, 0])
            gt_ar.append(gt[idxs_gt, :])

            #Matching with gps
            idxs = self.__match_tt(gps_rtk_ar[-1][:, 0], gps[:, 0])
            gps_ar.append(gps[idxs, :])

        return gps_ar, gps_rtk_ar, gps_rtk_err_ar, gt_ar

    def __decompose(self, data, date):
        if date == '2012-01-22':
            return [data[100:2054], data[2054:4009], data[4147:6400], data[6400:8890], data[9103:10856], data[11113:12608],
                    data[12733:13525]]#, [0, 4147, 9103, 11113, 12733]
        else:
            return data

    def concatenate(self, arrays):
        return np.concatenate(arrays, axis=0)

    def __process_data(self):
        '''
        lat0 = self.gps_rtk[0][0, 1]
        lng0 = self.gps_rtk[0][0, 2]
        bias = [self.gt[0][0, 1], self.gt[0][0, 2]]
        '''

        for i in range(len(self.gps_rtk)):
            self.gps_rtk[i][:, 1:] = polar2cartesian(self.gps_rtk[i][:, 1], self.gps_rtk[i][:, 2], self.lat0,
                                                     self.lng0)
            self.gps[i][:, 1:] = polar2cartesian(self.gps[i][:, 1], self.gps[i][:, 2], self.lat0,
                                                 self.lng0)

            self.gt[i][:, 1:] = remove_bias(self.gt[i][:, 1:], self.bias)

    def __match_tt(self, tt1, tt2):
        print("\tMatching gps and gt timestamps")
        arr_idx = []
        for i, ti in enumerate(tt1):
            diff = np.abs(tt2 - ti)
            min_idx = np.argmin(diff)
            arr_idx.append(min_idx)
        return arr_idx

    def _match_gt_step1(self, gps, gps_err, gt, margin=5):
        gt_aux = gt.copy()
        min_err = 1e10
        min_x, min_y = 0, 0
        for x in np.linspace(-margin, margin, 200):
            for y in np.linspace(-margin, margin, 200):
                gt_aux[:, 0] = gt[:, 0] + x
                gt_aux[:, 1] = gt[:, 1] + y
                err = mse(gps, gps_err, gt_aux)
                if err < min_err:
                    min_err = err
                    min_x = x
                    min_y = y
                    #print("x: %.4f \t y:%.4f \t err:%.4f" % (min_x, min_y, err))

        print(err)
        print("Fixing GT bias x: %.4f \t y:%.4f \t error:%.4f" % (min_x, min_y, min_err))
        return (min_x, min_y)

    def _match_gt_step2(self, gt, err):
        (min_x, min_y) = err
        gt[:, 0] = gt[:, 0] + min_x
        gt[:, 1] = gt[:, 1] + min_y
        return gt

    def __filer_freq(self, ts, f=1., window=5):
        arr_idx = []
        last_id = 0
        arr_idx.append(last_id)
        check = False
        while last_id < len(ts) - window:
            rel_j = []
            for j in range(1, window):
                rel_j.append(np.abs(f - (ts[last_id+j] - ts[last_id])/1000000))
            last_id = last_id + 1 + np.argmin(rel_j)

            min_val = np.min(rel_j)
            if min_val > 0.05:
                check = True
            arr_idx.append(last_id)
        if check:
            print("\tWarning: Not all frequencies are %.3fHz" % f)
        print("\tFiltering finished!")
        return arr_idx
    
    def cut_array(self, array, ratio):
        length = len(array)
        return array[0:int(round(ratio*length))]


def mse(gps, gps_err, gt, th=2):
    error = np.mean(np.square(gps - gt), axis=1)
    mapping = (gps_err < th).astype(np.float32)
    return np.mean(error*mapping)

def polar2cartesian(lat, lng, lat0, lng0):
    dLat = lat - lat0
    dLng = lng - lng0

    r = 6400000  # approx. radius of earth (m)
    x = r * np.cos(lat0) * np.sin(dLng)
    y = r * np.sin(dLat)
    return np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), 1)


def remove_bias(vector, bias):
    for i in range(vector.shape[1]):
        vector[:, i] = vector[:, i] - bias[i]
    return vector


def NCLT_DG(split_size):
    Train = NCLT('2012-01-22', partition='train')
    Valid = NCLT('2012-01-22', partition='val')
    Test = NCLT('2012-01-22', partition='test')
    
   
    ###
    _ , GT_TRAIN0, GPS_TRAIN0, TRAIN0_x0, TRAIN0_P0, _ = Train[0]
    _ , GT_TRAIN1, GPS_TRAIN1, TRAIN1_x0, TRAIN1_P0, _ = Train[1]
    
    
    ###
    _ , GT_VALID0, GPS_VALID0, VALID0_x0, VALID0_P0, _ = Valid[0]
    _ , GT_VALID1, GPS_VALID1, VALID1_x0, VALID1_P0, _ = Valid[1]
    
    
    ###
    _ , GT_TEST0, GPS_TEST0, TEST0_x0, TEST0_P0, _ = Test[0]
    _ , GT_TEST1, GPS_TEST1, TEST1_x0, TEST1_P0, _ = Test[1]
    _ , GT_TEST2, GPS_TEST2, TEST2_x0, TEST2_P0, _ = Test[2]
    ###
    
    #
    train_obs = np.concatenate((GPS_TRAIN0[:(math.floor(GPS_TRAIN0.shape[0] / split_size) ) *  split_size, :],
                                GPS_TRAIN1[:(math.floor(GPS_TRAIN1.shape[0] / split_size) ) *  split_size, :],
                                GPS_VALID0[:(math.floor(GPS_VALID0.shape[0] / split_size) ) *  split_size, :],
                                GPS_VALID1[400:(math.floor(GPS_VALID1.shape[0] / split_size)*  split_size ), :],
                                GPS_TEST0[:(math.floor(GPS_TEST0.shape[0] / split_size) ) *  split_size, :],
                                GPS_TEST1[:(math.floor(GPS_TEST1.shape[0] / split_size) ) *  split_size, :],
                                ), axis=0)
    train_targets = np.concatenate((GT_TRAIN0[:(math.floor(GT_TRAIN0.shape[0] / split_size) ) *  split_size, :],
                                    GT_TRAIN1[:(math.floor(GT_TRAIN1.shape[0] / split_size) ) *  split_size, :],
                                    GT_VALID0[:(math.floor(GT_VALID0.shape[0] / split_size) ) *  split_size, :],
                                    GT_VALID1[400:(math.floor(GT_VALID1.shape[0] / split_size)*  split_size ), :],
                                    GT_TEST0[:(math.floor(GT_TEST0.shape[0] / split_size) ) *  split_size, :],
                                    GT_TEST1[:(math.floor(GT_TEST1.shape[0] / split_size) ) *  split_size, :],
                                    ), axis=0)
    
    #
    valid_obs = GPS_VALID1[:400, :]
    valid_targets = GT_VALID1[:400, :]
    
    #
    test_obs = GPS_TEST2[:(math.floor(GPS_TEST2.shape[0] / split_size)*  split_size ), :]
    test_targets = GT_TEST2[:(math.floor(GT_TEST2.shape[0] / split_size)*  split_size ), :]
    

    return train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets

# if __name__ == '__main__':
#     for date in dates:
#         dataset = NCLT('2012-01-22', partition='train')
#         dataset = NCLT('2012-01-22', partition='val')
#         dataset = NCLT('2012-01-22', partition='test')





