
import matplotlib
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt
import argparse
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from GIN import GIN
from LayerNormalizer import LayerNormalizer
from PolyboxData import BallBox
from PymunkData import PymunkData



def read_json(path):
    with open(path) as f:
        d = json.load(f)
    f.close()
    return d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="polybox state estimation/config.json")
    return parser.parse_args()
        

def generate_poly_filter_dataset( num_seqs_train, num_seqs_test, num_seqs_valid,
                                  seq_length_train, seq_length_test, seq_length_valid):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    scale = 1

    np.random.seed(1234)

    # Create data dir
    if not os.path.exists('./data'):
        os.makedirs('./data')

    cannon = BallBox(dt=0.2, res=(32*scale, 32*scale), init_pos=(16*scale, 16*scale), init_std=2.5, wall=None)
    cannon.run(delay=None, iterations=seq_length_train, sequences=num_seqs_train, radius=3*scale, angle_limits=(0, 360), shape=2,
               velocity_limits=(10.0*scale, 15.0*scale), filepath='./data/polygon.npz', save='npz')

    np.random.seed(5678)
    cannon = BallBox(dt=0.2, res=(32*scale, 32*scale), init_pos=(16*scale, 16*scale), init_std=2.5, wall=None)
    cannon.run(delay=None, iterations=seq_length_test, sequences=num_seqs_test, radius=3*scale, angle_limits=(0, 360), shape=2,
               velocity_limits=(10.0*scale, 15.0*scale), filepath='./data/polygon_test.npz', save='npz')
    
    np.random.seed(7788)
    cannon = BallBox(dt=0.2, res=(32*scale, 32*scale), init_pos=(16*scale, 16*scale), init_std=2.5, wall=None)
    cannon.run(delay=None, iterations=seq_length_valid, sequences=num_seqs_valid, radius=3*scale, angle_limits=(0, 360), shape=2,
               velocity_limits=(10.0*scale, 15.0*scale), filepath='./data/polygon_valid.npz', save='npz')
    
    rs = np.random.RandomState(seed=1515)
    train_valid = rs.rand(num_seqs_train, seq_length_train, 1) < 0.5
    train_valid[:, :20] = True

    rs = np.random.RandomState(seed=1516)
    test_valid = rs.rand(num_seqs_test, seq_length_test, 1) < 0.5
    test_valid[:, :20] = True

    rs = np.random.RandomState(seed=1517)
    valid_valid = rs.rand(num_seqs_valid, seq_length_valid, 1) < 0.5
    valid_valid[:, :20] = True
    
    train_data = PymunkData("./data/{}.npz".format("polygon"))
    test_data = PymunkData("./data/{}_test.npz".format("polygon"))
    valid_data = PymunkData("./data/{}_valid.npz".format("polygon"))
    # train_data = PymunkData("./data/{}.npz".format("polygon"))
    # test_data = PymunkData("./data/{}_test.npz".format("polygon"))
    # valid_data = PymunkData("./data/{}_valid.npz".format("polygon"))
    return train_data, test_data, valid_data, train_valid, test_valid, valid_valid


# Implement Encoder and Decoder hidden layers
class PolyStateEstemGIN(GIN):

    def build_encoder_hidden(self):
        return [
            # 1: Conv Layer
            k.layers.Conv2D(12, kernel_size=5, padding="same"),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            # 2: Conv Layer
            k.layers.Conv2D(12, kernel_size=3, padding="same", strides=2),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            k.layers.Flatten(),
            # 3: Dense Layer
            k.layers.Dense(30, activation=k.activations.relu)]

    def build_decoder_hidden(self):
        return [k.layers.Dense(units=10, activation=k.activations.relu)]

    def build_var_decoder_hidden(self):
        return [k.layers.Dense(units=10, activation=k.activations.relu)]
     
    


def main():


    args = parse_args()

    # config_path = "./config.json"
    config_path = args.config
    configs = read_json(config_path)

    

    for key in configs.keys():
        
        result_path = "./results/" + key
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        config_name = ""
        for elems in configs[key].keys():
            config_name += elems + "_" + str(configs[key][elems]) + "\n"
        with open(result_path + '/config.txt', 'w') as config_wr:
            config_wr.write(config_name)
        config_wr.close()

        print("running config: {}".format(config_name))

        train_data, test_data, valid_data, train_valid, test_valid, valid_valid = generate_poly_filter_dataset(1000, 100, 100, 70, 70, 70)

        #Build Model
        gin = PolyStateEstemGIN(observation_shape=train_data.images.shape[-3:],
                                latent_observation_dim = configs[key]["lod"], 
                                latent_state_dim = configs[key]["lsd"], 
                                output_dim=train_data.state_dim,
                                num_basis=configs[key]["Num_Bases"],
                                never_invalid=True, 
                                cell_type = configs[key]["cell"],
                                Qnetwork= configs[key]["QNetwork"], 
                                Smoothing=configs[key]["Smoothing"], 
                                USE_CONV= bool(configs[key]["Use_Conv_Covar"]), 
                                USE_MLP_AFTER_KGGRU = bool(configs[key]["USE_MLP_AFTER_KGGRU"]),
                                KG_Units = configs[key]["KG_Units"],
                                Xgru_Units =configs[key]["Xgru_Units"],
                                Fgru_Units =configs[key]["Fgru_Units"],
                                KG_InputSize = configs[key]["KG_InputSize"],
                                Xgru_InputSize = configs[key]["Xgru_InputSize"],
                                Fgru_InputSize = configs[key]["Fgru_InputSize"],
                                lr = configs[key]["lr"],
                                lr_decay = configs[key]["lr_decay"],
                                lr_decay_it = configs[key]["lr_decay_iteration"],
                                result_path = result_path)


        # Plot Loss
        if bool(configs[key]["draw_fig"]) == True:
            x_epoch = []
            record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
            fig = plt.figure()
            ax0 = fig.add_subplot(121, title="loss")
        # Train Model
        epochs, batch_size = configs[key]["epochs"], configs[key]["batch_size"]

        Training_Loss = gin.training( gin, train_data.images, train_data.state, valid_data.images, valid_data.state,
                                    test_data.images, test_data.state, epochs, batch_size,
                                    x_epoch, record, fig, ax0, draw_fig= bool(configs[key]["draw_fig"]))
        Test_Loss = gin.testing( gin, test_data.images, test_data.state, batch_size)

if __name__ == '__main__':
	main()
