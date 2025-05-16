
import tensorflow as tf
from tensorflow import keras as k
from LorenzSysModel import SystemModel
from parameters import m, n
import model
import numpy as np
from PiSSM import PiSSM


def Generate_Data(num_seqs_train=1, num_seqs_test=1, num_seqs_valid=1, seq_length_train=1, seq_length_test=1, seq_length_valid=1, q=1, r=1):
    
    obj_sysmodel = SystemModel( model.f, q, model.h, r, seq_length_train,  m, n)
    train_obs, train_targets = obj_sysmodel.GenerateBatch(num_seqs_train, seq_length_train, randomInit=True)
    obj_sysmodel = SystemModel( model.f, q, model.h, r, seq_length_test,  m, n)
    test_obs, test_targets = obj_sysmodel.GenerateBatch(num_seqs_test, seq_length_test, randomInit=True)
    obj_sysmodel = SystemModel( model.f, q, model.h, r, seq_length_valid,  m, n)
    valid_obs, valid_targets = obj_sysmodel.GenerateBatch(num_seqs_valid, seq_length_valid, randomInit=True)

    #reorder data to #(num of samples, length, features)
    train_obs = np.transpose(train_obs.numpy(), axes=[0,2,1]) 
    train_targets = np.transpose(train_targets.numpy(), axes=[0,2,1])
    test_obs = np.transpose(test_obs.numpy(), axes=[0,2,1])
    test_targets = np.transpose(test_targets.numpy(), axes=[0,2,1])
    valid_obs = np.transpose(valid_obs.numpy(), axes=[0,2,1])
    valid_targets = np.transpose(valid_targets.numpy(), axes=[0,2,1])
    return train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets


# Implement Encoder and Decoder hidden layers
class LorenzStateEstemPiSSM(PiSSM):
    def build_encoder_hidden(self):
        return [k.layers.Dense(units=3, activation=k.activations.relu)]

    def build_decoder_hidden(self):
        return [k.layers.Dense(units=3, activation=k.activations.relu)]

    def build_var_decoder_hidden(self):
        return [k.layers.Dense(units=3, activation=k.activations.relu)]
     

def main():

    r2 = 0.25 ### r^2 = 0.25
    r = np.sqrt(r2)
    vdB = -20 # ratio v=q2/r2
    v = 10**(vdB/10)
    q2 = v*r2
    q = np.sqrt(q2) 




    ##data gen
    train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets = Generate_Data(num_seqs_train=20, num_seqs_test=1,
                                       num_seqs_valid=2, seq_length_train=200, seq_length_test=200, seq_length_valid=280, q=q, r=r)

    ##
    ## Build Model
    Lorenz = LorenzStateEstemPiSSM(observation_shape=train_obs.shape[-1], latent_observation_dim=3, output_dim=3, num_basis=6,
                                 never_invalid=True)


    # # Train Model
    epochs= 100
    Training_Loss = Lorenz.training( Lorenz, train_obs, train_targets,
                                 valid_obs, valid_targets, epochs)
    Test_Loss = Lorenz.testing( Lorenz, test_obs, test_targets)

if __name__ == '__main__':
	main()
