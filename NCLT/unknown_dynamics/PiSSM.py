import tensorflow as tf

from tensorflow import keras as k
import numpy as np
from PiSSMTransitionCell import PiSSMTransitionCell, pack_input, unpack_state


class PiSSM(k.models.Model):

    def __init__(self, observation_shape, latent_observation_dim, output_dim, num_basis,
                 trans_net_hidden_units=[], never_invalid=False, cell_type="gin"):
        """
        :param observation_shape: shape of the observation to work with
        :param latent_observation_dim: latent observation dimension (m in paper)
        :param output_dim: dimensionality of model output
        :param num_basis: number of basis matrices (k in paper)
        :param trans_net_hidden_units: hidden units for transition network
        :param never_invalid: if you know a-priori that the observation valid flag will always be positive you can set
                              this to true for slightly increased performance (obs_valid mask will be ignored)
        :param cell_type: type of cell to use "gin" for our approach, "lstm" or "gru" for baselines
        """
        super().__init__()

        self._obs_shape = observation_shape
        self._lod = latent_observation_dim
        self._lsd = self._lod
        self._output_dim = output_dim
        self._never_invalid = never_invalid
        self._ld_output = np.isscalar(self._output_dim)

        # build encoder
        self._enc_hidden_layers = self._time_distribute_layers(self.build_encoder_hidden())

        # we need to ensure the bias is initialized with non-zero values to ensure the normalization does not produce
        # nan
        self._layer_w_mean = k.layers.TimeDistributed(
            k.layers.Dense(self._lod, activation=k.activations.linear,
                           bias_initializer=k.initializers.RandomNormal(stddev=0.05)))
        self._layer_w_mean_norm = k.layers.TimeDistributed(k.layers.Lambda(
            lambda x: x / tf.norm(x, ord='euclidean', axis=-1, keepdims=True)))
        self._layer_w_covar = k.layers.TimeDistributed(
            k.layers.Dense(self._lod, activation=lambda x: k.activations.elu(x) + 1))

        # build transition
        if cell_type.lower() == "gin":
            self._cell = PiSSMTransitionCell(self._lsd, self._lod,
                                           number_of_basis=num_basis,
                                           init_kf_matrices=0.,
                                           init_Q_matrices = 0.,
                                           init_KF_matrices = 0.1,
                                           trans_net_hidden_units=trans_net_hidden_units,
                                           never_invalid=never_invalid)
        elif cell_type.lower() == "lstm":
            print("Running LSTM Baseline")
            self._cell = k.layers.LSTMCell(2 * self._lsd)
        elif cell_type.lower() == "gru":
            print("Running GRU Baseline")
            self._cell = k.layers.GRUCell(2 * self._lsd)
        else:
            raise AssertionError("Invalid Cell type, needs tp be 'rkn', 'lstm' or 'gru'")

        self._layer_rkn = k.layers.RNN(self._cell, return_sequences=True)

        self._dec_hidden = self._time_distribute_layers(self.build_decoder_hidden())
        if self._ld_output:
            # build decoder mean
            self._layer_dec_out = k.layers.TimeDistributed(k.layers.Dense(units=self._output_dim))

            # build decoder variance
            self._var_dec_hidden = self._time_distribute_layers(self.build_var_decoder_hidden())
            self._layer_var_dec_out = k.layers.TimeDistributed(
                k.layers.Dense(units=self._output_dim, activation=lambda x: k.activations.elu(x) + 1))

        else:
            self._layer_dec_out = k.layers.TimeDistributed(
                k.layers.Conv2DTranspose(self._output_dim[-1], kernel_size=3, padding="same",
                                         activation=k.activations.sigmoid))

    def build_encoder_hidden(self):
        """
        Implement encoder hidden layers
        :return: list of encoder hidden layers
        """
        raise NotImplementedError

    def build_decoder_hidden(self):
        """
        Implement mean decoder hidden layers
        :return: list of mean decoder hidden layers
        """
        raise NotImplementedError

    def build_var_decoder_hidden(self):
        """
        Implement var decoder hidden layers
        :return: list of var decoder hidden layers
        """
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: model inputs (i.e. observations)
        :param training: required by k.models.Models
        :param mask: required by k.models.Model
        :return:
        """
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            img_inputs, obs_valid = inputs
        else:
            assert self._never_invalid, "If invalid inputs are possible, obs_valid mask needs to be provided"
            img_inputs = inputs
            obs_valid = tf.ones([tf.shape(img_inputs)[0], tf.shape(img_inputs)[1], 1])

        enc_last_hidden = self._prop_through_layers(img_inputs, self._enc_hidden_layers)
        w_mean = self._layer_w_mean_norm(self._layer_w_mean(enc_last_hidden))
        w_covar = self._layer_w_covar(enc_last_hidden)
        

        # transition
        rkn_in = pack_input(w_mean, w_covar, obs_valid)
        z = self._layer_rkn(rkn_in)

        # post_mean, post_covar = unpack_state(z, self._lsd)
        post_mean, post_covar, prior_mean, prior_covar, self.transition_matrix, logp_list = z
        post_covar = tf.concat(post_covar, -1)

        # decode
        pred_mean = self._layer_dec_out(self._prop_through_layers(post_mean, self._dec_hidden))
        if self._ld_output:
            pred_var = self._layer_var_dec_out(self._prop_through_layers(post_covar, self._var_dec_hidden))
            return tf.concat([pred_mean, pred_var], -1), logp_list
        else:
            return pred_mean, logp_list

    # loss functions
    def gaussian_nll(self, target, pred_mean_var):
        """
        gaussian nll
        :param target: ground truth positions
        :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
        :return: gaussian negative log-likelihood
        """
        pred_mean, pred_var = pred_mean_var[..., :self._output_dim], pred_mean_var[..., self._output_dim:]
        pred_var += 1e-8
        element_wise_nll = 0.5 * (np.log(2 * np.pi) + tf.math.log(pred_var) + ((target - pred_mean)**2) / pred_var)
        sample_wise_error = tf.reduce_sum(element_wise_nll, axis=-1)
        return tf.reduce_mean(sample_wise_error)

    def rmse(self, target, pred_mean_var):
        """
        root mean squared error
        :param target: ground truth positions
        :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
        :return: root mean squared error between targets and predicted mean, predicted variance is ignored
        """
        pred_mean = pred_mean_var[..., :self._output_dim]
        return tf.sqrt(tf.reduce_mean((pred_mean - target) ** 2))

    def reinforce_loss(self, target, pred_mean_var, logp_list):
            """
            output reinforce+basement loss
            target: ground truth
            pred_mean_var: mean and covar 
            calculate reinforce+baseline
            
            """
            pred_mean, pred_var = pred_mean_var[..., :self._output_dim], pred_mean_var[..., self._output_dim:]
            pred_var += 1e-8
            element_wise_nll = 0.5 * (np.log(2 * np.pi) + tf.math.log(pred_var) + ((target - pred_mean)**2) / pred_var)
            sample_wise_error = tf.reduce_sum(element_wise_nll, axis=-1) # [batch, T]
            base_loss = tf.reduce_mean(tf.reduce_sum(sample_wise_error, axis=1)) #scalar 

            reward = -sample_wise_error 
            ###
            ### Baseline+ REINFORCE
            baseline = tf.reduce_mean(reward, axis=0)  # shape [T]
            baseline = tf.expand_dims(baseline, axis=0)  # [1, T]
            baseline = tf.tile(baseline, [pred_mean.shape[0], 1])  # [batch, T]
            logps = tf.squeeze(logp_list, axis=-1) 

            reward = tf.stop_gradient(reward)
            baseline = tf.stop_gradient(baseline)
            reinforce_term = - (reward - baseline) * logps  # shape [batch, T]
            reinforce_loss = tf.reduce_mean(reinforce_term)  # scalar

            return reinforce_loss
    

    def bernoulli_nll(self, targets, predictions, uint8_targets=True):
        """ Computes Binary Cross Entropy
        :param targets:
        :param predictions:
        :param uint8_targets: if true it is assumed that the targets are given in uint8 (i.e. the values are integers
        between 0 and 255), thus they are devided by 255 to get "float image representation"
        :return: Binary Crossentropy between targets and prediction
        """
        if uint8_targets:
            targets = targets / 255.0
        point_wise_error = - (
                    targets * tf.math.log(predictions + 1e-12) + (1 - targets) * tf.math.log(1 - predictions + 1e-12))
        red_axis = [i + 2 for i in range(len(targets.shape) - 2)]
        sample_wise_error = tf.reduce_sum(point_wise_error, axis=red_axis)
        return tf.reduce_mean(sample_wise_error)
    
    def training(self, model, Train_Obs, Train_Target, Valid_Obs, Valid_Target, epochs, batch_size, ratio):
        
        ##val_batching
        Ybatch_val = []
        Ubatch_val = []
        
        for bid in range(int(len(Valid_Target)/batch_size)):
            Ybatch_val.append( Valid_Target[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(Valid_Obs)/batch_size)):
            Ubatch_val.append( Valid_Obs[bid*batch_size:(bid+1)*batch_size])
        
        Ybatch_val = np.array(Ybatch_val)
        Ubatch_val = np.array(Ubatch_val)
        
        ##train_batching
        batch_size = batch_size
        Ybatch = []
        Ubatch = []
        
        for bid in range(int(len(Train_Target)/batch_size)):
            Ybatch.append( Train_Target[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(Train_Obs)/batch_size)):
            Ubatch.append( Train_Obs[bid*batch_size:(bid+1)*batch_size])
        
        Ybatch = np.array(Ybatch)
        Ubatch = np.array(Ubatch)
        
        
        Training_Loss = []
        for epoch in range(epochs):
            loss_show_tr = 0.
            loss_show_val = 0.
            for i in range(len(Ybatch)):
                
                NetIn = Ubatch[i]
                with tf.GradientTape() as tape:
                    preds, logp_list = model(NetIn)
                    # loss = self.rmse(Ybatch[i], preds)
                    reinforce_loss = self.reinforce_loss(Ybatch[i], preds, logp_list)
                    loss_show_tr += loss
                
                ##
                print('epoch: %d  reinforce_loss: %s' % (epoch, reinforce_loss.numpy()))
                if np.isnan(reinforce_loss.numpy()):
                    break
                dynamic_variables = model._layer_rkn.cell._coefficient_net.weights
                # dynamic_variables = model._layer_rkn.cell._coefficient_net.trainable_variables
                gradients = tape.gradient(reinforce_loss, dynamic_variables)
                tf.keras.optimizers.Adam(learning_rate = self.lr, clipnorm=5.0).apply_gradients(zip(gradients, dynamic_variables))


                ##
                with tf.GradientTape() as tape2:
                    preds, _ = model(NetIn)  # recompute
                    phi_loss = self.gaussian_nll(Ybatch[i], preds)
                
                print('epoch: %d  base_loss: %s' % (epoch, phi_loss.numpy()))
                if np.isnan(phi_loss.numpy()):
                    break

                phi_vars =  [v for v in model.trainable_variables if all(v is not d for d in dynamic_variables)]
                grads_phi = tape2.gradient(phi_loss, phi_vars)
                tf.keras.optimizers.Adam(learning_rate = self.lr, clipnorm=5.0).apply_gradients(zip(grads_phi, phi_vars))
                ##
                if i %10==0:
                    rand_sel = np.random.randint(0, len(Valid_Ubatch))
                    val_preds, val_logp_list = model(Valid_Ubatch[rand_sel])
                    val_reinforce_loss = self.reinforce_loss(Valid_Ybatch[rand_sel], val_preds, val_logp_list)
                    val_phi_loss = self.gaussian_nll(Valid_Ybatch[rand_sel], val_preds)
                    val_loss = val_reinforce_loss + val_phi_loss
                    print('val loss: %s' % (val_loss.numpy()))
                
                loss = phi_loss + reinforce_loss
                print('epoch: %d  total_loss: %s' % (epoch, loss.numpy()))
                if np.isnan(loss.numpy()):
                    break
                            
                # variables = model.trainable_variables
                # if i%(ratio-1)==0 and i!= 0:
                #     gradients = tape.gradient(loss_show_tr, variables)
                #     tf.keras.optimizers.Adam(clipnorm=5.0).apply_gradients(zip(gradients, variables))
                        
                # rand_sel = np.random.randint(0, len(Ubatch_val))
                # val_preds = model(Ubatch_val[rand_sel])
                # loss_show_val += self.rmse(Ybatch_val[rand_sel], val_preds)    
                # if i%(ratio-1)==0 and i!= 0:
                #     print('val loss %s' % (loss_show_val.numpy() ))
                #     loss_show_val = 0
                
                #print('epoch %d  loss %s' % (epoch, self.rmse(Ybatch[i], preds).numpy() ))
                if i%(ratio-1) ==0 and i !=0:
                    print('epoch %d  loss %s' % (epoch, loss.numpy() ))
                    loss = 0
                
                
                Training_Loss.append(loss/batch_size)  
        return Training_Loss
    
    def testing(self, model, test_obs, test_targets, batch_size, ratio):
        batch_size = 1
        Ybatch = []
        Ubatch = []
        
        for bid in range(int(len(test_targets)/batch_size)):
            Ybatch.append( test_targets[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(test_obs)/batch_size)):
            Ubatch.append( test_obs[bid*batch_size:(bid+1)*batch_size])        
        Ybatch = np.array(Ybatch)
        Ubatch = np.array(Ubatch)

        Test_Loss = []
        Test_loss_show = 0
        Test_loss_show_arr = []
        for i in range(len(Ybatch)):
            NetIn = Ubatch[i]
            preds = model(NetIn)
            loss = self.rmse(Ybatch[i], preds)
            #print('test loss: %s' % (loss))
            Test_Loss.append(loss.numpy())
            
            Test_loss_show += loss
            if i% (ratio-1) ==0 and i!=0:
                print('test loss: %s' % (Test_loss_show.numpy()))
                Test_loss_show_arr.append(Test_loss_show.numpy())
                Test_loss_show = 0
            
        print('total test_loss %s' % (tf.reduce_mean(Test_loss_show_arr)))
        return Test_Loss

    @staticmethod
    def _prop_through_layers(inputs, layers):
        """propagates inputs through layers"""
        h = inputs
        for layer in layers:
            h = layer(h)
        return h

    @staticmethod
    def _time_distribute_layers(layers):
        """wraps layers with k.layers.TimeDistributed"""
        td_layers = []
        for l in layers:
            td_layers.append(k.layers.TimeDistributed(l))
        return td_layers