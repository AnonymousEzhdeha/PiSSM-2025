import tensorflow as tf
from tensorflow import keras as k
import numpy as np
from LayerNormalization import LayerNormalization



# Math Util
def elup1(x):
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return tf.nn.elu(x) + 1


# Pack and Unpack functions

def pack_state(mean, covar):
    """ packs system state (either prior or posterior) into single vector
    :param mean: state mean as vector
    :param covar: state covar as list [upper, lower, side]
    :return: state as single vector of size 5 * latent observation dim,
    order of entries: mean, covar_upper, covar_lower, covar_side
    """
    return tf.concat([mean] + [covar], -1)


def unpack_state(state, lsd):
    """ unpacks system state packed by 'pack_state', can be used to unpack cell output (in non-debug case)
    :param state: packed state, containg mean and covar as single vector
    :return: mean, list of covariances (upper, lower, side)
    """
    
    mean = state[..., :lsd]
    covar = state[..., lsd: ]
    
    return mean, covar


def pack_input(obs_mean, obs_covar, obs_valid):
    """ packs cell input. All inputs provided to the cell should be packed using this function
    :param obs_mean: observation mean
    :param obs_covar: observation covariance
    :param obs_valid: flag indication if observation is valid
    :return: packed input
    """
    if not obs_valid.dtype == tf.float32:
        obs_valid = tf.cast(obs_valid, tf.float32)
    return tf.concat([obs_mean, obs_covar, obs_valid], axis=-1)


def unpack_input(input_as_vector):
    """ used to unpack input vectors that where packed with 'pack_input
    :param input_as_vector packed input
    :return: observation mean, observation covar, observation valid flag
    """
    lod = int((input_as_vector.get_shape().as_list()[-1] - 1) / 2)
    obs_mean = input_as_vector[..., :lod]
    obs_covar = input_as_vector[..., lod: -1]
    obs_valid = tf.cast(input_as_vector[..., -1], tf.bool)
    return obs_mean, obs_covar, obs_valid



class TransitionNet:
    """Implements a simple dense network, used as coefficient network to get the state dependent coefficentes for the
       transition model """

    def __init__(self, lsd, number_of_basis, hidden_units):
        """
        :param lsd: latent state size (i.e. network input dimension)
        :param number_of_basis: number of basis matrices (i.e. network output dimension)
        :param hidden_units: list of numbers of hidden units
        """
        self._hidden_layers = []
        cur_in_shape = lsd
        for u in hidden_units:
            layer = k.layers.Dense(u, activation=k.activations.relu)
            layer.build([None, cur_in_shape])
            cur_in_shape = u
            self._hidden_layers.append(layer)
        self._out_layer = k.layers.Dense(number_of_basis, activation=k.activations.softmax)
        self._out_layer.build([None, cur_in_shape])

    def __call__(self, latent_state):
        """
        :param latent_state: current latent state
        :return: coefficents for transition basis matrices
        """
        h = latent_state
        for hidden_layer in self._hidden_layers:
            h = hidden_layer(h)
        return self._out_layer(h)

    @property
    def weights(self):
        weigths = self._out_layer.trainable_weights
        for hidden_layer in self._hidden_layers:
            weigths += hidden_layer.trainable_weights
        return weigths


class GINTransitionCell(k.layers.Layer):
    """Implementation of the actual transition cell. This is implemented as a subclass of the Keras Layer Class, such
     that it can be used with tf.keras.layers.RNN"""

    def __init__(self,
                 latent_state_dim,
                 latent_obs_dim,
                 number_of_basis,
                 init_kf_matrices,
                 init_Q_matrices,
                 init_KF_matrices,
                 trans_net_hidden_units=[],
                 never_invalid=False):

        """
        :param latent_state_dim: dimensionality of latent state (n in paper)
        :param latent_obs_dim: dimensionality of latent observation (m in paper)
        :param number_of_basis: number of basis matrices used (k in paper)
        :param trans_net_hidden_units: list of number (numbers of hidden units per layer in coefficient network)
        :param never_invalid: if you know a-priori that the observation valid flag will always be positive you can set
                              this to true for slightly increased performance (obs_valid mask will be ignored)
        """

        super().__init__()

        self._lsd = latent_state_dim
        self._lod = latent_obs_dim
        self._num_basis = number_of_basis
        self._never_invalid = never_invalid
        self._trans_net_hidden_units = trans_net_hidden_units
        self.init_kf_matrices = init_kf_matrices
        self.init_Q_matrices = init_Q_matrices
        self.init_KF_matrices = init_KF_matrices
        
        self.GRUKGunit = 2 * self._lsd**2 * 10
        
        self.onelayervar = False # F and H are one layer variable
        self.Qnetwork = "Xgru"
        
    def build(self, input_shape):
        
        if self.onelayervar:
            ##### (lsd, lsd) weight for F
            ##### (lod, lsd) weight for H
            self._num_basis_vec = tf.constant([self._num_basis, 1, 1], tf.int32)
            #build F_weight
            F_init = np.expand_dims(np.ones((self._lsd, self._lsd), dtype=np.float32), 0)
            self.F_weight = self.add_weight(shape=[ self._lsd, self._lsd], name = "F_weight", initializer=k.initializers.Constant(F_init))
            
            #build H_weight
            Init_Hmatrix_Identity =  tf.eye(self._lod, num_columns=self._lsd)
            self.H_weight = self.add_weight(shape=[self._lod, self._lsd], name="H_weight",
                                         initializer=k.initializers.Constant(Init_Hmatrix_Identity))
        else:
            # build F matrix basis: (num_basis, lsd, lsd) weights
            Init_Fmatrix = np.tile( np.expand_dims(np.array(self.init_kf_matrices * np.random.randn(self._lsd, self._lsd).astype(np.float32)),0) ,
                                    [1, self._num_basis, 1, 1])
            Init_Fmatrix_Identity = np.tile( np.expand_dims(np.array(1 * np.eye(self._lsd).astype(np.float32)),0) ,
                                    [1, self._num_basis, 1, 1])
            self.Fmatrix = tf.Variable(Init_Fmatrix,  trainable=True)
            
                    
            # build H matrix basis: (num_basis, lod, lsd) weights
            Init_Hmatrix = np.tile( np.expand_dims(np.array(self.init_kf_matrices * np.random.randn(self._lod, self._lsd).astype(np.float32)),0) ,
                                    [1, self._num_basis, 1, 1])
            self.Hmatrix = tf.Variable(Init_Hmatrix,  trainable=True)

        # build state dependent coefficent network
        self._coefficient_net = TransitionNet(self._lsd, self._num_basis, self._trans_net_hidden_units)
        self._trainable_weights += self._coefficient_net.weights

        
        if self.Qnetwork == "Fmlp":
            #build Q mlp parameters
            self._layer_Q_MLP = k.layers.Dense(self._lsd, activation=lambda x: k.activations.elu(x) + 1)
            
            
        if self.Qnetwork == "Xmlp":
            #build Q mlp parameters
            self._layer_Q_MLP = k.layers.Dense(self._lsd, activation=lambda x: k.activations.elu(x) + 1)
            
            
        if self.Qnetwork == "Fgru":
            #build Q gru parameters
            self.GRUQunit = 15
            self.NextWeightGRUQ = self.add_weight(shape=[self.GRUQunit , self._lsd], name="grunextweight", initializer='random_normal') #(gru out, Q)
            self.PrevWeightGRUQ = self.add_weight(shape=[  self._lsd**2 , self.GRUQunit], name="gruprevweight", initializer='random_normal')# (2*lsd, gru in)
            self.GRUQ = k.layers.GRUCell( self.GRUQunit)
            self.GRUQ_state = self.init_Q_matrices * tf.ones([input_shape[0],  self.GRUQunit ])
            
        if self.Qnetwork == "Xgru":
            #build Q gru parameters
            self.GRUQunit = 15
            self.NextWeightGRUQ = self.add_weight(shape=[self.GRUQunit , self._lsd], name="grunextweight", initializer='random_normal') #(gru out, Q)
            self.PrevWeightGRUQ = self.add_weight(shape=[  self._lsd , self.GRUQunit], name="gruprevweight", initializer='random_normal')# (2*lsd, gru in)
            self.GRUQ = k.layers.GRUCell( self.GRUQunit)
            self.GRUQ_state = self.init_Q_matrices * tf.ones([input_shape[0],  self.GRUQunit ])
        
        
        #build KG gru parameters
        # self.CholeskyKG = self.add_weight(shape=[self._lsd * self._lod , self._lod * self._lod], name="grulastweight", initializer='random_normal') #(KG, lod^2)
        self.LastWeightKG = self.add_weight(shape=[4 * self._lsd * self._lod , self._lsd * self._lod], name="grulastweight", initializer='random_normal') #(4*KG, KG)
        self.NextWeightKG = self.add_weight(shape=[self.GRUKGunit ,4 * self._lsd * self._lod], name="grunextweight", initializer='random_normal') #(gru out, KG*4)
        self.PrevWeightKG = self.add_weight(shape=[self._lsd**2 + self._lod, self.GRUKGunit * 2], name="gruprevweight", initializer='random_normal')# (lod + lsd^2, gru in)
        self.GRUKG = k.layers.GRUCell( self.GRUKGunit)
        self.GRUKG_state = self.init_KF_matrices * tf.ones([input_shape[0],  self.GRUKGunit ])
         
        #build dense layer for diag covariance
        self._layer_covar_gru = k.layers.Dense(self._lsd, activation=lambda x: k.activations.elu(x) + 1)


        super().build(input_shape)

    def call(self, inputs, states, **kwargs):
        """Performs one transition step (prediction followed by update in Kalman Filter terms)
        Parameter names match those of superclass - same signature as k.layers.LSTMCell
        :param inputs: Latent Observations (mean and covariance vectors concatenated)
        :param states: Last Latent Posterior State (mean and covariance vectors concatenated)
        :param scope: See super
        :return: cell output: current posterior (if not debug, else current posterior, prior and kalman gain)
                 cell state: current posterior
        """
        # unpack inputs
        obs_mean, obs_covar, obs_valid = unpack_input(inputs)
        state_mean, state_covar = unpack_state(states[0], self._lsd) # mu_t-1 and sigma_t-1 at time t

        # predict step (next prior from current posterior (i.e. cell state))
        prior_mean, prior_covar = self._predict(state_mean, state_covar) # mu_t|t-1 and sigma_t|t-1 at time t
        

        # update step (current posterior from current prior)
        if self._never_invalid:
            dec_mean, dec_covar = self._update(prior_mean, prior_covar, obs_mean, obs_covar)
        else:
            dec_mean, dec_covar = self._masked_update(prior_mean, prior_covar, obs_mean, obs_covar, obs_valid)
        

        # pack outputs
        post_state = pack_state(dec_mean, dec_covar)
        return post_state, [post_state]
    
    

    def _predict(self, post_mean, post_covar):
        """ Performs prediction step
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :return: current prior latent state mean and covariance
        """
        # compute state dependent transition matrix and Hmatrix
        coefficients = self._coefficient_net(post_mean)
        
        if self.onelayervar:
            #stack F matrix 
            F_diag = tf.expand_dims(self.F_weight, 0)
            self._basis_matrices = tf.tile(F_diag, self._num_basis_vec)
            self.Fmatrix = tf.expand_dims(self._basis_matrices, 0)
            
            #stack H matrix
            H_diag = tf.expand_dims(self.H_weight, 0)
            self.H_tiled = tf.tile(H_diag, self._num_basis_vec)
            self.Hmatrix = tf.expand_dims(self.H_tiled, 0)
        
        scaled_matrices = tf.reshape(coefficients, [-1, self._num_basis, 1, 1]) * self.Fmatrix
        self.transition_matrix = tf.reduce_sum(scaled_matrices, 1)
        
        scaled_H = tf.reshape(coefficients, [-1, self._num_basis, 1, 1]) * self.Hmatrix
        self.H_matrix = tf.reduce_sum(scaled_H, 1)

        # predict next prior mean
        expanded_state_mean = tf.expand_dims(post_mean, -1)
        new_mean = tf.squeeze(tf.matmul(self.transition_matrix, expanded_state_mean), -1)
        
        #compute Q by gru cell
        prior_covar_matrix = tf.reshape(post_covar, [post_covar.shape[0], self._lsd, self._lsd])
        if self.Qnetwork == "Fmlp":
            Q = self._predict_q_Fmlp(self.transition_matrix)
            new_covar = tf.matmul (tf.matmul(self.transition_matrix, prior_covar_matrix), tf.transpose(self.transition_matrix, perm=[0, 2, 1])) + tf.linalg.diag(Q)
        if self.Qnetwork == "Fgru":
            Q = self._predict_q_Fgru(self.transition_matrix)
            new_covar = tf.matmul (tf.matmul(self.transition_matrix, prior_covar_matrix), tf.transpose(self.transition_matrix, perm=[0, 2, 1])) + tf.linalg.diag(Q)
        if self.Qnetwork == "Xmlp":
            Q = self._predict_q_Xmlp(post_mean)
            new_covar = tf.matmul (tf.matmul(self.transition_matrix, prior_covar_matrix), tf.transpose(self.transition_matrix, perm=[0, 2, 1])) + tf.linalg.diag(Q)
        if self.Qnetwork == "Xgru":
            Q = self._predict_q_Xgru(post_mean)
            new_covar = tf.matmul (tf.matmul(self.transition_matrix, prior_covar_matrix), tf.transpose(self.transition_matrix, perm=[0, 2, 1])) + tf.linalg.diag(Q)
        if self.Qnetwork == "nothing":
            # Q = self._predict_q_Xgru(post_mean)
            new_covar = tf.matmul (tf.matmul(self.transition_matrix, prior_covar_matrix), tf.transpose(self.transition_matrix, perm=[0, 2, 1])) #+ tf.linalg.diag(Q)
        new_covar = tf.reshape(new_covar, [new_covar.shape[0], -1])
     
        return new_mean, new_covar
    
    def _predict_q_Fmlp(self, transition_matrix): # F_t is used
        stacked_states = tf.reshape(transition_matrix, [transition_matrix.shape[0], -1])
        Q = self._layer_Q_MLP(stacked_states)   
        return Q
    
    def _predict_q_Xmlp(self, state_mean): # state_mean = mu_t-1|t-1, prior_mean = mu_t|t-1
        stacked_states = tf.reshape(state_mean, [state_mean.shape[0], -1])
        Q = self._layer_Q_MLP(stacked_states)   
        return Q
    
    def _predict_q_Fgru(self, transition_matrix): # F_t is used
        stacked_states = tf.reshape(transition_matrix, [transition_matrix.shape[0], -1])
        in_GRU = tf.matmul(stacked_states, self.PrevWeightGRUQ)
        Q, _ = self.GRUQ(in_GRU, self.GRUQ_state)
        self.GRUQ_state = Q # next self.GRUQ_state
        Q = tf.matmul(Q, self.NextWeightGRUQ)
        Q = elup1(Q)
        return Q
    
    def _predict_q_Xgru(self, state_mean): # state_mean = mu_t-1|t-1, prior_mean = mu_t|t-1
        # stacked_states = tf.concat([state_mean, prior_mean], axis=-1)
        in_GRU = tf.matmul(state_mean, self.PrevWeightGRUQ)
        Q, _ = self.GRUQ(in_GRU, self.GRUQ_state)
        self.GRUQ_state = Q # next self.GRUQ_state
        Q = tf.matmul(Q, self.NextWeightGRUQ)
        Q = elup1(Q)
        return Q
    
    def _predict_kg_gru(self, prior_covar, obs_covar):

        stacked_covars = tf.concat([prior_covar, obs_covar], axis=-1)
        in_GRU = tf.matmul(stacked_covars, self.PrevWeightKG)
        KG, _ = self.GRUKG(in_GRU, self.GRUKG_state)
        self.GRUKG_state = KG # next self.GRUKG_state
        KG = tf.matmul(KG, self.NextWeightKG)
        KG = tf.matmul(KG, self.LastWeightKG)
        KG = tf.reshape(KG, [KG.shape[0], self._lsd, self._lod])

        # KG = tf.matmul(KG, self.NextWeightKG)
        # KG = tf.matmul(KG, self.LastWeightKG)
        # KG = tf.matmul(KG, self.CholeskyKG)
        # KG = tf.reshape(KG, [KG.shape[0], self._lod, self._lod])
        # Diag_KG = tf.linalg.diag_part(KG)
        # Diag_elements_dense = self._layer_covar_gru(Diag_KG)
        # elup_Diag_elements = tf.linalg.diag(elup1(Diag_elements_dense))
        # Positive_KG = elup_Diag_elements + ( KG - tf.linalg.diag(tf.linalg.diag_part(KG)))
        # KG = tf.matmul(tf.matmul(prior_covar, tf.transpose(self.H_matrix)), tf.matmul(Positive_KG, tf.transpose(Positive_KG)))

        return KG
    
    def _masked_update(self, prior_mean, prior_covar, obs_mean, obs_covar, obs_valid):
        """ Ensures update only happens if observation is valid
        CAVEAT: You need to ensure that obs_mean and obs_covar do not contain NaNs, even if they are invalid.
        If they do this will cause problems with gradient computation (they will also be NaN) due to how tf.where works
        internally (see: https://github.com/tensorflow/tensorflow/issues/2540)
        :param prior_mean: current prior latent state mean
        :param prior_covar: current prior latent state convariance
        :param obs_mean: current latent observation mean
        :param obs_covar: current latent observation covariance
        :param obs_valid: indicating if observation is valid
        :return: current posterior latent state mean and covariance
        """

        posterior_mean, posterior_covar_vector = self._update(prior_mean, prior_covar, obs_mean, obs_covar)
        
        #select posterior if obs is available, otherwise select prior
        #select mean
        # masked_mean = tf.where(obs_valid, posterior_mean, prior_mean)
        masked_mean = tf.squeeze(tf.convert_to_tensor([tf.where(obs_valid[i], posterior_mean[i], prior_mean[i]) for i in range(len(obs_valid))],
                                 dtype=tf.float32)) # masked_mean = posterior_mean if obs_valid else prior_mean
        
        
        #select covar        
        masked_covar = tf.convert_to_tensor([tf.where(obs_valid[i], posterior_covar_vector[i], prior_covar[i]) for i in range(len(obs_valid))],
                                 dtype=tf.float32) # masked_mean = posterior_mean if obs_valid else prior_mean
        
        return masked_mean, masked_covar


    def _update(self, prior_mean, prior_covar, obs_mean, obs_covar):
        #(mu_t|t-1, sigma_t|t-1, obs_mu_t, obs_covar_t)
        """Performs update step
        :param prior_mean: current prior latent state mean
        :param prior_covar: current prior latent state covariance
        :param obs_mean: current latent observation mean
        :param obs_covar: current latent covariance mean
        :return: current posterior latent state and covariance
        """
        
        KG = self._predict_kg_gru( prior_covar, obs_covar)
        
        # posterior mean
        expanded_prior_mean_mean = tf.expand_dims(prior_mean, -1)
        expanded_obs_mean = tf.expand_dims(obs_mean, -1)
        diff_y = expanded_obs_mean - tf.matmul(self.H_matrix, expanded_prior_mean_mean)
        posterior_mean = prior_mean - tf.squeeze(tf.matmul(KG, diff_y))
        
        #posterior covar
        prior_covar_matrix = tf.reshape(prior_covar, [prior_covar.shape[0], self._lsd, self._lsd])
        S = tf.matmul( tf.matmul(self.H_matrix , prior_covar_matrix), tf.transpose(self.H_matrix, perm=[0, 2, 1])) + tf.linalg.diag(obs_covar)
        posterior_covar_matrix = prior_covar_matrix - tf.matmul(tf.matmul(KG,S), tf.transpose(KG, perm=[0, 2, 1]))
        #
        Diag_elements = tf.linalg.diag_part(posterior_covar_matrix)
        Diag_elements_dense = self._layer_covar_gru(Diag_elements)
        elup_Diag_elements = tf.linalg.diag(elup1(Diag_elements_dense))
        posterior_covar_matrix = elup_Diag_elements + ( posterior_covar_matrix - tf.linalg.diag(tf.linalg.diag_part(posterior_covar_matrix)))
        #
        posterior_covar_vector = tf.reshape(posterior_covar_matrix, [posterior_covar_matrix.shape[0], -1])
        return posterior_mean, posterior_covar_vector # mu_t|t, sigma_t|t at t
        
    
    def get_initial_state(self, inputs, batch_size, dtype):
        """
        Signature matches the run required by k.layers.RNN
        :param inputs:
        :param batch_size:
        :param dtype:
        :return:
        """
        initial_mean = tf.zeros([batch_size,  self._lsd], dtype=dtype)
        initial_covar = tf.ones([batch_size,  self._lsd * self._lsd], dtype=dtype)
        
        return tf.concat([initial_mean, initial_covar], -1)
    
    @staticmethod
    def _prop_to_layers(inputs, convlayers):
        """propagates inputs through layers"""
        h = inputs
        for layer in convlayers:
            h = layer(h)
        return h
    
    @property
    def state_size(self):
        """ required by k.layers.RNN"""
        return self._lsd + self._lsd**2