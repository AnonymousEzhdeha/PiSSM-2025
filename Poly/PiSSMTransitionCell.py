
import tensorflow as tf
from tensorflow import keras as k
import numpy as np
from LayerNormalizer import LayerNormalizer



# Math Util and packing functions
def elup1(x):
    """
    elu + 1 activation faction
    exp(x) if x < 0 else x + 1
    """
    return tf.nn.elu(x) + 1




def pack_state(mean, covar):

    return tf.concat([mean] + [covar], -1)


def unpack_state(state, lsd):
    
    mean = state[..., :lsd]
    covar = state[..., lsd: ]
    
    return mean, covar


def pack_input(obs_mean, obs_covar, obs_valid):
    
    if not obs_valid.dtype == tf.float32:
        obs_valid = tf.cast(obs_valid, tf.float32)
    return tf.concat([obs_mean, obs_covar, obs_valid], axis=-1)


def unpack_input(input_as_vector):
    
    lod = int((input_as_vector.get_shape().as_list()[-1] - 1) / 2)
    obs_mean = input_as_vector[..., :lod]
    obs_covar = input_as_vector[..., lod: -1]
    obs_valid = tf.cast(input_as_vector[..., -1], tf.bool)
    return obs_mean, obs_covar, obs_valid


class DynamicsNet:
    """Implementing the dynamics network in the paper, used as coefficient network to get the state dependent coefficentes to construct
    transition and emission matrices"""

    def __init__(self, lsd, number_of_basis, hidden_units):
        """
        lsd: latent state size 
        number_of_basis: number of basis matrices (k)
        hidden_units: hidden units in the network
        """
        self._hidden_layers = []
        cur_in_shape = lsd
        for u in hidden_units:
            layer = k.layers.Dense(u, activation=k.activations.relu)
            layer.build([None, cur_in_shape])
            cur_in_shape = u
            self._hidden_layers.append(layer)
        # self._out_layer = k.layers.Dense(number_of_basis, activation=k.activations.softmax)
        self._out_layer = k.layers.Dense(units=number_of_basis, activation=None)
        self._out_layer.build([None, cur_in_shape])

    def __call__(self, latent_state):
        """
        latent_state: x_t^+
        
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


class PiSSMTransitionCell(k.layers.Layer):
    """Implementing the GIN cell. This implementation is a subclass of the Keras Layer Class, such
     that it can be used with tf.keras.layers.RNN"""

    def __init__(self,
                 latent_state_dim,
                 latent_obs_dim,
                 number_of_basis,
                 init_kf_matrices,
                 init_Q_matrices,
                 init_KF_matrices,
                 Qnetwork,
                 USE_CONV,
                 USE_MLP_AFTER_KGGRU = False,
                 KG_Units = 50,
                 Xgru_Units =15,
                 Fgru_Units =15,
                 KG_InputSize = 100,
                 Xgru_InputSize = 15,
                 Fgru_InputSize = 15,
                 trans_net_hidden_units=[],
                 never_invalid=False):

        """
        latent_state_dim: dimension of the latent state 
        latent_obs_dim: dimension of the latent observation 
        number_of_basis: number of basis matrices (k)
        init_kf_matrices: initialization of transition matrix 
        init_Q_matrices: initialization of process noise matrix
        init_KF_matrices: initialization of gru cell of the kalman gain network
        KG_Units: state size of gru cell in the GIN cell
        trans_net_hidden_units: list of number 
        never_invalid: boolean indicating whether all observations are available or a part of it is missing
        
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
        self.eye_init = lambda shape, dtype=np.float32: np.eye(*shape, dtype=dtype)
        
        
        self.onelayervar = False # F and H are one layer variable
        self.Qnetwork = Qnetwork
        self.USE_CONV = USE_CONV 
        self.USE_MLP_AFTER_KGGRU = USE_MLP_AFTER_KGGRU
        self.KG_Units = KG_Units
        self.Xgru_Units = Xgru_Units
        self.Fgru_Units = Fgru_Units

        self.KG_InputSize = KG_InputSize
        self.Xgru_InputSize = Xgru_InputSize
        self.Fgru_InputSize = Fgru_InputSize
        
        
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
        self._coefficient_net = DynamicsNet(self._lsd, self._num_basis, self._trans_net_hidden_units)
        self._trainable_weights += self._coefficient_net.weights

        
        
        if self.Qnetwork == "Fmlp":
            #build Q mlp parameters
            self._layer_Q_MLP = k.layers.Dense(self._lsd, activation=lambda x: k.activations.elu(x) + 1)
            
            
        if self.Qnetwork == "Xmlp":
            #build Q mlp parameters
            self._layer_Q_MLP = k.layers.Dense(self._lsd, activation=lambda x: k.activations.elu(x) + 1)
            
            
        if self.Qnetwork == "Fgru":
            #build Q gru parameters
            self.GRUQunit = self.Fgru_Units
            # self.CholeskyKG = self.add_weight(shape=[ self._lsd * self._lod , self._lod * self._lod], name="grulastweight", initializer='random_normal') #(KG, lod^2)
            self.NextWeightGRUQ = self.add_weight(shape=[self.GRUQunit , self._lsd], name="grunextweight", initializer='random_normal') #(gru out, Q)
            self.PrevWeightGRUQ = self.add_weight(shape=[  self._lsd**2 , self.Fgru_InputSize ], name="gruprevweight", initializer='random_normal')# (2*lsd, gru in)
            self.GRUQ = k.layers.GRUCell( self.GRUQunit)
            self.GRUQ_state = self.init_Q_matrices * tf.ones([input_shape[0],  self.GRUQunit ])
            
        if self.Qnetwork == "Xgru":
            #build Q gru parameters
            self.GRUQunit = self.Xgru_Units
            # self.CholeskyKG = self.add_weight(shape=[ self._lsd * self._lod , self._lod * self._lod], name="grulastweight", initializer='random_normal') #(KG, lod^2)
            self.NextWeightGRUQ = self.add_weight(shape=[self.GRUQunit , self._lsd], name="grunextweight", initializer='random_normal') #(gru out, Q)
            self.PrevWeightGRUQ = self.add_weight(shape=[  self._lsd , self.Xgru_InputSize ], name="gruprevweight", initializer='random_normal')# (2*lsd, gru in)
            self.GRUQ = k.layers.GRUCell( self.GRUQunit)
            self.GRUQ_state = self.init_Q_matrices * tf.ones([input_shape[0],  self.GRUQunit ])
            
        
        
        if self.USE_CONV == True:
            #build KG gru parameters
            if self.USE_MLP_AFTER_KGGRU == True:
                self.GRUKGunit = self.KG_Units 
                self.LastWeightKG = self.add_weight(shape=[4 * self._lsd * self._lod , self._lsd * self._lod], name="grulastweight", initializer='random_normal') #(4*KG, KG)
                self.NextWeightKG = self.add_weight(shape=[self.GRUKGunit ,4 * self._lsd * self._lod], name="grunextweight", initializer='random_normal') #(gru out, KG*4)
                self.PrevWeightKG = self.add_weight(shape=[self._lsd**2 + self._lod, self.KG_InputSize], name="gruprevweight", initializer='random_normal')# (lod + lsd^2, gru in)
                self.GRUKG = k.layers.GRUCell( self.GRUKGunit)
                self.GRUKG_state = self.init_KF_matrices * tf.ones([input_shape[0],  self.GRUKGunit ])
            else:
                self.GRUKGunit = self.KG_Units 
                self.LastWeightKG = self.add_weight(shape=[self.GRUKGunit, self._lsd * self._lod], name="grulastweight", initializer='random_normal') #(gru out, KG)
                self.PrevWeightKG = self.add_weight(shape=[self._lsd**2 + self._lod, self.KG_InputSize], name="gruprevweight", initializer='random_normal')# (lod + lsd^2, gru in)
                self.GRUKG = k.layers.GRUCell( self.GRUKGunit)
                self.GRUKG_state = self.init_KF_matrices * tf.ones([input_shape[0],  self.GRUKGunit ])
        if self.USE_CONV == False:
            #build KG gru parameters
            if self.USE_MLP_AFTER_KGGRU == True:
                self.GRUKGunit = self.KG_Units 
                self.LastWeightKG = self.add_weight(shape=[4 * self._lsd * self._lod , self._lsd * self._lod], name="grulastweight", initializer='random_normal') #(4*KG, KG)
                self.NextWeightKG = self.add_weight(shape=[self.GRUKGunit ,4 * self._lsd * self._lod], name="grunextweight", initializer='random_normal') #(gru out, KG*4)
                self.PrevWeightKG = self.add_weight(shape=[self._lsd**2 + self._lod, self.KG_InputSize], name="gruprevweight", initializer='random_normal')# (lod + lsd^2, gru in)
                self.GRUKG = k.layers.GRUCell( self.GRUKGunit)
                self.GRUKG_state = self.init_KF_matrices * tf.ones([input_shape[0],  self.GRUKGunit ])
            else:
                self.GRUKGunit = self.KG_Units 
                self.LastWeightKG = self.add_weight(shape=[self.GRUKGunit, self._lsd * self._lod], name="grulastweight", initializer='random_normal') #(gru out, KG)
                self.PrevWeightKG = self.add_weight(shape=[self._lsd**2 + self._lod, self.KG_InputSize], name="gruprevweight", initializer='random_normal')# (lod + lsd^2, gru in)
                self.GRUKG = k.layers.GRUCell( self.GRUKGunit)
                self.GRUKG_state = self.init_KF_matrices * tf.ones([input_shape[0],  self.GRUKGunit ])
        
        #build dense layer for diag covariance
        self._layer_covar_gru = k.layers.Dense(self._lsd, activation=lambda x: k.activations.elu(x) + 1)
        
        super().build(input_shape)

    def call(self, inputs, states, **kwargs):
        """ similar to the LSTM and GRU cells. The names and parameters of the GIN cell 
        mathch with those of the RNN based cells
        inputs: Mean and covariance vectors 
        states: Last Latent Posterior State 
        
        """
        # unpack inputs
        obs_mean, obs_covar, obs_valid = unpack_input(inputs)
        state_mean, state_covar = states[0]  # mu_t-1 and sigma_t-1 at time t

        # save logp of categorical samples for REINFORCE
        logp_list = []
        # predict step (next prior from current posterior (i.e. cell state))
        prior_mean, prior_covar, logp_list = self._predict(state_mean, state_covar, logp_list) # mu_t|t-1 and sigma_t|t-1 at time t
        

        # update step (current posterior from current prior)
        if self._never_invalid:
            dec_mean, dec_covar = self._update(prior_mean, prior_covar, obs_mean, obs_covar)# mu_t|t  and sigma_t|t  at time t
        else:
            dec_mean, dec_covar = self._masked_update(prior_mean, prior_covar, obs_mean, obs_covar, obs_valid)
        

        # pack outputs;[ dec_mean = mu_t|t, (posterior_mean, i.e. mean filtered),
                       # dec_covar = sigma_t|t, (posterior_covar, i.e. covar filtered) 
                       # prior_mean = mu_t|t-1 = A_t mu_t-1|t-1, 
                       # prior_covar = sigma_t|t-1 = A_t sigma_t-1|t-1 A_t^T + Q_t
                       # transition_matrix = A_t
        output = [dec_mean, dec_covar, prior_mean, prior_covar, self.transition_matrix, logp_list]
        # pack states
        post_state = (dec_mean, dec_covar)
        
        return output, [post_state]
    
    

    def _predict(self, post_mean, prior_covar, logp_list):
        """ 
        Performs prediction step
        
        """
        # compute state dependent transition matrix and Hmatrix
        # coefficients = self._coefficient_net(post_mean)

        logits = self._coefficient_net(post_mean)  # shape: [batch_size, K]
        k_t = tf.random.categorical(logits, num_samples=1)  # shape: [batch_size, 1]
        k_t = tf.squeeze(k_t, axis=-1)  # shape: [batch_size]

        logp = tf.nn.log_softmax(logits)  # [batch, K]
        logp_k = tf.gather(logp, k_t[:, None], batch_dims=1)  # shape [batch, 1]
        logp_list.append(logp_k) # at the enf will be [batch, T, 1]

        if self.onelayervar:
            #stack F matrix 
            F_diag = tf.expand_dims(self.F_weight, 0)
            self._basis_matrices = tf.tile(F_diag, self._num_basis_vec)
            self.Fmatrix = tf.expand_dims(self._basis_matrices, 0)
            
            #stack H matrix
            H_diag = tf.expand_dims(self.H_weight, 0)
            self.H_tiled = tf.tile(H_diag, self._num_basis_vec)
            self.Hmatrix = tf.expand_dims(self.H_tiled, 0)
        
        ## Soft weight matrices
        # scaled_matrices = tf.reshape(coefficients, [-1, self._num_basis, 1, 1]) * self.Fmatrix
        # self.transition_matrix = tf.reduce_sum(scaled_matrices, 1)
        # scaled_H = tf.reshape(coefficients, [-1, self._num_basis, 1, 1]) * self.Hmatrix
        # self.H_matrix = tf.reduce_sum(scaled_H, 1)
            
        # Fetch F_k and H_k
        F_k = tf.gather(tf.squeeze(self.Fmatrix, axis=0) , k_t)  # shape: [batch_size, lsd, lsd]
        H_k = tf.gather(tf.squeeze(self.Hmatrix, axis=0) , k_t)  # shape: [batch_size, m, lsd]
        self.transition_matrix = F_k
        self.H_matrix = H_k

        # predict next prior mean
        expanded_state_mean = tf.expand_dims(post_mean, -1)
        new_mean = tf.squeeze(tf.matmul(self.transition_matrix, expanded_state_mean), -1)
        
        #compute Q 
        prior_covar_matrix = tf.reshape(prior_covar, [prior_covar.shape[0], self._lsd, self._lsd])
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
            new_covar = tf.matmul (tf.matmul(self.transition_matrix, prior_covar_matrix), tf.transpose(self.transition_matrix, perm=[0, 2, 1])) 
        new_covar = tf.reshape(new_covar, [new_covar.shape[0], -1])
     
        return new_mean, new_covar, logp_list
    
    def _predict_q_Fmlp(self, transition_matrix): # F_t is used
        stacked_states = tf.reshape(transition_matrix, [transition_matrix.shape[0], -1])
        Q = self._layer_Q_MLP(stacked_states)   
        return Q
    
    def _predict_q_Xmlp(self, state_mean): # state_mean = mu_t-1|t-1
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
    
    def _predict_q_Xgru(self, state_mean): # state_mean = mu_t-1|t-1
        # stacked_states = tf.concat([state_mean, prior_mean], axis=-1)
        in_GRU = tf.matmul(state_mean, self.PrevWeightGRUQ)
        Q, _ = self.GRUQ(in_GRU, self.GRUQ_state)
        self.GRUQ_state = Q # next self.GRUQ_state
        Q = tf.matmul(Q, self.NextWeightGRUQ)
        Q = elup1(Q)
        return Q
    
    def _predict_kg_gru(self, prior_covar, obs_covar):
        
        if self.USE_CONV == True:
            #propagate covar matrix through the conv2d
            prior_covar_matrix = tf.reshape(prior_covar, [prior_covar.shape[0], self._lsd, self._lsd])
            prior_covar_matrix = tf.expand_dims(prior_covar_matrix, -1)
            prior_covar = self._prop_to_layers(prior_covar_matrix, self.build_conv_gru())
            
        #
        stacked_covars = tf.concat([prior_covar, obs_covar], axis=-1)
        in_GRU = tf.matmul(stacked_covars, self.PrevWeightKG)
        KG, _ = self.GRUKG(in_GRU, self.GRUKG_state)
        self.GRUKG_state = KG # next self.GRUKG_state
        if self.USE_MLP_AFTER_KGGRU == True:
            KG = tf.matmul(KG, self.NextWeightKG)
            KG = tf.matmul(KG, self.LastWeightKG)
        else:
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
    
    def build_conv_gru(self):
        return [
            # 1: Conv Layer
            k.layers.Conv2D(4, kernel_size=5, padding="same", name="first conv2d gru"),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            # 2: Conv Layer
            # k.layers.Conv2D(4, kernel_size=3, padding="same", strides=2, name="second conv2d gru"),
            # LayerNormalizer(),
            # k.layers.Activation(k.activations.relu),
            # k.layers.MaxPool2D(2, strides=2),
            k.layers.Flatten(),
            # 3: Dense Layer
            k.layers.Dense(3*self._lsd, activation=k.activations.linear, name="dense gru")]

    def _masked_update(self, prior_mean, prior_covar, obs_mean, obs_covar, obs_valid):
        

        posterior_mean, posterior_covar_vector = self._update(prior_mean, prior_covar, obs_mean, obs_covar)
        
        #select posterior if obs is available, otherwise select prior
        #select mean
        masked_mean = tf.squeeze(tf.convert_to_tensor([tf.where(obs_valid[i], posterior_mean[i], prior_mean[i]) for i in range(len(obs_valid))],
                                 dtype=tf.float32)) # masked_mean = posterior_mean if obs_valid else prior_mean
        
        
        #select covar        
        masked_covar = tf.convert_to_tensor([tf.where(obs_valid[i], posterior_covar_vector[i], prior_covar[i]) for i in range(len(obs_valid))],
                                 dtype=tf.float32) # masked_mean = posterior_mean if obs_valid else prior_mean
        
        return masked_mean, masked_covar

    def _update(self, prior_mean, prior_covar, obs_mean, obs_covar):
        #(mu_t|t-1, sigma_t|t-1, obs_mu_t, obs_covar_t)
        
        
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
        one of the built in functions of k.layers.RNN, required for the initialization
        
        """
        initial_mean = tf.zeros([batch_size,  self._lsd], dtype=dtype)
        initial_covar = tf.ones([batch_size,  self._lsd * self._lsd], dtype=dtype)
        
        return [(initial_mean, initial_covar)]
    
    @staticmethod
    def _prop_to_layers(inputs, convlayers):
        """propagation"""
        h = inputs
        for layer in convlayers:
            h = layer(h)
        return h
    
    @property
    def state_size(self):
        """ state size as a required function of RNN based cell"""
        return self._lsd + self._lsd**2