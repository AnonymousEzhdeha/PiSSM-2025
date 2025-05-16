import tensorflow as tf
from tensorflow import keras as k
import numpy as np
from LayerNormalizer import LayerNormalizer



# Math Util
def elup1(x):
    """
    elu + 1 activation faction
    exp(x) if x < 0 else x + 1
    """
    return tf.nn.elu(x) + 1


# Pack and Unpack functions

def pack_state(mean, covar):
     
    return tf.concat([mean] + [covar], -1)


def unpack_state(state, lsd):
     
    
    mean = state[..., :lsd]
    covar = state[..., lsd: ]
    
    return mean, covar


def pack_input_smooth(obs_mean, obs_covar):
     
    return tf.concat([obs_mean, obs_covar], axis=-1)


def unpack_input_smooth(input_as_vector):
     
    lod = int((input_as_vector.get_shape().as_list()[-1]) / 2)
    filt_mean = input_as_vector[..., :lod]
    filt_covar = input_as_vector[..., lod:]
    return filt_mean, filt_covar


class PiSSMSmoothingCell(k.layers.Layer):
    """Implementing the GIN cell. This implementation is a subclass of the Keras Layer Class, such
     that it can be used with tf.keras.layers.RNN"""

    def __init__(self,
                 latent_state_dim,
                 latent_obs_dim,
                 init_kf_matrices,
                 init_KF_matrices,
                 USE_CONV):

        """
        latent_state_dim: dimension of the latent state 
        latent_obs_dim: dimension of the latent observation 
        number_of_basis: number of basis matrices (k)
        init_kf_matrices: initialization of transition matrix 
        init_Q_matrices: initialization of process noise matrix
        init_KF_matrices: initialization of gru cell of the kalman gain network
        trans_net_hidden_units: list of number 
        never_invalid: boolean indicating whether all observations are available or a part of it is missing
        
        """


        super().__init__()

        
        self._lsd = latent_state_dim
        self._lod = latent_obs_dim
        self.init_kf_matrices = init_kf_matrices
        self.init_KF_matrices = init_KF_matrices
        self.eye_init = lambda shape, dtype=np.float32: np.eye(*shape, dtype=dtype)
        self.USE_CONV = USE_CONV 
        
    def build(self, input_shape):
        input_shape = input_shape[0]      
        
        if self.USE_CONV == True:
            #build J gru parameters
            self.GRUJunit = 2 * self._lsd 
            # self.CholeskyKG = self.add_weight(shape=[ self._lsd * self._lod , self._lsd * self._lsd], name="grulastweight", initializer='random_normal') #(J, lsd^2)
            self.LastWeightKG = self.add_weight(shape=[2 * self._lsd * self._lsd , self._lsd * self._lsd], name="grulastweight", initializer='random_normal') #(4*J, J)
            self.NextWeightKG = self.add_weight(shape=[self.GRUJunit ,2 * self._lsd * self._lsd], name="grunextweight", initializer='random_normal') #(gru out, J*4)
            self.PrevWeightKG = self.add_weight(shape=[3*self._lsd , self.GRUJunit*2], name="gruprevweight", initializer='random_normal')# ( lsd^2, gru in)
            self.GRUJ = k.layers.GRUCell( self.GRUJunit)
            self.GRUJ_state = self.init_KF_matrices * tf.ones([input_shape[0],  self.GRUJunit ])
        if self.USE_CONV == False:
            #build J gru parameters
            self.GRUJunit = 2 * self._lsd 
            # self.CholeskyKG = self.add_weight(shape=[ self._lsd * self._lod , self._lsd * self._lsd], name="grulastweight", initializer='random_normal') #(J, lsd^2)
            self.LastWeightKG = self.add_weight(shape=[2 * self._lsd * self._lsd , self._lsd * self._lsd], name="grulastweight", initializer='random_normal') #(4*J, J)
            self.NextWeightKG = self.add_weight(shape=[self.GRUJunit ,2 * self._lsd * self._lsd], name="grunextweight", initializer='random_normal') #(gru out, J*4)
            self.PrevWeightKG = self.add_weight(shape=[self._lsd**2 , self.GRUJunit*2], name="gruprevweight", initializer='random_normal')# ( lsd^2, gru in)
            self.GRUJ = k.layers.GRUCell( self.GRUJunit)
            self.GRUJ_state = self.init_KF_matrices * tf.ones([input_shape[0],  self.GRUJunit ])
        
        #build dense layer for diag covariance
        self._layer_covar_gru = k.layers.Dense(self._lsd, activation=lambda x: k.activations.elu(x) + 1)
        
        super().build(input_shape)

    def call(self, inputs, states, **kwargs):
        """ similar to the LSTM and GRU cells. The names and parameters of the GIN cell 
        mathch with those of the RNN based cells
        inputs: Mean and covariance vectors 
        states: Last Latent Posterior State 
        
        """
        # unpack inputs; filt_mean_t = mu_t|t
        #                filt_covar_t = sigma_t|t
        #                prior_mean_tp1 = mu_t+1|t = A_t+1 mu_t|t
        #                prior_covar_tp1 = sigma_t+1|t = A|t+1 sigma_t|t A_t+1^T + Q_t+1
        #                transition_matrix_tp1 = A_t+1
        filt_t_mean, filt_t_covar, prior_tp1_mean, prior_tp1_covar, transition_tp1_matrix = inputs
        # self.A_tp1_matrix = transition_tp1_matrix
        smooth_tp1_mean, smooth_tp1_covar = unpack_state( states[0], self._lsd)  
        
        # # update step (current smooth from next smooth)
        smooth_t_mean, smooth_t_covar = self._update(smooth_tp1_mean, smooth_tp1_covar, filt_t_mean, 
                                filt_t_covar, prior_tp1_mean, prior_tp1_covar, transition_tp1_matrix)

        post_state = pack_state(smooth_t_mean, smooth_t_covar)

        
        return [smooth_t_mean, smooth_t_covar], [post_state]
    

    
    def _predict_J_gru(self, prior_covar):
        
        if self.USE_CONV == True:
            #propagate covar matrix through the conv2d
            prior_covar_matrix = tf.reshape(prior_covar, [prior_covar.shape[0], self._lsd, self._lsd])
            prior_covar_matrix = tf.expand_dims(prior_covar_matrix, -1)
            prior_covar = self._prop_to_layers(prior_covar_matrix, self.build_conv_gru())
            

        in_GRU = tf.matmul(prior_covar, self.PrevWeightKG)
        J, _ = self.GRUJ(in_GRU, self.GRUJ_state)
        self.GRUKG_state = J # next self.GRUKG_state
        J = tf.matmul(J, self.NextWeightKG)
        J = tf.matmul(J, self.LastWeightKG)
        J = tf.reshape(J, [J.shape[0], self._lsd, self._lsd])

        # J = tf.matmul(J, self.NextWeightKG)
        # J = tf.matmul(J, self.LastWeightKG)
        # J = tf.matmul(J, self.CholeskyKG)
        # J = tf.reshape(J, [J.shape[0], self._lod, self._lod])
        # Diag_J = tf.linalg.diag_part(J)
        # Diag_elements_dense = self._layer_covar_gru(Diag_J)
        # elup_Diag_elements = tf.linalg.diag(elup1(Diag_elements_dense))
        # Positive_J = elup_Diag_elements + ( J - tf.linalg.diag(tf.linalg.diag_part(J)))
        # J = tf.matmul(tf.matmul(prior_covar, tf.transpose(self.A_tp1_matrix)), tf.matmul(Positive_J, tf.transpose(Positive_J)))

        return J
    
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


    def _update(self, smooth_tp1_mean, smooth_tp1_covar, filt_t_mean, filt_t_covar, prior_tp1_mean,
                                             prior_tp1_covar, transition_tp1_matrix):
        
        J = self._predict_J_gru( prior_tp1_covar)

        smooth_tp1_covar = tf.reshape(smooth_tp1_covar, [smooth_tp1_covar.shape[0], self._lsd, self._lsd])
        filt_t_covar = tf.reshape(filt_t_covar, [filt_t_covar.shape[0], self._lsd, self._lsd])
        prior_tp1_covar = tf.reshape(prior_tp1_covar, [prior_tp1_covar.shape[0], self._lsd, self._lsd])

        
        mu_es = smooth_tp1_mean - tf.squeeze( tf.matmul(transition_tp1_matrix, tf.expand_dims( filt_t_mean, -1) ), -1)
        smooth_t_mean = filt_t_mean + tf.squeeze( tf.matmul(J, tf.expand_dims( mu_es,-1 )), -1)

        smooth_t_covar = smooth_tp1_covar - prior_tp1_covar
        smooth_t_covar = tf.matmul(smooth_t_covar, tf.transpose(J, perm=[0, 2, 1]))
        smooth_t_covar = tf.matmul(J, smooth_t_covar)
        smooth_t_covar = smooth_t_covar + filt_t_covar


        #
        Diag_elements = tf.linalg.diag_part(smooth_t_covar)
        Diag_elements_dense = self._layer_covar_gru(Diag_elements)
        elup_Diag_elements = tf.linalg.diag(elup1(Diag_elements_dense))
        smooth_t_covar = elup_Diag_elements + ( smooth_t_covar - tf.linalg.diag(tf.linalg.diag_part(smooth_t_covar)))
        #
        smooth_t_covar = tf.reshape(smooth_t_covar, [smooth_t_covar.shape[0], -1])
        return smooth_t_mean, smooth_t_covar # mu_t|t, sigma_t|t at t

    def _update_conventional(self, smooth_tp1_mean, smooth_tp1_covar, filt_t_mean, filt_t_covar, prior_tp1_mean,
                                             prior_tp1_covar, transition_tp1_matrix):
        
        smooth_tp1_covar = tf.reshape(smooth_tp1_covar, [smooth_tp1_covar.shape[0], self._lsd, self._lsd])
        filt_t_covar = tf.reshape(filt_t_covar, [filt_t_covar.shape[0], self._lsd, self._lsd])
        prior_tp1_covar = tf.reshape(prior_tp1_covar, [prior_tp1_covar.shape[0], self._lsd, self._lsd])

        #
        Diag_elements = tf.linalg.diag_part(prior_tp1_covar)
        Diag_elements_dense = self._layer_covar_gru(Diag_elements)
        elup_Diag_elements = tf.linalg.diag(elup1(Diag_elements_dense))
        prior_tp1_covar = elup_Diag_elements + ( prior_tp1_covar - tf.linalg.diag(tf.linalg.diag_part(prior_tp1_covar)))
        #

        sigmat_A_tp1 = tf.matmul(filt_t_covar, tf.transpose( transition_tp1_matrix, perm=[0,2,1]))
        J = tf.matmul(sigmat_A_tp1, tf.linalg.inv(prior_tp1_covar))

        
        mu_es = smooth_tp1_mean - tf.squeeze( tf.matmul(transition_tp1_matrix, tf.expand_dims( filt_t_mean, -1) ), -1)
        smooth_t_mean = filt_t_mean + tf.squeeze( tf.matmul(J, tf.expand_dims( mu_es,-1 )), -1)

        smooth_t_covar = smooth_tp1_covar - prior_tp1_covar
        smooth_t_covar = tf.matmul(smooth_t_covar, tf.transpose(J, perm=[0, 2, 1]))
        smooth_t_covar = tf.matmul(J, smooth_t_covar)
        smooth_t_covar = smooth_t_covar + filt_t_covar


        #
        Diag_elements = tf.linalg.diag_part(smooth_t_covar)
        Diag_elements_dense = self._layer_covar_gru(Diag_elements)
        elup_Diag_elements = tf.linalg.diag(elup1(Diag_elements_dense))
        smooth_t_covar = elup_Diag_elements + ( smooth_t_covar - tf.linalg.diag(tf.linalg.diag_part(smooth_t_covar)))
        #
        smooth_t_covar = tf.reshape(smooth_t_covar, [smooth_t_covar.shape[0], -1])
        return smooth_t_mean, smooth_t_covar # mu_t|t, sigma_t|t at t
        
    
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
        # return [(k.layers.Input(shape=(None, self._lsd)), k.layers.Input(shape=(None, self._lsd**2)))]
        
