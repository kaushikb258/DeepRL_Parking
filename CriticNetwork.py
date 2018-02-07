import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras import initializers
import keras



HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300

class CriticNetwork(object):
    def __init__(self, train_indicator, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.train_indicator = train_indicator
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        K.set_learning_phase(self.train_indicator) #set learning phase
        S = Input(shape=[state_size])  
        A = Input(shape=[action_dim],name='action2')   
        w1 = Dense(HIDDEN1_UNITS)(S)
        w1 = BatchNormalization()(w1)
        w1 = Activation('relu')(w1)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1],mode='sum')    
        h3 = Dense(HIDDEN2_UNITS)(h2)
        h3 = BatchNormalization()(h3)
        h3 = Activation('relu')(h3)
        V = Dense(action_dim,activation='linear',kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(h3)   
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 
