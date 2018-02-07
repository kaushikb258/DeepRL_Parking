import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.initializers import VarianceScaling, RandomNormal
from keras import initializers
import keras


HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300


class ActorNetwork(object):
    def __init__(self, train_indicator, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.train_indicator = train_indicator
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    
    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        K.set_learning_phase(self.train_indicator) #set learning phase
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN1_UNITS)(S)
        h0 = BatchNormalization()(h0)
        h0 = Activation('relu')(h0)
        h1 = Dense(HIDDEN2_UNITS)(h0)
        h1 = BatchNormalization()(h1)
        h1 = Activation('relu')(h1)
        Steering = Dense(1,activation='tanh',kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(h1) 
        Acceleration = Dense(1,activation='tanh',kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(h1)   
        V = merge([Steering,Acceleration],mode='concat')          
        model = Model(input=S,output=V)
        return model, model.trainable_weights, S

