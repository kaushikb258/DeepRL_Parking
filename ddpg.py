import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import matplotlib.pyplot as plt
from misc import *


#---------------------------------------------------

OU = OU()       #Ornstein-Uhlenbeck Process

#---------------------------------------------------
# distances in meters
# time in seconds
# angles in radians

xdim = 15.0
ydim = 25.0

carl = 4.4
carw = 1.8

lf = carl/2.0
lr = carl/2.0

dt = 0.1

v_max = 3.0
delta_max = 0.427
L_wheel_2_wheel = 0.8*carl

dv_dt_scale = 1.0
ddelta_dt_scale  = 0.427/2.0

#---------------------------------------------------

BUFFER_SIZE = 500000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001     #Target Network HyperParameters
LRA = 1e-4    #Learning rate for Actor
LRC = 1e-3     #Lerning rate for Critic

action_dim = 2  #Steering/Acceleration or Brake
state_dim = 11  #of sensors input

np.random.seed(258)

EXPLORE = 500000.

# total number of episodes
episode_count = 500000 

max_steps = 250

screen_out = 100

# if restart = 0, fresh start; else reload weights
restart = 0

#---------------------------------------------------


def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    
    if (train_indicator == 1):
     no_plt_until = 490000
    else: 
     no_plt_until = 0
    
    reward = 0
    done = False
    epsilon = 1

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(train_indicator,sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(train_indicator,sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer


    if (train_indicator == 0 or restart == 1):
      #Now load the weight
      print("Now we load the weight")
      try:
        actor.model.load_weights("out_files/actormodel.h5")
        critic.model.load_weights("out_files/criticmodel.h5")
        actor.target_model.load_weights("out_files/actormodel.h5")
        critic.target_model.load_weights("out_files/criticmodel.h5")
        print("Weight load successfully")
      except:
        print("Cannot find the weight")


#------------------------------------    

    for i in range(episode_count):

        if (i % screen_out == 0):
          print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        # initialize state
        x, y, v, zi, delta, dv_dt, ddelta_dt  = calc_init_state(i, xdim, ydim, carl, carw, train_indicator)


        # state vector
        d2left, d2top, d2bottom, d2right, d2goal, ang2goal = compute_statevars(xdim,ydim,carl,x,y)
        s_t = np.hstack((x, y, v, zi, delta, d2left, d2top, d2bottom, d2right, d2goal, ang2goal))
     
        if (i == no_plt_until):
          plt.ion()

        total_reward = 0.
     

        u1 = 0.0
        u2  = 0.0
        u1_old = 0.0
        u2_old = 0.0

        done = False


        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            # actor predictions
            # note: this will be in [-1,1] range as tanh activation function is used
            # add Ornstein-Uhlenbeck noise to it
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0, 0.15, 0.4, dt)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.0, 0.15, 0.25, dt)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]


            # controls
            u1 = a_t[0][0]
            u2 = a_t[0][1]


            # d(velocity)/dt
            dv_dt = u1*dv_dt_scale

            # d(steering angle)/dt
            ddelta_dt = u2*ddelta_dt_scale
        
            # move one time step
            x, y, v, zi, delta = move(dt, v_max, delta_max, L_wheel_2_wheel, x, y, zi, delta, v, dv_dt, ddelta_dt)

#-----------------------------------
            if (i >= no_plt_until and j%5==0):

             x1 = np.array([-xdim, 0.0])
             y1 = np.array([-ydim/2.0, -ydim/2.0])
             plt.plot(x1,y1,c='r')
             x1 = np.array([-xdim, 0.0])
             y1 = np.array([ydim/2.0, ydim/2.0])
             plt.plot(x1,y1,c='r')
             x1 = np.array([-xdim, -xdim])
             y1 = np.array([-ydim/2.0, ydim/2.0])
             plt.plot(x1,y1,c='r')
             x1 = np.array([0.0, 0.0])
             y1 = np.array([-ydim/2.0, -carw])
             plt.plot(x1,y1,c='r')
             x1 = np.array([0.0, 0.0])
             y1 = np.array([ydim/2.0, carw])
             plt.plot(x1,y1,c='r')
             x1 = np.array([0.0, 1.5*carl])
             y1 = np.array([-carw, -carw])
             plt.plot(x1,y1,c='r')
             x1 = np.array([0.0, 1.5*carl])
             y1 = np.array([carw, carw])
             plt.plot(x1,y1,c='r')
             x1 = np.array([1.5*carl, 1.5*carl])
             y1 = np.array([-carw, carw])
             plt.plot(x1,y1,c='r')

             plt.scatter(x,y,s=10,c='k')

             xa  = x + carw/2.0*np.sin(zi) + lf*np.cos(zi) 
             xb  = x - carw/2.0*np.sin(zi) + lf*np.cos(zi) 
             xc  = x - carw/2.0*np.sin(zi) - lr*np.cos(zi) 
             xd  = x + carw/2.0*np.sin(zi) - lr*np.cos(zi) 

             ya  = y - carw/2.0*np.cos(zi) + lf*np.sin(zi) 
             yb  = y + carw/2.0*np.cos(zi) + lf*np.sin(zi) 
             yc  = y + carw/2.0*np.cos(zi) - lr*np.sin(zi) 
             yd  = y - carw/2.0*np.cos(zi) - lr*np.sin(zi) 

             x1 = np.array([xa,xb])
             y1 = np.array([ya,yb])    
             plt.plot(x1,y1,c='b')
             x1 = np.array([xc,xb])
             y1 = np.array([yc,yb])    
             plt.plot(x1,y1,c='b')
             x1 = np.array([xc,xd])
             y1 = np.array([yc,yd])    
             plt.plot(x1,y1,c='b')
             x1 = np.array([xa,xd])
             y1 = np.array([ya,yd])    
             plt.plot(x1,y1,c='b')

             plt.axis('equal')
             plt.show()
             plt.pause(0.05)
#-----------------------------------

            print_screen = False
            if (i % screen_out == 0):
              print_screen = True 

            # reward
            r_t, done = compute_reward(xdim, ydim, carw, carl, x, y, v, zi, u1, u1_old, u2, u2_old, print_screen)
             

            # state vector
            d2left, d2top, d2bottom, d2right, d2goal, ang2goal = compute_statevars(xdim,ydim,carl,x,y)
            s_t1 = np.hstack((x, y, v, zi, delta, d2left, d2top, d2bottom, d2right, d2goal, ang2goal))
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])


            if (train_indicator):
              target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
              for k in range(len(batch)):
                 if dones[k]:
                     y_t[k] = rewards[k]
                 else:
                     y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            
              loss += critic.model.train_on_batch([states,actions], y_t) 
              a_for_grad = actor.model.predict(states)
              grads = critic.gradients(states, a_for_grad)
              actor.train(states, grads)
              actor.target_train()
              critic.target_train()


            total_reward += r_t
            s_t = s_t1
        
            if (i % screen_out == 0 and j % 50 == 0):
              print("Episode:", i, "Step:", j, "Action", np.round(a_t,3), "Reward", np.round(r_t,3), "Loss", np.round(loss,3))


            # current new is old for next step        
            u1_old = u1
            u2_old = u2


            if done:
                break

        
        if (i % screen_out == 0):
          print("number of substeps: ", j)
	
        if (i >= no_plt_until):  
          plt.clf()

        if np.mod(i, 100) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("out_files/actormodel.h5", overwrite=True)
                with open("out_files/actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("out_files/criticmodel.h5", overwrite=True)
                with open("out_files/criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)


#        with open("out_files/test.txt", "a") as myfile:
#          myfile.write(str(i) + " " + str(np.round(r_t,3)) + "\n")

        if (i % screen_out == 0):
         print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward)) 
         print("")
        if (train_indicator == 1): 
           with open("out_files/total_reward.txt", "a") as myfile:
             myfile.write(str(i) + " " + str(total_reward) + "\n")



    print("Finish.")

if __name__ == "__main__":
    train_test = input("enter 1 for training / 0 for testing")
    playGame(int(train_test))
    #1 means Train, 0 means simply Run
