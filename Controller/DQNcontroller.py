#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:47:08 2017

@author: danberenberg
"""

from subprocess import Popen, PIPE
import tensorflow as tf
import tensorflow.contrib.slim as slim # library to remove boilerplate code
import numpy as np
import random
import glob
import sys, os
import controller_utils as ut
import datetime
import warnings
import pickle

def save(data,filename):
    with open(filename,'wb') as f:
        pickle.dump(data,f)

def play_game_randomly(process,nsteps):
    """
    Play the game randomly
    """
    
    steps = 0
    
    left,right,space,up = random.randint(0,1),random.randint(0,1),random.randint(0,1),0
    
    while steps < nsteps:
        value = str(left) + str(right) + str(up) + str(space) + '\n'
        ut.write_to_game(process,value)
        
        if steps % 2 == 0:
            left = random.randint(0, 1)
            right = random.randint(0, 1)
            #up = random.randint(0, 1)
            up = 0
            space = random.randint(0, 1)
        
        steps+=1
        
    print('Game ended.')
    value = str('d') + str('d') + str('d') + str('d') + '\n'
    ut.write_to_game(process,value)
    wait_time = 0
    response = ""
    while (response != 'd'):
        ut.write_to_game(process,'k\n')
        response = str(ut.read_from_game(process))
        wait_time+=1
    
    process.kill()

def random_action():
    # choose a random action index
    action_ind = random.randint(0,2)
                    
    # initialize actions
    actions = np.zeros(3)
                
    actions[action_ind] = 1
                    
    left  = actions[0]
    right = actions[1]
    up    = 0          # ship stays in middle of board
    space = actions[2]

    
    return left,right,up,space,action_ind

def verify_screenshot_directory(screenshot_dir):
    """
    Find the next directory that is unoccupied to place screen shots into
    """
    screenshot_dir = screenshot_dir.strip('/')
    if not os.path.isdir(screenshot_dir):
        print ("[!] making a screenshot dir: {}".format(screenshot_dir))
        os.makedirs(screenshot_dir)
        return screenshot_dir
    
    
    num = 1
    screenshot_dir_ = screenshot_dir
    
    while os.path.isdir(screenshot_dir_):
        #print (ut.BUFFER_LINE + "trying {}".format(screenshot_dir_))
        if glob.glob(screenshot_dir_+ '/*' ) == []:
            return screenshot_dir_
        
        screenshot_dir_ = screenshot_dir + str(num)
        num +=1
        
    if screenshot_dir_ != screenshot_dir:
        print ("[!] making a screenshot dir: {}".format(screenshot_dir_))
        os.makedirs(screenshot_dir_)
        return screenshot_dir_


def updateTargetNet(tfVars,tau):
    """
    Update the target network's graph using the expriences of the behavior network
    """
    
    nvars = len(tfVars)
    op_holder = []
    
    for ind,var in enumerate(tfVars[0:nvars//2]): # extracting only the behavior net's values
    
        # update the parallel target net parameter
        op_holder.append(tfVars[ind+nvars//2].assign((var.value()*tau) + ((1-tau)*tfVars[ind+nvars//2].value())))
    
    return op_holder

def updateTarget(op_holder,sess):
    """
    Update the target network using the operations collected and updated in updateTargetNet
    """
    for op in op_holder:
        sess.run(op)
        
class QNetwork():
    """
    QNetwork Class code adapted from Arthur Juliani 
    ----------------------------------------------
    [https://medium.com/@awjuliani/]
    
    QNetwork Architecture implemented from Mnih,Kavukcuoglu, Silver, et al
    ---------------------------------------------------------------------
    'Human-level control through deep reinforcement learning'
    
    """
    
    def __init__(self,h_size,name=''):
        """
        Network recieves a flattened frame array from the game
        - The frame array is resized to an 84x84x1 image
        
        - The image is processed through 4 convolutional layers, ending
          with a one hot encoded action vector
        """
        self.name = name
        
        print("[+] Initializing {}".format(self.name))
        self.std = 0.1
        ###################### Input                       ####################

        self.frame_array = tf.placeholder(shape=[None,ut.X_SIZE*ut.Y_SIZE*ut.N_CHANNELS],dtype=tf.float32)
        
        self.in_image    = tf.reshape(self.frame_array,shape=[-1,ut.X_SIZE,ut.Y_SIZE,ut.N_CHANNELS])
        
        ###################### ConvLayer1: 8x8x4x32 stride 4 ##################
        print ("[+] Building ConvLayer1: 8x8x4x32 stride 4..")
        self.w_conv1 = self.weights([8,8,4,32])
        
        # no biases .. ?
        #self.b_conv1 = self.biases([32])
        self.conv1 = self.cnv_lyr(self.in_image,self.w_conv1,[1,4,4,1])
        self.h_conv1 = tf.nn.relu(self.conv1)
        print(ut.BUFFER_LINE + "[+] Built ConvLayer1: 8x8x4x32 stride 4")
       
        ###################### ConvLayer2: 4x4x64 stride 2 ####################
        print ("[+] Building ConvLayer2: 4x4x32x64 stride 4.. ")
        self.w_conv2 = self.weights([4,4,32,64])
        
        # no biases .. ?
        #self.b_conv2 = self.biases([64])
        self.conv2 = self.cnv_lyr(self.h_conv1,self.w_conv2,[1,2,2,1])
        self.h_conv2 = tf.nn.relu(self.conv2)
        print(ut.BUFFER_LINE + "[+] Built ConvLayer2: 4x4x32x64 stride 2")
        
        ###################### ConvLayer3: 3x3x64 stride 1 ####################
        print("[+] Building ConvLayer3: 3x3x64x64 stride 1.. ")
        self.w_conv3 = self.weights([3,3,64,64])
        
        # no biases .. ?
        #self.b_conv2 = self.biases([64])
        self.conv3 = self.cnv_lyr(self.h_conv2,self.w_conv3,[1,1,1,1])
        self.h_conv3 = tf.nn.relu(self.conv3)

        # take output of Conv3 and separate into Advantage and Value streams
        #self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
        print(ut.BUFFER_LINE + "[+] Built ConvLayer3: 3x3x64x64 stride 1")
        
        ###################### ConvLayer4: 7x7x3 stride 1 ####################
        print("[+] Building ConvLayer4: 7x7x512x1 stride 1")
        self.w_conv4 = self.weights([7,7,64,h_size])
        self.conv4   = self.cnv_lyr(self.h_conv3,self.w_conv4,[1,1,1,1])
        self.h_conv4 = tf.nn.relu(self.conv4)
        print(ut.BUFFER_LINE + "[+] Built ConvLayer4: 7x7x512x1 stride 1")
        
        ###################### Fully Connected (FC1): (3x3x64)x512 ############
        print("[+] Building Dual Layer: Separate Advantage and Value streams")
        
        self.streamAC, self.streamVC = tf.split(self.h_conv4,2,3)
        
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        
        xavier_init = tf.contrib.layers.xavier_initializer()
        
        self.AW = tf.Variable(xavier_init([h_size//2,3]))
        self.VW = tf.Variable(xavier_init([h_size//2,1]))
        
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        print(ut.BUFFER_LINE + "[+] Built Dual Layer: Separate Advantage and Value streams")
        
        ###################### Q values and Initial weights ################### 
        print("[+] Initializing parameters")
 
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
       
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)

        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)

        self.actions_onehot = tf.one_hot(self.actions,3,dtype=tf.float32)
    
        self.Q = tf.reduce_sum(tf.multiply(self.Qout,self.actions_onehot),axis=1)

        self.TD_error = tf.square(self.targetQ-self.Q)
        self.loss = tf.reduce_mean(self.TD_error)
        
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.optimizer = self.trainer.minimize(self.loss)
        print()
    def weights(self,dims):
        """
        Initialize a set of weights with dimensions
        """
        initial_wts = tf.truncated_normal(dims,stddev=self.std)
        return tf.Variable(initial_wts)
        
    def biases(self,dims):
        """
        Initialize a set of biases with dimensions
        """
        initial_biases = tf.constant(0.1,shape=dims)
        return tf.Variable(initial_biases)
        
    def cnv_lyr(self,in_lyr,weight_vec,strides):
        return tf.nn.conv2d(in_lyr,weight_vec,strides=strides,padding="VALID")
        
class ExperienceBuffer():
    """
    Implementation of the Experience Replay buffer described in:
    
    QNetwork Architecture implemented from Mnih,Kavukcuoglu, Silver, et al
    ---------------------------------------------------------------------
    'Human-level control through deep reinforcement learning'
    
    Adapted from Arthur Juliani
    ---------------------------------------------------------------------
    [https://medium.com/@awjuliani/]
    """
    
    def __init__(self,buffer_size=10000):
        """
        Initialize the empty buffer with size buffer_size and 
        """
        self.buffer = []
        self.buffer_size = buffer_size
        
    def add(self,experience):
        """
        Add an experience to the buffer
        """
        
        # determine whether the buffer is full - if it is,
        # remove the first k elements of the buffer where 
        # current_buf_size = k + buffer_size_limit
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = [] 
        
        # add the experience
        self.buffer.extend(experience)
        
    def sample(self,size):
        """
        Sample out size experiences from the buffer
        """
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,4])
        

# parameter intialization
BATCH_SIZE = 32
UPDATE_FREQ = 2
PLAY_FREQ = 3    # take action every 3 frames which is comprable to human speeds
N_STEPS = 400
N_EPISODES = 800

gamma = 0.99
starting_epsilon = 1   # starting epsilon
ending_epsilon   = 0.05  # decreases on some schedule until it reaches this epsilon
annealing_steps  = 5000  # number of steps to reduce starting_epsilon to ending_epsilon
pretraining_steps = 10000 # numbear of steps to play randomly and gather state information

tau = 0.001 # rate at which target network is updated toward primary network

h_size = 512

# variable initialization
tf.reset_default_graph()
mainQN = QNetwork(h_size,name="DQN behavior") # the behavior network
targetQN = QNetwork(h_size,name="DQN optimal") # the target network

init = tf.global_variables_initializer()
saver = tf.train.Saver()

trainable_vars = tf.trainable_variables()
target_ops     = updateTargetNet(trainable_vars,tau)

# initialize Experience Buffer
buff = ExperienceBuffer()

epsilon = starting_epsilon
step_drop = (starting_epsilon - ending_epsilon)/annealing_steps

# reward list over each episode
rwrds = []
steps_ctr = []
total_steps = 0

left,right,up,down = 0,0,0,0


params = [BATCH_SIZE,UPDATE_FREQ,PLAY_FREQ,N_STEPS,N_EPISODES,gamma,
          starting_epsilon,annealing_steps,pretraining_steps,tau,h_size,
          step_drop]

param_names = "BATCH_SIZE,UPDATE_FREQ,PLAY_FREQ,N_STEPS,N_EPISODES,gamma,starting_epsilon,annealing_steps,pretraining_steps,tau,h_size,step_drop".split(",")
              
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ########################### simulation setup ###############################
        ###########################################################################
        args =  ut.setup_args()
        
        asteroids_location  = args.ast_path
        out_dir = args.out_dir
        screenshot_location = args.picture_path
        screenshot_location = verify_screenshot_directory(screenshot_location)
        load_model = args.model_path
        # not output directory specified, create one inside 
        # of this directory
        
        if load_model != None and not os.path.exists(load_model):
            sys.exit("[!] Could not locate {}".format(load_model))
            
        
        if not out_dir:
            # make a rsz ( resize ) directory if necessary
            if not os.path.isdir(screenshot_location + "/rsz"):
                print ("[!] making a resize photos dir: {}".format(screenshot_location + "/rsz"))
                os.makedirs(screenshot_location + "/rsz")
                
            out_dir = screenshot_location + "/rsz/"
            
        #process = Popen([asteroids_location,screenshot_location + '/'],stdout=PIPE, stdin=PIPE)
        game_args = {"screenshot_location":screenshot_location,
                     "output_directory":out_dir,
                     'load_pth':load_model,
                     'load':type(load_model) == str}
        
        
        ########################## simulation #####################################
        ###########################################################################
        left,right,space,up = random.randint(0,1),random.randint(0,1),random.randint(0,1),0
            
        #img_location = game_args['screenshot_location']
        #out_dir      = game_args['output_directory']
        load_model   = game_args['load']
        load_pth     = game_args['load_pth']
        
        img_pth = screenshot_location.strip("/") + "/" + "*pgm"  # search string
        buffer_string = ""
        
        print()
        print("="*50)
        for p,n in zip(params,param_names):
            print("[*] {:<22}: {}".format(n,p))
        print("="*50)
        print()
        #process = None
        begin = datetime.datetime.now().strftime("%m/%d/%Y - %H:%M:%S")
        print ("[!] Starting session at {}".format(begin))
        with tf.Session() as sess:
            sess.run(init)
            
            if load_model == True:
                #print ("[+] loading model: {}".format(load_pth))
                checkpt = tf.train.get_checkpont_state(load_pth)
                saver.restore(sess,checkpt.model_checkpoint_path)
            
            dead = False
            for i in range(N_EPISODES):
                ########################### simulation setup ###############################
                ###########################################################################
    
                ep_buff = ExperienceBuffer() # new experience buffer
                
                if not os.path.isdir(screenshot_location + "/rsz"):
                    print ("[!] making a resize photos dir: {}".format(screenshot_location + "/rsz"))
                    os.makedirs(screenshot_location + "/rsz")
           
                # new regular expression
                img_pth = screenshot_location.strip("/") + "/" + "*pgm"
    
                out_dir = screenshot_location + "/rsz/"
                dead = False
                # start up a game
                process = Popen([asteroids_location,screenshot_location + '/'],stdout=PIPE, stdin=PIPE)        
                
                steps = 0 # reset steps
                
                ########################## simulation #####################################
                ###########################################################################
                state    = None
                state_p1 = None
                score_t  = None
                score_t1 = None
                buffer_string = ""
                running_reward = 0
                frame_buffer   = ut.FrameBuffer()
                # do gradient step for nsteps
                while steps < N_STEPS and not dead:
                    
                    old_photos = set()
                    curr_photos = set()
                    steps+=1
                    total_steps+=1
                    
                    if steps < 60:
                        value = str(0) + str(0) + str(0) + str(0) + '\n'
                        ut.write_to_game(process,value)
                        
                        continue # lots of black screen before anythign happens
                    else:    
                        if steps == 60:
                           state,score_t,buffer_string,curr_photos,old_photos = ut.next_state_and_reward(old_photos,curr_photos,
                                                                                                         img_pth,screenshot_location,
                                                                                                         out_dir,buffer_string,epsilon,
                                                                                                         total_steps,-1)      
                        else:
                            if steps % PLAY_FREQ == 0:
            
                                # with probability epsilon or if training hasn't begun yet ... 
                                if total_steps < pretraining_steps or np.random.rand(1) < epsilon:
                                    left,right,up,space,act = random_action()
                                    value = str(left) + str(right) + str(up) + str(space) + '\n'
                                    ut.write_to_game(process,value)
                                    action = act
                                    actions = np.zeros(3)
                                    actions[action] = 1
                                
                                else:
                                    # select an action
                                    if state != None:
                                        #print('getting an action')
                                        
                                        print(frame_buffer.stack)
                                        fb  = np.vstack(frame_buffer.stack)
                                        while fb.shape[0] < 4:
                                                fb = np.vstack([fb,fb[-1]])
                                        print(fb)
                                        
                                        action = sess.run(mainQN.predict,
                                                          feed_dict={mainQN.frame_array:np.reshape(fb,[1,
                                                                                                   7056*ut.N_CHANNELS]
                                                                                                    )
                                                                    }
                                                          )
                                    
                                    else:
                                        action = [-1]  
                                    
                                    action = action[0]
                                    
                                    actions = np.zeros(3)
                                    actions[action] = 1
                                    
                                    if action == -1:
                                        actions[action] = 0
                                    
                                    
                                    # left right up down
                                    value = str(actions[0]) + str(actions[1]) + str(0) +  str(actions[2]) + '\n'
                                    ut.write_to_game(process,value)
                                    
                                # get the next photo and resize it
                                state_p1, score_t1, buffer_string,curr_photos,old_photos = ut.next_state_and_reward(old_photos,curr_photos,
                                                                                                                    img_pth,screenshot_location,
                                                                                                                    out_dir,buffer_string,epsilon,
                                                                                                                    total_steps,action)
                                terminal = 0
                                #print(score_t1,score_t)
                                if score_t1 == score_t:
                                    score_t1 -= 1000
                                    #print("[-] foul play {}".format(steps))
                                    terminal = 0
                                    dead = True
                                    
                                old_buff = frame_buffer.deep_copy()
                                frame_buffer.add(state_p1)
                                transition = [old_buff.stack,action,score_t1,frame_buffer.stack]
                                #transition = [state,action,score_t1,state_p1]
                                if state_p1 != None and state != None:
                                    ep_buff.add(np.reshape(transition,[1,4]))
                                
                                if total_steps > pretraining_steps:
                                    #print ('[!] here: steps --> {}  total --> {}'.format(steps,total_steps))
                                    
                                    if epsilon > ending_epsilon:
                                        epsilon -= step_drop
                                    
                                    if steps % UPDATE_FREQ == 0 and steps != 0 and action != -1:
                                        #print('updating...')
                                        # sample out a random batch
                                        trainBatch = buff.sample(BATCH_SIZE)
                                        print (trainBatch[:,3].shape)
                                        stp1 =  np.vstack(trainBatch[:,3])
                                        print(stp1.shape)
                                        stp1 = np.reshape(stp1,[32,7056*4])
                                        print(stp1)
                                        
                                        #stp1 = np.reshape(stp1,[32,4])
                                        # belief of the optimal action value of next state
                                        #print("maxQ")
                                        #or exp in stp1:
                                        #print(exp)
                                        maxQ = sess.run(mainQN.predict,feed_dict={mainQN.frame_array:stp1})
                                        #,[-1,ut.N_CHANNELS*ut.X_SIZE*ut.Y_SIZE]))})
    
                                        # action value of previous state
                                        print("Q1")
                                        Q1 = sess.run(mainQN.predict,feed_dict={mainQN.frame_array:stp1})
                                        print("Q2")
                                        Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.frame_array:stp1})
                                        
                                        #print("doubleQ")
                                        doubleQ = Q2[range(BATCH_SIZE),Q1]
                                        
                                        #print("targetQ")
                                        targetQ = trainBatch[:,2] + (gamma*doubleQ * terminal)
                                        
                                        #print("frame_arr")
                                        frame_arr = np.vstack(trainBatch[:,0])
                                        print (trainBatch[:,0].shape)
                                        st =  np.vstack(trainBatch[:,0])
                                        print(st.shape)
                                        print(st)
                                        if st.shape[0] != 128:
                                            while st.shape[0] < 128:
                                                st = np.vstack([st,st[-1]])
                                        
                                        print(st)
                                        print(st.shape)
                                        st = np.reshape(st,[32,7056*4])
                                        print(st)
                        
                                        #print("myactions")
                                        myactions = np.vstack(trainBatch[:,1])
                                       
                                        print("_")
                                        _ = sess.run(mainQN.optimizer,feed_dict={mainQN.frame_array:st,
                                                                             mainQN.targetQ:targetQ,
                                                                             mainQN.actions:trainBatch[:,1]})
                                        updateTarget(target_ops,sess)
                                        print ("[*] Updated model!")
                            else:
                                
                                value = str(0) + str(0) + str(0) + str(0) + '\n'
                                ut.write_to_game(process,value)
                                frame = ut.next_frame_thats_it(curr_photos,old_photos,img_pth,screenshot_location,out_dir)
                                #print("adding a frame")
                                frame_buffer.add(frame)
                                
                            # end update sequence 
                            if state_p1 != None:
                                state = state_p1    
                            score_t = score_t1
                            
                            if score_t != None:
                                running_reward += score_t
                # finish the game
                buff.add(ep_buff.buffer)
                # process killed
                screenshot_location = verify_screenshot_directory(args.picture_path)         # new shots directory
                process.kill()
                
                rwrds.append(running_reward/steps)
                steps_ctr.append(steps)
                #print ("here we go again {} {}".format(dead,screenshot_location))
                saver.save(sess,"./model/model-"+str(i)+".ckpt")
                save(rwrds,"./pickles/avg_rwd_m{}".format(str(i))+".pkl")
                save(steps_ctr,"./pickles/step_ct_m{}".format(str(i)+".pkl"))
                print("Saved model")
                
                # make the new screen shot directory            
                # reset the resized photo directory to trigger creation
                out_dir = None
            
            saver.save(sess,"./model-"+str(i)+".ckpt")

end = datetime.datetime.now().strftime("%m/%d/%Y - %H:%M:%S")
print ("[!] Started at {}, ended at {}".format(begin,end))
      
