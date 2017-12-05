#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:47:08 2017

@author: danberenberg
"""

from subprocess import Popen, PIPE
import tensorflow as tf
import numpy as np
import random

import controller_utils as ut

def play_game(process):
    """
    Play the game
    """
    
    result = ""
    steps = 0
    
    left,right,space,up = random.randint(0,1),random.randint(0,1),random.randint(0,1),0
    
    while steps < 200:
        value = str(left) + str(right) + str(up) + str(space) + '\n'
        ut.write_to_game(process,value)
        result = ut.read_from_game(process)
        
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
        
if __name__ == '__main__':
    args =  ut.setup_args()
    
    asteroids_location  = args.ast_path
    screenshot_location = args.picture_path
    
    process = Popen([asteroids_location,screenshot_location],stdout=PIPE, stdin=PIPE)
    play_game(process)
    #process.kill()