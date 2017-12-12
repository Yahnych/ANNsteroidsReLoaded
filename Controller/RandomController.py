from subprocess import Popen, PIPE
import random
import sys
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from sklearn.preprocessing import normalize
N_STEPS = 400
N_EPISODES = 200
PLAY_FREQ = 3



def play_game(p):
    
    
    steps = 0

    result = ""
    
    left = random.randint(0, 1)
    right = random.randint(0, 1)
    up = 0
    space = random.randint(0, 1)

    dead = False
    over = False
    total_reward = 0    
    while (steps < N_STEPS and 'e' not in str(result)):
        if steps % PLAY_FREQ == 0:
            left = random.randint(0, 1)
            right = random.randint(0, 1)
            up = 0
            space = random.randint(0, 1)
        else:
            left = 0
            right = 0
            up = 0
            space = 0
            
        value = str(left) + str(right) + str(up) + str(space) + '\n'
        value = bytes(value, 'UTF-8')  # Needed in Python 3.
        p.stdin.write(value)
        p.stdin.flush()
        result = p.stdout.readline().strip()
        result = result.decode("utf-8")

        if 'e' in str(result):
            dead = True
        elif str(result) == '':
            pass
        else:
            total_reward = int(result)
        
        steps+=1
    print(steps)
    if dead == True:
        total_reward -= 1000
    
    p.kill()
    return total_reward, steps

def main(argv):
    parser = argparse.ArgumentParser(description='Run DRL agent')
    parser.add_argument('ast_path', type=str,
                    help='Path of asteroids executable')
    args = parser.parse_args()
    
    asteroids_location = args.ast_path
    
    
 #   all_rewards = np.zeros(shape=N_EPISODES)
    all_rewards = []
    all_steps = []
    try:
        for episode in range(N_EPISODES):
            process = Popen([asteroids_location, "!"],  stdout=PIPE, stdin=PIPE)
            episode_reward, steps = play_game(process)
            print('Average score for episode ', episode, ": ",
                  episode_reward, sep='')
            all_rewards.append(episode_reward)
            all_steps.append(steps)
    except KeyboardInterrupt:
        process.kill()
    except Exception as e:
        print(e)
    
    all_rewards = np.array(all_rewards)
    all_steps = np.array(all_steps)
    plot_it(all_rewards, all_steps)
##    mean = np.mean(all_rewards)
##    std = np.std(all_rewards)
##    normalized = (all_rewards - mean)/std
##    
##    plt.plot(normalized)
##    print(normalized)
##    #plt.plot([-760.00, -800.55, 900])
##    plt.ylabel('Reward')
##    plt.xlabel('Episodes')
##    plt.title('Random Controller')
##    axes = plt.gca()
##    #axes.set_ylim([-1000,1200])
##    plt.show()
##    sys.exit("Simulation over")
def plot_it(overall_rwd_3L, step_ct_r3L):
    fig, axarr = plt.subplots(1,2,figsize=(12,5))
    mean_rwd = np.mean(overall_rwd_3L)
    std_rwd  = np.std(overall_rwd_3L)

    plot_mea = (np.array(overall_rwd_3L) - mean_rwd)/mean_rwd

    plt.subplots_adjust(wspace=0.3)
    plt.suptitle("Random Run",fontsize=14)
    axarr[0].set_title("Normalized Reward per Episode",fontsize=14)
    axarr[0].plot(plot_mea)
    axarr[0].set_xlabel("Episode",fontsize=14)
    axarr[0].set_ylabel("Normalized Reward",fontsize=14)

    axarr[1].set_title("Step Count per Episode", fontsize=14)
    axarr[1].plot(step_ct_r3L)
    axarr[1].set_xlabel("Episode", fontsize=14)
    axarr[1].set_ylabel("Step Count",fontsize=14)


    plt.show()


if __name__ == "__main__":
    main(sys.argv)
