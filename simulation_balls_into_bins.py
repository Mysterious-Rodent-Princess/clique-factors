#!/usr/bin/python3

import math
import random


def balls_into_bins_experiment_with_disapperance(k, m, p):

    # should store key-value pairs like {3: [5, 2, 8]}
    bin_status_dict = {}
    for i in range(m):
        bin_status_dict[i] = []

    # should store key-value pairs like {2:3, 5:3, 8:3}
    ball_status_dict = {}
    
    balls_thrown = 0


    # evaluation list: round (i.e. current ball), simple expectation, complicated expectation, actual current value
    evaluation_list = []
    

    for current_ball in range(k):
        
        selected_bin = random.choice(list(bin_status_dict.keys()))
        bin_status_dict[selected_bin].append(current_ball)
        ball_status_dict[current_ball] = selected_bin


        #with probability p, a bin disappears
        
        random_number = random.random()
        
        if random_number < p:
            random_ball = random.choice(list(ball_status_dict.keys()))
            belonging_bin = ball_status_dict[random_ball]

            # EVALUATION #########################################################################
            
            k_curr = len(ball_status_dict.keys())
            m_curr = len(bin_status_dict.keys())
            
            simple_expectation = 1 + k_curr / m_curr
            complicated_expectation = sum([((len(load))**2)/k_curr for load in bin_status_dict.values()])
            actual_current_value = len(bin_status_dict[belonging_bin])
            
            evaluation_list.append([current_ball, simple_expectation, complicated_expectation, actual_current_value])
            
            # END EVALUATION #####################################################################
            
            all_balls_of_belonging_bin = bin_status_dict[belonging_bin]
            
            for ball in all_balls_of_belonging_bin:
                ball_status_dict.pop(ball)
            
            bin_status_dict.pop(belonging_bin)

        current_ball += 1

    return bin_status_dict, ball_status_dict, evaluation_list


k=100000
m=25000
p=0.05 # set this to zero if we do not want disappearance

bin_status_dict, balls_status_dict, evaluation_list = balls_into_bins_experiment_with_disapperance(k, m, p)
#print(bin_status_dict)
#print(balls_status_dict)
print(evaluation_list)
print()
sum_simple = sum([entry[1] for entry in evaluation_list])
sum_complicated = sum([entry[2] for entry in evaluation_list])
quotient = sum_simple / sum_complicated
print(quotient)

















