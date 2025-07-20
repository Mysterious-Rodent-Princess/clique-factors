#!/usr/bin/python3

import math
import numpy as np

from sympy import symbols, Eq, Function, sqrt, atan, exp, sin, cos
from sympy.solvers.ode.systems import dsolve_system

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt


##############################
### get precision ############
##############################

#sympy
precision_parameter = 50

#scipy
from mpmath import mp
mp.dps = 30

######################################################
### get evaluation time granularity 10**(-7) #########
######################################################
granularity = 10**(-7)


#############################################################################################################################
#############################################################################################################################
############## PLOTS ########################################################################################################
#############################################################################################################################
#############################################################################################################################


def create_plot_for_phase_2(right_boundary, initial_values):

	################################
	# integration interval
	################################

	integration_interval = [0., right_boundary]

	################################
	# solution 
	################################

	sol = solve_ivp(phase_2_system, integration_interval, initial_values, dense_output=True) 

	number_of_samples_to_generate = 30000
	s = np.linspace(0, right_boundary, number_of_samples_to_generate)
	z = sol.sol(s) # the field sol of the output only exists if dense_output is set to True
	result_values = z.T

	################################
	# plot 
	################################

	values_m = [x[0] for x in result_values]
	values_r = [x[1] for x in result_values]


	plt.plot(s, values_m, color='black')
	plt.plot(s, values_r, color='red')
	
	plt.xlabel('x')
	plt.legend(['m', 'r'], shadow=True) 
	plt.title('PM')
	plt.show()





#############################################################################################################################
#############################################################################################################################
############## PHASE 2. #####################################################################################################
#############################################################################################################################
#############################################################################################################################



def phase_2_system(t, y):
	m, r  = y

	eq_m = 2 * (1-m+r)
	eq_r = m - 2*r - 2 *r + r * (-1 - 2 *r /(1-m))
	
	equations = [eq_m, eq_r]

	return equations





def calculate_ode_system_for_phase_2(left_boundary, right_boundary, target_value_for_f_precise, initial_values):

	################################
	# integration interval
	################################

	integration_interval = [0., right_boundary]

	################################
	# evaluation times
	################################


	#evaluation_times = np.arange(0, right_boundary, granularity)
	evaluation_times = np.arange(mp.mpf(left_boundary), mp.mpf(right_boundary-granularity), mp.mpf(granularity))
	# there are also contexts where we just would want:
	# evaluation_times = [right_boundary]


	################################
	# initial values
	################################

	initial_values_precise = [mp.mpf(x) for x in initial_values]


	################################
	# solution - numbers
	################################
	
	sol = solve_ivp(phase_2_system, integration_interval, initial_values_precise, t_eval=evaluation_times) # "dense output = True" doesn't help!!!! 

	################################
	# create result list
	################################



	if target_value_for_f_precise != None:

		for i,s in enumerate(sol.t):

			values_at_time_s = [item[i] for item in sol.y]
			values_at_time_s_precise = [mp.mpf(x) for x in values_at_time_s]

			for value in values_at_time_s_precise:
				if value > 1 or value < 0:
					print("already too much")
					print("time = " + str(s))
					print(values_at_time_s_precise)
					return


			m_at_time_s = values_at_time_s_precise[0]
			
			if m_at_time_s > target_value_for_f_precise:
				print("m_at_time_s = " + str(m_at_time_s))
				print("target_value_for_f_precise = " + str(target_value_for_f_precise))
				result_time = s
				result_list = values_at_time_s_precise
				return result_time, result_list

		last = [item[-1] for item in sol.y]
		print(last)

	else:
		result_time = sol.t[-1]
		result_list = [item[-1] for item in sol.y]
		return result_time, result_list




#############################################################################################################################
#############################################################################################################################
############## MAIN FUNCTION ################################################################################################
#############################################################################################################################
#############################################################################################################################



plot_boolean = True


initial_values_for_phase_2 = [0, 0]
target_value_for_f_precise = mp.mpf(1-(0.5 * 10**(-14)))

left_boundary = 1.274
right_boundary = 1.2769



if plot_boolean == True:
	create_plot_for_phase_2(right_boundary, initial_values_for_phase_2)
result_time_phase_2, result_list_phase_2 = calculate_ode_system_for_phase_2(left_boundary, right_boundary, target_value_for_f_precise, initial_values_for_phase_2)

print("result_time_phase_2 = " + str(result_time_phase_2))
print("result_list_phase_2 = " + str(result_list_phase_2))



