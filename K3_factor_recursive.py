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
precision_parameter = 30

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

def create_plot_for_pm(right_boundary):

	################################
	# integration interval
	################################

	integration_interval = [0., right_boundary]

	################################
	# initial values
	################################

	initial_values = [0, 0]

	################################
	# solution 
	################################
	
	sol = solve_ivp(pm_system_1, integration_interval, initial_values, dense_output=True)


	number_of_samples_to_generate = 300
	s = np.linspace(0, right_boundary, number_of_samples_to_generate)
	z = sol.sol(s) # the field sol of the output only exists if dense_output is set to True
	result_values = z.T

	################################
	# plot 
	################################



	values_m = [y[0] for y in result_values]
	values_r = [y[1] for y in result_values]


	plt.plot(s, values_m, color='black')
	plt.plot(s, values_r, color='red')

	plt.xlabel('x')
	plt.legend(['m', 'r'], shadow=True)
	plt.title('Phase 1 (p.m.-strat.)')
	plt.show()




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

	values_a_1 = [x[0] for x in result_values]
	values_a_2 = [x[1] for x in result_values]
	values_f = [x[2] for x in result_values]
	values_d = [x[3] for x in result_values]
	values_r_3 = [x[4] for x in result_values]
	values_r_2 = [x[5] for x in result_values]
	values_r_1 = [x[6] for x in result_values]

	plt.plot(s, values_a_1, color='deepskyblue')
	plt.plot(s, values_a_2, color='lime')
	plt.plot(s, values_f, color='black')
	plt.plot(s, values_d, color='mistyrose')
	plt.plot(s, values_r_3, color='hotpink')
	plt.plot(s, values_r_2, color='darkorange')
	plt.plot(s, values_r_1, color='red')
	
	plt.xlabel('x')
	plt.legend(['a_1', 'a_2', 'f', 'd', 'r_3', 'r_2', 'r_1'], shadow=True)
	plt.title('Phase 2')
	plt.show()










#############################################################################################################################
#############################################################################################################################
############## PHASE 1. #####################################################################################################
#############################################################################################################################
#############################################################################################################################


def pm_system_1(t, y):
	m, r  = y

	eq_m = 2 - 2 * m + 2 * r
	eq_r = m - 4*r

	equations = [eq_m, eq_r]

	return equations




def calculate_ode_system_for_pm_phase_1(right_boundary, initial_values):

	################################
	# integration interval
	################################

	integration_interval = [0., right_boundary]

	################################
	# evaluation times
	################################

	evaluation_times = np.arange(mp.mpf(right_boundary-0.01), mp.mpf(right_boundary-granularity), mp.mpf(granularity))

	################################
	# initial values
	################################

	initial_values_precise = [mp.mpf(x) for x in initial_values]

	################################
	# solution - numbers
	################################

	sol = solve_ivp(pm_system_1, integration_interval, initial_values_precise, t_eval=evaluation_times) 

	################################
	# create result list
	################################

	for i,s in enumerate(sol.t):

		values_at_time_s = [item[i] for item in sol.y]
		values_at_time_s_precise = [mp.mpf(a) for a in values_at_time_s]

		m_at_time_s = values_at_time_s_precise[0]
		r_at_time_s = values_at_time_s_precise[1]

		if m_at_time_s >= 2/3:
			result_time = s
			result_list = values_at_time_s_precise

			return result_time, result_list

	print("last = " + str([item[-1] for item in sol.y]))
	print()





#############################################################################################################################
#############################################################################################################################
############## PHASE 2. #####################################################################################################
#############################################################################################################################
#############################################################################################################################



def phase_2_system(t, y):
	a_1, a_2, f, d, r_3, r_2, r_1  = y

	den = a_1 + a_2
	fac = 2*a_2 + 3*r_1

	eq_a_1 = - 3*a_1 - 3 * (a_1*r_1 / den)
	eq_a_2 = 3*a_1 - 2*a_2  - 3 * (a_2*r_1 / den)
	eq_f = fac
	
	eq_d = - 3*d + 2*a_2 * (1 + ((r_3 + 1.5 * r_2 + 3 * r_1) / den)) + 3*r_1 * (2 + ((r_3 + 1.5 * r_2 + 3 * r_1) / den))
	
	eq_r_3 = 3*d - 3*r_3 - fac * (r_3 / den)
	eq_r_2 = 2*r_3 - 2*r_2 - fac * (r_2 / den)
	
	eq_r_1 = r_2 - 2*a_2 * (r_1 / den) - 3*r_1 * (1/3 + r_1 / den)
	
	equations = [eq_a_1, eq_a_2, eq_f, eq_d, eq_r_3, eq_r_2, eq_r_1]

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

	for i,s in enumerate(sol.t):

		values_at_time_s = [item[i] for item in sol.y]
		values_at_time_s_precise = [mp.mpf(x) for x in values_at_time_s]

		for value in values_at_time_s_precise:
			if value > 1 or value < 0:
				print("already too much")
				print("time = " + str(s))
				print(values_at_time_s_precise)
				return



		f_at_time_s = values_at_time_s_precise[2]
		
		if f_at_time_s > target_value_for_f_precise:
			result_time = s
			result_list = values_at_time_s_precise
			return result_time, result_list

	print("last = " + str([item[-1] for item in sol.y]))
	print()

	# there are also contexts where we just would want:
	# result_time = sol.t[-1]
	# result_list = [item[-1] for item in sol.y]
	# return result_time, result_list


#############################################################################################################################
#############################################################################################################################
############## PHASE 3. #####################################################################################################
#############################################################################################################################
#############################################################################################################################


def phase_3(alpha):
	return ((2 * alpha)/(1-alpha))**(1/8) * 1/(1-alpha**(1/8))





#############################################################################################################################
#############################################################################################################################
############## MAIN FUNCTION ################################################################################################
#############################################################################################################################
#############################################################################################################################




	
"""
This is how long phase 1 of the matching algorithm takes altogether: 
s = 0.73680000000000003530873510238
m_at_time_s = 0.841855608372715573117839159475
r_at_time_s = 0.158146261519498065554285162675
at this time, m is already > 2/3, so we can stop before (and we don't need the other phases)
"""



initial_values_for_phase_1 = [0,0]
right_boundary_for_phase_1 = 0.51
plot_boolean = True
if plot_boolean == True:
	create_plot_for_pm(right_boundary_for_phase_1)
result_time_phase_1, result_list_phase_1 = calculate_ode_system_for_pm_phase_1(right_boundary_for_phase_1, initial_values_for_phase_1)
print("result_time_phase_1 = " + str(result_time_phase_1))
print(result_list_phase_1)
print()

m_phase_1, r_phase_1 = result_list_phase_1

a_2 = r_phase_1 * 3
a_1 = 1 - a_2

initial_values_for_phase_2 = [a_1, a_2, 0, 0, 0, 0, 0]

alpha = 10**(-15)

target_value_for_f_precise = mp.mpf(1-(0.5 * alpha))


left_boundary = 2.44
right_boundary = 2.4485

plot_boolean = True

if plot_boolean == True:
	create_plot_for_phase_2(right_boundary, initial_values_for_phase_2)
result_time_phase_2, result_list_phase_2 = calculate_ode_system_for_phase_2(left_boundary, right_boundary, target_value_for_f_precise, initial_values_for_phase_2)
print("result_time_phase_2 = " + str(result_time_phase_2))
print(result_list_phase_2)
print()

result_time_phase_3 = phase_3(alpha)
print("result_time_phase_3 = " + str(result_time_phase_3))
print()

print("total time = " + str(result_time_phase_1 + result_time_phase_2 + result_time_phase_3))
