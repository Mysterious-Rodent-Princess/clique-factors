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




def create_plot_for_phase_2_k3(right_boundary, initial_values):

	################################
	# integration interval
	################################

	integration_interval = [0., right_boundary]

	################################
	# solution 
	################################

	sol = solve_ivp(phase_2_k3_system, integration_interval, initial_values, dense_output=True)

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
	plt.title('Phase 2 -- K_3')
	plt.show()



def create_plot_for_phase_2_k4(right_boundary, initial_values):

	################################
	# integration interval
	################################

	integration_interval = [0., right_boundary]

	################################
	# solution 
	################################

	sol = solve_ivp(phase_2_k4_system, integration_interval, initial_values, dense_output=True) 

	number_of_samples_to_generate = 30000
	s = np.linspace(0, right_boundary, number_of_samples_to_generate)
	z = sol.sol(s) # the field sol of the output only exists if dense_output is set to True
	result_values = z.T

	################################
	# plot 
	################################


	values_a_1 = [x[0] for x in result_values]
	values_a_2 = [x[1] for x in result_values]
	values_a_3 = [x[2] for x in result_values]

	values_f = [x[3] for x in result_values]
	values_d = [x[4] for x in result_values]

	values_r_52 = [x[5] for x in result_values] # with m if dist
	values_r_41b = [x[6] for x in result_values] # with m if dist
	values_r_42 = [x[7] for x in result_values] # with m if dist
	values_r_31b = [x[8] for x in result_values]
	values_r_32 = [x[9] for x in result_values]
	values_r_21 = [x[10] for x in result_values]
	values_r_22 = [x[11] for x in result_values]
	values_r_11 = [x[12] for x in result_values]

	plt.plot(s, values_a_1, color='deepskyblue')
	plt.plot(s, values_a_2, color='lime')
	plt.plot(s, values_a_3, color='green')

	plt.plot(s, values_f, color='black')
	plt.plot(s, values_d, color='gray')

	plt.plot(s, values_r_52, color='darkviolet')
	plt.plot(s, values_r_41b, color='mediumorchid')
	plt.plot(s, values_r_42, color='violet')
	plt.plot(s, values_r_31b, color='magenta')
	plt.plot(s, values_r_32, color='deeppink')
	plt.plot(s, values_r_21, color='hotpink')
	plt.plot(s, values_r_22, color='darkorange')
	plt.plot(s, values_r_11, color='red')

	
	plt.xlabel('x')
	plt.legend(['a_1', 'a_2', 'a_3', 'f', 'd', 'r_52', 'r_41b', 'r_42', 'r_31b', 'r_32', 'r_21', 'r_22', 'r_11'], shadow=True) 
	plt.title('Phase 2 -- K_4')
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
############## PHASE 2. (K_3) ###############################################################################################
#############################################################################################################################
#############################################################################################################################



def phase_2_k3_system(t, y):
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




def calculate_ode_system_for_phase_2_k3(left_boundary, right_boundary, initial_values):

	################################
	# integration interval
	################################

	integration_interval = [0., right_boundary]

	################################
	# evaluation times
	################################

	evaluation_times = np.arange(mp.mpf(left_boundary), mp.mpf(right_boundary-granularity), mp.mpf(granularity))


	################################
	# initial values
	################################

	initial_values_precise = [mp.mpf(x) for x in initial_values]


	################################
	# solution - numbers
	################################
	
	sol = solve_ivp(phase_2_k3_system, integration_interval, initial_values_precise, t_eval=evaluation_times) 

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
		
		if f_at_time_s >= 3/4:
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

def phase_2_k4_system(t, y):
	a_1, a_2, a_3, f, d, r_52, r_41b, r_42, r_31b, r_32, r_21, r_22, r_11  = y

	den = a_1 + a_2 + a_3
	p_creation_k4 = 0.5*a_3 + r_11

	r = 4 * r_52 + r_41b + 4 * r_42 + (4/3) * r_31b + 4 * r_32 + 2 * r_21 + 4 * r_22 + 4 * r_11

	def ex_diff_a_something_after_r_11_off_times_prob(a_something):
		return - 4 * (a_something / den) * r_11

	def ex_diff_red_something_vert_after_k4_creation_times_prob(red_something):
		return - p_creation_k4 * (4 * red_something)/den


	#########################################################
	eq_a_1 = -4 * a_1 + ex_diff_a_something_after_r_11_off_times_prob(a_1)
	eq_a_2 = 4 * a_1 - 3 * a_2 + ex_diff_a_something_after_r_11_off_times_prob(a_2)
	eq_a_3 = 3 * a_2 - 2 * a_3 + ex_diff_a_something_after_r_11_off_times_prob(a_3)

	eq_f = 4 * p_creation_k4
	eq_d = -4 * d + 4 * 0.5*a_3 * (1 + (r / den)) + + 4 * r_11 * (2 + (r / den))

	eq_r_52 = d - 4 * r_52 + ex_diff_red_something_vert_after_k4_creation_times_prob(r_52)
	eq_r_41b = 4 * r_52 - 4 * r_41b + ex_diff_red_something_vert_after_k4_creation_times_prob(r_41b)
	eq_r_42 = 3 * r_52 - 3 * r_42 + ex_diff_red_something_vert_after_k4_creation_times_prob(r_42)
	eq_r_31b = 3 * (r_41b + r_42) - 3 * r_31b + ex_diff_red_something_vert_after_k4_creation_times_prob(r_31b)
	eq_r_32 = 2 * r_42 - 2 * r_32 + ex_diff_red_something_vert_after_k4_creation_times_prob(r_32)
	eq_r_21 = 2 * (r_31b + r_32) - 2 * r_21 + ex_diff_red_something_vert_after_k4_creation_times_prob(r_21)
	eq_r_22 = r_32 - r_22 + ex_diff_red_something_vert_after_k4_creation_times_prob(r_22)
	eq_r_11 = (r_21 + r_22) - 0.5*a_3 * (4 * r_11)/den - r_11 * (1 + (4 * r_11)/den)

	equations = [eq_a_1, eq_a_2, eq_a_3, eq_f, eq_d, eq_r_52, eq_r_41b, eq_r_42, eq_r_31b, eq_r_32, eq_r_21, eq_r_22, eq_r_11]

	return equations




def calculate_ode_system_for_phase_2_k4(left_boundary, right_boundary, target_value_for_f_precise, initial_values):

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
	
	sol = solve_ivp(phase_2_k4_system, integration_interval, initial_values_precise, t_eval=evaluation_times)  

	################################
	# create result list
	################################

	for i,s in enumerate(sol.t):

		values_at_time_s = [item[i] for item in sol.y]
		values_at_time_s_precise = [mp.mpf(x) for x in values_at_time_s]

		##################### start of sanity check #####################

		for value in values_at_time_s_precise:
			if value > 1 or value < 0:
				print("already too much - values out of range")
				print("time = " + str(s))
				print(values_at_time_s_precise)
				return None

		sum_augmentable_complete = values_at_time_s_precise[0] + values_at_time_s_precise[1] + values_at_time_s_precise[2] + values_at_time_s_precise[3]
		if sum_augmentable_complete > 1 + 10**(-15) or sum_augmentable_complete < 1 - 10**(-15):
				print("already too much - sum is not 1")
				print("time = " + str(s))
				print(values_at_time_s_precise)
				return None

		f_at_time_s = values_at_time_s_precise[3]
		
		if f_at_time_s > target_value_for_f_precise:
			result_time = s
			result_list = values_at_time_s_precise
			return result_time, result_list

	print("last = " + str([item[-1] for item in sol.y]))
	print()



#############################################################################################################################
#############################################################################################################################
############## PHASE 3. #####################################################################################################
#############################################################################################################################
#############################################################################################################################


def phase_3(alpha):
	return ((2 * alpha)/(1-alpha))**(1/12) * 1/(1-alpha**(1/12))




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

initial_values_for_phase_2_k3 = [a_1, a_2, 0, 0, 0, 0, 0]


left_boundary_for_phase_2_k3 = 0.978
right_boundary_for_phase_2_k3 = 0.98

if plot_boolean == True:
	create_plot_for_phase_2_k3(right_boundary_for_phase_2_k3, initial_values_for_phase_2_k3)
result_time_phase_2_k3, result_list_phase_2_k3 = calculate_ode_system_for_phase_2_k3(left_boundary_for_phase_2_k3, right_boundary_for_phase_2_k3, initial_values_for_phase_2_k3)
print("result_time_phase_2_k3 = " + str(result_time_phase_2_k3))
print(result_list_phase_2_k3)
print()


left_boundary_for_phase_2_k4 = 3.604
right_boundary_for_phase_2_k4 = 3.607

alpha_exponent = 15
target_value_for_f_precise = mp.mpf(1-(0.5 * 10**(-alpha_exponent)))
alpha_precise = mp.mpf(10**(-alpha_exponent))


initial_values_phase_2_k4 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


create_plot_for_phase_2_k4(right_boundary_for_phase_2_k4, initial_values_phase_2_k4)

result_time_phase_2_k4, result_list_phase_2_k4 = calculate_ode_system_for_phase_2_k4(left_boundary_for_phase_2_k4, right_boundary_for_phase_2_k4, target_value_for_f_precise, initial_values_phase_2_k4)
print("result_time_phase_2_k4 = " + str(result_time_phase_2_k4))
print(result_list_phase_2_k4)
print()


result_time_phase_3 = phase_3(alpha_precise)
print("result_time_phase_3 = " + str(result_time_phase_3))
print()

total_time = result_time_phase_1 + result_time_phase_2_k3 + result_time_phase_2_k4 + result_time_phase_3
print("total_time = " + str(total_time))
