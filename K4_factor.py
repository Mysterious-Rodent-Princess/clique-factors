#!/usr/bin/python3

import math
import numpy as np

from sympy import symbols, Eq, Function, sqrt, atan, exp, sin, cos
from sympy import solve
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
### get evaluation time granularity 10**(-8) #########
######################################################

granularity = 10**(-8)




#############################################################################################################################
#############################################################################################################################
############## PLOTS #####################################################################################################
#############################################################################################################################
#############################################################################################################################



def create_plot_for_phase_1(ode_system, subphase_index, right_boundary, initial_values):

	################################
	# integration interval
	################################

	integration_interval = [0., right_boundary]

	################################
	# solution 
	################################

	sol = solve_ivp(ode_system, integration_interval, initial_values, dense_output=True) 

	number_of_samples_to_generate = 30000
	s = np.linspace(0, right_boundary, number_of_samples_to_generate)
	z = sol.sol(s) # the field sol of the output only exists if dense_output is set to True
	result_values = z.T

	################################
	# plot 
	################################


	values_u = [x[0] for x in result_values]
	values_m = [x[1] for x in result_values]
	values_p_3 = [x[2] for x in result_values]

	values_s_4 = [x[3] for x in result_values]
	values_p_4 = [x[4] for x in result_values]
	values_k_3 = [x[5] for x in result_values]

	values_a_2 = [x[6] for x in result_values]
	values_a_3 = [x[7] for x in result_values]
	values_f = [x[8] for x in result_values]

	plt.plot(s, values_u, color='silver')
	plt.plot(s, values_m, color='deepskyblue')
	plt.plot(s, values_p_3, color='mediumblue')

	plt.plot(s, values_s_4, color='darkorange')
	if subphase_index == 2:
		plt.plot(s, values_p_4, color='brown')
	plt.plot(s, values_k_3, color='fuchsia')

	plt.plot(s, values_a_2, color='lime')
	plt.plot(s, values_a_3, color='green')
	plt.plot(s, values_f, color='black')


	plt.xlabel('x')
	if subphase_index == 1:
		plt.legend(['u', 'm', 'p_3', 's_4', 'k_3', 'a_2', 'a_3', 'f'], shadow=True) 
	if subphase_index == 2:
		plt.legend(['u', 'm', 'p_3', 's_4', 'p_4', 'k_3', 'a_2', 'a_3', 'f'], shadow=True) 
	plt.title('Phase 1.'+str(subphase_index))
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
	plt.title('Phase 2')
	plt.show()



#############################################################################################################################
#############################################################################################################################
############## PHASE 1. #####################################################################################################
#############################################################################################################################
#############################################################################################################################

#############################################################################################################################
############## Sympy #####################################################################################################
#############################################################################################################################


########### the sympy-code for phase 1.1 that does not generate an analytical result... 
"""
def solve_phase_1_1_ode_system(none_or_x):


	# functions

	x = symbols("x")
	u, m, p_3, s_4, k_3, a_2, a_3, f = symbols("u m p_3 s_4 k_3 a_2 a_3 f", cls=Function)


	# equations

	#eq_u = Eq(u(x).diff(x), - 2 * u(x) - m(x) - (1/3) * p_3(x) - k_3(x))
	eq_u = Eq(u(x).diff(x), - 2 * u(x) - m(x) - k_3(x))
	eq_m = Eq(m(x).diff(x), 2 * u(x) - 2 * m(x))
	#eq_p_3 = Eq(p_3(x).diff(x), 3 * m(x) - 3 * p_3(x))
	eq_p_3 = Eq(p_3(x).diff(x), 3 * m(x) - 2 * p_3(x))
	#eq_s_4 = Eq(s_4(x).diff(x), (4/3) * p_3(x) - 3 * s_4(x))
	eq_s_4 = Eq(s_4(x).diff(x), 0)

	eq_k_3 = Eq(k_3(x).diff(x), 2 * p_3(x) - 3 * k_3(x))

	#eq_a_2 = Eq(a_2(x).diff(x), 3 * s_4(x) + 4 * k_3(x) - 3 * a_2(x))
	eq_a_2 = Eq(a_2(x).diff(x), 4 * k_3(x) - 3 * a_2(x))
	eq_a_3 = Eq(a_3(x).diff(x), 3 * a_2(x) - 2 * a_3(x))
	eq_f = Eq(f(x).diff(x), 2 * a_3(x))

	equations = [eq_u, eq_m, eq_p_3, eq_s_4, eq_k_3, eq_a_2, eq_a_3, eq_f]


	# initial values

	x_in = 0
	u_in, m_in, p_3_in, s_4_in, k_3_in, a_2_in, a_3_in, f_in = [1, 0, 0, 0, 0, 0, 0, 0]


	initial_values = {u(x_in): u_in, m(x_in): m_in, p_3(x_in): p_3_in, s_4(x_in): s_4_in, k_3(x_in): k_3_in, a_2(x_in): a_2_in, a_3(x_in): a_3_in, f(x_in): f_in}


	# solution

	solution_array = dsolve_system(equations, ics=initial_values)[0]

	if none_or_x == None:
		for function in solution_array: 
			print(function)

	else:
		function_values_at_x = [solution_function.evalf(precision_parameter, subs={x: none_or_x}).rhs for solution_function in solution_array]
		return function_values_at_x
"""




def solve_phase_1_2_ode_system(initial_values_for_phase_1_2, none_or_x):

	# functions

	x = symbols("x")
	u, m, p_3, s_4, p_4, k_3, a_2, a_3, f = symbols("u m p_3 s_4 p_4 k_3 a_2 a_3 f", cls=Function)


	# equations

	eq_u = Eq(u(x).diff(x), 0)
	eq_m = Eq(m(x).diff(x), - 4 * m(x))
	eq_p_3 = Eq(p_3(x).diff(x),- 2 * p_3(x))
	eq_s_4 = Eq(s_4(x).diff(x), - 3 * s_4(x))
	eq_p_4 = Eq(p_4(x).diff(x), 4 * m(x) - 4 * p_4(x))

	eq_k_3 = Eq(k_3(x).diff(x), 2 * p_3(x))

	eq_a_2 = Eq(a_2(x).diff(x), 3 * s_4(x) + 4 * p_4(x) - 3 * a_2(x))
	eq_a_3 = Eq(a_3(x).diff(x), 3 * a_2(x) - 2 * a_3(x))
	eq_f = Eq(f(x).diff(x), 2 * a_3(x))

	equations = [eq_u, eq_m, eq_p_3, eq_s_4, eq_p_4, eq_k_3, eq_a_2, eq_a_3, eq_f]


	# initial values

	x_in = 0
	u_in, m_in, p_3_in, s_4_in, p_4_in, k_3_in, a_2_in, a_3_in, f_in = initial_values_for_phase_1_2


	initial_values = {u(x_in): u_in, m(x_in): m_in, p_3(x_in): p_3_in,  s_4(x_in): s_4_in, p_4(x_in): p_4_in, k_3(x_in): k_3_in, a_2(x_in): a_2_in, a_3(x_in): a_3_in, f(x_in): f_in}


	# solution

	solution_array = dsolve_system(equations, ics=initial_values)[0]

	if none_or_x == None:
		for function in solution_array: 
			print(function)
				

	else:
		function_values_at_x = [solution_function.evalf(precision_parameter, subs={x: none_or_x}).rhs for solution_function in solution_array]
		return none_or_x, function_values_at_x



#############################################################################################################################
############## Scipy #####################################################################################################
#############################################################################################################################


# this alone does not get us to phase 2 (neither with nor without s_4)
def phase_1_1_system(t, y):
	u, m, p_3, s_4, p_4, k_3, a_2, a_3, f = y

	eq_u = -2 * u - m - (1/3) * p_3 - k_3 
	eq_m = 2 * u - 2 * m
	eq_p_3 = 3 * m - 3 * p_3
	eq_s_4 = (4/3) * p_3 - 3 * s_4
	eq_p_4 = 0 # only to have same number of variables
	eq_k_3 = 2 * p_3 - 3 * k_3
	eq_a_2 = 3 * s_4 + 4 * k_3 - 3 * a_2
	eq_a_3 = 3 * a_2 - 2 * a_3
	eq_f = 2 * a_3

	equations = [eq_u, eq_m, eq_p_3, eq_s_4, eq_p_4, eq_k_3, eq_a_2, eq_a_3, eq_f]

	return equations


def phase_1_2_system(t, y):
	u, m, p_3, s_4, p_4, k_3, a_2, a_3, f = y

	eq_u = 0
	eq_m = -4 * m
	eq_p_3 = - 2 * p_3
	eq_s_4 = -3 * s_4
	eq_p_4 = 4 * m - 4 * p_4
	eq_k_3 = 2 * p_3 
	eq_a_2 = 3 * s_4 + 4 * p_4 - 3 * a_2
	eq_a_3 = 3 * a_2 - 2 * a_3
	eq_f = 2 * a_3

	equations = [eq_u, eq_m, eq_p_3, eq_s_4, eq_p_4, eq_k_3, eq_a_2, eq_a_3, eq_f]

	return equations


def calculate_ode_system_for_phase_1(ode_system, left_boundary, right_boundary, initial_values, ode_system_index):

	integration_interval = [0., right_boundary]

	evaluation_times = None 

	
	evaluation_times = np.arange(mp.mpf(left_boundary), mp.mpf(right_boundary), mp.mpf(granularity))

	initial_values_precise = [mp.mpf(x) for x in initial_values]

	################################
	# solution - numbers
	################################
	
	sol = solve_ivp(ode_system, integration_interval, initial_values_precise, t_eval=evaluation_times) 

	################################
	# create result list
	################################

	if ode_system_index == 1:

		for i,s in enumerate(sol.t):

			values_at_time_s = [item[i] for item in sol.y]
			values_at_time_s_precise = [mp.mpf(x) for x in values_at_time_s]

			u_at_time_s = values_at_time_s_precise[0]

			if u_at_time_s < 10**(-5):
				result_time = s
				result_list = values_at_time_s_precise
				return result_time, result_list


	else:
		for i,s in enumerate(sol.t):

			values_at_time_s = [item[i] for item in sol.y]
			values_at_time_s_precise = [mp.mpf(x) for x in values_at_time_s]

			u_at_time_s = values_at_time_s_precise[0]
			m_at_time_s = values_at_time_s_precise[1]
			p_3_at_time_s = values_at_time_s_precise[2]
			s_4_at_time_s = values_at_time_s_precise[3]
			p_4_at_time_s = values_at_time_s_precise[4]
			k_3_at_time_s = values_at_time_s_precise[5]

			non_augmentable = u_at_time_s + m_at_time_s + p_3_at_time_s + s_4_at_time_s + p_4_at_time_s

			##################### start of sanity check #####################

			for value in values_at_time_s_precise:
				if value > 1 or value < 0:
					print("already too much - values out of range")
					print("time = " + str(s))
					print(values_at_time_s_precise)
					print("non_augmentable = " + str(non_augmentable))
					print("k_3_at_time_s/3 = " + str(k_3_at_time_s/3))
					return None

			##################### end of sanity check #####################

			if non_augmentable < k_3_at_time_s/3:
				result_time = s
				result_list = values_at_time_s_precise
				return result_time, result_list

		last = [item[-1] for item in sol.y]
		print("last = " + str(last))
		return None




#############################################################################################################################
#############################################################################################################################
############## PHASE 2. #####################################################################################################
#############################################################################################################################
#############################################################################################################################

def phase_2_system(t, y):
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
	
	sol = solve_ivp(phase_2_system, integration_interval, initial_values_precise, t_eval=evaluation_times)  

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



right_boundary_for_phase_1_1 = 0.872

left_boundary_for_phase_2 = 3.235
right_boundary_for_phase_2 = 3.2364

alpha_exponent = 15
target_value_for_f_precise = mp.mpf(1-(0.5 * 10**(-alpha_exponent)))
alpha_precise = mp.mpf(10**(-alpha_exponent))




initial_values_for_phase_1_1 = [1,0,0,0,0,0,0,0,0]
result_time_1_1, result_list_1_1 = calculate_ode_system_for_phase_1(phase_1_1_system, right_boundary_for_phase_1_1-0.005, right_boundary_for_phase_1_1, initial_values_for_phase_1_1, 1)
print("result_time_1_1 = " + str(result_time_1_1))
print(result_list_1_1)
print()

create_plot_for_phase_1(phase_1_1_system, 1, float(result_time_1_1), initial_values_for_phase_1_1)


initial_values_for_phase_1_2 = result_list_1_1


result_time_1_2, result_list_1_2 = solve_phase_1_2_ode_system(initial_values_for_phase_1_2, 0.7383820)
print("result_time_1_2 = " + str(result_time_1_2))
print(result_list_1_2)
print()

create_plot_for_phase_1(phase_1_2_system, 2, float(result_time_1_2), initial_values_for_phase_1_2)


u, m, p_3, s_4, p_4, k_3, a_2, a_3, f = result_list_1_2
a_1 = u + m + p_3 + s_4 + p_4 + k_3


initial_values_phase_2 = [a_1, a_2, a_3, f, f, 0, 0, 0, 0, 0, 0, 0, 0]


create_plot_for_phase_2(right_boundary_for_phase_2, initial_values_phase_2)

result_time_phase_2, result_list_phase_2 = calculate_ode_system_for_phase_2(left_boundary_for_phase_2, right_boundary_for_phase_2, target_value_for_f_precise, initial_values_phase_2)
print("result_time_phase_2 = " + str(result_time_phase_2))
print(result_list_phase_2)
print()

result_time_phase_3 = phase_3(alpha_precise)
print("result_time_phase_3 = " + str(result_time_phase_3))
print()

total_time = result_time_1_1 + result_time_1_2 + result_time_phase_2 + result_time_phase_3 
print("total_time = " + str(total_time))




"""
Eq(u(x), 9.99205144086067e-6)
Eq(m(x), 0.208523106671333*exp(-4*x))
Eq(p_3(x), 0.252583815989712*exp(-2*x))
Eq(s_4(x), 0.0913479514171854*exp(-3*x))
Eq(p_4(x), 0.834092426685332*x*exp(-4*x) - 2.22044604925031e-16*exp(-4*x))
Eq(k_3(x), 0.38960574311549 - 0.252583815989712*exp(-2*x))
Eq(a_2(x), 0.274043854251556*x*exp(-3*x) - 3.33636970674133*x*exp(-4*x) + 3.50637021820822*exp(-3*x) - 3.33636970674133*exp(-4*x))
Eq(a_3(x), -0.822131562754668*x*exp(-3*x) + 5.00455456011199*x*exp(-4*x) + 3.93351425947344*exp(-2*x) - 11.3412422173793*exp(-3*x) + 7.50683184016798*exp(-4*x))
Eq(f(x), 0.548087708503112*x*exp(-3*x) - 2.50227728005599*x*exp(-4*x) + 0.610384264833069 - 3.93351425947344*exp(-2*x) + 7.74352404775392*exp(-3*x) - 4.37898524009799*exp(-4*x))
"""

#u, m, p_3, s_4, p_4, k_3, _, _, _ = result_list_1_2
#print(u + m + p_3 + s_4 + p_4)
#print(k_3 / 3)





