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

def create_plot_for_phase_1():

	def phase_1_system_with_r_3(t, y):
		u, m, a_2, f, d, r_3m, r_3a = y

		eq_u = -2*u - m
		eq_m = 2*u - 2*m
		eq_a_2 = 3*m - 2*a_2
		eq_f = 2*a_2
		eq_d = -3*d + 2*a_2 + 2* r_3a 
		eq_r_3m = 3*d - 2*r_3m
		eq_r_3a = 2*r_3m - 2*r_3a

		equations = [eq_u, eq_m, eq_a_2, eq_f, eq_d, eq_r_3m, eq_r_3a]

		return equations



	################################
	# integration interval
	################################

	right_boundary = math.atan(math.sqrt(2)) / math.sqrt(2)
	integration_interval = [0., right_boundary]

	################################
	# initial values
	################################

	initial_values =  [1, 0, 0, 0, 0, 0, 0] 

	################################
	# solution 
	################################
	
	sol =  solve_ivp(phase_1_system_with_r_3, integration_interval, initial_values, dense_output=True)


	number_of_samples_to_generate = 300
	s = np.linspace(0, right_boundary, number_of_samples_to_generate)
	z = sol.sol(s) # the field sol of the output only exists if dense_output is set to True
	result_values = z.T

	################################
	# plot 
	################################

	values_u = [x[0] for x in result_values]
	values_m = [x[1] for x in result_values]
	values_a_2 = [x[2] for x in result_values]
	values_f = [x[3] for x in result_values]
	values_d = [x[4] for x in result_values]
	values_r_3m = [x[5] for x in result_values]
	values_r_3a = [x[6] for x in result_values]

	plt.plot(s, values_u, color='silver')
	plt.plot(s, values_m, color='deepskyblue')
	plt.plot(s, values_a_2, color='lime')
	plt.plot(s, values_f, color='black')
	plt.plot(s, values_d, color='mistyrose')
	plt.plot(s, values_r_3m, color='hotpink')
	plt.plot(s, values_r_3a, color='darkorange')
	

	plt.xlabel('x')
	plt.legend(['u', 'm', 'a_2', 'f', 'd', 'r_3m', 'r_3a'], shadow=True)
	plt.title('Phase 1')
	plt.show()




def create_plot_for_phase_2(right_boundary, initial_values):

	################################
	# integration interval
	################################

	integration_interval = [0., right_boundary]

	################################
	# solution 
	################################

	sol = solve_ivp(phase_2_system_with_distinction, integration_interval, initial_values, dense_output=True)

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
	values_r_3a = [x[7] for x in result_values]
	values_r_2a = [x[8] for x in result_values]
	values_r_1a = [x[9] for x in result_values]

	plt.plot(s, values_a_1, color='deepskyblue')
	plt.plot(s, values_a_2, color='lime')
	plt.plot(s, values_f, color='black')
	plt.plot(s, values_d, color='mistyrose')
	plt.plot(s, values_r_3, color='hotpink')
	plt.plot(s, values_r_2, color='darkorange')
	plt.plot(s, values_r_1, color='red')
	plt.plot(s, values_r_3a, color='tan')
	plt.plot(s, values_r_2a, color='navajowhite')
	plt.plot(s, values_r_1a, color='papayawhip')
	
	plt.xlabel('x')
	plt.legend(['a_1', 'a_2', 'f', 'd', 'r_3m', 'r_2m', 'r_1m', 'r_3a', 'r_2a', 'r_1a'], shadow=True) 
	plt.title('Phase 2')
	plt.show()



#############################################################################################################################
#############################################################################################################################
############## PHASE 1. #####################################################################################################
#############################################################################################################################
#############################################################################################################################


# input: None -  to get analytical solution 
# input: some number - to get values of the function at that number
def solve_phase_1_ode_system_with_r_3(none_or_x):

	# functions

	x = symbols("x")
	u, m, a_2, f, d, r_3m, r_3a = symbols("u m a_2 f d r_3m r_3a", cls=Function)

	# equations

	eq_u = Eq(u(x).diff(x), - 2 * u(x) - m(x))
	eq_m = Eq(m(x).diff(x), 2 * u(x) - 2 * m(x))
	eq_a_2 = Eq(a_2(x).diff(x), 3 * m(x) - 2 * a_2(x))
	eq_f = Eq(f(x).diff(x), 2 * a_2(x))
	eq_d = Eq(d(x).diff(x), - 3 * d(x) + 2 * a_2(x) + 2 * r_3a(x)) 
	eq_r_3m = Eq(r_3m(x).diff(x), 3 * d(x) - 2 * r_3m(x))
	eq_r_3a = Eq(r_3a(x).diff(x), 2 * r_3m(x) - 2 * r_3a(x))
	

	equations = [eq_u, eq_m, eq_a_2, eq_f, eq_d, eq_r_3m, eq_r_3a]

	# initial values

	x_in = 0

	u_in = 1
	m_in = 0
	a_2_in = 0
	f_in = 0
	d_in = 0
	r_3m_in = 0
	r_3a_in = 0

	initial_values = {u(x_in): u_in, m(x_in): m_in, a_2(x_in): a_2_in, f(x_in): f_in, d(x_in): d_in, r_3m(x_in): r_3m_in, r_3a(x_in): r_3a_in}

	# solution

	solution_array = dsolve_system(equations, ics=initial_values)[0]

	if none_or_x == None:
		for function in solution_array: 
			print(function)

	else:
		function_values_at_x = [solution_function.evalf(precision_parameter, subs={x: none_or_x}).rhs for solution_function in solution_array]
		return function_values_at_x



#############################################################################################################################
#############################################################################################################################
############## PHASE 2. #####################################################################################################
#############################################################################################################################
#############################################################################################################################


def phase_2_system_with_distinction(t, y):
	a_1, a_2, f, d, r_3m, r_2m, r_1m, r_3a, r_2a, r_1a  = y

	p_disapp_a_1 = a_1 + r_1m
	p_disapp_a_2 = (2/3)*a_2 + r_1a
	e_resp_a_1 = (r_3m + 1.5 * r_2m + 3 * r_1m) / a_1
	e_resp_a_2 = (r_3a + 1.5 * r_2a + 3 * r_1a) / a_2
	e_r_3m_for_a_1 = 3 * r_3m / a_1
	e_r_2m_for_a_1 = 3 * r_2m / a_1
	e_r_1m_for_a_1 = 3 * r_1m / a_1
	e_r_3a_for_a_2 = 3 * r_3a / a_2
	e_r_2a_for_a_2 = 3 * r_2a / a_2
	e_r_1a_for_a_2 = 3 * r_1a / a_2


	#########################################################
	eq_a_1 = -3 * p_disapp_a_1
	eq_a_2 = 3*a_1 - 3 * p_disapp_a_2
	eq_f = 3 * (r_1m + p_disapp_a_2)
	eq_d = - 3*d + 3 * r_1m * (2 + e_resp_a_1) + 2 * a_2 * (1 +  e_resp_a_2)+ 3 * r_1a * (2 + e_resp_a_2)

	#eq_r_3m = 3*d - 3*r_3m - p_disapp_a_1 * e_r_3m_for_a_1
	eq_r_3m = 3*d*(a_1 / (a_1 + a_2)) - 3*r_3m - p_disapp_a_1 * e_r_3m_for_a_1
	eq_r_2m = 2*r_3m - 2*r_2m - p_disapp_a_1 * e_r_2m_for_a_1
	eq_r_1m = r_2m - a_1 * e_r_1m_for_a_1 - r_1m * (1 + e_r_1m_for_a_1)

	eq_r_3a = 3*d*(a_2 / (a_1 + a_2)) -3*r_3a + a_1 * e_r_3m_for_a_1 - p_disapp_a_2 * e_r_3a_for_a_2
	eq_r_2a = 2*r_3a - 2*r_2a + a_1 * e_r_2m_for_a_1 - p_disapp_a_2 * e_r_2a_for_a_2
	eq_r_1a = r_2a + a_1 * e_r_1m_for_a_1 - (2/3)*a_2 * e_r_1a_for_a_2 - r_1a * (1 + e_r_1a_for_a_2)

	
	equations = [eq_a_1, eq_a_2, eq_f, eq_d, eq_r_3m, eq_r_2m, eq_r_1m, eq_r_3a, eq_r_2a, eq_r_1a]

	return equations



def calculate_ode_system_for_phase_2(left_boundary, right_boundary, target_value_for_f_precise, initial_values):

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
	
	sol = solve_ivp(phase_2_system_with_distinction, integration_interval, initial_values_precise, t_eval=evaluation_times) # "dense output = True" doesn't help!!!! 

	################################
	# create result list
	################################



	if target_value_for_f_precise != None:

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

			sum_augmentable_complete = values_at_time_s_precise[0] + values_at_time_s_precise[1] + values_at_time_s_precise[2]
			if sum_augmentable_complete > 1 + 10**(-15) or sum_augmentable_complete < 1 - 10**(-15):
					print("already too much - sum is not 1")
					print("time = " + str(s))
					print(values_at_time_s_precise)
					return None

			##################### end of sanity check #####################

			f_at_time_s = values_at_time_s_precise[2]
			
			if f_at_time_s > target_value_for_f_precise:
				result_time = s
				result_list = values_at_time_s_precise
				return result_time, result_list

		last = [item[-1] for item in sol.y]
		print("last = " + str(last))
		return None

	else:
		result_time = sol.t[-1]
		result_list = [item[-1] for item in sol.y]
		return result_time, result_list



			


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




plot_boolean = True

expected_time_phase_1 = math.atan(math.sqrt(2)) / math.sqrt(2)

if plot_boolean == True:
	create_plot_for_phase_1()

result_list_phase_1 = solve_phase_1_ode_system_with_r_3(expected_time_phase_1)
u, m, a_2, f, d, r_3m, r_3a = result_list_phase_1

a_1 = u + m

initial_values_for_phase_2 = [a_1, a_2, f, d, r_3m, 0, 0, r_3a, 0, 0]

left_boundary_2 = 2.2506
right_boundary_2 = 2.25085

alpha_exponent = 15
target_value_for_f_precise = mp.mpf(1-(0.5 * 10**(-alpha_exponent)))


if plot_boolean == True:
	create_plot_for_phase_2(right_boundary_2, initial_values_for_phase_2)
result_phase_2 = calculate_ode_system_for_phase_2(left_boundary_2, right_boundary_2, target_value_for_f_precise, initial_values_for_phase_2)

if result_phase_2 != None:
	result_time_phase_2, result_list_phase_2 = result_phase_2

	print("result_time_phase_2 = " + str(result_time_phase_2))
	print("result_list_phase_2 = " + str(result_list_phase_2))

	alpha_precise = mp.mpf(10**(-alpha_exponent))
	result_time_phase_3 = phase_3(alpha_precise)

	print("total number of rounds = " + str(0.675511 + result_time_phase_2 + result_time_phase_3))



