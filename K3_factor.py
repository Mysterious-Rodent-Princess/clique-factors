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

	def phase_1_system_without_r_3(t, y):
		u, m, a_2, f = y

		eq_u = -2*u - m
		eq_m = 2*u - 2*m
		eq_a_2 = 3*m - 2*a_2
		eq_f = 2*a_2

		equations = [eq_u, eq_m, eq_a_2, eq_f]

		return equations



	################################
	# integration interval
	################################

	right_boundary = math.atan(math.sqrt(2)) / math.sqrt(2)
	integration_interval = [0., right_boundary]

	################################
	# initial values
	################################

	initial_values = [1, 0, 0, 0] 

	################################
	# solution 
	################################
	
	sol = solve_ivp(phase_1_system_without_r_3, integration_interval, initial_values, dense_output=True) 

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
	plt.plot(s, values_u, color='silver')
	plt.plot(s, values_m, color='deepskyblue')
	plt.plot(s, values_a_2, color='lime')
	plt.plot(s, values_f, color='black')

	plt.xlabel('x')
	plt.legend(['u', 'm', 'a_2', 'f'], shadow=True) 
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
	values_r_3 = [x[4] for x in result_values] # with m if dist
	values_r_2 = [x[5] for x in result_values] # with m if dist
	values_r_1 = [x[6] for x in result_values] # with m if dist

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



# input: None -  to get analytical solution 
# input: some number - to get values of the function at that number
def solve_phase_1_ode_system(none_or_x):

	# functions

	x = symbols("x")
	u, m, a_2, f = symbols("u m a_2 f", cls=Function)


	# equations

	eq_u = Eq(u(x).diff(x), - 2 * u(x) - m(x))
	eq_m = Eq(m(x).diff(x), 2 * u(x) - 2 * m(x))
	eq_a_2 = Eq(a_2(x).diff(x), 3 * m(x) - 2 * a_2(x))
	eq_f = Eq(f(x).diff(x), 2 * a_2(x))

	equations = [eq_u, eq_m, eq_a_2, eq_f]


	# initial values

	x_in = 0

	u_in = 1
	m_in = 0
	a_2_in = 0
	f_in = 0

	initial_values = {u(x_in): u_in, m(x_in): m_in, a_2(x_in): a_2_in, f(x_in): f_in}


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


			f_at_time_s = values_at_time_s_precise[2]
			
			if f_at_time_s > target_value_for_f_precise:
				print("f_at_time_s = " + str(f_at_time_s))
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


#############################################
# to only get analytical solution of phase 1
#############################################
#solve_phase_1_ode_system(None)
#solve_phase_1_ode_system_with_r_3(None)


#############################################
# to only get values of phase 1 at a certain time
#############################################
#function_values_atan_plus_a_bit = solve_phase_1_ode_system(0.675511)
#print(function_values_atan_plus_a_bit)
"""
u, m, a_2, f, d, r_3m, r_3a = solve_phase_1_ode_system_with_r_3(math.atan(math.sqrt(2)) / math.sqrt(2))
a_1 = u+m
print("a_1 = " + str(a_1) + ", r_3m = " + str(r_3m) + "; fraction = " + str(r_3m / a_1))
print("a_2 = " + str(a_2) + ", r_3a = " + str(r_3a) + "; fraction = " + str(r_3a / a_2))


#############################################
# to get a plot for phase 1
#############################################
ode_system_index = 2
create_plot_for_phase_1(ode_system_index)
"""






plot_boolean = True



expected_time_phase_1 = math.atan(sqrt(2)) / sqrt(2)

if plot_boolean == True:
	create_plot_for_phase_1()

result_list_phase_1 = solve_phase_1_ode_system(expected_time_phase_1)


print("result_list_phase_1 = " + str(result_list_phase_1))
print()

u, m, a_2, f = result_list_phase_1

a_1 = u + m
d = f

initial_values_for_phase_2 = [a_1, a_2, f, d, 0, 0, 0]
target_value_for_f_precise = mp.mpf(1-(0.5 * 10**(-15)))

left_boundary = 2.2592
right_boundary = 2.2594 

"""
if we wanna hard-code initial values for phase 2 for whatever reason:

denominator = math.exp(math.sqrt(2) * math.atan(math.sqrt(2)))
a_1_in = math.sqrt(3) / denominator
a_2_in = (3 - math.sqrt(3)) / denominator
f_in = 1 - (3 / denominator)
d_in = 1 - (3 / denominator) 			
r_3_in = 0 					
r_2_in = 0
r_1_in = 0
"""

if plot_boolean == True:
	create_plot_for_phase_2(right_boundary, initial_values_for_phase_2)
result_time_phase_2, result_list_phase_2 = calculate_ode_system_for_phase_2(left_boundary, right_boundary, target_value_for_f_precise, initial_values_for_phase_2)

print("result_time_phase_2 = " + str(result_time_phase_2))
print("result_list_phase_2 = " + str(result_list_phase_2))
print()

alpha_precise = mp.mpf(10**(-15))
result_time_phase_3 = phase_3(alpha_precise)
print("result_time_phase_3 = " + str(result_time_phase_3))
print()

print("total number of rounds = " + str(0.675511 + result_time_phase_2 + result_time_phase_3))





