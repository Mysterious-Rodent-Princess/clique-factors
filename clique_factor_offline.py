#!/usr/bin/python3
from scipy.optimize import minimize
import numpy as np
from functools import partial

# the next two lines are necessary to fix a weird error...
import math
np.math = math



def increase_x_entries_by_1(x, input_list):

	if x > len(input_list):
		print("x > len(input_list)")
		return None

	if x == len(input_list):
		return [[a + 1 for a in input_list]]

	if x == 0:
		return [input_list]

	else:
		start_element = input_list[0]
		list_ass = input_list[1:]

		part_1 = [[start_element] + nicelist for nicelist in increase_x_entries_by_1(x, list_ass)]
		part_2 = [[start_element+1] + nicelist for nicelist in increase_x_entries_by_1(x-1, list_ass)]

		result_list = part_1 + part_2

		return result_list



def get_all_out_degree_sequences(k):

	if k == 2:
		return [[1,0]]

	else:
		sequences_for_k_minus_1 = get_all_out_degree_sequences(k-1)
		
		result_list = []

		for deg in range(k):
			for old_sequence in sequences_for_k_minus_1:

				increasable_entries = (k-1)-deg
				partial_result_list = [sorted([deg] + nicelist) for nicelist in increase_x_entries_by_1(increasable_entries, old_sequence)]

				relevant_partial_result_list = [nicelist for nicelist in partial_result_list if nicelist not in result_list]
				result_list = result_list + relevant_partial_result_list

		return result_list



def number_entries_at_least_i(i, input_list):
	if len(input_list) == 0:
		return 0

	else:
		start_element = input_list[0]
		list_ass = input_list[1:]

		if start_element >= i:
			return 1 + number_entries_at_least_i(i, list_ass)

		else:
			return 0 + number_entries_at_least_i(i, list_ass)




def optimize_x_for_K_k(k):

	out_deg_patterns = get_all_out_degree_sequences(k)
	p = len(out_deg_patterns)

	print("k = " + str(k))
	print("out degree patterns: " + str(out_deg_patterns))

	trivial_lower_bound_for_x = (k-1)/2
	sum_of_c_s = 1/k


	#############################################
	# Objective function to minimize x
	#############################################
	def objective_function(variables):
		x = variables[0]
		return x


	#############################################
	# Initial guess for x and c
	# e.g. for k = 4: [1.5, 1/16, 1/16, 1/16, 1/16]
	#############################################
	initial_guess = [trivial_lower_bound_for_x] + [sum_of_c_s / p] * p


	#############################################
	# Bounds for x and c
	# e.g. for k = 4: [(1.5, None), (0, None), (0, None), (0, None), (0, None)]
	#############################################
	bounds = [(trivial_lower_bound_for_x, None)] + [(0, None)] * p


	# Constraint: c_0 + c_1 + c_2 + ... +  c_p = 1/k
	def c_sum(variables):
		return sum(variables[1:]) - sum_of_c_s

	# Inequality l: 1 - (1 + x + x^2/2 + ... + x^(l-1)/(l-1)!) / e^x >= c_0 * (# entries >= l in pattern 0) + ... + c_(p-1) * (# entries >= l in pattern p-1)
	def inequality(l, variables):
		x = variables[0]
		c_vector = variables[1:]

		lhs_summands = 0
		for j in range(l):
			lhs_summands += (x**j)/np.math.factorial(j)
		lhs =  1 - (np.exp(-x) * lhs_summands)

		number_large_degree_vertices = [number_entries_at_least_i(l, pattern) for pattern in out_deg_patterns]
		rhs_summands = 0
		for i in range(p):
			rhs_summands += c_vector[i] * number_large_degree_vertices[i]
		rhs = rhs_summands

		return lhs - rhs

	#############################################
	# Constraints for x and c (equality for the sum of c, inequalities for the other conditions)
	#############################################
	constraints = [{'type': 'eq', 'fun': c_sum}] + [{'type': 'ineq', 'fun': partial(inequality, l)} for l in range(1,k)]


	#############################################
	# Perform the minimization
	#############################################
	result = minimize(objective_function, initial_guess, bounds=bounds, constraints=constraints)

	variable_values = result.x 
	x = variable_values[0]
	c_vector = variable_values[1:]

	# Extract the result
	print("x = " + str(x))
	print("c = " + str(c_vector))
	print()

	return x



def optimize_c_sparsity_for_K_k_and_x(k, x_k):

	out_deg_patterns = get_all_out_degree_sequences(k)
	p = len(out_deg_patterns)
	sum_of_c_s = 1/k

	print("k = " + str(k) + ", plugging in x = " + str(x_k))
	print("sum of c's: " + str(sum_of_c_s))


	#############################################
	# Objective function to minimize the number of out-degree patterns actually used
	# the first one (all out degrees different) is favored
	#############################################

	def objective_function(variables):
		weighted_c_sum = variables[1]
		for c in variables[2:]:
			weighted_c_sum += 100 * c # other out degree patterns get weight 5
		return weighted_c_sum
	

	#############################################
	# Initial guess for x and c
	# e.g. for k = 4: [x_4, 1/4, 0, 0, 0]
	#############################################
	initial_guess = [x_k, sum_of_c_s] + [0] * (p-1)


	
	# Constraint: x = x_k
	def x_equals_x_k(variables):
		return variables[0] - x_k

	# Constraint: c_0 + c_1 + c_2 + ... +  c_p = 1/k
	def c_sum(variables):
		return sum(variables[1:]) - sum_of_c_s

	# Inequality l: 1 - (1 + x + x^2/2 + ... + x^(l-1)/(l-1)!) / e^x >= c_0 * (# entries >= l in pattern 0) + ... + c_(p-1) * (# entries >= l in pattern p-1)
	def inequality(l, variables):
		x = variables[0]
		c_vector = variables[1:]

		lhs_summands = 0
		for j in range(l):
			lhs_summands += (x**j)/np.math.factorial(j)
		lhs =  1 - (np.exp(-x) * lhs_summands)

		number_large_degree_vertices = [number_entries_at_least_i(l, pattern) for pattern in out_deg_patterns]
		rhs_summands = 0
		for i in range(p):
			rhs_summands += c_vector[i] * number_large_degree_vertices[i]
		rhs = rhs_summands

		return lhs - rhs

	#############################################
	# Constraints for c (equality for the sum of c, inequalities for the other conditions)
	#############################################
	constraints = [{'type': 'eq', 'fun': x_equals_x_k}, {'type': 'eq', 'fun': c_sum}] + [{'type': 'ineq', 'fun': partial(inequality, l)} for l in range(1,k)]



	for n in range(1, p+1):

		existence_of_c_vector_with_n_nonzero_entries = False

		zero_one_arrays_with_n_ones = increase_x_entries_by_1(n, [0]*p)

		for zero_one_array in zero_one_arrays_with_n_ones:
			#############################################
			# Bounds for c
			# e.g. for k = 4 and zero_one_array = [1, 0, 0, 1]
			# [(x_k, None), (0, 1), (0, 0), (0, 0), (0, 1)]
			#############################################
			bounds = [(x_k, None)] + [(0, zero_or_one) for zero_or_one in zero_one_array]


			#############################################
			# Perform the minimization
			#############################################
			result = minimize(objective_function, initial_guess, bounds=bounds, constraints=constraints)

			if result.success == True:
				existence_of_c_vector_with_n_nonzero_entries = True

				variable_values = result.x 
				x = variable_values[0]
				c_vector = variable_values[1:]

				# Extract the result
				print("c = " + str(c_vector))

		if existence_of_c_vector_with_n_nonzero_entries == True:
			print()
			break




"""
print(len(get_all_out_degree_sequences(5)))
print(len(get_all_out_degree_sequences(6)))
print(len(get_all_out_degree_sequences(7)))
print(len(get_all_out_degree_sequences(8)))
print(len(get_all_out_degree_sequences(9)))
print(len(get_all_out_degree_sequences(10)))
"""

x_3 = optimize_x_for_K_k(3)
optimize_c_sparsity_for_K_k_and_x(3, x_3)

x_4 = optimize_x_for_K_k(4)
optimize_c_sparsity_for_K_k_and_x(4, x_4)


x_5 = optimize_x_for_K_k(5)
optimize_c_sparsity_for_K_k_and_x(5, x_5)


x_6 = optimize_x_for_K_k(6)




"""
optimize_x_for_K_k(7)

optimize_x_for_K_k(8)

optimize_x_for_K_k(9)

optimize_x_for_K_k(10)

"""




