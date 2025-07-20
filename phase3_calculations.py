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



def general_clique_factor_alpha_and_c_requirements(k, alpha, c):

	q_constant = 0.9999**(2*k-2) * k * c**(2*k-2) / ((2*k-2) * math.factorial(k-2) * math.exp(c*0.1*k)) 

	rhs_1 = 1 / (10**(2 * (2*k-2)) + 2)
	ineq_1 = (alpha <= rhs_1)

	lhs_2 = q_constant * math.sqrt((1/alpha)-2) 
	rhs_2 = 120**2 * (2*k-2)
	ineq_2 = (lhs_2 > rhs_2)

	lhs_3 = -math.log(alpha)
	rhs_3 = (q_constant / (128 * (2*k-2) * 2)) * alpha**(-0.5) * math.log(alpha**(-0.5))
	ineq_3 = (lhs_3 <= rhs_3)

	lhs_4 = math.log(k+1) - 2 * math.log(alpha)
	rhs_4 = (q_constant / (128 * (2*k-2) * 2)) * alpha**(-0.5)
	ineq_4 = (lhs_4 <= rhs_4)

	return ineq_1 and ineq_2 and ineq_3 and ineq_4



def general_clique_factor_rounds_calculation(k, alpha, c):
	exponent = 1 / (2 * (2*k-2))
	factor_1 = (2/(1-alpha))**exponent
	factor_2 = alpha**exponent / (1 - alpha**exponent)
	return c * factor_1 * factor_2


### check for k=4,5,6
for k in range(50,51):
	i = 24
	while True:
		if general_clique_factor_alpha_and_c_requirements(k, 10**(-i), 1):
			print("k = " + str(k))
			print("i = " + str(i))
			print(general_clique_factor_rounds_calculation(k, 10**(-i), 1))
			break
		i+=1



"""
k = 4
i = 13
0.09531594523401048

k = 4
i = 14
0.0774574982922152

k = 4
i = 15
0.06312793257320153

k = 4
i = 16
0.05156956799356095


k = 4
i = 24
0.010701647417770662

k = 4
i = 25
0.008817633840976712
##########################################

k = 5
i = 17
0.09900375599748812

True
k = 5
i = 18
0.08465793582491617

k = 5
i = 19
0.07252271303628088
######################################

k = 6
i = 20
0.11502943598237528

k = 6
i = 40
0.010457221452943205

k = 6
i = 41
0.00930978186410795
#########################################

k = 7
i = 24
0.11436691518261023

k = 7
i = 48
0.010396992289328206

k = 7
i = 49
0.009437129708521005

"""





"""
def k_3_factor_induction_lemma_check(l):
	# slope:
	lhs = l * math.log(10) * 10**5
	rhs = 10**(l/2) * math.log(10**(l/2))
	# p = 1
	lhs_2 = l * math.log(10) * 10**5 * 2 + math.log(4) * 10**5
	rhs_2 = 10**(l/2) 
	print(l)
	print(lhs <= rhs)
	print(lhs_2 <= rhs_2)
	print()

for l in range(9, 20):
	k_3_factor_induction_lemma_check(l)



def aaa(alpha):
	print(alpha)
	constant = 0.9999**6  / (3 * math.exp(0.4) * 192 * 2 * 4)

	lhs_slope = - math.log(alpha)
	rhs_slope = constant* alpha**(-0.5) * (-math.log(alpha) *0.5)
	print(lhs_slope <= rhs_slope)

	lhs_f_1 = math.log(5) - math.log(alpha) * 2
	rhs_f_1 =  constant * alpha**(-0.5)
	print(lhs_f_1 <= rhs_f_1)


for i in range(7, 17):
	aaa(10**(-i))

"""
