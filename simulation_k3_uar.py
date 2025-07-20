#!/usr/bin/python3

import math
import random



def offer_random_vertex(n):
        return random.randint(0,n-1)



def conduct_experiment(n, alpha):

        n_over_3 = n/3	

        round_counter = 0

        u_list = [i for i in range(n)]
        m_list = [] # contains pairs of vertices
        a_2_list = [] # contains triples of vertices

        number_of_triangles = 0

        size_X = 0

        # [status, neighbors, responsibilities]
        #vertex_status_list = [["u", None, None]] * n
        vertex_status_list = []
        for i in range(n):
                vertex_status_list.append(["u", None, None])


	#################################################################################
	#################################################################################
	# PHASE 1: build matching of size n/3
	#################################################################################
	#################################################################################

        while size_X < n_over_3:

                v_t = offer_random_vertex(n)
                v_t_status = vertex_status_list[v_t][0]


                #############################################################################
                # rule (I)
                #############################################################################

                if v_t_status == "u": 

                        w_t = random.choice(u_list)

                        if w_t == v_t:
                                pass

                        else:
                                vertex_status_list[v_t][0] = "m"
                                vertex_status_list[v_t][1] = sorted([v_t, w_t])

                                vertex_status_list[w_t][0] = "m"
                                vertex_status_list[w_t][1] = sorted([v_t, w_t])

                                m_list.append(sorted([v_t, w_t]))
                                size_X += 1

                                u_list.remove(v_t)
                                u_list.remove(w_t)



                #############################################################################
                # rule (II)
                #############################################################################

                elif v_t_status == "m":

                        matching_edge_sorted = vertex_status_list[v_t][1]
                        v_t_partner = [entry for entry in matching_edge_sorted if entry != v_t][0]

                        w_t = random.choice(u_list)

                        vertex_status_list[v_t][0] = "midpoint_a_2"
                        vertex_status_list[v_t][1] = sorted([v_t, v_t_partner, w_t])
                        vertex_status_list[v_t_partner][0] = "a_2"
                        vertex_status_list[v_t_partner][1] = sorted([v_t, v_t_partner, w_t])
                        vertex_status_list[w_t][0] = "a_2"
                        vertex_status_list[w_t][1] = sorted([v_t, v_t_partner, w_t])

                        a_2_list.append(sorted([v_t, v_t_partner, w_t]))

                        m_list.remove(matching_edge_sorted)
                        u_list.remove(w_t)


                #############################################################################
                # rule (III)
                #############################################################################

                elif v_t_status == "a_2":

                        a_2_structure_sorted = vertex_status_list[v_t][1]

                        vertex_a, vertex_b, vertex_c = a_2_structure_sorted

                        vertex_status_list[vertex_a][0] = "d"
                        vertex_status_list[vertex_b][0] = "d"
                        vertex_status_list[vertex_c][0] = "d"

                        a_2_list.remove(a_2_structure_sorted)

                        number_of_triangles += 1
		


                #############################################################################
                # rule (IV)
                #############################################################################

                else:
                        pass

                round_counter += 1


        rounds_after_phase_1 = round_counter
        result_after_phase_1 = [rounds_after_phase_1 / n, number_of_triangles/n_over_3, len(a_2_list)/n_over_3, len(m_list)/n_over_3]
        print("phase 1 (rounds, frac triangles, frac a2, frac a1) = " + str(result_after_phase_1))
	

        #################################################################################
        #################################################################################
        # PHASE 2: build triangles until |F| >= (1 - alpha) * n
        #################################################################################
        #################################################################################

        a_1_list = []

        # fix bijection between matching edges and untouched vertices 
        for i in range(len(u_list)):

                vertex = u_list[i]
                m_1, m_2 = m_list[i]

                a_1_list.append(sorted([vertex, m_1, m_2]))

                # [status, neighbors, responsibilities]
                vertex_status_list[vertex][0] = "a_1_u"
                vertex_status_list[vertex][1] = sorted([vertex, m_1, m_2])

                vertex_status_list[m_1][0] = "a_1_m"
                vertex_status_list[m_1][1] = sorted([vertex, m_1, m_2])

                vertex_status_list[m_2][0] = "a_1_m"
                vertex_status_list[m_2][1] = sorted([vertex, m_1, m_2])
		
        #######################################################################
        ############################
        # Evaluation
        ############################
        evaluation_list_aug_str_of_red = []
        evaluation_list_a_2_completion = []

        number_of_r_3 = 0
        number_of_r_2 = 0
        number_of_r_1 = 0

        #evaluation(vertex_status_list[v_t][2], "c", round_counter)
        #evaluation(vertex_status_list[s_1][2], "a", round_counter)
        def evaluation(resp_triangles, creation_type, round_counter):
                
                den = (len(a_1_list) + len(a_2_list)) * 3
                expected_triangles_per_aug_str = (number_of_r_3 + 1.5 * number_of_r_2 + 3 * number_of_r_1) / den
                expected_r_3_per_aug_str = (number_of_r_3 * 3)/den
                expected_r_2_per_aug_str = (number_of_r_2 * 3)/den
                expected_r_1_per_aug_str = (number_of_r_1 * 3)/den
                
                number_triangles_this_aug_str = 0
                number_r_3_this_aug_str = 0
                number_r_2_this_aug_str = 0
                number_r_1_this_aug_str = 0
                
                if resp_triangles != None:
                        number_triangles_this_aug_str = len(resp_triangles)
                        for resp_triangle in resp_triangles:
                                a, b, c = resp_triangle
                                if vertex_status_list[a][0].startswith("r_3"):
                                        number_r_3_this_aug_str += 3
                                elif vertex_status_list[a][0].startswith("r_2") or vertex_status_list[b][0].startswith("r_2"):
                                        number_r_2_this_aug_str += 2
                                else:
                                        # debugging
                                        if not(vertex_status_list[a][0].startswith("r_1") or vertex_status_list[b][0].startswith("r_1") or vertex_status_list[c][0].startswith("r_1")):
                                                print("mistake evaluation !!!!!")
                                        # debugging end
                                        number_r_1_this_aug_str += 1
                                        
                data = [round_counter, expected_triangles_per_aug_str, expected_r_3_per_aug_str, expected_r_2_per_aug_str, expected_r_1_per_aug_str, number_triangles_this_aug_str, number_r_3_this_aug_str, number_r_2_this_aug_str, number_r_1_this_aug_str]
                if creation_type == "c":
                        evaluation_list_a_2_completion.append(data)
                if creation_type == "a":
                        evaluation_list_aug_str_of_red.append(data)

        #######################################################################

        while number_of_triangles < (1 - alpha) * n_over_3:

                v_t = offer_random_vertex(n)
                v_t_status = vertex_status_list[v_t][0]

                #############################################################################
                # rule (I)a)
                #############################################################################

                if v_t_status == "a_1_u":

                        a_1_structure_sorted = vertex_status_list[v_t][1]
                        w_t, w_t_partner = [entry for entry in a_1_structure_sorted if entry != v_t]

                        vertex_status_list[w_t][0] = "midpoint_a_2"
                        vertex_status_list[w_t_partner][0] = "a_2"
                        vertex_status_list[v_t][0] = "a_2"

                        a_2_list.append(a_1_structure_sorted)

                        a_1_list.remove(a_1_structure_sorted)


                #############################################################################
                # rule(I)b)
                #############################################################################

                elif v_t_status == "a_1_m": 

                        a_1_structure_sorted = vertex_status_list[v_t][1]
                        vertex_1, vertex_2 = [entry for entry in a_1_structure_sorted if entry != v_t]

                        vertex_status_list[v_t][0] = "midpoint_a_2"
                        vertex_status_list[vertex_1][0] = "a_2"
                        vertex_status_list[vertex_2][0] = "a_2"

                        a_2_list.append(a_1_structure_sorted)

                        a_1_list.remove(a_1_structure_sorted)


                #############################################################################
                # rule (II)
                #############################################################################

                elif v_t_status == "a_2":

                        a_2_structure_sorted = vertex_status_list[v_t][1]

                        vertex_a, vertex_b, vertex_c = a_2_structure_sorted

                        evaluation(vertex_status_list[v_t][2], "c", round_counter)
			
                        # uncolor all responsible triangles and remove all responsibilities
                        if vertex_status_list[v_t][2] == None:
                                pass

                        else:
                                for responsible_triangle in vertex_status_list[v_t][2]:
                                        v_a, v_b, v_c = responsible_triangle

                                        if vertex_status_list[v_a][0].startswith("r_3"):
                                                number_of_r_3 -= 3
                                        elif vertex_status_list[v_a][0].startswith("r_2") or vertex_status_list[v_b][0].startswith("r_2"):
                                                number_of_r_2 -= 2
                                        else:
                                                number_of_r_1 -= 1
                                        
                                        vertex_status_list[v_a][0] = "d"
                                        vertex_status_list[v_a][2] = None
                                        vertex_status_list[v_b][0] = "d"
                                        vertex_status_list[v_b][2] = None
                                        vertex_status_list[v_c][0] = "d"
                                        vertex_status_list[v_c][2] = None

                        vertex_status_list[vertex_a][0] = "d"
                        vertex_status_list[vertex_b][0] = "d"
                        vertex_status_list[vertex_c][0] = "d"

                        vertex_status_list[vertex_a][2] = None
                        vertex_status_list[vertex_b][2] = None
                        vertex_status_list[vertex_c][2] = None

                        a_2_list.remove(a_2_structure_sorted)

                        number_of_triangles += 1


                #############################################################################
                # rule (III)
                #############################################################################

                elif v_t_status == "d":

                        triangle_sorted = vertex_status_list[v_t][1]
                        vertex_a, vertex_b = [entry for entry in triangle_sorted if entry != v_t]

                        # find the three vertices of the augmentable structure
                        aug_a = None 
                        aug_b = None
                        aug_c = None

                        # yields a number between 0 and (len(a_1_list) + len(a_2_list) -1); both included
                        augmentable_structure_index = random.randrange(0, len(a_1_list) + len(a_2_list))

                        if augmentable_structure_index <= len(a_1_list)-1:
                                chosen_a_1_str = a_1_list[augmentable_structure_index]
                                aug_a, aug_b, aug_c = chosen_a_1_str

                        else:
                                a_2_list_index = augmentable_structure_index - len(a_1_list)
                                chosen_a_2_str = a_2_list[a_2_list_index]
                                aug_a, aug_b, aug_c = chosen_a_2_str


                        # our triangle will now be responsible for the augmentable structure
                        vertex_status_list[v_t][0] = "r_3_L" # leading vertex
                        vertex_status_list[v_t][2] = [aug_a, aug_b, aug_c] # simple list
			
                        vertex_status_list[vertex_a][0] = "r_3"
                        vertex_status_list[vertex_a][2] = [aug_a, aug_b, aug_c] # simple list

                        vertex_status_list[vertex_b][0] = "r_3"
                        vertex_status_list[vertex_b][2] = [aug_a, aug_b, aug_c] # simple list

			
                        # at the augmentable structure, we list that the triangle is among the responsible triangles now
                        if vertex_status_list[aug_a][2] == None:
                                vertex_status_list[aug_a][2] = [triangle_sorted] # list of lists
                                vertex_status_list[aug_b][2] = [triangle_sorted] # list of lists
                                vertex_status_list[aug_c][2] = [triangle_sorted] # list of lists
                        else:
                                vertex_status_list[aug_a][2].append(triangle_sorted) # list of lists
                                vertex_status_list[aug_b][2].append(triangle_sorted) # list of lists
                                vertex_status_list[aug_c][2].append(triangle_sorted) # list of lists

                        number_of_r_3 += 3

                #############################################################################
                # rule (IV) 
                #############################################################################

                elif v_t_status == "r_3_L":

                        triangle_sorted = vertex_status_list[v_t][1]
                        vertex_a, vertex_b = [entry for entry in triangle_sorted if entry != v_t]

                        vertex_status_list[v_t][0] = "g_L"
                        vertex_status_list[vertex_a][0] = "r_2"
                        vertex_status_list[vertex_b][0] = "r_2"

                        number_of_r_3 -= 3
                        number_of_r_2 += 2
                        

                elif v_t_status == "r_3":

                        triangle_sorted = vertex_status_list[v_t][1]
                        vertex_a, vertex_b = [entry for entry in triangle_sorted if entry != v_t]

                        vertex_status_list[v_t][0] = "g"

                        if vertex_status_list[vertex_a][0] == "r_3":
                                vertex_status_list[vertex_a][0] = "r_2"
                                vertex_status_list[vertex_b][0] = "r_2_L"
                        else:
                                # debugging
                                if not (vertex_status_list[vertex_a][0] == "r_3_L" and vertex_status_list[vertex_b][0] == "r_3"):
                                        print("mistake rule iv")
                                # debugging end
                                vertex_status_list[vertex_a][0] = "r_2_L"
                                vertex_status_list[vertex_b][0] = "r_2"

                        number_of_r_3 -= 3
                        number_of_r_2 += 2
	
                #############################################################################
                # rule (V) 
                #############################################################################

                elif v_t_status == "r_2_L":

                        triangle_sorted = vertex_status_list[v_t][1]
                        vertex_a, vertex_b = [entry for entry in triangle_sorted if entry != v_t]

                        vertex_status_list[v_t][0] = "g_L"

                        if vertex_status_list[vertex_a][0] == "r_2":
                                vertex_status_list[vertex_a][0] = "r_1"
                        else:
                                # debugging
                                if vertex_status_list[vertex_b][0] != "r_2":
                                        print("mistake rule v")
                                # end debugging
                                vertex_status_list[vertex_b][0] = "r_1"

                        number_of_r_2 -= 2
                        number_of_r_1 += 1


                elif v_t_status == "r_2":

                        triangle_sorted = vertex_status_list[v_t][1]
                        vertex_a, vertex_b = [entry for entry in triangle_sorted if entry != v_t]

                        vertex_status_list[v_t][0] = "g"

                        if vertex_status_list[vertex_a][0] == "r_2":
                                vertex_status_list[vertex_a][0] = "r_1"
                        elif vertex_status_list[vertex_b][0] == "r_2":
                                # debugging
                                if vertex_status_list[vertex_a][0] != "g_L":
                                        print("mistake rule v part 2 1")
                                # end debugging
                                vertex_status_list[vertex_b][0] = "r_1"    
                        elif vertex_status_list[vertex_a][0] == "r_2_L":
                                vertex_status_list[vertex_a][0] = "r_1_L"
                        else: # if vertex_status_list[vertex_b][0] == "r_2_L":
                                # debugging
                                if vertex_status_list[vertex_b][0] != "r_2_L":
                                        print("mistake rule v part 2 1")
                                # end debugging
                                vertex_status_list[vertex_b][0] = "r_1_L"

                        number_of_r_2 -= 2
                        number_of_r_1 += 1
			
                #############################################################################
                # rule (VI) 
                #############################################################################

                # if A_1-str: leading vertex goes with the two "a_1_m"-vertices
                # if A_2-str: leading vertex goes with "midpoint_a_2" and arbitrary second "a_2"-vertex
                elif v_t_status == "r_1_L" or v_t_status == "r_1":

                        v_L = None # leading vertex of responsible triangle
                        v_1 = None
                        v_2 = None 

                        triangle_sorted = vertex_status_list[v_t][1]
                        vertex_a, vertex_b = [entry for entry in triangle_sorted if entry != v_t]
                        # if v_t_status == "r_1_L": vertex_a and vertex_b have status "g"

                        if v_t_status == "r_1_L":
                                v_L = v_t
                                v_1 = vertex_a
                                v_2 = vertex_b

                        else: # i.e. if v_t_status == "r_1"
                                if vertex_status_list[vertex_a][0] == "g_L":
                                        v_L = vertex_a
                                        v_1 = v_t
                                        v_2 = vertex_b
                                else: # i.e. if vertex_status_list[vertex_b][0] == "g_L"
                                        # debugging
                                        if vertex_status_list[vertex_b][0] != "g_L":
                                                print("mistake rule vi")
                                        # end debugging
                                        v_L = vertex_b
                                        v_1 = v_t
                                        v_2 = vertex_a
                                        

                        augmentable_structure = vertex_status_list[v_t][2] # simple list
                        
                        s_1, s_2, s_3 = augmentable_structure

                        evaluation(vertex_status_list[s_1][2], "a", round_counter)

                        aug_1_with_L = None 
                        aug_2_with_L = None
                        aug_without_L = None # does NOT go with leading vertex

                        if vertex_status_list[s_1][0] == "a_1_u":
                                aug_without_L = s_1
                                aug_1_with_L = s_2
                                aug_2_with_L = s_3
                                a_1_list.remove(augmentable_structure)

                        elif vertex_status_list[s_2][0] == "a_1_u":
                                aug_without_L = s_2
                                aug_1_with_L = s_1
                                aug_2_with_L = s_3
                                a_1_list.remove(augmentable_structure)

                        elif vertex_status_list[s_3][0] == "a_1_u":
                                aug_without_L = s_3
                                aug_1_with_L = s_1
                                aug_2_with_L = s_2
                                a_1_list.remove(augmentable_structure)
                                
                        elif vertex_status_list[s_1][0] == "midpoint_a_2":
                                aug_without_L = s_2
                                aug_1_with_L = s_1
                                aug_2_with_L = s_3
                                a_2_list.remove(augmentable_structure)

                        else: # i.e. A_2-structure where s_2 or s_3 is the midpoint
                                # debugging
                                if not (vertex_status_list[s_2][0] == "midpoint_a_2" or vertex_status_list[s_3][0] == "midpoint_a_2"):
                                        print("mistake rule vi b")
                                # end debugging
                                aug_without_L = s_1
                                aug_1_with_L = s_2
                                aug_2_with_L = s_3
                                a_2_list.remove(augmentable_structure)
                                
                        ##################################################################
                        # we arbitrarily exploit s_1 to make the updates in vertex_status_list for the other responsible triangles
                        ##################################################################

                        vertex_status_list[s_1][2].remove(triangle_sorted)
                        number_of_r_1 -= 1

                        for responsible_triangle in vertex_status_list[s_1][2]:
                                v_a, v_b, v_c = responsible_triangle

                                if vertex_status_list[v_a][0].startswith("r_3"):
                                        number_of_r_3 -= 3
                                elif vertex_status_list[v_a][0].startswith("r_2") or vertex_status_list[v_b][0].startswith("r_2"):
                                        number_of_r_2 -= 2
                                else:
                                        number_of_r_1 -= 1
                                      
                                vertex_status_list[v_a][0] = "d"
                                vertex_status_list[v_a][2] = None
                                vertex_status_list[v_b][0] = "d"
                                vertex_status_list[v_b][2] = None
                                vertex_status_list[v_c][0] = "d"
                                vertex_status_list[v_c][2] = None

                        ###################################################################

                        new_triangle_1_sorted = sorted([v_L, aug_1_with_L, aug_2_with_L])
                        new_triangle_2_sorted = sorted([aug_without_L, v_1, v_2])

                        # new triangle: v_L, aug_1_with_L, aug_2_with_L
                        vertex_status_list[v_L][0] = "d"
                        vertex_status_list[v_L][1] = new_triangle_1_sorted
                        vertex_status_list[v_L][2] = None

                        vertex_status_list[aug_1_with_L][0] = "d"
                        vertex_status_list[aug_1_with_L][1] = new_triangle_1_sorted
                        vertex_status_list[aug_1_with_L][2] = None
			
                        vertex_status_list[aug_2_with_L][0] = "d"
                        vertex_status_list[aug_2_with_L][1] = new_triangle_1_sorted
                        vertex_status_list[aug_2_with_L][2] = None

                        # new triangle: aug_without_L, v_1, vertex_b
                        vertex_status_list[aug_without_L][0] = "d"
                        vertex_status_list[aug_without_L][1] = new_triangle_2_sorted
                        vertex_status_list[aug_without_L][2] = None

                        vertex_status_list[v_1][0] = "d"
                        vertex_status_list[v_1][1] = new_triangle_2_sorted
                        vertex_status_list[v_1][2] = None

                        vertex_status_list[v_2][0] = "d"
                        vertex_status_list[v_2][1] = new_triangle_2_sorted
                        vertex_status_list[v_2][2] = None
			

                        number_of_triangles += 1

			

                #############################################################################
                # --- 4.2. --- otherwise
                #############################################################################

                else:
                        pass


                round_counter += 1


        rounds_after_phase_2 = round_counter - rounds_after_phase_1
        result_after_phase_2 = [round_counter/n, rounds_after_phase_2 / n, number_of_triangles/n_over_3, len(a_2_list)/n_over_3, len(a_1_list)/n_over_3]
        print("phase 2 (rounds total, rounds phase 2, frac triangles, frac a2, frac a1) = " + str(result_after_phase_2))
			
        return round_counter, evaluation_list_aug_str_of_red, evaluation_list_a_2_completion


		


n = 300000
alpha = 10**(-5)
print("n = " + str(n))
print("alpha = " + str(alpha))
print()

round_counter, evaluation_list_aug_str_of_red, evaluation_list_a_2_completion = conduct_experiment(n, alpha)
print("rounds = " + str(round_counter / n))
print()
print()

print("[round_counter, triangles_exp, r_3_exp, r_2_exp, r_1_exp, triangles_curr, r_3_curr, r_2_curr, r_1_curr]")
print()

print("expected vs actual: completion")
sum_triangles_actual = sum([data[5] for data in evaluation_list_a_2_completion])
sum_triangles_expected = sum([data[1] for data in evaluation_list_a_2_completion])
print(sum_triangles_actual / sum_triangles_expected)
print()
sum_r3_actual = sum([data[6] for data in evaluation_list_a_2_completion])
sum_r3_expected = sum([data[2] for data in evaluation_list_a_2_completion])
print(sum_r3_actual / sum_r3_expected)
print()
sum_r2_actual = sum([data[7] for data in evaluation_list_a_2_completion])
sum_r2_expected = sum([data[3] for data in evaluation_list_a_2_completion])
print(sum_r2_actual / sum_r2_expected)
print()
sum_r1_actual = sum([data[8] for data in evaluation_list_a_2_completion])
sum_r1_expected = sum([data[4] for data in evaluation_list_a_2_completion])
print(sum_r1_actual / sum_r1_expected)
print()
#print("first = " + str(evaluation_list_a_2_completion[0]))
#print("last = " + str(evaluation_list_a_2_completion[-1]))
print()



print("expected vs actual: augmentation")
sum_triangles_actual_2 = sum([data[5] for data in evaluation_list_aug_str_of_red])
#sum_triangles_expected_2 = sum([data[1] for data in evaluation_list_aug_str_of_red])
sum_triangles_expected_2_plus_one = sum([data[1]+1 for data in evaluation_list_aug_str_of_red])
print(sum_triangles_actual_2 / sum_triangles_expected_2_plus_one)
print()
sum_r3_actual_2 = sum([data[6] for data in evaluation_list_aug_str_of_red])
sum_r3_expected_2 = sum([data[2] for data in evaluation_list_aug_str_of_red])
print(sum_r3_actual_2 / sum_r3_expected_2)
print()
sum_r2_actual_2 = sum([data[7] for data in evaluation_list_aug_str_of_red])
sum_r2_expected_2 = sum([data[3] for data in evaluation_list_aug_str_of_red])
print(sum_r2_actual_2 / sum_r2_expected_2)
print()
sum_r1_actual_2 = sum([data[8] for data in evaluation_list_aug_str_of_red])
#sum_r1_expected_2 = sum([data[4] for data in evaluation_list_aug_str_of_red])
sum_r1_expected_2_plus_one = sum([data[4]+1 for data in evaluation_list_aug_str_of_red])
print(sum_r1_actual_2 / sum_r1_expected_2_plus_one)
print()
