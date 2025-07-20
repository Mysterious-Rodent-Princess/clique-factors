#!/usr/bin/python3

import math
import random



def offer_random_vertex(n):
    return random.randint(0,n-1)



def pm_experiment(n, alpha):
    round_counter = 0

    unsat_vertex_list = [i for i in range(n)]
    number_vertices_in_matching = 0
    number_red_vertices = 0


    # [status, structure-neighbors-list, responsibilities]
    #vertex_status_list = [["u", None, None]] * n
    vertex_status_list = []
    for i in range(n):
        vertex_status_list.append(["u", None, None])
		
    ############################
    # Evaluation
    ############################
    evaluation_list_unsat_of_red = []
    evaluation_list_random_unsat = []


    
    #############################################################################
    # creating new matching edge 
    #############################################################################

    def resp_release(unsat_vertex):
        if vertex_status_list[unsat_vertex][2] == None:
            pass

        else:               
            for resp_matching_edge in vertex_status_list[unsat_vertex][2]:
                                
                vertex_a, vertex_b = resp_matching_edge
                                
                vertex_status_list[vertex_a][0] = "m"
                vertex_status_list[vertex_a][2] = None
                                
                vertex_status_list[vertex_b][0] = "m"
                vertex_status_list[vertex_b][2] = None
                                


    def matching_edge_creation(unsat_vertex_1, unsat_vertex_2):
        resp_release(unsat_vertex_1)
        resp_release(unsat_vertex_2)
                    
        matching_edge_sorted = sorted([unsat_vertex_1, unsat_vertex_2])

        vertex_status_list[unsat_vertex_1][0] = "m"
        vertex_status_list[unsat_vertex_1][1] = matching_edge_sorted
        vertex_status_list[unsat_vertex_1][2] = None

        vertex_status_list[unsat_vertex_2][0] = "m"
        vertex_status_list[unsat_vertex_2][1] = matching_edge_sorted
        vertex_status_list[unsat_vertex_2][2] = None

        unsat_vertex_list.remove(unsat_vertex_1)
        unsat_vertex_list.remove(unsat_vertex_2)


    while number_vertices_in_matching < (1 - alpha) * n:
                
        v_t = offer_random_vertex(n)
        v_t_status = vertex_status_list[v_t][0]

        #############################################################################
        # rule (I)
        #############################################################################

        if v_t_status == "u": 

            w_t = random.choice(unsat_vertex_list)

            if w_t == v_t:
                pass
            
            else:
                ###################
                # EVALUATION!!!!!
                ###################
                if number_red_vertices > 0:
                    expected_red_per_unsat = number_red_vertices / len(unsat_vertex_list)
                    number_red_v_t = 0 if vertex_status_list[v_t][2] == None else len(vertex_status_list[v_t][2])
                    number_red_w_t = 0 if vertex_status_list[w_t][2] == None else len(vertex_status_list[w_t][2])
                
                    evaluation_list_random_unsat.append([round_counter, expected_red_per_unsat, number_red_v_t])
                    evaluation_list_random_unsat.append([round_counter, expected_red_per_unsat, number_red_w_t])
                ###################
                
                number_vertices_in_matching += 2
                number_red_vertices -= 0 if vertex_status_list[v_t][2] == None else len(vertex_status_list[v_t][2])
                number_red_vertices -= 0 if vertex_status_list[w_t][2] == None else len(vertex_status_list[w_t][2])
                
                matching_edge_creation(v_t, w_t)
    
        #############################################################################
        # rule (II) 
        #############################################################################

        elif v_t_status == "r":
                        
            matching_edge_sorted = vertex_status_list[v_t][1]
            v_t_partner = [entry for entry in matching_edge_sorted if entry != v_t][0]

            our_unsat_vertex = vertex_status_list[v_t][2]

            w_t = random.choice(unsat_vertex_list)

            if w_t == our_unsat_vertex:
                pass

            else:
                ###################
                # EVALUATION!!!!!
                ###################
                expected_red_per_unsat = number_red_vertices / len(unsat_vertex_list)
                number_red_our_unsat = len(vertex_status_list[our_unsat_vertex][2])
                number_red_w_t = 0 if vertex_status_list[w_t][2] == None else len(vertex_status_list[w_t][2])
                
                evaluation_list_unsat_of_red.append([round_counter, expected_red_per_unsat, number_red_our_unsat])
                evaluation_list_random_unsat.append([round_counter, expected_red_per_unsat, number_red_w_t])
                ###################
                                
                number_vertices_in_matching += 2
                number_red_vertices -= len(vertex_status_list[our_unsat_vertex][2])
                number_red_vertices -= 0 if vertex_status_list[w_t][2] == None else len(vertex_status_list[w_t][2])
                # I think this is wrong... there should be augmentation...
                
                resp_release(our_unsat_vertex)
                resp_release(w_t)
                    
                matching_edge_1_sorted = sorted([v_t, w_t])
                matching_edge_2_sorted = sorted([v_t_partner, our_unsat_vertex])

                vertex_status_list[v_t][0] = "m"
                vertex_status_list[v_t][1] = matching_edge_1_sorted
                vertex_status_list[v_t][2] = None

                vertex_status_list[w_t][0] = "m"
                vertex_status_list[w_t][1] = matching_edge_1_sorted
                vertex_status_list[w_t][2] = None

                vertex_status_list[v_t_partner][0] = "m"
                vertex_status_list[v_t_partner][1] = matching_edge_2_sorted
                vertex_status_list[v_t_partner][2] = None

                vertex_status_list[our_unsat_vertex][0] = "m"
                vertex_status_list[our_unsat_vertex][1] = matching_edge_2_sorted
                vertex_status_list[our_unsat_vertex][2] = None

                unsat_vertex_list.remove(w_t)
                unsat_vertex_list.remove(our_unsat_vertex)
				

        #############################################################################
        # rule (III)
        #############################################################################

        elif v_t_status == "m":

            matching_edge_sorted = vertex_status_list[v_t][1]
            v_t_partner = [entry for entry in matching_edge_sorted if entry != v_t][0]

            w_t = random.choice(unsat_vertex_list)

            vertex_status_list[v_t][0] = "g"
            vertex_status_list[v_t][2] = w_t
                        
            vertex_status_list[v_t_partner][0] = "r"
            vertex_status_list[v_t_partner][2] = w_t
                        

            if vertex_status_list[w_t][2] == None:
                vertex_status_list[w_t][2] = [matching_edge_sorted]
            else:
                vertex_status_list[w_t][2].append(matching_edge_sorted)

                        
            number_red_vertices += 1                
                        
        #############################################################################
        # rule (IV)
        #############################################################################

        else:
            pass 
				
        round_counter += 1
        
    return round_counter, evaluation_list_unsat_of_red, evaluation_list_random_unsat



n = 30000
alpha = 10**(-4)
print("n = " + str(n))
print("alpha = " + str(alpha))
print()

round_counter, evaluation_list_unsat_of_red, evaluation_list_random_unsat = pm_experiment(n, alpha)
print("rounds = " + str(round_counter / n))
print()


print("expected plus 1 vs actual: unsat of red")
sum_actual = sum([data[2] for data in evaluation_list_unsat_of_red])
sum_expected = sum([data[1] for data in evaluation_list_unsat_of_red])
sum_expected_plus_one = sum([data[1]+1 for data in evaluation_list_unsat_of_red])

#print(sum_actual)
#print(sum_expected)
#print(sum_actual / sum_expected)
print(sum_actual / sum_expected_plus_one)
print()

print("first = " + str(evaluation_list_unsat_of_red[0]))
print("last = " + str(evaluation_list_unsat_of_red[-1]))
print()



print("expected vs actual: random unsat")
sum_actual_2 = sum([data[2] for data in evaluation_list_random_unsat])
sum_expected_2 = sum([data[1] for data in evaluation_list_random_unsat])
#print(sum_actual_2)
#print(sum_expected_2)
print(sum_actual_2 / sum_expected_2)
print()

print("first = " + str(evaluation_list_random_unsat[0]))
print("last = " + str(evaluation_list_random_unsat[-1]))










