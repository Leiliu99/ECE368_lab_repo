import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    # forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps


    # TODO: Compute the forward messages
    # Initialize forward message: the first node
    forward_messages[0] = rover.Distribution()
    for states in prior_distribution:  # all states in alpha 0
        possible_locations_for_state = observation_model(states)
        for observe_location in possible_locations_for_state:  # all possible locations given this state
            if (observations[0] == None):  # if this location is missing in txt
                forward_messages[0][states] = prior_distribution[states] * 1
            elif (observations[0] == observe_location):  # if the location is this possible location
                forward_messages[0][states] = prior_distribution[states] * possible_locations_for_state[
                    observe_location]
    forward_messages[0].renormalize()

    for i in range(1, num_time_steps):
        forward_messages[i] = rover.Distribution()
        for states in all_possible_hidden_states: # all possible Zn
            inner_segma = 0
            for prev_state in forward_messages[i-1]: # sum all possible Zn-1
                # alpha(Zn-1) * p(Zn|Zn-1)
                inner_segma += forward_messages[i-1][prev_state] * transition_model(prev_state)[states]

            # compute observe_given_state = P((xn, yn)|Zn)
            possible_locations_for_state = observation_model(states)
            if (observations[i] == None):  # if this location is missing in txt
                observe_given_state = 1
            else:
                if observations[i] not in possible_locations_for_state:
                    continue
                observe_given_state = possible_locations_for_state[observations[i]]
            result = inner_segma * observe_given_state
            if(result != 0): # avoid normalization division by 0
                forward_messages[i][states] = result
        forward_messages[i].renormalize()

    # TODO: Compute the backward messages
    # Initialize backward message: the last node
    backward_messages[num_time_steps-1] = rover.Distribution()
    for state in all_possible_hidden_states:
        backward_messages[num_time_steps-1][state] = 1
    backward_messages[num_time_steps-1].renormalize()

    for i in range(num_time_steps-2, -1, -1):
        backward_messages[i] = rover.Distribution()
        for states in all_possible_hidden_states:  # all possible state: Zn-1
            result = 0
            for next_state in backward_messages[i + 1]:  # all possible states: Zn
                # beta(Zn) * p(Zn|Zn-1)
                inner_product = backward_messages[i+1][next_state] * transition_model(states)[next_state]

                # compute observe_given_state = P((xn, yn)|Zn)
                possible_locations_for_state = observation_model(next_state)
                if (observations[i + 1] == None):  # if this location is missing in txt
                    observe_given_state = 1
                else:
                    if observations[i + 1] not in possible_locations_for_state:
                        continue
                    observe_given_state = possible_locations_for_state[observations[i + 1]]

                result += inner_product * observe_given_state
            if(result != 0): # avoid normalization division by 0
                backward_messages[i][states] = result
        backward_messages[i].renormalize()

    # TODO: Compute the marginals
    for i in range(0, num_time_steps):
        marginals[i] = rover.Distribution()
        for state in forward_messages[i]:
            if state in backward_messages[i].keys():
                marginals[i][state] = forward_messages[i][state] * backward_messages[i][state]

        marginals[i].renormalize()

    return marginals


def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps
    w = [None] * num_time_steps
    track = [None] * num_time_steps# a dictionary key is the next state, value is current state

    # initialization
    w[0] = rover.Distribution()
    for states in prior_distribution:  # all states in alpha 0
        possible_locations_for_state = observation_model(states)
        for observe_location in possible_locations_for_state:  # all possible locations given this state
            if (observations[0] == None):  # if this location is missing in txt
                w[0][states] = np.log(prior_distribution[states])
            elif (observations[0] == observe_location):  # if the location is this possible location
                w[0][states] = np.log(prior_distribution[states]) + np.log(possible_locations_for_state[
                    observe_location])
    w[0].renormalize()

    for j in range(0, num_time_steps):
        track[j] = rover.Distribution()

    # recursion
    for i in range(1, num_time_steps):
        w[i] = rover.Distribution()
        for states in all_possible_hidden_states:# all possible Zn
            max_term = -np.Infinity
            for prev_state in w[i-1]:# sum all possible Zn-1
                if(states in transition_model(prev_state)):
                    if(w[i-1][prev_state] != 0):
                        # update the max term here
                        if((w[i-1][prev_state] + np.log(transition_model(prev_state)[states])) > max_term):
                            max_term = w[i-1][prev_state] + np.log(transition_model(prev_state)[states])
                            track[i-1][states] = prev_state

            # compute observe_given_state = P((xn, yn)|Zn)
            possible_locations_for_state = observation_model(states)
            if (observations[i] == None):  # if this location is missing in txt
                observe_given_state = 1
            else:
                if observations[i] not in possible_locations_for_state:
                    continue
                observe_given_state = possible_locations_for_state[observations[i]]

            if(max_term != -np.Infinity):
                w[i][states] = max_term + np.log(observe_given_state)

    # find the end point which is the start point to back track
    maximum = -np.Infinity
    start_state = None
    for last_state in w[num_time_steps - 1]:
        if(w[num_time_steps - 1][last_state] > maximum):
            maximum = w[num_time_steps - 1][last_state]
            start_state = last_state

    # backtrack to get the path
    for i in range(num_time_steps - 1, -1, -1):
        estimated_hidden_states[i] = rover.Distribution()
        if(i == num_time_steps - 1):# start point
            estimated_hidden_states[i] = start_state
        else:
            estimated_hidden_states[i] = track[i][estimated_hidden_states[i+1]]

    return estimated_hidden_states

def error_probability(true_hidden_states, max_marginals, estimated_hidden_states, observations):
    num_time_steps = len(observations)

    sum_marginal = 0
    sum_estimate = 0
    for k in range(0, num_time_steps):
        if(max_marginals[k] == true_hidden_states[k]):
            sum_marginal += 1
        if(estimated_hidden_states[k] == true_hidden_states[k]):
            sum_estimate += 1

    err_prob_marginal = 1 - (sum_marginal/100)
    err_prob_estimate = 1 - (sum_estimate/100)

    return err_prob_marginal, err_prob_estimate

def detect_invalid_sequence(max_marginals, transition_model):
    has_invalid_sequence = False
    for i in range(0, 100):
        valid_step = False
        possible_transition_state = transition_model(max_marginals[i])
        for state in possible_transition_state:
            if(max_marginals[i + 1] == state): # in this time step, it is valid
                valid_step = True
                break
        if(valid_step == False):
            print("At time ", i, " = ", max_marginals[i])
            print("However, at time ", i + 1, " = ", max_marginals[i + 1])
            has_invalid_sequence = True
            break

    if(has_invalid_sequence == False):
        print("No valid sequence detected")
    return

if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')
    timestep = num_time_steps - 1
    timestep = 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    timestep = 99
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    #Question 4
    max_marginals = [None] * num_time_steps

    for i in range(0, num_time_steps):
        maximum = -np.Infinity
        for state in marginals[i]:
            if (marginals[i][state] > maximum):
                maximum = marginals[i][state]
                max_marginals[i] = state

    err_marginal, err_estimate = error_probability(hidden_states, max_marginals, estimated_states, observations)
    print("The error probability for estimate (Viterbi algo): ", err_estimate)
    print("The error probability for marginal (forward-backward algo): ", err_marginal)

    #Question 5
    detect_invalid_sequence(max_marginals, rover.transition_model)
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps

    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
