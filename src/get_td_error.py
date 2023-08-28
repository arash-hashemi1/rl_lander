import numpy as np
from softmax import softmax

def get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau):
    """
    Args:
        states (Numpy array): The batch of states with the shape (batch_size, state_dim).
        next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
        actions (Numpy array): The batch of actions with the shape (batch_size,).
        rewards (Numpy array): The batch of rewards with the shape (batch_size,).
        discount (float): The discount factor.
        terminals (Numpy array): The batch of terminals with the shape (batch_size,).
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets,
                                        and particularly, the action-values at the next-states.
    Returns:
        The TD errors (Numpy array) for actions taken, of shape (batch_size,)
    """

    # Note: Here network is the latest state of the network that is getting replay updates. In other words,
    # the network represents Q_{t+1}^{i} whereas current_q represents Q_t, the fixed network used for computing the
    # targets, and particularly, the action-values at the next-states.
    # Compute action values at next states using current_q network
    # Note that q_next_mat is a 2D array of shape (batch_size, num_actions)

    q_next_mat = current_q.get_action_values(next_states)

    # Compute policy at next state by passing the action-values in q_next_mat to softmax()
    # Note that probs_mat is a 2D array of shape (batch_size, num_actions)

    probs_mat = softmax(q_next_mat, tau)

    # Compute the estimate of the next state value, v_next_vec.
    # Hint: sum the action-values for the next_states weighted by the policy, probs_mat. Then, multiply by
    # (1 - terminals) to make sure v_next_vec is zero for terminal next states.
    # Note that v_next_vec is a 1D array of shape (batch_size,)

    v_next_vec = np.sum(probs_mat * q_next_mat, axis=1)
    v_next_vec *= (1 - terminals)


    # Compute Expected Sarsa target
    # Note that target_vec is a 1D array of shape (batch_size,)

    target_vec = rewards + discount * v_next_vec


    # Compute action values at the current states for all actions using network
    # Note that q_mat is a 2D array of shape (batch_size, num_actions)

    q_mat = network.get_action_values(states)


    # Batch Indices is an array from 0 to the batch size - 1.
    batch_indices = np.arange(q_mat.shape[0])

    # Compute q_vec by selecting q(s, a) from q_mat for taken actions
    # Use batch_indices as the index for the first dimension of q_mat
    # Note that q_vec is a 1D array of shape (batch_size)


    q_vec = np.array([q_mat[i, actions[i]] for i in range(len(actions))])


    # Compute TD errors for actions taken
    # Note that delta_vec is a 1D array of shape (batch_size)

    delta_vec = target_vec - q_vec

    return delta_vec