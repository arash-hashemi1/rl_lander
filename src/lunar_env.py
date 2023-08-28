import numpy as np
import environment
from utils import get_landing_zone, get_angle, get_velocity, get_position, get_fuel, tests
get_landing_zone()
# Lunar Lander Environment
class LunarLanderEnvironment(environment.BaseEnvironment):
    def __init__(self):
        self.current_state = None
        self.count = 0
    def env_init(self, env_info):
        # users set this up
        self.state = np.zeros(6) # velocity x, y, angle, distance to ground, landing zone x, y
    def env_start(self):
        land_x, land_y = get_landing_zone() # gets the x, y coordinate of the landing zone
        # At the start we initialize the agent to the top left hand corner (100, 20) with 0 velocity
        # in either any direction. The agent's angle is set to 0 and the landing zone is retrieved and set.
        # The lander starts with fuel of 100.
        # (vel_x, vel_y, angle, pos_x, pos_y, land_x, land_y, fuel)
        self.current_state = (0, 0, 0, 100, 20, land_x, land_y, 100)
        return self.current_state
    def env_step(self, action):

        land_x, land_y = get_landing_zone() # gets the x, y coordinate of the landing zone
        vel_x, vel_y = get_velocity(action) # gets the x, y velocity of the lander
        angle = get_angle(action) # gets the angle the lander is positioned in
        pos_x, pos_y = get_position(action) # gets the x, y position of the lander
        fuel = get_fuel(action) # get the amount of fuel remaining for the lander
        terminal = False
        reward = 0.0
        observation = (vel_x, vel_y, angle, pos_x, pos_y, land_x, land_y, fuel)
        # use the above observations to decide what the reward will be, and if the
        # agent is in a terminal state.
        # Recall - if the agent crashes or lands terminal needs to be set to True
        # your code here
        rev_x = - (pos_x - land_x) ** 2
        rev_y = - (pos_y - land_y) ** 2
        rev_fuel = - (fuel ** 2)

        y_vel_flag = (pos_y == 0) and (vel_y < -3)
        x_vel_flag = (pos_y == 0) and ( (vel_x < -10) or (vel_x > 10) )
        angle_flag = (pos_y == 0) and (5 < angle < 355)
        fuel_flag = (pos_y != 0) and (fuel <= 0)
        land_flag = ((pos_x != land_x) and (pos_y == land_y))

        crash_flag = True if (y_vel_flag or x_vel_flag or angle_flag or fuel_flag or land_flag) else False
        land_successful_flag = True if ( (pos_x == land_x) and (pos_y == land_y) ) else False
        terminal = (crash_flag or land_successful_flag)

        if crash_flag:

            rev_crash = -100000
        else:
            rev_crash = 0

        if land_successful_flag:

            rev_success = 100000
        else:
            rev_success = 0

        reward = rev_x + rev_y + rev_fuel + rev_crash + rev_success

        self.reward_obs_term = (reward, observation, terminal)
        return self.reward_obs_term

    def env_cleanup(self):
        return None

    def env_message(self):
        return None