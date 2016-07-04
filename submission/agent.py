import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import random
import time
import pandas as pd
import os

df = pd.DataFrame(index=np.arange(0, 0), columns=['attempt', 't', 'alpha', 'gamma', 'rar', 'radr', 'plr', 'plrr',
                                                  'minv', 'maxv', 'light', 'left', 'right', 'oncoming', 'planner',
                                                  'state', 'action', 'reward', 'deadline', 'total_reward', 'successes',
                                                  'at-none', 'at-forward', 'at-left', 'at-right'])


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    valid_actions = [None, 'forward', 'left', 'right']
    valid_lights = ['red', 'green']

    def __init__(self, env, params):
        super(LearningAgent, self).__init__(env)
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.attempt = 0
        self.successes = 0
        self.state = -1
        self.last_state = -1
        self.last_action = -1
        self.last_reward = -1
        self.total_rewards = 0

        self.alpha, self.gamma, self.rar, self.radr, self.plr, self.plrr, self.minv, self.maxv, self.verbose = params

        self.action_taken = [0, 0, 0, 0]

        print "alpha, gamma, rar, radr, min, max, verbose = {}".format(params)
        # 2 lights * 4 {left, right, oncoming} * 3 planner * 4 actions
        self.Q = np.random.uniform(self.minv, self.maxv, size=(2, 4, 4, 4, 3, 4))

    def clear_states(self):
        self.state = -1
        self.last_state = -1
        self.last_action = -1
        self.last_reward = -1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.clear_states()

    @staticmethod
    def inputs_to_state(inputs_tuple):
        """
        # 2 lights * 4 {oncoming, left, right} * 2 deadline * 4 actions
        """
        light, left, right, oncoming, planner = inputs_tuple
        assert light in ('red', 'green')
        assert left in (None, 'forward', 'right', 'left')
        assert right in (None, 'forward', 'right', 'left')
        assert oncoming in (None, 'forward', 'right', 'left')
        assert planner in ('forward', 'right', 'left')

        il = LearningAgent.valid_lights.index(light)
        ilf = LearningAgent.valid_actions.index(left)
        ir = LearningAgent.valid_actions.index(right)
        io = LearningAgent.valid_actions.index(oncoming)
        ip = 0 if planner == 'forward' else 1 if planner == 'right' else 2

        state = int(str(il) + str(ilf) + str(ir) + str(io) + str(ip))
        return state

    def state_to_indeces(self, state):
        state_str = '0' * (len(self.Q.shape) - 1 - len(str(state))) + str(state)
        return tuple([int(i) for i in state_str])

    def update_q_table(self, max_action):
        last_state = self.state_to_indeces(self.last_state)
        current_state = self.state_to_indeces(self.state)

        exp_disc_rewards = self.last_reward + self.gamma * self.Q[current_state + (max_action, )]
        self.Q[last_state + (self.last_action, )] = (1 - self.alpha) * self.Q[last_state + (self.last_action, )] + \
                                                    self.alpha * exp_disc_rewards

    @staticmethod
    def get_planned_action(inputs_tuple):
        light, left, right, oncoming, planner = inputs_tuple
        # None, 'forward', 'left', 'right'
        # valid_actions = [None, 'forward', 'left', 'right']

        if light == 'red' and oncoming != 'left' and left != 'oncoming' and planner == 'right':
            return LearningAgent.valid_actions.index('right')

        if light == 'red' and oncoming != 'right' and right != 'oncoming' and planner == 'left':
            return LearningAgent.valid_actions.index('left')

        if light == 'green' and oncoming is not None and planner == 'left':
            return LearningAgent.valid_actions.index(None)

        return LearningAgent.valid_actions.index(planner)

    def update(self, t):

        # sense the world
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        inputs_tuple = (inputs['light'], inputs['left'], inputs['right'], inputs['oncoming'], self.next_waypoint)

        # determine the new state
        self.state = self.inputs_to_state(inputs_tuple)
        if self.verbose:
            print "t {} - inputs {} - state {}".format(t, inputs_tuple, self.state)

        # select best action
        max_action = np.argmax(self.Q[self.state_to_indeces(self.state)])

        # UPDATE PREVIOUS STATE, ACTION Q-VALUE
        if t != 0:
            self.update_q_table(max_action)

        # potentially act as bootstrapped knowledge
        act_planned = np.random.choice(a=[True, False], p=[self.plr, 1 - self.plr])
        planned_action = self.get_planned_action(inputs_tuple)

        # potentially act randomly for enhancing learning
        act_randomly = np.random.choice(a=[True, False], p=[self.rar, 1 - self.rar])
        random_action = random.randint(0, 3)

        if self.verbose:
            print "Choosing action based on:"
            print "\trandom action? ", act_randomly, " it would be ", random_action
            print "\trar =", self.rar, "radr =", self.radr
            print "\tplanned action? ", act_planned, " it would be ", planned_action
            print "\tplr =", self.plr, "plrr =", self.plrr
            print "\tmax action at [", self.state_to_indeces(self.state), "]=", \
                max_action, \
                LearningAgent.valid_actions[max_action]
            print "\t", self.Q[self.state_to_indeces(self.state)]

        action = planned_action if act_planned else random_action if act_randomly else max_action
        if self.verbose:
            print 'we are selecting:'
            print '\taction: ', action
            print "\t(" + str(LearningAgent.valid_actions[action]) + ")"
            print "\t" + ("planned" if act_planned else "random" if act_randomly else "max")

        reward = self.env.act(self, LearningAgent.valid_actions[action])
        self.total_rewards += reward
        if self.verbose and reward > 5:
            print '#################### GREAT #######################'
            print '#################### GREAT #######################'
            print '#################### GREAT #######################'
            print '#################### GREAT #######################'
            print '#################### GREAT #######################'
            print '#################### GREAT #######################'

        if self.verbose:
            print "decision: action = {}, action reward = {}, deadline = {}". \
                format(action, reward, self.env.get_deadline(self))

        # hold the experience for this timestep
        self.last_state = self.state
        self.last_action = action
        self.last_reward = reward

        # update counters
        self.attempt = self.attempt + 1 if t == 0 else self.attempt
        self.successes = self.successes + 1 if reward > 5 else self.successes
        self.action_taken[action] += 1

        # log t for further analysis
        df.loc[len(df)] = [
            self.attempt, t, self.alpha, self.gamma, self.rar, self.radr, self.plr, self.plrr, self.minv, self.maxv,
            inputs_tuple[0], 'None' if inputs_tuple[1] is None else inputs_tuple[1],
            'None' if inputs_tuple[2] is None else inputs_tuple[2],
            'None' if inputs_tuple[3] is None else inputs_tuple[3], inputs_tuple[4], self.state,
            str(LearningAgent.valid_actions[action]), reward, self.env.get_deadline(self), self.total_rewards,
            self.successes, self.action_taken[0], self.action_taken[1], self.action_taken[2], self.action_taken[3]
        ]

        if act_planned:
            # decay 'training wheels'
            self.plr *= self.plrr
        elif act_randomly:
            # decay randomness
            self.rar *= self.radr

        if self.verbose:
            print "total_rewards = {}, attempts = {}, successes = {}, actions_distribution {}". \
                format(self.total_rewards, self.attempt, self.successes, self.action_taken)

        if False and self.attempt % 10 == 0 and t == 0:
            # self.alpha, self.gamma, self.rar, self.radr, min, max, self.verbose
            directory = '../alpha-' + str(self.alpha) + '-gamma-' + str(self.gamma )+ '-rand-' + \
                        str(self.rar) + '-randr-' + str(self.radr) + '-plr-' + \
                        str(self.plr) + '-plrr-' + str(self.plrr) + \
                        '-minv-' + str(self.minv) + '-maxv-' + str(self.maxv)
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = 'q-table-' + str(self.attempt) + '-' + str(int(time.time())) + '.gz'

            full_path = directory + '/' + filename
            if self.verbose:
                print 'saving q table to file: ', full_path

            with open(full_path, 'w') as f:
                np.savetxt(f, self.Q, fmt="%s")

        if t == 0 and (self.attempt == 0 or self.attempt == 100):
            directory = '../analyses'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = 'analysis-' + str(int(time.time())) + '.csv'
            full_path = directory + '/' + filename

            if self.verbose:
                print 'saving analysis file at: ', full_path

            with open(full_path, 'w') as f:
                df.to_csv(f)


def run():
    """Run the agent for a finite number of trials."""


    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)

    a = e.create_agent(LearningAgent, (0.7, 0.5, 1.0, 0.9,
                                       1.0, 0.9, 0, 5, True))  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.2, )  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    return

    # self.alpha, self.gamma, self.rar, self.radr, min, max, self.verbose
    for alpha in [0.2, 0.5, 0.8]:
        for gamma in [0.2, 0.5, 0.8]:
            for rand in [0.0, 1.0]:
                for randr in [0.9]:
                    for plr in [0.0, 1.0]:
                        for plrr in [0.9]:
                            for minv in [-5, -2, 0]:
                                for maxv in [0, 2, 5]:

                                    # Set up environment and agent
                                    e = Environment()  # create environment (also adds some dummy traffic)

                                    a = e.create_agent(LearningAgent, (alpha, gamma, rand, randr,
                                                                       plr, plrr, minv, maxv, False))  # create agent
                                    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

                                    # Now simulate it
                                    sim = Simulator(e, update_delay=0.0, )  # reduce update_delay to speed up simulation
                                    sim.run(n_trials=100)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()
