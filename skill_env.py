import numpy as np

class Spec:
    def __init__(self, name, len_skill, action_space):
        self.id = '{}-skill{}-action_space{}'.format(RMSenv, len_skill, action_space)

class RMSenv:
    def __init__(self, len_skill=4, action_space=5):
        self.len_skill = len_skill
        self.action_space = action_space

        self.state = np.zeros((len_skill, action_space))
        self.ac_counter = 0
        self.ground_truth_skill = np.random.randint(action_space-1, size=len_skill)
        # attribute
        self.spec = Spec("RMSenv", len_skill, action_space)

        self.reset_counter = 0

    def reset(self):
        self.state = np.zeros((self.len_skill, self.action_space))
        self.ac_counter = 0

        self.reset_counter += 1
        print('{}th run environment'.format(self.reset_counter))
        return self.state

    def step(self, action):
        '''
        Args
            action: int, vary from 0 to action space
        Output
            next_state: np.array, shape=(self.len_skill, self.action_space)
            reward:     int, basically zero, when complete a skill, get a final reward.
            done:       bool, False when not done. True when the whole skill is generated.
            _:          Whatever

        '''
        # one hot encoding
        onehotaction = np.zeros(self.action_space)
        onehotaction[action] = 1

        # insert one-hot action into current action
        self.state[self.ac_counter] = onehotaction


        self.ac_counter += 1
        if self.ac_counter == self.len_skill:
            reward = self.get_reward()
            done = True
        else:
            reward = 0
            done = False

        return self.state, reward, done, None

    def get_reward(self):
        """
        each action is either [0,1,2,3,4]
        length of skill is set to 6
        ground truth answer: [0,3,4,2]
        say input skill is [a1, a2, a3, a4], then reward is:
            200 - root_mean_square(ground_truth_answer, input_skill)
        """
        skill = np.argmax(self.state, axis=1)
        reward = 700 - 4 * np.sum( (skill-self.ground_truth_skill)**2 )
        # reward = reward / 200

        return reward

    def close(self):
        '''
        Actually nothing to do
        '''
        pass

# demo code for AtariPolicyManager in manager.py
import gym
from skill_lib.env_wrapper import SkillWrapper, ActionRemapWrapper
from skill_lib.manager import AtariPolicyManager
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


class AtariSkillenv:
    def __init__(self, ENV="Alien-ramDeterministic-v4", len_skill=4, action_space=5, 
                 model=PPO2, policy=MlpPolicy, save_path = "./path/to/store/location",
                 verbose=1, num_cpu=15):
        self.len_skill = len_skill
        self.action_space = action_space

        self.state = np.zeros((len_skill, action_space))    # observation
        self.ac_counter = 0 # Counter for actions choosed. When ac_counter equals to len_skill, game end and return a reward.
        
        env_creator = lambda:ActionRemapWrapper(gym.make(ENV))
        self.atari_manager = AtariPolicyManager(env_creator=env_creator, model=model, policy=policy, save_path=save_path, verbose=verbose, num_cpu=num_cpu)

        self.spec = Spec("RMSenv", len_skill, action_space)

        self.reset_counter = 0

    def reset(self):
        self.state = np.zeros((self.len_skill, self.action_space))
        self.ac_counter = 0     

        self.reset_counter += 1
        print('{}th run environment'.format(self.reset_counter))
        return self.state

    def step(self, action):
        '''
        Args
            action: int, vary from 0 to action space
        Output
            next_state: np.array, shape=(self.len_skill, self.action_space)
            reward:     int, basically zero, when complete a skill, get a final reward.
            done:       bool, False when not done. True when the whole skill is generated.
            _:          Whatever

        '''
        # one hot encoding
        onehotaction = np.zeros(self.action_space)
        onehotaction[action] = 1

        # insert one-hot action into current action
        self.state[self.ac_counter] = onehotaction


        self.ac_counter += 1
        if self.ac_counter == self.len_skill:
            reward = self.get_reward()
            done = True
        else:
            reward = 0
            done = False

        return self.state, reward, done, None

    def get_reward(self):
        """
        
        """
        skills = np.argmax(self.state, axis=1).tolist()
        skills = [skills]
        episode_ave_reward, action_ave_reward = self.atari_manager.get_rewards(skills, add_info={"example":"value"})

        return episode_ave_reward


if __name__ == '__main__':
    env = AtariSkillenv()
    env.reset()
    for action in [0,3,4,2]:
        state, reward, done, _ = env.step(action)
        print(reward)




