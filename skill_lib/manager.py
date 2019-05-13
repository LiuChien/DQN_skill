import gym
import time
from skill_lib.env_wrapper import SkillWrapper
from collections import deque, OrderedDict, Counter
import os
import datetime
import yaml
import shutil
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines.common import set_global_seeds
import numpy as np
import glob
import time
#TODO: Define abstract class for policy manager
# provide logger with skills used frequency
#
class PolicyManager(object):
    
    def get_rewards():
        raise NotImplementedError





class AtariPolicyManager(object):
    def __init__(self, env_creator, model, policy, save_path, preserve_model=10, num_cpu=20, pretrain=False, log_action_skill=True, verbose=0):
        """
        :(deprecate)param env: (gym.core.Env) gym env with discrete action space
        :env_creator: (lambda function) environment constructor to create an env with type (gym.core.Env )
        :param model: any model in stable_baselines e.g PPO2, TRPO...
        :param policy: (ActorCriticPolicy) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
        :param save_path: (str) path to store model and log
        :param preserve_model: (int) how much history model will be preserved
        :param num cpu: (int) train on how many process
        :param pretrain: (bool) if the model passed in are pretrained
        :param log_action_skill(bool) wheather to count frequency of actions
        :param verbose: (int) 0,1 wheather or not to print the training process
        """
        # super(AtariPolicyManager, self).__init__()
        # self.env=env
        self.env_creator = env_creator
        self.model = model
        self.policy = policy
        self.preserve_model = preserve_model
        self.verbose = verbose
        self._save_model_name=deque()
        self._serial_num = 1
        self.num_cpu = num_cpu
        self.reset_num_timesteps = not pretrain
        if save_path is None:
            self.save_path = None
        elif os.path.exists(save_path):
            while(True):
                i = input("save path already exist: {}\n new name(n)/keep(k)/exit(e)?".format(save_path))
                if i=="n":
                    dir_name = input("new dir name: ")
                    new_path = os.path.abspath(os.path.join(save_path, dir_name))
                    os.makedirs(new_path)
                    self.save_path = new_path
                    break
                elif i=="k":
                    self.save_path = os.path.abspath(save_path)
                    break
                elif i=="R":
                    shutil.rmtree(save_path)
                    os.makedirs(save_path)
                    self.save_path = os.path.abspath(save_path)
                    break
                elif i=="e":
                    exit(0)
        else:
            self.save_path = save_path
            os.makedirs(save_path)
        self.log_action_skill = log_action_skill
        
    def make_env(self, env_creator, rank, skills=[], action_table=None,seed=0):
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = env_creator()
            env = SkillWrapper(env, skills=skills)
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
    def evaluate(self, env, model, eval_times, eval_max_steps, render=False):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: info
        """

        # evaluate with multiprocess env
        if self.num_cpu>1:
            episode_rewards = [[] for _ in range(env.num_envs)]
            ep_rew = [0.0 for _ in range(env.num_envs)]

            action_statistic = OrderedDict()
            for i in range(env.action_space.n):
                    action_statistic[str(env.action_space[i])]=0
            # print(action_statistic)
            act_log = [[]for _ in range(env.num_envs)]

            obs = env.reset()
            # for i in range(num_steps):
            ep_count = 0
            total_actions_count=0
            print("start to eval agent...")
            while True:
                # _states are only useful when using LSTM policies
                actions, _states = model.predict(obs)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                obs, rewards, dones, info = env.step(actions)

                # Stats
                for i in range(env.num_envs):
                    # episode_rewards[i][-1] += rewards[i]
                    ep_rew [i] = ep_rew[i] + rewards[i]
                    act_log[i].append(actions[i])
                    if render:
                        env.render()
                        time.sleep(0.05)
                    if dones[i]:
                        # episode_rewards[i].append(0.0)
                        episode_rewards[i].append(ep_rew[i])
                        
                        act_count = Counter(np.asarray(act_log[i]).flatten())
                        for key in act_count:
                            action_statistic[str(env.action_space[key])] +=  act_count[key]
                            total_actions_count += act_count[key]
                        ep_rew[i] = 0
                        act_log[i] = []
                        ep_count = ep_count + 1
                if ep_count >= eval_times:
                    break
            print("Finish eval agent")
            print("Elapsed: {} sec".format(round(time.time()-self.strat_time, 3)))

            # mean_rewards =  [0.0 for _ in range(env.num_envs)]
            total_reward = []
            # n_episodes = 0
            for i in range(env.num_envs):
                total_reward.extend(episode_rewards[i])
                # mean_rewards[i] = np.mean(episode_rewards[i])     
                # n_episodes += len(episode_rewards[i])   

            # Compute mean reward
            # mean_reward = round(np.mean(mean_rewards), 1)
            # print("Mean reward:", mean_reward, "Num episodes:", n_episodes)
            print(total_reward)
            info = OrderedDict()
            info["ave_score"] = round(np.mean(total_reward), 1)
            info["ave_score_std"] = round(np.std(np.array(total_reward)),3)
            info["ave_action_reward"] = sum(total_reward)/total_actions_count
            if self.log_action_skill:
                info.update(action_statistic)
        else:
            # evaluate with single process env
            info = OrderedDict()
            if self.log_action_skill:
                action_statistic = OrderedDict()
                for i in range(env.action_space.n):
                    action_statistic[str(env.action_space[i])]=0
            ep_reward = []
            ep_ave_reward = []
            print("start to eval agent...")
            for i in range(eval_times):
                obs = env.reset()
                total_reward = []
                for i in range(eval_max_steps):
                    action, _states = model.predict(obs)
                    obs, rewards, dones, info_ = env.step(action)
                    total_reward.append(rewards[0])

                    if self.log_action_skill is True:
                        action_statistic[str(env.action_space[action[0]])] = action_statistic[str(env.action_space[action[0]])] + 1

                    if bool(dones[0]) is True:
                        break
                
                ep_reward.append(sum(total_reward))
                ep_ave_reward.append(sum(total_reward)/len(total_reward))
            
            
            print("Finish eval agent")
            print("Elapsed: {} sec".format(round(time.time()-self.strat_time, 3)))
            ave_score = sum(ep_reward)/len(ep_reward)
            ave_action_reward = sum(ep_ave_reward)/len(ep_ave_reward)
            ave_score_std = round(np.std(np.array(ep_reward)),3)

            # info.update({"ave_score":ave_score, "ave_score_std":ave_score_std, "ave_reward":ave_reward})
            info["ave_score"] = ave_score
            info["ave_score_std"] = ave_score_std
            info["ave_action_reward"] = ave_action_reward
            if self.log_action_skill:
                info.update(action_statistic)
        return info
    def get_rewards(self, skills=[], train_total_timesteps=5000000, eval_times=10, eval_max_steps=10000, model_save_name=None, add_info={}):
    # def get_rewards(self, skills=[], train_total_timesteps=10, eval_times=10, eval_max_steps=10, model_save_name=None, add_info={}):

        """
        
        :param skills: (list) the availiable action sequence for agent 
        e.g [[0,2,2],[0,1,1]]
        :param train_total_timesteps: (int)total_timesteps to train 
        :param eval_times: (int)the evaluation times
        e.g eval_times=100, evalulate the policy by averageing the reward of 100 episode
        :param eval_max_steps: (int)maximum timesteps per episode when evaluate
        :param model_save_name: (str)specify the name of saved model (should not repeat)
        :param add_info: (dict) other information to log in log.txt
        """

        # env = SkillWrapper(self.env, skills=skills)
        if self.num_cpu>1:
            env = SubprocVecEnv([self.make_env(self.env_creator, i, skills) for i in range(self.num_cpu)])
        else:
            env = DummyVecEnv([lambda: self.env_creator()])
        model = self.model(self.policy, env, verbose=self.verbose)
        
        self.strat_time = time.time()
        print("start to train agent...")
        model.learn(total_timesteps=train_total_timesteps, reset_num_timesteps=self.reset_num_timesteps)
        print("Finish train agent")

        if self.save_path is not None:
            if self.preserve_model>0:
                self.save_model(model, model_save_name, skills=skills)
        
        # evaluate
        info = self.evaluate(env, model, eval_times, eval_max_steps)
        env.close()    
        
        #log result
        info.update(add_info)
        self.log(info)

        self._serial_num = self._serial_num + 1
        return info["ave_score"], info["ave_action_reward"]
    
    def save_model(self, model, name=None, **kwargs):
        if name is None:
            name = "model_" + str(self._serial_num)
        
        save_name = os.path.join(self.save_path, name)
        if os.path.isfile(save_name+".pkl"):
            print("Warning: overwrite model: {}".format(save_name+".pkl"))
        model.save(save_name)

        if len(kwargs)!=0:
            with open('{}.yml'.format(save_name), 'w') as outfile:
                yaml.dump(kwargs, outfile, default_flow_style=False)

        
        if save_name not in self._save_model_name:
            self._save_model_name.append(save_name)

        if len(self._save_model_name) > self.preserve_model:
            remove_name = self._save_model_name.popleft()
            remove_name = remove_name + ".*"
            for rm_name in glob.glob(remove_name):
                os.remove(rm_name)
        
    def log(self, info=None):
        if info is not None:
            filename = os.path.join(self.save_path, "log.txt")
            # assert all(key in info for key in ["ave_reward", "ave_score"])
            with open(filename, 'a') as f:
                # if "ave_total_scroe" in info:
                # print("{s:{c}^{n}}".format(s=(" Episode: " + str(self._serial_num) + " "), c='*', n=27), file=f)
                print("Episode: {}".format(str(self._serial_num)), file=f)
                keys = info.keys()
                # keys.sort()
                for key in keys:
                    print("{}: {}".format(key, info[key]), file=f)
                print("{s:{c}^{n}}".format(s="", c='*', n=27), file=f)




                





        

        

