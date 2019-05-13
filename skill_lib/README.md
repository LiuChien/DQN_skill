# skill_lib


## Example

### mamager
```python
# demo code for AtariPolicyManager in manager.py
from env_wrapper import SkillWrapper
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
...
# (deprecated)env = gym.make("Alien-ram-v0")
# [new feature]the new AtariPolicyManager support multiprocess of PPO1, PPO2, A2C...etc (NOT INCLUDE TRPO)
# [new feature] num_cpu decide how many parallel processs 
# [new feature] the add_info argumrnt of get_rewards will log to the log.txt
env_creator = lambda:ActionRemapWrapper(gym.make(ENV))
atari_manager = AtariPolicyManager(env_creator=env_creator, model=PPO2, policy=MlpPolicy, save_path = "/path/to/store/location", verbose=1, num_cpu=15)
...
skills= [[2,2,2,2],[3,3,3,3],[4,4,4],[5,5,5]]
episode_ave_reward, action_ave_reward = atari_manager.get_rewards(skills, add_info={"example":"value"})

```

### env_wrapper.py
#### SkillWrapper
```python
# demo code for SkillWrapper in env_wrapper.py
import gym
from env_wrapper import SkillWrapper

SKILLS = [[2,2,2,2],[3,3,3,3],[4,4,4],[5,5,5]]
...
env = gym.make("Alien-ram-v0")
env = SkillWrapper(env, SKILLS)
...
```
#### ActionRemapWrapper
Already predefined table: Alien-ram-v0, Alien-ram-v4

```python
# demo code for ActionRemapWrapper in env_wrapper.py
import gym
from env_wrapper import ActionRemapWrapper

       
# Usage1
# the env id "Alien-ram-v0" has a default action remap table
...
env = gym.make("Alien-ram-v0")
env = ActionRemapWrapper(env)
...

# Usage2
# specify the name of predefined table
...
env = gym.make("Alien-ram-v0", table_name="alien")
env = ActionRemapWrapper(env)
...

# Usage3
# predefined your own action remap table
...
TABLE={0:0,
       1:1,
       2:2,
       3:3,
       4:4,
       5:5}
#or
TABLE = [0,1,2,3,4,5]
env = gym.make("Alien-ram-v0")
env = ActionRemapWrapper(env, action_table=TABLE)
...
```
