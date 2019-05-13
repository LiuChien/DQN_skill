### Pre-requirements
#### stable_baselines
Please add stable_baselines into ./

#### Environment
Requirments are the same as skill_lib, which can be found at "./skill_lib/requirements.txt"

### Run
Two different envs are provides.
1. RMSenv (Root Mean Square)
  It's a test env. The reward is calculate by RMS.
2. AtariSkillenv
  Reward is gave by AtariPolicyManager
  
One can modify by changing ```line 20, 21 ``` in dqn.py.

Simply run the program by
```
python dqn.py
```



### Acknowledgement
This code is highly inspired by [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning) and [AllenChen0958/skill_lib](https://github.com/AllenChen0958/skill_lib)

