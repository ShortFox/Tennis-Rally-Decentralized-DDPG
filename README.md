[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/Unity-Technologies/ml-agents/master/docs/images/tennis.png "Image of environment."

# Tennis Rally: Multiagent reinforcement learning for a cooperative task.
[Unity ML-Agents](https://unity3d.com/machine-learning) + [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) using [PyTorch](https://pytorch.org/). Adapted for multiagent reinforcement learning.

Project on use of policy-based methods for continuous control in multiagent task scenarios for partial fulfillment of [Udacity's Deep Reinforcement Learning Nanodegree.](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

### Background

See attached ```report.pdf``` for background information on reinforcement learning and the multiagent DDPG architecture utilized in this repository. DDPG utilizes an "Actor" (which learns which actions to take) and a "Critic" (which informs the Actor on the value of these actions). Here, separate Actor/Critic networks are used to train separate agents to act cooperatively.

### Introduction

The goal of the agents is to maintain a tennis rally for as long as possible.

![Trained Agent][image1]

In this modified version of [ML-Agent's Tennis example](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) provided by Udacity, the agents are rewarded (+0.1 score) every time they hit the ball over the net. If the ball hits the ground, or goes out of bounds, the agents will receive a punishment (-0.01 score). The Tennis agents can observe 8 variables: The 2D positions and velocities of the ball and racket. The agents can act by moving towards (or away) from the net, and by jumping (total action dimension = 2).

To solve the task, the average maximum score for one of the agents needs to be +0.5 over 100 consecutive episodes.

### Getting Started
1. [Download the Anaconda Python Distribution](https://www.anaconda.com/download/)

2. Once installed, open the Anaconda command prompt window and create a new conda environment that will encapsulate all the required dependencies. Replace **"desired_name_here"** with whatever name you'd like. Python 3.6 is required.

    `conda create --name "desired_name_here" python=3.6`  
    `activate "desired_name_here"`

3. Clone this repository on your local machine.

    `git clone https://github.com/ShortFox/Tennis-Rally.git`  

4. Navigate to the `python/` subdirectory in this repository and install all required dependencies

    `cd Learning-to-Reach/python`  
    `pip install .`  

    **Windows Users:** When installing the required dependencies, you may receive an error when trying to install "torch." If that is the case, then install pytorch manually, and then run `pip install .` again. Note that pytorch version 0.4.0 is required:

    `conda install pytorch=0.4.0 -c pytorch`

5. Download the Reacher Unity environment from one of the links below.  Note you do not need to have Unity installed for this repository to function.

    Download the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

6. Place the unzipped file inside the repository location. Note that the Windows (x86_64) version of the environment is already included (must be unzipped).

### Instructions to train the Tennis agents

1. Open `main.py` and edit the `file_name` variable to correctly locate the location of the Banana Man Unity environment.

2. Within the same anaconda prompt, return to the `Tennis-Rally` subfolder, and then train the agent:

    `cd ..`  
    `python main.py`

3. Following training, `DDPG_Actor1.pth/DDPG_Actor2.pth` and `DDPG_Critic1.pth/DDPG_Critic2.pth` will be created which represents the neural network weights for the Actor and Critic, respectively for Tennis agent 1 and 2. Additionally, `tennis_scores.csv` will contain the scores from training (with the row indicating episode number).

4. To watch your trained agent, first open `main.py` and edit the input parameter for the `run(train_agent)` function call so that it is the following:  

    `run(train_agent = False)`

    By default, `train()` will load the `DDPG_Actor1.pth/DDPG_Actor2.pth` and `DDPG_Critic1.pth/DDPG_Critic2.pth` files.

5. Then, execute `main.py` again:

    `python main.py`

### Interested to make your own changes?

- `model.py` - defines the neural network architecture for the Actor and Critic.
- `ddpg_agent.py` - contains the classes `Agent()`, which defines the agent's behavior and neural network update rules.
- `memory.py` - contains the class `ReplayBuffer()`, which keeps track of the agent's state-action-reward experiences in memory for randomized batch learning for the neural network.
- `noise.py` - simple noise class that adds random variation to the Reacher agent's movements to improve exploration.
- `main.py` - Defines the Unity environment and function that performs the training protocol.

See comments in associated files for more details regarding specific functions/variables.
