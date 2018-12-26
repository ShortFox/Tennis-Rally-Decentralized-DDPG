from ddpg_agent import Agent
from unityagents import UnityEnvironment
import numpy as np
from collections import deque


#Download and select appropraite executable based on your operating system.
#Mac:
    #filename = "Tennis.app"
#Windows (x86):
    #file_name = Tennis_Windows_x86/Tennis.exe
#Windows (x86_64):
file_name = "Tennis_Windows_x86_64/Tennis.exe"
#Linux (x86):
    #file_name = "Tennis_Linux/Tennis.x86"
#Linux (x86_64)
    #file_name = "Tennis_Linux/Tennis.x86_64"

#Function that runs simulation. train_agent parameter determines if network is trained, or if a saved version is used
def run(num_episodes = 10000, train_agent = True):
    #Define the Unity environment from executable
    env = UnityEnvironment(file_name)

    #Get the "brain" name of learning interface
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    #Get environment information.
    env_info = env.reset(train_mode=train_agent)[brain_name]
    states = env_info.vector_observations.reshape(1,-1)
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)

    #Define the players in learning environment
    agent1 = Agent(state_size, action_size, random_seed = 0)
    agent2 = Agent(state_size, action_size, random_seed = 0)
    
    #If training set to false, load previous network weights if possible.
    if train_agent == False:
        agent1.load(agent1.actor_local, 'DDPG_Actor1.pth')
        agent1.load(agent1.critic_local, 'DDPG_Critic1.pth')
        agent2.load(agent2.actor_local, 'DDPG_Actor2.pth')
        agent2.load(agent2.critic_local, 'DDPG_Critic2.pth')

    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1,num_episodes+1):
        
        #Get environment information and current state.
        env_info = env.reset(train_mode=train_agent)[brain_name]
        states = env_info.vector_observations.reshape(1,-1)
        
        #Update mean of noise distribution for agents.
        agent1.reset()
        agent2.reset()
        
        score = np.zeros(num_agents)

        while True:
            #Given state, have each actor perform an action
            action1 = agent1.act(states,add_noise=train_agent)
            action2 = agent2.act(states,add_noise=train_agent)
            
            #Combine agent actions to send to Unity. Receive next state, rewards, and whether task is done
            action_combined = np.concatenate((action1, action2), axis=0)
            env_info = env.step(action_combined)[brain_name]
            next_states = env_info.vector_observations.reshape(1,-1)
            rewards = env_info.rewards
            dones = env_info.local_done

            #If agent set to training, take a step. This will add experience to each agent's memory and perform a learning iteration.
            if train_agent:
                agent1.step(states, action1, rewards[0], next_states, dones[0])
                agent2.step(states, action2, rewards[1], next_states, dones[1])

            #Update score.
            score += rewards
            states = next_states

            #If episode ends, break out of while loop.
            if np.any(dones):
                break
      
        #Save the max score obtained by a player during the episode.
        scores.append(np.max(score))
        scores_window.append(np.max(score))

        #Print statements and save score and model weights periodically.
        print('Episode', i_episode, 'Score:', scores[-1], end='\r')
        if i_episode >= 100:
            # Every 100 episodes, save scores and models.
            if (i_episode)%100 ==0:
                print("Episode: {0:d}, Average score {1:f}".format(i_episode,np.mean(scores_window)))
                save_scores(scores)
                if train_agent:
                    agent1.save(agent1.actor_local, 'DDPG_Actor1.pth')
                    agent1.save(agent1.critic_local, 'DDPG_Critic1.pth')
                    agent2.save(agent2.actor_local, 'DDPG_Actor2.pth')
                    agent2.save(agent2.critic_local, 'DDPG_Critic2.pth')
            #If environment is solved early, break
            if np.mean(scores_window) >= 0.5:
                break
    save_scores(scores)
    if train_agent:
        print("Saving final model...")
        agent1.save(agent1.actor_local, 'DDPG_Actor1.pth')
        agent1.save(agent1.critic_local, 'DDPG_Critic1.pth')
        agent2.save(agent2.actor_local, 'DDPG_Actor2.pth')
        agent2.save(agent2.critic_local, 'DDPG_Critic2.pth')
    env.close()
    
def save_scores(output):
    """Save scores to a .csv file"""
    np.savetxt('tennis_scores.csv',output, delimiter=',')
        
run(train_agent = True)