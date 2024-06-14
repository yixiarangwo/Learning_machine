import cv2
from pathlib import Path
import numpy as np
import random
import copy
import pickle
import math


from data_files import FIGRURES_DIR, RESULT_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

from robobo_interface.datatypes import (
    Emotion,
    LedColor,
    LedId,
    Acceleration,
    Position,
    Orientation,
    WheelPosition,
    SoundEmotion,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1
    
class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99, #gamma
		tau=0.005,
		policy_noise=0.2, #target_policy_noise
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=128):
		self.total_it += 1
		print("---------------------------------------")
		print(f"self.total_it: {self.total_it}")

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Check gradients optimization step
		if self.total_it > self.policy_freq:
			for name, param in self.critic.named_parameters():
				if param.grad is None:
					error_save()
					raise ValueError(f"Gradient for parameter {name} in critic network is None.")
                
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
        
        # Check gradients optimization step
		if self.total_it > self.policy_freq:
			for name, param in self.critic.named_parameters():
				if param.grad is None:
					error_save()
					raise ValueError(f"Gradient for parameter {name} in critic network is None.")

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
# 			print(f"actor_loss: {actor_loss}")
			
            # Check gradients optimization step
			if self.total_it > self.policy_freq:
				before_optimization = {}
				for name, param in self.actor.named_parameters():
					before_optimization[name] = param.data.clone()
					if param.grad is None:
						print(f"actor_loss: {actor_loss}")
						print(param)
						error_save()
						raise ValueError(f"Gradient for parameter {name} in actor network is None.")

			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			if self.total_it > self.policy_freq:
				for name, param in self.actor.named_parameters():
					if torch.equal(before_optimization[name], param.data):
						print(f"actor_loss: {actor_loss}")
						print(f"Gradient descent not performed.")
# 						error_save()
						raise ValueError(f"Gradient descent not performed.")

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
			print("---------------------------------------")


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
    
#TD3 parameters
start_timesteps = 100 # Time steps initial random policy is used
eval_freq = 1000  # After 1000 steps (1 stage / 5 episodes) to perform the evaluation
max_timesteps = int(1e6)  # Maximum number of steps to perform (35 stages=35000 steps), but we set it to 1e6 to ensure it's large enough (sometimes we load old buffer)
expl_noise = 0.2  # Initial Gaussian exploration noise starting value 
batch_size = 128  # Batch size for both actor and critic
discount = 0.99  # Discount factor 
tau = 0.01  # # Target network soft update rate 
policy_noise = 0.2  # Noise added to target policy during critic update
noise_clip = 0.5  # Range to clip target policy noise
policy_freq = 2  # Frequency of delayed policy updates for Actor network

state_dim = 8
action_dim = 2
max_action = 100 # torch.tanh range:(-1, 1) -> torch.tanh*max_action: (-100,100) for both left speed and right speed

policy = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=max_timesteps)    


# Load previous model weights
#critic
critic_path = RESULT_DIR / f"model_stage_0_critic"
if critic_path.exists():
    policy.critic.load_state_dict(torch.load(critic_path))
    print("model_stage_0_critic loaded")
    
critic_optimizer_path = RESULT_DIR / f"model_stage_0_critic_optimizer"
if critic_optimizer_path.exists():
    policy.critic_optimizer.load_state_dict(torch.load(critic_optimizer_path))
    print("model_stage_0_critic_optimizer loaded")
    
if critic_path.exists() and critic_optimizer_path.exists():
    policy.critic_target = copy.deepcopy(policy.critic)
    print("critic_target copied from policy.critic")

print("---------------------------------------")

# actor    
actor_path = RESULT_DIR / f"model_stage_0_actor"
if actor_path.exists():
    policy.actor.load_state_dict(torch.load(actor_path))
    print("model_stage_0_actor loaded")
    
actor_optimizer_path = RESULT_DIR / f"model_stage_0_actor_optimizer"
if actor_optimizer_path.exists():
    policy.actor_optimizer.load_state_dict(torch.load(actor_optimizer_path))
    print("model_stage_0_actor_optimizer loaded")
    
if actor_path.exists() and actor_optimizer_path.exists():
    policy.actor_target = copy.deepcopy(policy.actor)
    print("actor_target copied from policy.actor")
    
print("---------------------------------------")    
    
#load previous replay_buffer
filename = RESULT_DIR / "replay_buffer.pkl"
if filename.exists():
    with open(filename, 'rb') as f:
        replay_buffer = pickle.load(f)
    print("replay_buffer loaded")
        
print("---------------------------------------")        
        
#######################Connecting the TD3 network to IRobobo###########################   .
def error_save():
    critic_path = RESULT_DIR / f"model_stage_E_critic"
    torch.save(policy.critic.state_dict(), critic_path)
    critic_optimizer_path = RESULT_DIR / f"model_stage_E_critic_optimizer"
    torch.save(policy.critic_optimizer.state_dict(), critic_optimizer_path)

    actor_path = RESULT_DIR / f"model_stage_E_actor"
    torch.save(policy.actor.state_dict(), actor_path)
    actor_optimizer_path = RESULT_DIR / f"model_stage_E_actor_optimizer"
    torch.save(policy.actor_optimizer.state_dict(), actor_optimizer_path)

def normalize_sensor(v):
    if v == 0:
        return 0
    else:
        return 1 / math.log2(v)

def read_irs_data(rob):
    IRS_data = rob.read_irs()
    normalized_data = [normalize_sensor(v) for v in IRS_data]
    
    return torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0).to(device)

def is_spin(last_position, current_position):
    if abs(current_position.x - last_position.x) < 0.1 and abs(current_position.y - last_position.y) < 0.1:
        print("spin!")
        return True
    
    return False

def is_stuck(last_position, current_position):
    if abs(current_position.x - last_position.x) < 0.01 and abs(current_position.y - last_position.y) < 0.01:
        print("stuck!")
        return True
    
    return False
    
def get_move_distacne(last_position, current_position):
    move_distance = (current_position.x - last_position.x)**2 + (current_position.y - last_position.y)**2
    move_distance = np.sqrt(move_distance)
    
    return move_distance

    
def calculate_reward(rob, last_position, current_position, current_left_speed, current_right_speed, last_stuck, current_stuck):
    penalty = 0
    avoid_reward = 0
    linear_speed_reward = 0
    angular_speed_reward = 0
    distance_reward = get_move_distacne(last_position, current_position)*100
    
    if (current_left_speed >=0 and current_right_speed >=0):
        if (abs(abs(current_left_speed)-abs(current_left_speed))<=10):
            linear_speed_reward = abs(current_left_speed)/2.0 + abs(current_right_speed)/2.0 #forward linear speed
        else:
            linear_speed_reward = abs(current_left_speed)/10.0 + abs(current_right_speed)/10.0 
    elif (current_left_speed <=0 and current_right_speed <=0):
        linear_speed_reward = -abs(current_left_speed)/2.0 - abs(current_right_speed)/2.0 #backward linear speed  
    else:
        angular_speed_reward = -(abs(current_left_speed)/10.0) - (abs(current_right_speed)/10.0) #angular speed
        
    if current_stuck:
        penalty -= 100
    elif is_spin(last_position, current_position):
        penalty -= 50
        
    if last_stuck and current_stuck:
        linear_speed_reward *= -1 #help robot to escape
        angular_speed_reward *= -1
    
    if last_stuck and not current_stuck:
        print("escape")
        avoid_reward = 100
        
    reward = linear_speed_reward + angular_speed_reward + penalty + avoid_reward + distance_reward
    scaled_reward = reward / 300.0 #Scale the theoretical range (-200,300) to the range (-0.66, 1). Assume max move_distance is 1
    
    return scaled_reward 

def get_state(rob):
    IRS_data = read_irs_data(rob)
    state_data = IRS_data.unsqueeze(1)
    
    return state_data

def one_episode_initial(rob):
    move_index = 0
    current_left_speed = 0
    current_right_speed = 0
    
    done = False
    rob.play_simulation()  
    while move_index < start_timesteps:
        # Get the current state
        if move_index > 0:
            state_data = next_state_data
            
            # only for reward calculation
            last_position = current_position
            last_stuck = current_stuck
        else: 
            state_data = get_state(rob)
            
            # only for reward calculation
            last_position = rob.get_position()
            last_stuck = False
        
        # Predefined behavior
        if move_index <= 10:
            left_speed, right_speed = 100,100
        elif move_index <= 20:
            left_speed, right_speed = -100,-100
        elif move_index <= 30:
            left_speed, right_speed = 100,100
        elif move_index <= 40:
            left_speed, right_speed = -100,-100
        elif move_index <= 50:
            left_speed, right_speed = 100,100
        elif move_index <= 60:
            left_speed, right_speed = -100,-100
        elif move_index <= 70:
            left_speed, right_speed = 100,100
        elif move_index <= 80:
            left_speed, right_speed = -100,-100
        elif move_index <= 90:
            left_speed, right_speed = 100,100
        else:
            left_speed, right_speed = -100,-100
        
             
         #OR Select action randomly
#         left_speed = random.randint(-100, 100)
#         right_speed = random.randint(-100, 100)
        action = [left_speed, right_speed]
        
        # Perform action
        left_speed = int(action[0])
        right_speed = int(action[1])
        rob.move_blocking(left_speed, right_speed, millis=400)
        
        # Getting the state and rewards after performing an action
        move_index += 1
        
        next_state_data = get_state(rob)
        
        # only for reward calculation
        current_position = rob.get_position()
        current_left_speed, current_right_speed = left_speed, right_speed
        current_stuck = is_stuck(last_position, current_position)

        # Calculate the reward of this action
        reward = calculate_reward(rob, last_position, current_position, current_left_speed, current_right_speed, last_stuck, current_stuck)
        
        print(f"move: {move_index}, left_speed: {left_speed}, right_speed: {right_speed}, reward: {reward}\n")
        
        # Add state, action, reward, etc. to the replay buffer
        replay_buffer.add(state_data, action, next_state_data, reward, done)
        
        done = move_index == start_timesteps
        
    rob.stop_simulation()  
    
    #save buffer
    filename = RESULT_DIR / "replay_buffer.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(replay_buffer, f)

def one_episode(rob, stage, episode):
    move_index = 0
    
    rob.play_simulation()  
    done = False
    while not done:
       # Get the current state
        if move_index > 0:
            state_data = next_state_data

            # only for reward calculation
            last_position = current_position
            last_stuck = current_stuck
        else: 
            state_data = get_state(rob)

            # only for reward calculation
            last_position = rob.get_position()
            last_stuck = False

        # Select action according to policy
        action = (
            policy.select_action(np.array(state_data))
            + np.random.normal(0, max_action * expl_noise, size=action_dim)
        ).clip(-max_action, max_action)    

        # Perform action
        left_speed = int(action[0])
        right_speed = int(action[1])
        rob.move_blocking(left_speed, right_speed, millis=400)

        # Getting the state and rewards after performing an action
        move_index += 1

        next_state_data = get_state(rob)

        # only for reward calculation
        current_position = rob.get_position()
        current_left_speed, current_right_speed = left_speed, right_speed
        current_stuck = is_stuck(last_position, current_position)

        # Calculate the reward of this action
        reward = calculate_reward(rob, last_position, current_position, current_left_speed, current_right_speed, last_stuck, current_stuck)

        print(f"move: {move_index}, left_speed: {left_speed}, right_speed: {right_speed}, reward: {reward}")

        # Add state, action, reward, etc. to the replay buffer
        replay_buffer.add(state_data, action, next_state_data, reward, done)

        done = move_index == 200

        # Train agent after collecting sufficient data
        policy.train(replay_buffer, batch_size=128)

    rob.stop_simulation() 
    
    #save buffer
    filename = RESULT_DIR / "replay_buffer.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(replay_buffer, f)
    
    #save the model
#     critic_path = RESULT_DIR / f"model_stage_{stage}_episode_{episode+1}_critic"
#     torch.save(policy.critic.state_dict(), critic_path)
#     critic_optimizer_path = RESULT_DIR / f"model_stage_{stage}_episode_{episode+1}_critic_optimizer"
#     torch.save(policy.critic_optimizer.state_dict(), critic_optimizer_path)

#     actor_path = RESULT_DIR / f"model_stage_{stage}_episode_{episode+1}_actor"
#     torch.save(policy.actor.state_dict(), actor_path)
#     actor_optimizer_path = RESULT_DIR / f"model_stage_{stage}_episode_{episode+1}_actor_optimizer"
#     torch.save(policy.actor_optimizer.state_dict(), actor_optimizer_path)
        
def one_stage(rob, stage):
    episode = 0
    while episode < 5:
        print(f"episode: {episode+1}")
        one_episode(rob, stage, episode)
        episode+=1
        
    #save the model
    critic_path = RESULT_DIR / f"model_stage_{stage}_critic"
    torch.save(policy.critic.state_dict(), critic_path)
    critic_optimizer_path = RESULT_DIR / f"model_stage_{stage}_critic_optimizer"
    torch.save(policy.critic_optimizer.state_dict(), critic_optimizer_path)

    actor_path = RESULT_DIR / f"model_stage_{stage}_actor"
    torch.save(policy.actor.state_dict(), actor_path)
    actor_optimizer_path = RESULT_DIR / f"model_stage_{stage}_actor_optimizer"
    torch.save(policy.actor_optimizer.state_dict(), actor_optimizer_path)
    

def train_model(rob):
    #one_episode_initial(rob) #collect initial data
    
    stage = 0
    while stage < 1000:
        one_stage(rob, stage+1)
        stage += 1
    
def run_all_actions(rob: IRobobo):
    train_model(rob)
    
