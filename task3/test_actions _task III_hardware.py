﻿import cv2
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
tau = 0.005  # # Target network soft update rate 
policy_noise = 0.2  # Noise added to target policy during critic update
noise_clip = 0.5  # Range to clip target policy noise
policy_freq = 2  # Frequency of delayed policy updates for Actor network

state_dim = 17
action_dim = 2
max_action = 100 # torch.tanh range:(-1, 1) -> torch.tanh*max_action: (-100,100) for both left speed and right speed

policy = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
# replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=max_timesteps)    

# actor    
actor_path = RESULT_DIR / f"model_final_actor"
if actor_path.exists():
    policy.actor.load_state_dict(torch.load(actor_path))
    print("model_final_actor loaded")
    
    
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
    if v == 0 or math.log2(v) == 0: #reality gap: math.log2(v)==0 only in hardware
        return 0
    else:
        return 1 / math.log2(v)

def read_irs_data(rob):
    IRS_data = rob.read_irs()
    normalized_data = [normalize_sensor(v) for v in IRS_data]
    
    return torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0).to(device)

def image_features(rob: IRobobo):
    # Capture image from the front camera
    image = rob.get_image_front()
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color range for green
    green_mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    green_pixels = np.count_nonzero(green_mask)
    
    # Define color range for red
    red_mask = cv2.inRange(hsv, (78,158,124), (138,255,255))
    red_pixels = np.count_nonzero(red_mask)

    total_pixels = image.shape[0] * image.shape[1]

    # Calculate green and red proportion
    green_percentage = green_pixels / total_pixels
    red_percentage = red_pixels / total_pixels
    other_percentage = 1 - green_percentage - red_percentage

    # Get the coordinates of green pixels
    green_coords = np.column_stack(np.where(green_mask > 0))

    # Calculate the average coordinates for green pixels
    if green_pixels > 0:
        avg_green_x = np.mean(green_coords[:, 1]) / image.shape[1]
        avg_green_y = np.mean(green_coords[:, 0]) / image.shape[0]
    else:
        avg_green_x, avg_green_y = 0, 0
        
    # Get the coordinates of red pixels
    red_coords = np.column_stack(np.where(red_mask > 0))

    # Calculate the average coordinates for red pixels
    if red_pixels > 0:
        avg_red_x = np.mean(red_coords[:, 1]) / image.shape[1]
        avg_red_y = np.mean(red_coords[:, 0]) / image.shape[0]
    else:
        avg_red_x, avg_red_y = 0, 0
        
    # Calculate the average coordinates for other pixels
    avg_other_x = 1 - avg_green_x - avg_red_x
    avg_other_y = 1 - avg_green_y - avg_red_y
    
    # Return all features
    return green_percentage, red_percentage, other_percentage, avg_green_x, avg_green_y, avg_red_x, avg_red_y, avg_other_x, avg_other_y

def set_tilt(rob): #works in new camera environment
    while rob.read_phone_tilt()<80:
            rob.set_phone_tilt_blocking(tilt_position=80, tilt_speed=20)

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

def get_food_position(self) -> Position:
    self._food = self._get_object("/Food")
    
    if self._food is None:
        raise AttributeError("Scene does not have a food")

    pos = self._sim.getObjectPosition(self._food, self._sim.handle_world)
    return Position(*pos) 

def distance_food_base(rob): #arena_push_easy.ttt: Maximum distance not exceeding 2.5
    food_position = get_food_position(rob)
    base_postion = rob.base_position()
    distance = np.sqrt((food_position.x - base_postion.x)**2 + (food_position.y - base_postion.y)**2)
    
    return distance

def distance_food_robot(rob):
    food_position = get_food_position(rob)
    robot_postion = rob.get_position()
    distance = np.sqrt((food_position.x - robot_postion.x)**2 + (food_position.y - robot_postion.y)**2)
    
    return distance
    
def calculate_reward(rob, green_percentage, last_distance, current_distance, last_position, current_position, base_detects_food, left_speed, right_speed):
    speed_reward = 0
    green_reward = 0
    base_distance_reward = 0
    food_distance_reward = (3 - distance_food_robot(rob))*30
    spin_penalty = 0
    stuck_penalty = 0

    if distance_food_robot(rob) < 0.20:
        food_distance_reward += 50

    if left_speed > 0 and right_speed > 0:
        speed_reward = abs(left_speed)/10.0 + abs(right_speed)/10.0 
        
    if abs(last_distance - current_distance) > 0.01:
        base_distance_reward += (last_distance - current_distance)*100
        
    if base_distance_reward > 1 and left_speed > 0 and right_speed > 0 and green_percentage > 0:  
        speed_reward += abs(left_speed)/2.0 + abs(right_speed)/2.0 
        green_reward += green_percentage*100
        
    if is_stuck(last_position, current_position):
        stuck_penalty -= 100
        speed_reward *= -1    
    elif is_spin(last_position, current_position):
        spin_penalty -= 40
    
    print(f"base_distance_reward:{base_distance_reward}, distance_food_robot(rob):{distance_food_robot(rob)}")
    reward = base_distance_reward  + food_distance_reward + base_detects_food*100.0 + spin_penalty + stuck_penalty + green_reward + speed_reward
    
    scaled_reward = reward / 500.0 ##Scale the theoretical range (-120,500) to the range (-0.24, 1).
    return scaled_reward

def get_state(rob):
    IRS_data = read_irs_data(rob)
    green_percentage, red_percentage, other_percentage, avg_green_x, avg_green_y, avg_red_x, avg_red_y, avg_other_x, avg_other_y = image_features(rob)
    image_features_tensor = torch.tensor([green_percentage, red_percentage, other_percentage, avg_green_x, avg_green_y, avg_red_x, avg_red_y, avg_other_x, avg_other_y], dtype=torch.float32).unsqueeze(0).to(device)
    state_data = torch.cat((IRS_data, image_features_tensor), dim=1)
    state_data = state_data.unsqueeze(1)
    
    return state_data


def perform_policy(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        set_tilt(rob)
        speed_rate = 1
    else: 
        speed_rate = 0.8 #hardware
    
    state, done = get_state(rob), False
    while not done:
        action = policy.select_action(np.array(state))
        # Perform action
        left_speed = int(action[0]*speed_rate)
        right_speed = int(action[1]*speed_rate)
        rob.move_blocking(left_speed, right_speed, millis=400)
        state = get_state(rob)
        if isinstance(rob, SimulationRobobo) and rob.base_detects_food():
            done = True
        
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        
def run_all_actions(rob: IRobobo):
    perform_policy(rob)