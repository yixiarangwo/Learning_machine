import cv2
from pathlib import Path
import numpy as np
import pickle
import time

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

# Load the Q-table from the file
def load_Q_table():
    global Q_table
    filename = RESULT_DIR / "Q_table.pkl"
    if filename.exists():
        with open(filename, 'rb') as f:
            Q_table = pickle.load(f)

# Define the number of states and actions
num_states = 5
action_dim = 6

# Function to choose an action based on the Q-table
def choose_action(state):
    return np.argmax(Q_table[state])

# Function to execute the action
def take_action(rob, action):
    if action == 0:
        rob.move_blocking(left_speed=20, right_speed=20, millis=300)  # Slowly moving forward
    elif action == 1:
        rob.move_blocking(left_speed=30, right_speed=30, millis=500)  # Fast forward
    elif action == 2:
        rob.move_blocking(left_speed=-20, right_speed=-20, millis=500) 
        rob.move_blocking(left_speed=-10, right_speed=20, millis=500)  # Turn left
    elif action == 3:
        rob.move_blocking(left_speed=-20, right_speed=-20, millis=500) 
        rob.move_blocking(left_speed=20, right_speed=-10, millis=500)  # Turn right
    elif action == 4:
        rob.move_blocking(left_speed=-20, right_speed=-20, millis=300)
        rob.move_blocking(left_speed=-10, right_speed=20, millis=300)  # Slowly moving backward
    elif action == 5:
        rob.move_blocking(left_speed=-20, right_speed=-20, millis=300)
        rob.move_blocking(left_speed=20, right_speed=-10, millis=300)  # Fast backward

# Function to define the state based on sensor data
def get_state(rob):
    IRS_data = rob.read_irs()
    FrontL_threshold = 200
    FrontLL_threshold = 200
    FrontC_threshold = 100
    FrontR_threshold = 200
    FrontRR_threshold = 200

    if (IRS_data[2] + IRS_data[7] + IRS_data[4]) > (FrontL_threshold + FrontLL_threshold + FrontC_threshold) or (IRS_data[3] + IRS_data[5] + IRS_data[4]) > (FrontR_threshold + FrontRR_threshold + FrontC_threshold):
        max_sensor = IRS_data.index(max(IRS_data))
        if max_sensor == 4:  # Obstacle at FrontC
            temp_IRS_data = IRS_data[:]
            temp_IRS_data[4] = float('-inf')
            max_sensor_2 = temp_IRS_data.index(max(temp_IRS_data))

            if max_sensor_2 == 3 or max_sensor_2 == 5:  # Obstacle on the right
                return 1  # State 1
            else:  # Obstacle on the left or directly in front
                return 2  # State 2
        elif max_sensor == 3 or max_sensor == 5:  # Obstacle on the right
            return 3  # State 1
        elif max_sensor == 2 or max_sensor == 7:  # Obstacle on the left
            return 4  # State 2
    elif (IRS_data[0] > FrontLL_threshold or IRS_data[6] > FrontC_threshold):  # Obstacle at back
        return 0  # State 3
    return 0  # State 0, no significant obstacle

# Check if the robot is spinning in place
def is_spin(last_position, current_position):
    if abs(current_position.x - last_position.x) < 0.05 and abs(current_position.y - last_position.y) < 0.05:
        print("spin!")
        return True
    return False

# Check if the robot is stuck
def is_stuck(last_position, current_position):
    if abs(current_position.x - last_position.x) < 0.01 and abs(current_position.y - last_position.y) < 0.01:
        print("stuck!")
        return True
    return False

# Calculate the distance moved by the robot
def get_move_distance(last_position, current_position):
    move_distance = (current_position.x - last_position.x) ** 2 + (current_position.y - last_position.y) ** 2
    move_distance = np.sqrt(move_distance)
    return move_distance

 

# Function to evaluate the policy
def eval_policy(rob):
    print("Evaluation start")

    avg_fitness = 0  

    for i in range(3):
        rob.play_simulation()
        state, done = get_state(rob), False
        move_index = 0
        n_stuck = 0
        n_spin = 0

        while not done:
            if move_index > 0:
                # only for reward calculation
                last_position = current_position
            else: 
                state= get_state(rob)
                # only for reward calculation
                last_position = rob.get_position()

            action = choose_action(state)
            take_action(rob, action)
            state = get_state(rob)
            current_position = rob.get_position()

            if is_stuck(last_position, current_position):
                n_stuck += 1
            elif is_spin(last_position, current_position):
                n_spin += 1
                
            move_index += 1
            if move_index >= 100:
                done = True

        rob.stop_simulation()      
        fitness = 100 - n_stuck * 10 - n_spin * 2
        avg_fitness += fitness

    avg_fitness /= 3  # Averaging over 3 runs
    print("Evaluation end")
    return avg_fitness


# Updated run_all_actions to include eval_policy
def run_all_actions(rob: IRobobo):
    load_Q_table()
    log_file = "evaluation_results.txt"
    
    for i in range(1, 25):  

        avg_fitness = eval_policy(rob)
        print(f"actor_{i}_avg_fitness: {avg_fitness}")

        with open(log_file, "a") as log:
            log_entry = f"actor {i} average fitness: {avg_fitness}\n"
            log.write(log_entry)
            log.flush()  # Ensure the log entry is written to the file immediately