import numpy as np
import myenv  # Assuming 'myenv' is the compiled Cython module
import os
from ppo_agent import Agent
from asset_components import create_ant_model
import matplotlib.pyplot as plt

EPISODES = 2
NUM_CREATURES = 9  # Adjust based on your environment setup
ACTION_DIMS = 12  # Adjusted based on the new understanding

# Initialize the PPO agent with correct dimensions
# Assuming each creature's observation is split into 38 parts
cwd = os.getcwd()

# Correctly setting the model file path for the initial environment setup
model_file_path = os.path.join(cwd, "xml_world_episode_0.xml")
#model_file_path = os.path.join(cwd, "soccer_environment.xml")

print(f"Checking XML file at: {model_file_path}")
assert os.path.exists(model_file_path), f"File does not exist: {model_file_path}"

env = myenv.PyEnvironment(model_file_path=model_file_path,
                          leg_info=[[1, 2, 3] for _ in range(NUM_CREATURES)],
                          max_steps=1000,
                          num_creatures=NUM_CREATURES)

agent = Agent(n_actions=ACTION_DIMS, input_dims=[41], env=env)  

cwd = os.getcwd()
print("Current Working Directory:", 2)

flag_starting_radius = 1  # Start with flags close to the creature
flag_increment = 0.5       # Increase radius by this amount as creatures improve
local_python_counter = 0
counter = 0

episodes_we_care = []
steps_we_care = []

for episode in range(0, 15):
    print(f"episode {episode}")
    xml_string, leg_info = create_ant_model(flag_starting_radius + flag_starting_radius * counter)
    print("next")
    file_path = os.path.join(cwd, f"xml_world_episode_0.xml")
    print("next again")
    with open(file_path, "w") as file:
        file.write(xml_string)

    print("next next again")
    # Load the new XML file into the environment
    env.load_new_model(file_path)
    #env.load_new_model(model_file_path)
    print("here")
    env.reset()

    score = 0
    steps = 0
    done = False

    actions = np.random.rand(9, 12)
    print(actions.shape)
    done, observation, reward = env.step(actions)

    while not done:
        env.render_environment()
        print(local_python_counter, env.get_hit_counter())

        if local_python_counter != env.get_hit_counter():
            local_python_counter += 1

            if local_python_counter > 5 ** counter:
                if counter <= 6:
                    counter += 1

                    episodes_we_care.append(episode)
                    steps_we_care.append(counter)
                    
                local_python_counter += 1
            

        #actions = np.random.rand(9, 12)
        #print(actions.shape)
        #done, observation, reward = env.step(actions)

        
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
    
        for creature_id in range(9):
            # Extract the observation for the current creature
            start_idx = creature_id * 38  # Each creature has 38 observations
            end_idx = start_idx + 38
            _observation = observation[creature_id]

            action, log_prob, value = agent.choose_action(_observation)
            
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        

        combined_actions = np.concatenate(actions)
        done, observation, reward  = env.step(combined_actions.astype('float'))
        
        for creature_id in range(9):
            _reward = reward[creature_id]
            score += reward

            # print("learning")
            # print(actions[creature_id], log_probs[creature_id], values[creature_id], _reward, done)
            _observation = observation[creature_id]
            agent.remember(_observation, actions[creature_id], log_probs[creature_id], values[creature_id], _reward, done)

        if steps % 10 == 0 and episode % 2 == 0:
            print(f"Learning at step {steps} for episode {episode}")
            agent.learn()
        if episode % 2 == 1:
            agent.memory.clear_memory()  
        steps += 1
        
        if done:
            break
    
    print(episode)

    print(f'Episode: {episode+1}, Total Score: {sum(score)}, Score: {score}')
    agent.save_models(episode, directory=cwd)
    agent.load_models(episode-1, directory=cwd)

plt.plot(episodes_we_care, steps_we_care)
plt.show()
