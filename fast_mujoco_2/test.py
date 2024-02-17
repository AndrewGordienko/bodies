import numpy as np
import myenv  # Assuming 'myenv' is the compiled Cython module
import os
from ppo_agent import Agent
from asset_components import create_ant_model

EPISODES = 100
NUM_CREATURES = 9  # Adjust based on your environment setup
ACTION_DIMS = 12  # Adjusted based on the new understanding

# Initialize the PPO agent with correct dimensions
# Assuming each creature's observation is split into 38 parts
cwd = os.getcwd()

# Correctly setting the model file path for the initial environment setup
model_file_path = os.path.join(cwd, "xml_world_episode_0.xml")
print(f"Checking XML file at: {model_file_path}")
assert os.path.exists(model_file_path), f"File does not exist: {model_file_path}"

env = myenv.PyEnvironment(model_file_path=model_file_path,
                          leg_info=[[1, 2, 3] for _ in range(NUM_CREATURES)],
                          max_steps=1000,
                          num_creatures=NUM_CREATURES)

agent = Agent(n_actions=ACTION_DIMS, input_dims=[38], env=env)  

cwd = os.getcwd()
print("Current Working Directory:", cwd)

for episode in range(EPISODES):
    xml_string, leg_info = create_ant_model()
    file_path = os.path.join(cwd, f"xml_world_episode_{episode}.xml")

    with open(file_path, "w") as file:
        file.write(xml_string)

    # Use the same leg_info structure that worked previously
    environment = myenv.PyEnvironment(model_file_path=file_path,
                                      leg_info=[[1, 2, 3] for _ in range(NUM_CREATURES)],  # Same as before
                                      max_steps=500,
                                      num_creatures=NUM_CREATURES)
    environment.reset()  # Reset the environment to start a new episode

    score = 0
    steps = 0
    done = False

    actions = np.random.rand(108).reshape(1, 108)
    print(actions.shape)
    done, observation, reward = environment.step(actions)

    while not done:
        environment.render_environment()

        actions = np.random.rand(108).reshape(1, 108)
        print(actions.shape)
        done, observation, reward = environment.step(actions)

        """
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
    
        for creature_id in range(9):
            # Extract the observation for the current creature
            start_idx = creature_id * 38  # Each creature has 38 observations
            end_idx = start_idx + 38
            _observation = observation[start_idx:end_idx]

            action, log_prob, value = agent.choose_action(_observation)
            
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        

        combined_actions = np.concatenate(actions)
        done, observation, reward  = environment.step(combined_actions.astype('float'))
        
        for creature_id in range(9):
            _reward = reward[creature_id]
            score += reward

            # print("learning")
            # print(actions[creature_id], log_probs[creature_id], values[creature_id], _reward, done)
            _observation = observation[creature_id * 38:(creature_id + 1) * 38]
            agent.remember(_observation, actions[creature_id], log_probs[creature_id], values[creature_id], _reward, done)

        if steps % 20 == 0:
            agent.learn()
        
        steps += 1
        """
        if done:
            break
    
    print(episode)

    print(f'Episode: {episode+1}, Score: {score}')
