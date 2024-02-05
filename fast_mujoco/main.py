import numpy as np
import myenv  # Assuming 'myenv' is the compiled Cython module
import os
import time

cwd = os.getcwd()
print("Current Working Directory:", cwd)

from asset_components import create_ant_model

xml_string, leg_info = create_ant_model()

file_path = "xml_world.xml"  # Define the file path 

# Writing the XML string to the file
with open(file_path, "w") as file:
    file.write(xml_string)

# Initialize the environment
# The parameters would depend on how you've set up your Cython wrapper
environment = myenv.PyEnvironment(model_file_path="/Users/andrewgordienko/Documents/fib/xml_world.xml", leg_info=[[1, 2, 3] for _ in range(9)], max_steps=1000, num_creatures=9)

# Reset the environment at the start
environment.reset()

# Example simulation loop
for step in range(10000):  # Run for 100 steps as an example
    environment.render_environment()

    # Generate a random action for each creature
    actions = np.random.rand(9, environment.getActionSize())
    
    # Step through the environment with the generated actions
    done, observation, reward = environment.step(actions)
    
    #print(f"Step: {step}, Observation {observation}, Reward: {reward}")
    
    if done:
        print("Resetting environment.")
        environment.reset()
    
# Additional code to clean up or analyze the simulation results
