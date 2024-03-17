working on writing code to find the optimal set of hyperparameters for training.

the code works through cycling through all the possible combinations of these hyperparameters for twenty episodes:
hyperparameters = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [5, 10, 15],
    'gae_lambda': [0.9, 0.95],
    'entropy_beta': [0.01, 0.05, 0.1]
}

The next steps would be to reduce the noise. This should be done by making a window where after each configs run we smooth out the graph and plot it.
We should also make it so it runs for more than 20 episodes. It should be visible what the best configs are. 

the resulting is these graphs: 
![image](https://github.com/AndrewGordienko/bodies/assets/60705784/ce9c365c-a2d9-4994-892a-ea10394d13c0)

<img width="357" alt="image" src="https://github.com/AndrewGordienko/bodies/assets/60705784/2c601f87-aac5-468f-a800-68d2f9f7a8f5">

<img width="914" alt="image" src="https://github.com/AndrewGordienko/bodies/assets/60705784/02c2296b-a6d4-4fb4-9f8e-6729541b89ef">

