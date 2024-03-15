the aim of the game is to figure out what the best network configuration is. I did this by running different network architectures on MNIST handwritten digits and graphing the accuracy and data samples needed over 10 runs for each configuration, to look at what network needs the least amount of samples to reach 98% accuracy and the trajectory of the accuracy graph to see how they learn.

![image](https://github.com/AndrewGordienko/bodies/assets/60705784/a5752174-ba74-40d9-9c1d-2f10543d0a4b)

![image](https://github.com/AndrewGordienko/bodies/assets/60705784/4799e5ce-0466-4258-8d10-47f196797cc5)

based on the data it appears that the architecture of input, 1024, 512, 256, learns the fastest and needs the least amount of data samples (300k) to reach 98% accuracy.
