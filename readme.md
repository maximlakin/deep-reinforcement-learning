# Sources
* [FlankMe general-gym-player](https://github.com/FlankMe/general-gym-player/blob/master/GeneralGymPlayerWithTF.py)
* [John Schulman's Deep Reinforcement Learning Lectures](https://www.youtube.com/watch?v=aUrX-rP_ss4)
* [David Silver's Reinforcement Learning Lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0)
* [UC Berkeley Intro to AI](http://ai.berkeley.edu/home.html)

# Overview
Optimization problems are all around, especially in business operations: inventory management, routing and scheduling deliveries, resource allocation etc.  Reinforcement learning is a useful technique in solving these problems autonomously.

In this project I used reinforcement learning to teach autonomous agents how to navigate in open environments because because it’s a great example of a dynamic problem with a complex parameter space that can’t be solved using a search algorithm.

# Implementation
Reinforcement learning is based on a Markov Decision Process where transition probabilities are unknown.  This solution learns a Q function by training a deep neural network and uses it to predict the right actions by constantly improving it's policy based on new observations.

To train, the agent first does a few runs to collect learning data and then uses mini-batches gradient descent.  The hyper parameters here are the discount factor (to control iteration depth), epsilon (to control exploration), and replay set size.

After running the algorithm on a simple stick balancing problem, I was able to successfully run it on a much more complicated Lander problem without knowing anything about the details of these new actions and observations

# Conclusion
Navigation is only one example of the many types of optimization problems that can be solved by reinforcement learning and I’m excited to keep exploring many more applications and techniques.  Thanks for watching, and feel free to get in touch if you’d like to chat about these results in more detail.
