# Offline Reinforcement Learning
 Implemented Logistic Regression and fitted Q-Learning for Open AI Cartpole environment.

## Offline reinforcement learning (RL) 

Offline reinforcement learning is a machine learning approach where an agent learns to make decisions and improve its behavior using a fixed dataset of pre-recorded experiences, without interacting with the environment in real time. In contrast to traditional online reinforcement learning, where the agent learns by actively interacting with the environment, i.e., receiving feedback (rewards or penalties), and updating its policy based on these interactions, in offline RL learns from a dataset that was collected separately by some other means, such as from human demonstrations, expert agent or previously saved interactions. 

Offline RL is particularly beneficial in situations where online interactions with the environment are costly, dangerous, or impractical. By using a fixed dataset, the agent can learn from historical data without the need for additional interactions with the environment, which would save time and resources.  

## Environment 

The "CartPole" environment is another classic reinforcement learning environment provided by the OpenAI Gym toolkit. It simulates a simple physics problem where an inverted pendulum (pole) is attached to a moving cart. The goal of the agent is to balance the pole on the cart by applying appropriate forces to the cart. 

## Dataset 
For this assignment we have collected data from three resources. 

- **Expert Agent:** For Expert Agent we Q-learning algorithm with alpha = 1/8. In Q-learning , the Q-function estimates the expected cumulative reward an agent can achieve by taking a specific action in a given state and then following the optimal policy from that point onward. 

- **Random Selection:** Here we selected random action using binomial distribution 

- **Half Expert and Half Random Selection:** Here we collected odd no action using expert agent and even no. action using random agent  

## Imitation learning 

Imitation learning is a type of machine learning technique where an agent learns a task by imitating demonstrations of the task performed by an expert. Instead of explicitly specifying a reward function or trying to explore the environment on its own (trying to figure out the task from scratch), the agent learns from examples provided by a human or an expert policy.  

The goal of imitation learning is to create a model that can replicate the expert's actions based on the expert's demonstrations or examples. For this we project have used logistic regression to train a statistical model to mimic the expert's decisions based on the input data. This approach is also known as learning from demonstration (LfD) and behavioral cloning. 

Logistic regression is a supervised linear classification learning algorithm that is used to predict the probability that an instance belongs to a particular class, given its features (input variables). In context of reinforcement learning (RL), logistic regression can be used as a function approximator to estimate action probabilities in certain scenarios; The logistic regression model maps the observed states to the probabilities of taking each possible action. In imitation learning, the dataset consists of pairs of states and corresponding expert actions, which are used to train the logistic regression model. 

## Fitted q learning  

Fitted Q-learning improves the efficiency of the Q-learning algorithm by fitting a function approximator to estimate the Q-values of state-action pairs, rather than maintaining a tabular representation of the Q-values. the core idea is to fit a Q-function approximator to a dataset of experiences (state-action-next state-reward tuples) collected from the environment. This enables Q-learning to scale better to high-dimensional and continuous state spaces. 

 I have set hyperparameter K to 5. I observed that the Q-function values drift very quickly when k is more than 5 

## Performance  

### Policy  
I used the greedy policy to select the action to evaluate the performance to two models on three datasets. 

### Experiment
I, first, define the discrete state for the cart pole environment to create Q-function value table for each of the two agents. Then, I pre-trained the two agents with 80,000 samples (Independent episodes) and update their Q-table. I use these Q-value tables to collect observation datasets of sizes 100, 250 and 500. I use these datasets to train Imitation Learning (Logistic Regression) and Fitted Q-Learning estimator, with two different learning rates. I then run Greedy Algorithm for 100 episodes and collect the returns. I plot these returns to visualize their performance. 

### Result

#### when dataset size is 100 and learning rate is 1/8 

Fitted Q-learning performs slightly better than Imitation Learning, when trained with Expert policy dataset and Half policy dataset. Fitted Q-learning trained with Expert and half policy dataset, and Imitation Learning perform trained with Expert policy dataset have a high average episode returns, Fitted Q-learning trained with Expert agent and Imitation Learning perform trained with Expert have higher returns than expert policy. Both of the algorithms perform very poorly when trained with Random policy dataset. 

#### when dataset size is 100 and learning rate is 1/16 

Again, Fitted Q-learning performs slightly better than Imitation Learning, when trained with Expert policy dataset and Half policy dataset. Fitted Q-learning trained with Expert and half policy dataset, and Imitation Learning perform trained with Expert and Half policy dataset have a high average episode returns, Fitted Q-learning trained with Expert agent and Imitation Learning perform trained with Expert have equal or higher average returns than expert policy. Both of the algorithms perform very poorly when trained with Random policy dataset. 

#### Impact of Learning rate 

The learning rate does have any significant changes in the performance of the two algorithms trained under different datasets; the performance of Imitation learning trained with Half policy dataset is higher with the learning rate is 0.001 than 0.01 

#### when dataset size is 250 and learning rate is 1/8 

Fitted Q-learning performs slightly better than Imitation Learning, when trained with Expert policy dataset and Half policy dataset. Fitted Q-learning trained with Expert and half policy dataset, and Imitation Learning perform trained with Expert and Half policy dataset have a high average episode returns, Fitted Q-learning trained with Expert agent and Imitation Learning perform trained with Expert have equal or higher average returns than expert policy. Both of the algorithms perform very poorly when trained with Random policy dataset. 

#### when dataset size is 250 and learning rate is 1/16 

Again, Fitted Q-learning performs slightly better than Imitation Learning, when trained with Expert policy dataset and Half policy dataset. Fitted Q-learning trained with Expert and half policy dataset, and Imitation Learning perform trained with Expert and Half policy dataset have a high average episode returns, Fitted Q-learning trained with Expert agent and Imitation Learning perform trained with Expert have equal or higher average returns than expert policy. Both of the algorithms perform very poorly when trained with Random policy dataset. 

#### Impact of Learning rate 

The change in the learning rate does not make any significant changes in the performance of the algorithm under different datasets. 

#### when dataset size is 500 and learning rate is 1/8 

Fitted Q-learning trained with Expert and half policy dataset, and Imitation Learning perform trained with Half policy dataset have a high average episode returns, Fitted Q-learning trained with Expert policy dataset have higher average returns than expert agent. Both of the algorithms perform very poorly when trained with Random policy dataset. 

#### when dataset size is 500 and learning rate is 1/16 

Fitted Q-learning trained with Expert and Half policy dataset, and Imitation Learning perform trained with Expert and Half policy dataset have a high average episode returns, Fitted Q-learning trained with Expert policy dataset and Imitation Learning perform trained with Expert policy dataset have higher average returns than expert agent. Both of the algorithms perform very poorly when trained with Random policy dataset. 

#### Impact of Learning rate 

The learning rate does have any significant changes in the performance of the two algorithms trained under different datasets; the performance of Fitted Q-learning trained with Half policy dataset is higher with the learning rate is 0.001 than 0.01 

### Conclusion

The Fitted Q-Learning performs slightly better than the Imitation learning. It performs better when trained with the dataset collected with the Expert policy or the Half policy than the Random policy. 

### Impact of size

Increase in the size of dataset does have any significant changes in the performance of the two algorithms, however both algorithms (Imitation Learning and Fitted Q-learning) tends to perform better with a dataset trained with Expert policy as the size of dataset increases. 

 

 
