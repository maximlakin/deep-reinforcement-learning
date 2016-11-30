import gym, numpy as np, tensorflow as tf

def main():
    # env = gym.make('CartPole-v0')
    env = gym.make('LunarLander-v2')
    env.monitor.start('LunarLander-v2', force=True)
    with open("lander-log.csv",'a') as f:
        f.write(",".join(["episode","episode_reward","batch_reward","epsilon"])+"\n")

    agent = Agnet(env)

    reward, done = 0.0, False

    for i in range(1,530):
        if i == 25: print("****STARTING TRAINGING****")
        observation, done = env.reset(), False
        action = agent.act(observation, reward, done, i)
        while not done:
            observation, reward, done, info = env.step(action)
            action = agent.act(observation, reward, done, i)

    env.close()
    env.monitor.close()

class Agnet():

    def __init__(self, env):
        obs_space = env.observation_space.shape[0]
        self.total_reward = 0.0
        self.env = env
        self.nn = NeuralNet([obs_space, 25*obs_space, 25*obs_space, env.action_space.n])
        self._nn = self.nn
        self.eps = 1.0
        self.eps_decay = 0.98
        self.eps_min = 0.1
        self.previous_observations = []
        self.last_action = np.zeros(env.action_space.n)
        self.last_state = None
        self.batch_size = 128
        self.list_rewards = []


    def act(self, obs, reward, done, ep):
        self.total_reward+=reward

        if done:
            self.list_rewards.append(self.total_reward)
            self.eps = max(self.eps*self.eps_decay,self.eps_min)
            wr = [ep,self.total_reward,np.mean(self.list_rewards[-100:]),self.eps]
            print(wr)
            with open("lander-log.csv",'a') as f:
                f.write(",".join([str(w) for w in wr])+"\n")
            self.total_reward = 0

        current_state = obs.reshape((1, len(obs)))

        if self.last_state is None:
            self.last_state = current_state
            q = self.nn.predict(current_state)
            next_action = np.argmax(q)
            self.last_action = np.zeros(self.env.action_space.n)
            self.last_action[next_action] = 1
            return next_action

        new_observation = [self.last_state.copy(), self.last_action.copy(), reward, current_state.copy(), done]
        self.previous_observations.append(new_observation)
        self.last_state = current_state.copy()
        while len(self.previous_observations)>100000: self.previous_observations.pop(0)

        if ep > 25:
            self.train()
            next_action = np.argmax(self.nn.predict(self.last_state)) if np.random.random() > self.eps else np.random.randint(0,self.env.action_space.n)
        else:
            next_action = np.random.randint(0,self.env.action_space.n)

        self.last_action = np.zeros(self.env.action_space.n)
        self.last_action[next_action]=1
        return next_action

    def train(self):
        batch = np.random.permutation(len(self.previous_observations))[:self.batch_size]
        states = np.concatenate([self.previous_observations[i][0] for i in batch],axis=0)
        actions = np.concatenate([[self.previous_observations[i][1]] for i in batch],axis=0)
        rewards = np.array([self.previous_observations[i][2] for i in batch]).astype("float")
        current_states = np.concatenate([self.previous_observations[i][3] for i in batch],axis=0)
        done = np.array([self.previous_observations[i][4] for i in batch]).astype("bool")
        target = rewards.copy()+(1.0-done)*0.99*(self.nn.predict(current_states).max(axis=1))
        self.nn.fit(states,actions,target)

class NeuralNet():

    def __init__(self, layers):
        self.layers = layers
        self.generate()

    def generate(self):
        self.activation = lambda x : tf.maximum(0.01*x, x)
        self.session = tf.Session()
        self.input_layer = tf.placeholder("float", [None, self.layers[0]])
        self.hidden_layer = []
        self.ff_weights = []
        self.ff_bias = []

        for i in range(len(self.layers[:-1])):
            self.ff_weights.append(tf.Variable(tf.truncated_normal([self.layers[i],self.layers[i+1]], mean=0.0, stddev=0.1)))
            self.ff_bias.append(tf.Variable(tf.constant(-0.01, shape = [self.layers[i+1]])))
            if i==0:
                activation = self.input_layer
            else:
                activation = self.activation(tf.matmul(self.hidden_layer[i-1],self.ff_weights[i-1])+self.ff_bias[i-1])
            self.hidden_layer.append(activation)

        self.state_value_layer = tf.matmul(self.hidden_layer[-1], self.ff_weights[-1])+self.ff_bias[-1]
        self.actions = tf.placeholder("float", [None,self.layers[-1]])
        self.target = tf.placeholder("float", [None])
        self.action_value_vector = tf.reduce_sum(tf.mul(self.state_value_layer,self.actions),1)
        self.cost = tf.reduce_sum(tf.square(self.target - self.action_value_vector))
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

        self.session.run(tf.initialize_all_variables())
        self.feed_forward = lambda state: self.session.run(self.state_value_layer, feed_dict={self.input_layer: state})
        self.back_prop = lambda states, actions, target: self.session.run(
            self.optimizer,
            feed_dict={
                self.input_layer: states,
                self.actions: actions,
                self.target: target
            })


    def fit(self, states, actions, target):
        self.back_prop(states, actions, target)

    def predict(self, state):
        return self.feed_forward(state)

if __name__=="__main__":
   main()
