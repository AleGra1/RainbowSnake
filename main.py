import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import matplotlib.pyplot as plt
from categorical_dqn import Agent
import gym_snake


def plot_learning_curve(xs, scores, eps_history, avg_scores, filename):
    fig, host = plt.subplots(figsize=(8, 5))
    par1 = host.twinx()

    host.set_xlim(0, len(xs))
    host.set_ylim(min(avg_scores)-1, max(avg_scores)+1)
    par1.set_ylim(-0.05, 1.05)

    host.set_xlabel("Steps")
    host.set_ylabel("Score")
    par1.set_ylabel("Epsilon")

    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)

    #p1 = host.scatter(xs, scores, color=color1, label="Score")
    p1, = host.plot(xs, avg_scores, color=color1,
                    label="Avg Score last 100 games")
    p2, = par1.plot(xs, eps_history, color=color2, label="Epsilon")

    lns = [p1, p2]
    host.legend(handles=lns, loc='best')

    # host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    fig.tight_layout()
    plt.savefig(filename)


def build_model(model_name='gym_snake/Snake-v0', width=5, height=5, fps=10, gamma=0.99, epsilon=1.0, lr=5e-4, n_actions=5, mem_size=1000000, eps_min=0.01, batch_size=64, eps_dec=1e-3, tau=100, plot_filename='snake-DDQN.png', num_episodes=500, load_checkpoint=False):
    if model_name == 'gym_snake/Snake-v0':
        env = FlattenObservation(gym.make(model_name, render_mode=None, width=width,
                                          height=height, window_width=500, window_height=500, fps=fps))
    else:
        env = gym.make(model_name)

    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, input_dims=env.observation_space.shape,
                  n_actions=n_actions, mem_size=mem_size, eps_min=eps_min, batch_size=batch_size, eps_dec=eps_dec, tau=tau)

    if load_checkpoint:
        agent.load_models()

    scores, eps_history, avg_scores = [], [], []

    for i in range(num_episodes):
        done = False
        observation, _ = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, _ = env.step(action)

            score += reward
            agent.store_transition(observation, action,
                                   observation_, reward, done)

            agent.learn()
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        print("Episode", i, "score %.2f" % score, "avg_score %.2f" %
              avg_score, "epsilon %.2f" % agent.epsilon)
        if i >= 10 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)
    xs = np.arange(num_episodes)
    plot_learning_curve(xs, scores, eps_history, avg_scores, plot_filename)
    
def build_model_prb(model_name='gym_snake/Snake-v0', width=5, height=5, fps=10, gamma=0.99, epsilon=1.0, lr=5e-4, n_actions=5, mem_size=1000000, eps_min=0.01, batch_size=64, eps_dec=1e-3, tau=100, plot_filename='snake-DDQN.png', num_episodes=500, load_checkpoint=False):
    if model_name == 'gym_snake/Snake-v0':
        env = FlattenObservation(gym.make(model_name, render_mode=None, width=width,
                                          height=height, window_width=500, window_height=500, fps=fps))
    else:
        env = gym.make(model_name)

    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, input_dims=env.observation_space.shape,
                  n_actions=n_actions, mem_size=mem_size, eps_min=eps_min, batch_size=batch_size, eps_dec=eps_dec, tau=tau)

    if load_checkpoint:
        agent.load_models()

    scores, eps_history, avg_scores = [], [], []

    for i in range(num_episodes):
        done = False
        observation, _ = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, _ = env.step(action)

            score += reward
            agent.store_transition(observation, action,
                                   observation_, reward, done)

            agent.learn((i+1)/num_episodes)
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        print("Episode", i, "score %.2f" % score, "avg_score %.2f" %
              avg_score, "epsilon %.2f" % agent.epsilon)
        if i >= 10 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)
    xs = np.arange(num_episodes)
    plot_learning_curve(xs, scores, eps_history, avg_scores, plot_filename)


def demo_model(model_name='gym_snake/Snake-v0', width=5, height=5, fps=10, gamma=0.99, epsilon=1.0, lr=5e-4, n_actions=5, mem_size=1000000, eps_min=0.01, batch_size=64, eps_dec=1e-3, tau=100, num_episodes=500):
    if model_name == 'gym_snake/Snake-v0':
        env = FlattenObservation(gym.make(model_name, render_mode="human", width=width,
                                          height=height, window_width=500, window_height=500, fps=fps))
    else:
        env = gym.make(model_name, render_mode="human")

    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, input_dims=env.observation_space.shape,
                  n_actions=n_actions, mem_size=mem_size, eps_min=eps_min, batch_size=batch_size, eps_dec=eps_dec, tau=tau)
    agent.load_models()
    agent.q_eval.eval()
    agent.q_next.eval()

    for _ in range(num_episodes):
        done = False
        observation, _ = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, _, _ = env.step(action)
            score += reward
        print(score)


if __name__ == '__main__':
    do_training = True
    show_demo = False
    load_checkpoint = False
    model_name = 'gym_snake/Snake-v0'
    width = 5
    height = 5
    fps = 5
    gamma = 0.99
    epsilon = 1.0
    lr = 5e-4
    n_actions = 4
    mem_size = 1000000
    eps_min = 0.01
    batch_size = 64
    eps_dec = 1e-3
    tau = 1000
    plot_filename = 'snake-DDQN.png'
    num_episodes = 3000
    showcase_episodes = 10

    if load_checkpoint:
        epsilon = eps_min

    if do_training:
        build_model(model_name=model_name, width=width, height=height, fps=fps, gamma=gamma, epsilon=epsilon, lr=lr, n_actions=n_actions, mem_size=mem_size,
                    eps_min=eps_min, batch_size=batch_size, eps_dec=eps_dec, tau=tau, plot_filename=plot_filename, num_episodes=num_episodes, load_checkpoint=load_checkpoint)

    if show_demo:
        demo_model(model_name=model_name, width=width, height=height, fps=fps, gamma=gamma, epsilon=0.0, lr=lr, n_actions=n_actions,
                   mem_size=mem_size, eps_min=eps_min, batch_size=batch_size, eps_dec=eps_dec, tau=tau, num_episodes=showcase_episodes)
