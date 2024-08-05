import gymnasium as gym
import numpy as np
from actor_critic import Agent

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1800


    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            if not load_checkpoint:
                agent.learn(observation,reward,observation_,done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print(f"Episode {i} done with avergae score {avg_score}")

    # plot learning curves here