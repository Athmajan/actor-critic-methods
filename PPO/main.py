import gymnasium as gym
import numpy as np
import wandb
from ppo import Agent

if __name__ == "__main__":
    wandb.init(project="ActorCriticMethods",name="ppo")
    env = gym.make("CartPole-v1")
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n,
                  batch_size=batch_size,
                  alpha=alpha,
                  n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 300

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            n_steps += 1
            done = terminated or truncated
            score += reward
            agent.remember(observation,action,prob,val,reward,done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters +=1 
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f"Episode {i} ended with score {avg_score} with steps {n_steps}.learning steps {learn_iters}")
        wandb.log({'Reward':avg_score},step=i) 

