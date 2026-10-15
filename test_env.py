import gymnasium as gym

env = gym.make("Taxi-v4")
state, info = env.reset(seed=0)

print("Number of states:", env.observation_space.n)
print("Number of actions:", env.action_space.n)
print("Starting state:", state)

total_reward = 0
for step in range(20):
    action = env.action_space.sample()  # random action
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"Step {step}: action={action}, reward={reward}, next_state={state}")
    if terminated or truncated:
        break

print("Total reward from random actions:", total_reward)
env.close()