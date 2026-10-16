import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from math import *

# baseline settings
alpha=0.1
gamma=0.99
epsilon=1.0
epsilon_min=0.01
epsilonDecay=0.995
total_episodes=5000
seed=0

env_id="Taxi-v4"

# deterministic taxi
env_kwargs={"is_rainy":False,"fickle_passenger":False}

window=100

def movavg(x,w=window):
    return np.convolve(x,np.ones(w)/w,mode="valid")

def run_q_learning(alpha,gamma,epsilon,epsilon_min,epsilonDecay,total_episodes,seed,verbose=True):
    np.random.seed(seed)
    env=gym.make(env_id,**env_kwargs)
    nStates=env.observation_space.n
    nActions=env.action_space.n
    Q=np.zeros((nStates,nActions))

    rewards,stepsPerEp,success=[],[],[]

    for ep in range(1,total_episodes+1):
        s,_=env.reset(seed=seed+ep)
        totalReward=0; steps=0
        done=trunc=False

        while not (done or trunc):
            # epsilon greedy, break ties randomly so it doesn't always pick action 0 early on
            if np.random.rand()<epsilon:
                a=np.random.randint(nActions)
            else:
                best=np.flatnonzero(Q[s]==Q[s].max())
                a=np.random.choice(best)

            s2,r,done,trunc,_=env.step(int(a))

            if done: target=r
            else: target=r+gamma*np.max(Q[s2])
            Q[s,a]+=alpha*(target-Q[s,a])

            s=s2
            totalReward+=r
            steps+=1

        epsilon=max(epsilon_min,epsilon*epsilonDecay)

        rewards.append(totalReward)
        stepsPerEp.append(steps)
        success.append(done)

        if verbose==True and ep%500==0:
            print("ep %d  avg reward (last 100) = %.2f  eps = %.3f" % (ep,np.mean(rewards[-100:]),epsilon))

    env.close()
    return Q,np.array(rewards),np.array(stepsPerEp),np.array(success)

def evalPolicy(Q,n_episodes=100,eval_seed=10000):
    # greedy only, no exploration
    env=gym.make(env_id,**env_kwargs)
    rewards,Steps,wins=[],[],[]
    for i in range(n_episodes):
        s,_=env.reset(seed=eval_seed+i)
        tot,n=0,0
        done=trunc=False
        while not (done or trunc):
            s,r,done,trunc,_=env.step(int(np.argmax(Q[s])))
            tot+=r; n+=1
        rewards.append(tot); Steps.append(n); wins.append(done)
    env.close()
    return np.array(rewards),np.array(Steps),np.array(wins)

def first_positive(rLog):
    # first episode where the 100 ep moving avg gets to >= 0
    idx=np.flatnonzero(movavg(rLog)>=0)
    if len(idx)==0: return "never"
    return idx[0]+window

Q,reward_log,stepsLog,success_log=run_q_learning(alpha,gamma,epsilon,epsilon_min,epsilonDecay,total_episodes,seed)

print("\ndone training")
print("success rate, last 100 eps: %d%%" % round(success_log[-100:].mean()*100))

# ---- Part C ----
os.makedirs("results",exist_ok=True)

epsEp=int(ceil(log(epsilon_min/epsilon)/log(epsilonDecay)))
x_all=np.arange(1,total_episodes+1)

plt.figure(figsize=(10,5))
plt.plot(x_all,reward_log,color="lightsteelblue",lw=0.5,label="reward per episode")
plt.plot(x_all[window-1:],movavg(reward_log),color="navy",lw=2,label="100 ep moving avg")
plt.axvline(epsEp,color="gray",ls="--",label="ε = {} at ep {}".format(epsilon_min,epsEp))
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Baseline Q-learning, Taxi-v4 (α=%s, γ=%s, decay=%s)" % (alpha,gamma,epsilonDecay))
plt.legend(loc="lower right")
plt.grid(True,ls=":",alpha=0.6)
plt.tight_layout()
plt.savefig("results/baseline_rewards.png",dpi=150)

# ---- Part D: alpha ----
alphaResults={}
for a in [0.01,0.1,0.5,0.9]:
    print("\nalpha =",a)
    _,r,st,sc=run_q_learning(a,gamma,epsilon,epsilon_min,epsilonDecay,total_episodes,seed,verbose=False)
    alphaResults[a]=(r,st,sc)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
x=np.arange(window,total_episodes+1)
for a,(r,st,sc) in alphaResults.items():
    ax1.plot(x,movavg(r),lw=2,label="α = %s" % a)
    ax2.plot(x,movavg(r),lw=2,label="α = %s" % a)
ax1.set_title("Learning rate comparison (100 ep moving avg)")
ax2.set_title("Zoomed in")
ax2.set_ylim(-20,12)
for ax in (ax1,ax2):
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.grid(True,ls=":",alpha=0.6)
    ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/alpha_comparison.png",dpi=150)

print("\nalpha, eps to avg>=0, final mean, final std, success, steps  (final = last 1000 eps)")
for a,(r,st,sc) in alphaResults.items():
    print("%s  %s  %.2f  %.2f  %d%%  %.1f" % (a,first_positive(r),r[-1000:].mean(),r[-1000:].std(),round(sc[-1000:].mean()*100),st[-1000:].mean()))

# ---- Part E: gamma ----
gamma_results={}
for g in [0.1,0.5,0.9,0.99]:
    print("\ngamma =",g)
    _,r,st,sc=run_q_learning(alpha,g,epsilon,epsilon_min,epsilonDecay,total_episodes,seed,verbose=False)
    gamma_results[g]=(r,st,sc)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
x=np.arange(window,total_episodes+1)
for g,(r,st,sc) in gamma_results.items():
    ax1.plot(x,movavg(r),lw=2,label="γ = %s" % g)
    ax2.plot(x,movavg(r),lw=2,label="γ = %s" % g)
ax1.set_title("Discount factor comparison (100 ep moving avg)")
ax2.set_title("Zoomed in")
ax2.set_ylim(-20,12)
for ax in (ax1,ax2):
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.grid(True,ls=":",alpha=0.6)
    ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/gamma_comparison.png",dpi=150)

print("\ngamma, eps to avg>=0, final mean, final std, success, steps")
for g,(r,st,sc) in gamma_results.items():
    print("%s  %s  %.2f  %.2f  %d%%  %.1f" % (g,first_positive(r),r[-1000:].mean(),r[-1000:].std(),round(sc[-1000:].mean()*100),st[-1000:].mean()))

# ---- Part F: epsilon decay ----
DecayResults={}
for d in [0.9,0.99,0.995,0.999]:
    print("\ndecay =",d)
    Qd,r,_,_=run_q_learning(alpha,gamma,epsilon,epsilon_min,d,total_episodes,seed,verbose=False)
    DecayResults[d]=(r,*evalPolicy(Qd))

plt.figure(figsize=(10,5))
x=np.arange(window,total_episodes+1)
for d,(r,er,es,esc) in DecayResults.items():
    fl=int(ceil(log(epsilon_min/epsilon)/log(d)))
    plt.plot(x,movavg(r),lw=2,label="decay {} (ε min at ep {})".format(d,fl))
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("ε decay comparison (100 ep moving avg)")
plt.grid(True,ls=":",alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/epsilon_decay_comparison.png",dpi=150)

print("\ndecay, eps to avg>=0, final train mean, greedy mean, greedy success, greedy steps")
for d,(r,er,es,esc) in DecayResults.items():
    print("%s  %s  %.2f  %.2f  %d%%  %.1f" % (d,first_positive(r),r[-1000:].mean(),er.mean(),round(esc.mean()*100),es.mean()))

# ---- Part G ----
bestAlpha,best_gamma,bestDecay=0.5,0.99,0.99
nEval=100

bestQ,best_rewards,_,_=run_q_learning(bestAlpha,best_gamma,epsilon,epsilon_min,bestDecay,total_episodes,seed,verbose=False)
np.save("results/best_q_table.npy",bestQ)

def printEval(name,res):
    r,s,sc=res
    print("%-9s mean %.2f (std %.2f)  min %.0f  max %.0f  success %d%%  steps %.1f" % (name,r.mean(),r.std(),r.min(),r.max(),round(sc.mean()*100),s.mean()))

print("\nepisodes until moving avg >= 0 -> baseline: {}  best: {}".format(first_positive(reward_log),first_positive(best_rewards)))

print("\ngreedy eval (%d eps) after full %d episode training" % (nEval,total_episodes))
printEval("baseline",evalPolicy(Q,n_episodes=nEval))
printEval("best",evalPolicy(bestQ,n_episodes=nEval))

# how do they compare with less training
for budget in (500,1000):
    Qb,_,_,_=run_q_learning(alpha,gamma,epsilon,epsilon_min,epsilonDecay,budget,seed,verbose=False)
    Qt,_,_,_=run_q_learning(bestAlpha,best_gamma,epsilon,epsilon_min,bestDecay,budget,seed,verbose=False)
    print("\nafter only %d episodes:" % budget)
    printEval("baseline",evalPolicy(Qb,n_episodes=nEval))
    printEval("best",evalPolicy(Qt,n_episodes=nEval))

plt.show()