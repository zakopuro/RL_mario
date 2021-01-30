from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,RIGHT_ONLY
import time
from stable_baselines import PPO2,DQN
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from baselines.common.retro_wrappers import *
from stable_baselines.bench import Monitor
from util import CustomRewardAndDoneEnv,log_dir,CustomCallback
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import *
from stable_baselines.deepq.policies import CnnPolicy as DQNCnnPolicy

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = CustomRewardAndDoneEnv(env) # 報酬とエピソード完了の変更
    env = StochasticFrameSkip(env, n=4, stickprob=0.25) # スティッキーフレームスキップ
    env = Downsample(env, 2) # ダウンサンプリング
    env = FrameStack(env, 4) # フレームスタック
    env = ScaledFloatFrame(env) # 状態の正規化
    env = Monitor(env, log_dir, allow_early_resets=True)
    env.seed(0) # シードの指定
    set_global_seeds(0)
    env = DummyVecEnv([lambda: env]) # ベクトル環境の生成

    print('行動空間: ', env.action_space)
    print('状態空間: ', env.observation_space)

    return env


env = make_env()
custom_callback = CustomCallback(env,render=True)
model = PPO2(policy=CnnPolicy, env=env, verbose=0,learning_rate=0.000025,tensorboard_log=log_dir)
model = PPO2.load('./agents/best_mario_ppo2model', env=env, verbose=0)


state = env.reset()
total_reward = 0
while True:
    # 環境の描画
    env.render()

    # スリープ
    time.sleep(1/25)

    # モデルの推論
    action, _ = model.predict(state)

    # 1ステップ実行
    state, reward, done, info = env.step(action)
    total_reward += reward[0]

    # _, _, _, _ = env2.step(action)

    # エピソード完了
    if done:
        print('reward:', total_reward)
        state = env.reset()
        total_reward = 0
        # state = env2.reset()
