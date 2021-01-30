import gym
import os
import numpy as np
import datetime
import pytz
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import BaseCallback
import time

# ログフォルダの生成
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# コールバック
# best_mean_reward = -np.inf
# nupdates = 1

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env,render=False,verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf
        self.nupdates = 1
        self.env = env
        self.render = render
        self.y_len = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        # 10更新毎
        if (self.nupdates + 1) % 10 == 0:
            # 平均エピソード長、平均報酬の取得
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(y) > self.y_len:
                self.y_len = len(y)
                # 最近10件の平均報酬
                mean_reward = np.mean(y[-10:])

                # 平均報酬がベスト報酬以上の時はモデルを保存
                update_model = mean_reward > self.best_mean_reward

                if update_model:
                    self.best_mean_reward = mean_reward
                    self.model.save('./agents/best_mario_ppo2model')

                # ログ
                print('time: {}, nupdates: {}, mean: {:.2f}, best_mean: {:.2f}, model_update: {}'.format(
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')),
                    self.nupdates, mean_reward, self.best_mean_reward, update_model))

        self.nupdates += 1

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass





# CustomRewardAndDoneラッパー
class CustomRewardAndDoneEnv(gym.Wrapper):
    # 初期化
    def __init__(self, env):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self.reward = 0

    # リセット
    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        self.reward = 0
        return self.env.reset(**kwargs)

    # ステップ
    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # 報酬の変更
        if (info['x_pos'] > self._cur_x) & (self._cur_x != 0):
            # reward += 1
            self.reward += 1
        else:
            # reward -= 1
            self.reward -= 1
        self.reward /= 1000
        self._cur_x = info['x_pos']

        if info['life'] <= 1:
            self.reward -= 0.3

        if info['life'] == 1:
            done = True

        # エピソード完了の変更
        if info['flag_get']:
            self.reward += 2
            done =True
            print('GOAAL')
        return state, self.reward, done, info