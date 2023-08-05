from collections import defaultdict
from collections import deque

import numpy as np

from metamorph.config import cfg


class AgentMeter:
    def __init__(self, name):
        self.name = name
        self.mean_ep_rews = defaultdict(list)
        self.mean_pos = []
        self.mean_vel = []
        self.mean_metric = []
        self.mean_ep_len = []

        self.ep_rew = defaultdict(lambda: deque(maxlen=10))
        self.ep_pos = deque(maxlen=10)
        self.ep_vel = deque(maxlen=10)
        self.ep_metric = deque(maxlen=10)
        self.ep_count = 0
        self.ep_len = deque(maxlen=10)
        self.ep_len_ema = -1

        # statistics for UED
        self.ep_positive_gae = deque(maxlen=10)
        # self.return_buffer = deque(maxlen=50)
        # self.iter_buffer = deque(maxlen=50)
        # self.iter_returns = []
        self.iter_mean_return = deque(maxlen=5)
        self.iter_ep_num = deque(maxlen=5)
        self.iter_idx = deque(maxlen=5)
        self.LP = 0.
        # this needs to be reset for every iteration
        self.iter_ep_returns = []

    def add_ep_info(self, infos, cur_iter):
        for info in infos:
            if info["name"] != self.name:
                continue
            if "episode" in info.keys():
                self.ep_rew["reward"].append(info["episode"]["r"])
                self.iter_ep_returns.append(info["episode"]["r"])                
                # self.return_buffer.append(info["episode"]["r"])
                # self.iter_buffer.append(cur_iter)
                # while (1):
                #     if len(self.iter_returns) > cur_iter:
                #         break
                #     self.iter_returns.append([])
                self.ep_count += 1
                self.ep_len.append(info["episode"]["l"])
                if self.ep_count == 10:
                    self.ep_len_ema = np.mean(self.ep_len)
                elif self.ep_count >= 10:
                    alpha = cfg.TASK_SAMPLING.EMA_ALPHA
                    self.ep_len_ema = (
                        alpha * self.ep_len[-1] + (1 - alpha) * self.ep_len_ema
                    )

                for rew_type, rew_ in info["episode"].items():
                    if "__reward__" in rew_type:
                        self.ep_rew[rew_type].append(rew_)

                if "x_pos" in info:
                    self.ep_pos.append(info["x_pos"])
                if "x_vel" in info:
                    self.ep_vel.append(info["x_vel"])
                if "metric" in info:
                    self.ep_metric.append(info["metric"])
                
                self.ep_positive_gae.append(info['positive_gae'])

    def update_mean(self):
        if len(self.ep_rew["reward"]) == 0:
            return False

        for rew_type, rews_ in self.ep_rew.items():
            self.mean_ep_rews[rew_type].append(round(np.mean(rews_), 2))

        self.mean_pos.append(round(np.mean(self.ep_pos), 2))
        self.mean_vel.append(round(np.mean(self.ep_vel), 2))
        self.mean_metric.append(round(np.mean(self.ep_metric), 2))
        self.mean_ep_len.append(round(np.mean(self.ep_len), 2))
        return True

    def log_stats(self, max_name_len):
        if len(self.ep_rew["reward"]) == 0:
            return
        ep_rew = self.ep_rew["reward"]
        print(
            "Agent {:>{size}}: mean/median reward {:>4.0f}/{:<4.0f}, "
            "min/max reward {:>4.0f}/{:<4.0f}, "
            "#Ep: {:>7.0f}, avg/ema Ep len: {:>4.0f}/{:>4.0f}".format(
                self.name,
                np.mean(ep_rew),
                np.median(ep_rew),
                np.min(ep_rew),
                np.max(ep_rew),
                self.ep_count,
                np.mean(self.ep_len),
                self.ep_len_ema,
                size=max_name_len
            )
        )

    def update_iter_returns(self, cur_iter):
        if len(self.iter_ep_returns) > 0:
            self.iter_mean_return.append(np.mean(self.iter_ep_returns))
            self.iter_ep_num.append(len(self.iter_ep_returns))
            self.iter_idx.append(cur_iter)
            self.iter_ep_returns = []

    def get_learning_speed(self):

        def solve_linear_regression(X, y, w):
            X = np.array(X).reshape(-1, 1)
            X = np.c_[np.ones((X.shape[0], 1)), X]
            y = np.array(y)
            weight = np.diag(w)
            beta = np.linalg.inv(X.T.dot(weight).dot(X)).dot(X.T).dot(weight).dot(y)
            return beta.ravel()[1]

        if len(self.iter_mean_return) == 1:
            xdata = np.insert(np.array(self.iter_idx), 0, 0.)
            ydata = np.insert(np.array(self.iter_mean_return), 0, 0.)
            w = np.insert(np.array(self.iter_ep_num), 0, 0.)
        else:
            xdata = np.array(self.iter_idx)
            ydata = np.array(self.iter_mean_return)
            w = np.clip(np.array(self.iter_ep_num), None, 10.)
        print (xdata, ydata, w)
        self.LP = abs(solve_linear_regression(xdata, ydata, w))
        return self.LP

        # slope = []
        # for i in range(10, 60, 10):
        #     buffer_range = -min(i, buffer_size)
        #     xdata = [self.iter_buffer[j] for j in range(buffer_range, 0)]
        #     ydata = [self.return_buffer[j] for j in range(buffer_range, 0)]
        #     slope.append(solve_linear_regression(xdata, ydata))
        # score = np.array([abs(x) for x in slope])
        # r = np.sort(score)[2]
        # return r

        # xdata, ydata = [], []
        # for i in range(-5, 0):
        #     if len(self.iter_returns[i]) == 0:
        #         continue
        #     xdata.append(i + 5)
        #     ydata.append(np.array(self.iter_returns[i]).mean())
        # print (xdata, ydata)
        # slope = solve_linear_regression(xdata, ydata)
        # return abs(slope)


class TrainMeter:
    def __init__(self):
        self.agents = cfg.ENV.WALKERS
        self.max_name_len = max([len(a) for a in self.agents])
        self.agent_meters = {agent: AgentMeter(agent) for agent in self.agents}

        # Env stats
        self.train_stats = defaultdict(list)
        self.mean_ep_rews = defaultdict(list)
        self.mean_pos = []
        self.mean_vel = []
        self.mean_metric = []
        self.mean_ep_len = []

    def add_new_agents(self, agents):
        self.agents = cfg.ENV.WALKERS
        for agent in agents:
            self.agent_meters[agent] = AgentMeter(agent)

    def add_train_stat(self, stat_type, stat_value):
        self.train_stats[stat_type].append(stat_value)

    def add_ep_info(self, infos, cur_iter):
        for _, agent_meter in self.agent_meters.items():
            agent_meter.add_ep_info(infos, cur_iter)

    def update_mean(self):
        for _, agent_meter in self.agent_meters.items():
            success = agent_meter.update_mean()
            if not success:
                return

        metrics = ["mean_pos", "mean_vel", "mean_metric", "mean_ep_len"]
        for metric in metrics:
            metric_list = []
            for _, agent_meter in self.agent_meters.items():
                metric_list.append(getattr(agent_meter, metric)[-1])

            getattr(self, metric).append(round(np.mean(metric_list), 2))

        rew_types = self.agent_meters[self.agents[0]].mean_ep_rews.keys()

        for rew_type in rew_types:
            rew_list = []
            for _, agent_meter in self.agent_meters.items():
                rew_list.append(agent_meter.mean_ep_rews[rew_type][-1])

            self.mean_ep_rews[rew_type].append(round(np.mean(rew_list), 2))

    def log_stats(self):
        for _, agent_meter in self.agent_meters.items():
            agent_meter.log_stats(self.max_name_len)

        if len(self.mean_ep_rews["reward"]) > 0:
            print("Agent {:>{size}}: mean/------ reward {:>4.0f}, ".format(
                    "__env__",
                    self.mean_ep_rews["reward"][-1],
                    size=self.max_name_len
                )
            )

    def get_stats(self):
        stats = {}
        for agent, agent_meter in self.agent_meters.items():
            stats[agent] = {
                "reward": agent_meter.mean_ep_rews,
                "pos": agent_meter.mean_pos,
                "vel": agent_meter.mean_vel,
                "metric": agent_meter.mean_metric,
                "ep_len": agent_meter.mean_ep_len,
            }

        stats["__env__"] = {
                "reward": self.mean_ep_rews,
                "pos": self.mean_pos,
                "vel": self.mean_vel,
                "metric": self.mean_metric,
                "ep_len": self.mean_ep_len,
        }
        stats["__env__"].update(dict(self.train_stats))
        return stats
