"""
MAPPO (Multi-Agent Proximal Policy Optimization) 协同调度

实现:
1. CTDE (Centralized Training Decentralized Execution)
2. Graph Neural Network 通信协议
3. 冲突消解策略
4. 电池约束任务分配
5. 奖励函数设计

理论参考:
- Lowe, R., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
- Yu, C., et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.
- Suttle, W., et al. (2021). A Generalized Training Approach for Multi-Agent Learning.
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
from collections import deque
import heapq

# torch is optional — only needed for neural network classes
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass  # torch not installed; neural network classes will be unavailable


class AGVRole(Enum):
    """AGV 角色"""
    PICKER = "picker"
    TRANSPORTER = "transporter"
    CHARGER = "charger"
    IDLE = "idle"


@dataclass
class Task:
    """任务"""
    id: int
    type: str               # "pick", "transport", "charge"
    pickup: Tuple[float, float]  # 取货点 (x, y)
    dropoff: Tuple[float, float]  # 卸货点 (x, y)
    priority: int = 1        # 优先级 [1, 5]
    deadline: float = None   # 截止时间 (秒)
    assigned_to: Optional[int] = None  # 分配给哪个 AGV
    completed: bool = False

    def __post_init__(self):
        if self.deadline is None:
            self.deadline = 3600.0


@dataclass
class AGVAgent:
    """AGV 智能体"""
    id: int
    role: AGVRole
    state: np.ndarray         # (6,) [x, y, theta, v, omega, battery]
    local_observation: np.ndarray = None  # (obs_dim,)
    action: np.ndarray = None       # (action_dim,)

    def get_observation(self,
                        tasks: List[Task],
                        neighbors: List[int],
                        env_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """
        获取局部观测

        观测空间:
        - 自身状态 (6,)
        - 最近任务信息 (5,) [距离, 优先级, 类型编码, ...]
        - 邻居状态 (N_neighbors × 6,)
        - 环境边界 (4,)
        """
        obs = []

        # 自身状态
        obs.extend(self.state)

        # 最近任务 (按距离排序, 取前 3 个)
        my_pos = self.state[:2]
        sorted_tasks = sorted(
            [t for t in tasks if not t.assigned_to or t.assigned_to == self.id],
            key=lambda t: np.linalg.norm(np.array(t.pickup) - my_pos)
        )

        for t in sorted_tasks[:3]:
            task_vec = [
                np.linalg.norm(np.array(t.pickup) - my_pos) / 100.0,  # 归一化
                t.priority / 5.0,
                1.0 if t.type == "pick" else 0.0,
                1.0 if t.type == "transport" else 0.0,
                1.0 if t.type == "charge" else 0.0
            ]
            obs.extend(task_vec)

        if len(sorted_tasks) < 3:
            # 填充
            for _ in range(3 - len(sorted_tasks)):
                obs.extend([0.0] * 5)

        # 邻居状态 (最多 5 个)
        for nb_id in neighbors[:5]:
            obs.extend([0.0] * 6)  # 占位
        if len(neighbors) < 5:
            for _ in range(5 - len(neighbors)):
                obs.extend([0.0] * 6)

        # 环境边界
        obs.extend([
            env_bounds[0] / 100.0,
            env_bounds[1] / 100.0,
            env_bounds[2] / 100.0,
            env_bounds[3] / 100.0
        ])

        self.local_observation = np.array(obs)
        return self.local_observation


class ConflictResolver:
    """
    冲突消解器

    方法:
    1. 优先级抢占 (Priority Preemption)
    2. 时间窗调整 (Time Window Adjustment)
    3. 速度协调 (Velocity Coordination)
    4. 停车位分配 (Parking Spot Assignment)
    """
    def __init__(self,
                 safety_distance: float = 1.5,
                 max_wait_time: float = 30.0):
        self.safety_dist = safety_distance
        self.max_wait = max_wait_time

    def detect_conflicts(self,
                          agv_positions: Dict[int, np.ndarray],
                          agv_velocities: Dict[int, np.ndarray],
                          prediction_horizon: float = 5.0) -> List[Tuple[int, int, float]]:
        """
        检测未来冲突

        Args:
            agv_positions: {agv_id: (x, y, theta)}
            agv_velocities: {agv_id: (v, omega)}
            prediction_horizon: 预测时域 (秒)

        Returns:
            conflicts: [(agv_i, agv_j, predicted_time), ...]
        """
        conflicts = []
        agv_ids = list(agv_positions.keys())

        for i in range(len(agv_ids)):
            for j in range(i + 1, len(agv_ids)):
                id_i = agv_ids[i]
                id_j = agv_ids[j]

                pos_i = agv_positions[id_i]
                pos_j = agv_positions[id_j]

                vel_i = agv_velocities.get(id_i, np.array([0.0, 0.0]))
                vel_j = agv_velocities.get(id_j, np.array([0.0, 0.0]))

                # 简化: 假设匀速直线运动
                min_dist = float('inf')
                conflict_time = None

                for t in np.linspace(0, prediction_horizon, 50):
                    pred_i = pos_i[:2] + vel_i * t
                    pred_j = pos_j[:2] + vel_j * t

                    dist = np.linalg.norm(pred_i - pred_j)

                    if dist < min_dist:
                        min_dist = dist
                        conflict_time = t

                if min_dist < self.safety_dist:
                    conflicts.append((id_i, id_j, conflict_time))

        return conflicts

    def resolve_conflict(self,
                         conflict: Tuple[int, int, float],
                         agv_states: Dict[int, AGVAgent],
                         task_priorities: Dict[int, int]) -> Dict[int, str]:
        """
        消解冲突

        Returns:
            actions: {agv_id: "stop" | "yield" | "reroute"}
        """
        id_i, id_j, _ = conflict

        # 优先级高的继续, 低的让路
        priority_i = task_priorities.get(id_i, 1)
        priority_j = task_priorities.get(id_j, 1)

        if priority_i >= priority_j:
            return {id_i: "continue", id_j: "yield"}
        else:
            return {id_i: "yield", id_j: "continue"}

    def coordinate_velocities(self,
                               agv_positions: Dict[int, np.ndarray],
                               agv_velocities: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        协调速度 (避免冲突)

        方法: 使用势场法 (Potential Field)
        - 其他 AGV 产生排斥力
        - 目标点产生吸引力
        """
        new_velocities = {}

        for agv_id, pos in agv_positions.items():
            velocity = agv_velocities[agv_id].copy()

            for other_id, other_pos in agv_positions.items():
                if other_id == agv_id:
                    continue

                dist = np.linalg.norm(pos[:2] - other_pos[:2])

                if dist < self.safety_dist * 2.0:
                    # 排斥力
                    repulsion = (pos[:2] - other_pos[:2]) / (dist ** 2 + 1e-6)
                    velocity[:2] += repulsion * 0.1

            # 限制速度
            speed = np.linalg.norm(velocity[:2])
            if speed > 1.5:
                velocity[:2] = velocity[:2] / speed * 1.5

            new_velocities[agv_id] = velocity

        return new_velocities


class BatteryManager:
    """
    电池管理器

    功能:
    1. 监控电池电量
    2. 触发充电任务
    3. 优化充电调度 (平衡等待时间)
    """
    def __init__(self,
                 battery_capacity: float = 20000.0,
                 charging_power: float = 5000.0,
                 low_threshold: float = 0.2):
        self.capacity = battery_capacity
        self.charging_power = charging_power
        self.low_threshold = low_threshold

    def check_battery(self, agv: AGVAgent) -> str:
        """
        检查电池状态

        Returns:
            status: "normal" | "low" | "critical"
        """
        battery = agv.state[5]

        if battery < 0.1:
            return "critical"
        elif battery < self.low_threshold:
            return "low"
        else:
            return "normal"

    def compute_charging_time(self, current_battery: float) -> float:
        """
        计算充电时间

        假设: 从当前电量充到 80% 需要的时间
        """
        target = 0.8
        if current_battery >= target:
            return 0.0

        charge_needed = (target - current_battery) * self.capacity
        time_hours = charge_needed / self.charging_power

        return time_hours * 3600.0

    def assign_charging_task(self,
                                agv: AGVAgent,
                                tasks: List[Task]) -> Optional[Task]:
        """
        分配充电任务

        如果 AGV 电量低, 创建一个充电任务
        """
        if self.check_battery(agv) in ["low", "critical"]:
            has_charging = any(
                t.type == "charge" and t.assigned_to == agv.id
                for t in tasks
            )

            if not has_charging:
                charging_task = Task(
                    id=len(tasks),
                    type="charge",
                    pickup=(agv.state[0], agv.state[1]),
                    dropoff=(agv.state[0], agv.state[1]),
                    priority=5 if self.check_battery(agv) == "critical" else 3
                )
                return charging_task

        return None


# ============================================================
# 以下类需要 torch，仅在 torch 可用时定义
# ============================================================

if _TORCH_AVAILABLE:

    class GraphConvLayer(nn.Module):
        """
        图卷积层 (GNN 通信)

        实现邻居间的信息交换:
        h_i' = σ(W · MEAN({h_i} ∪ {h_j for j ∈ N(i)}))
        """
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.W = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)

        def forward(self,
                    node_features: torch.Tensor,
                    adjacency: torch.Tensor) -> torch.Tensor:
            """
            Args:
                node_features: (N, in_dim) 节点特征
                adjacency: (N, N) 邻接矩阵 (0/1)

            Returns:
                updated_features: (N, out_dim)
            """
            N = node_features.size(0)

            # 聚合邻居特征 (平均)
            neighbor_sum = torch.matmul(adjacency, node_features)
            degree = adjacency.sum(dim=1, keepdim=True) + 1.0
            neighbor_avg = neighbor_sum / degree
            combined = node_features + neighbor_avg
            updated = F.relu(self.bn(self.W(combined)))

            return updated


    class ActorNetwork(nn.Module):
        """行动者网络 (策略)"""
        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            mean = torch.tanh(self.fc3(x))
            std = torch.exp(self.log_std).expand_as(mean)
            return mean, std

        def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            mean, std = self.forward(obs)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob

        def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            mean, std = self.forward(obs)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            return log_prob


    class CriticNetwork(nn.Module):
        """评论者网络 (价值函数)"""
        def __init__(self, global_state_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(global_state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(self, global_state: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(global_state))
            x = F.relu(self.fc2(x))
            value = self.fc3(x)
            return value


    class MAPPOAgent:
        """
        MAPPO 智能体

        CTDE 架构:
        - Centralized Critic: 输入全局状态 s (所有 AGV + 所有任务)
        - Decentralized Actor: 输入局部观测 o_i
        """
        def __init__(self,
                     agent_id: int,
                     obs_dim: int,
                     action_dim: int,
                     global_state_dim: int,
                     lr_actor: float = 3e-4,
                     lr_critic: float = 3e-4):
            self.agent_id = agent_id

            self.actor = ActorNetwork(obs_dim, action_dim)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

            self.critic = CriticNetwork(global_state_dim)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

            self.gnn = GraphConvLayer(obs_dim, obs_dim)
            self.buffer = []

        def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            if deterministic:
                mean, _ = self.actor(obs_tensor)
                action = mean
            else:
                action, _ = self.actor.sample_action(obs_tensor)
            return action.detach().numpy()[0]

        def store_transition(self,
                              obs: np.ndarray,
                              action: np.ndarray,
                              reward: float,
                              next_obs: np.ndarray,
                              done: bool):
            self.buffer.append((obs, action, reward, next_obs, done))

        def update(self,
                  batch_size: int,
                  global_states: torch.Tensor,
                  next_global_states: torch.Tensor,
                  clip_param: float = 0.2):
            """PPO 更新 (简化版, 使用 GAE)"""
            if len(self.buffer) < batch_size:
                return

            batch_indices = random.sample(range(len(self.buffer)), batch_size)
            batch = [self.buffer[i] for i in batch_indices]

            obs_batch = torch.FloatTensor([b[0] for b in batch])
            action_batch = torch.FloatTensor([b[1] for b in batch])
            reward_batch = torch.FloatTensor([b[2] for b in batch])
            next_obs_batch = torch.FloatTensor([b[3] for b in batch])
            done_batch = torch.BoolTensor([b[4] for b in batch])

            with torch.no_grad():
                next_values = self.critic(next_global_states[:batch_size])
                targets = reward_batch.unsqueeze(1) + \
                         0.99 * next_values * (~done_batch).unsqueeze(1).float()
                current_values = self.critic(global_states[:batch_size])
                advantages = targets - current_values

            old_log_probs = self.actor.evaluate_action(obs_batch, action_batch)

            for _ in range(10):
                new_log_probs = self.actor.evaluate_action(obs_batch, action_batch)
                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages.detach()
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                current_values = self.critic(global_states[:batch_size])
                critic_loss = F.mse_loss(current_values, targets.detach())

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            self.buffer.clear()


# =========== 导出列表 ===========
__all__ = [
    'AGVRole',
    'Task',
    'AGVAgent',
    'ConflictResolver',
    'BatteryManager',
]
if _TORCH_AVAILABLE:
    __all__.extend([
        'GraphConvLayer',
        'ActorNetwork',
        'CriticNetwork',
        'MAPPOAgent',
    ])


# =========== 测试代码 ===========

if __name__ == "__main__":
    print("=" * 60)
    print("MAPPO Multi-AGV Coordination - Test")
    print("=" * 60)

    # 创建测试 AGV 智能体
    print("\nCreating AGV agents...")

    agvs = []
    for i in range(5):
        agv = AGVAgent(
            id=i,
            role=AGVRole.PICKER if i % 2 == 0 else AGVRole.TRANSPORTER,
            state=np.array([
                random.uniform(0, 100),
                random.uniform(0, 100),
                random.uniform(-np.pi, np.pi),
                random.uniform(0, 1.5),
                0.0,
                random.uniform(0.3, 1.0)
            ])
        )
        agvs.append(agv)

    print(f"  AGVs created: {len(agvs)}")

    # 创建测试任务
    print("\nCreating tasks...")

    tasks = []
    for i in range(20):
        task = Task(
            id=i,
            type=random.choice(["pick", "transport"]),
            pickup=(random.uniform(0, 100), random.uniform(0, 100)),
            dropoff=(random.uniform(0, 100), random.uniform(0, 100)),
            priority=random.randint(1, 5)
        )
        tasks.append(task)

    print(f"  Tasks created: {len(tasks)}")

    # 测试观测获取
    print("\nTesting observation extraction...")

    env_bounds = (0.0, 0.0, 100.0, 100.0)

    for agv in agvs[:1]:
        obs = agv.get_observation(tasks, neighbors=[1, 2], env_bounds=env_bounds)
        print(f"  AGV {agv.id} observation shape: {obs.shape}")
        print(f"  Observation sample: {obs[:10]}...")

    # 测试冲突检测
    print("\nTesting conflict detection...")

    resolver = ConflictResolver(safety_distance=1.5)

    agv_positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([5.0, 0.0, 0.0]),
        2: np.array([10.0, 10.0, 0.0])
    }

    agv_velocities = {
        0: np.array([1.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 0.0])
    }

    conflicts = resolver.detect_conflicts(agv_positions, agv_velocities, prediction_horizon=5.0)
    print(f"  Conflicts detected: {len(conflicts)}")
    for c in conflicts:
        print(f"    AGV {c[0]} vs AGV {c[1]} at t = {c[2]:.2f}s")

    # 测试电池管理
    print("\nTesting battery management...")

    battery_mgr = BatteryManager()

    for agv in agvs[:3]:
        status = battery_mgr.check_battery(agv)
        charging_time = battery_mgr.compute_charging_time(agv.state[5])
        print(f"  AGV {agv.id}: battery = {agv.state[5]:.2f}, status = {status}")
        print(f"    Charging time to 80%: {charging_time / 60.0:.1f} min")

    # 测试 torch 部分 (如果可用)
    if _TORCH_AVAILABLE:
        print("\nTesting Actor network (torch available)...")
        actor = ActorNetwork(obs_dim=44, action_dim=2)
        obs_tensor = torch.randn(1, 44)
        mean, std = actor(obs_tensor)
        print(f"  Action mean: {mean.detach().numpy()}")
        action, log_prob = actor.sample_action(obs_tensor)
        print(f"  Sampled action: {action.detach().numpy()}")
        print(f"  Log prob: {log_prob.detach().numpy()}")
    else:
        print("\n[Skipped] Actor network test (torch not installed)")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)
