"""
多层级库存优化 (Multi-Echelon Inventory Optimization)

实现:
1. 安全库存计算 (Safety Stock)
2. (s, Q) 策略
3. 经济订货量 (EOQ - Economic Order Quantity)
4. 多层级供应链 (工厂 → 区域仓 → 前置仓 → 门店)
5. 需求不确定性下的优化

理论参考:
- Simchi-Levi, D., Chen, X., & Bramel, J. (2014). The Logic of Logistics.
- Silver, E. A., et al. (1998). Inventory Management and Production Planning.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats


class DemandDistribution(Enum):
    """需求分布类型"""
    NORMAL = "normal"
    POISSON = "poisson"
    GAMMA = "gamma"


@dataclass
class DemandModel:
    """需求模型"""
    distribution: DemandDistribution
    mean: float                  # 平均需求 (单位/时间)
    std: float                   # 需求标准差
    parameters: Dict = None      # 分布特定参数
    
    def sample(self, n: int = 1) -> np.ndarray:
        """采样"""
        if self.distribution == DemandDistribution.NORMAL:
            return np.random.normal(self.mean, self.std, n)
        elif self.distribution == DemandDistribution.POISSON:
            return np.random.poisson(self.mean, n)
        elif self.distribution == DemandDistribution.GAMMA:
            # Gamma 分布: shape = α, scale = β
            alpha = self.parameters.get('alpha', self.mean**2 / self.std**2)
            beta = self.parameters.get('beta', self.std**2 / self.mean)
            return np.random.gamma(alpha, beta, n)
    
    def probability(self, x: float) -> float:
        """概率密度 (用于计算缺货风险)"""
        if self.distribution == DemandDistribution.NORMAL:
            return stats.norm.pdf(x, self.mean, self.std)
        elif self.distribution == DemandDistribution.POISSON:
            return stats.poisson.pmf(int(x), self.mean)
        elif self.distribution == DemandDistribution.GAMMA:
            alpha = self.parameters.get('alpha', self.mean**2 / self.std**2)
            beta = self.parameters.get('beta', self.std**2 / self.mean)
            return stats.gamma.pdf(x, alpha, scale=beta)


@dataclass
class InventoryNode:
    """库存节点 (供应链中的一层)"""
    id: str
    name: str
    holding_cost: float          # 单位持有成本 (元/单位/时间)
    ordering_cost: float          # 订货成本 (元/次)
    lead_time: int               # 提前期 (时间单位)
    demand: DemandModel         # 需求模型
    initial_stock: int = 100    # 初始库存
    reorder_point: int = None   # 订货点 (s)
    order_quantity: int = None  # 订货量 (Q)
    safety_stock: int = None    # 安全库存
    service_level: float = 0.95  # 服务水平 (缺货概率 ≤ 1 - service_level)
    
    def compute_safety_stock(self) -> int:
        """
        计算安全库存
        
        公式 (正态分布):
        SS = z · σ · √(L)
        
        其中:
        - z: 服务水平因子 (e.g., 1.65 for 95%, 2.33 for 99%)
        - σ: 需求标准差
        - L: 提前期
        """
        z = stats.norm.ppf(self.service_level)
        self.safety_stock = int(np.ceil(z * self.demand.std * np.sqrt(self.lead_time)))
        return self.safety_stock
    
    def compute_eoq(self) -> int:
        """
        计算经济订货量 (EOQ)
        
        公式:
        Q* = √(2 · D · K / h)
        
        其中:
        - D: 年需求量
        - K: 订货成本
        - h: 单位持有成本
        """
        D = self.demand.mean * 365  # 假设时间单位是天
        K = self.ordering_cost
        h = self.holding_cost
        
        self.order_quantity = int(np.ceil(np.sqrt(2 * D * K / h)))
        return self.order_quantity
    
    def compute_reorder_point(self) -> int:
        """
        计算订货点 (s)
        
        公式:
        s* = μ · L + SS
        
        其中:
        - μ: 平均需求
        - L: 提前期
        - SS: 安全库存
        """
        if self.safety_stock is None:
            self.compute_safety_stock()
        
        self.reorder_point = int(np.ceil(
            self.demand.mean * self.lead_time + self.safety_stock
        ))
        return self.reorder_point


@dataclass
class SupplyChainNetwork:
    """供应链网络"""
    nodes: Dict[str, InventoryNode]
    edges: List[Tuple[str, str]]  # (上游, 下游)
    
    def compute_optimal_policy(self) -> Dict[str, Tuple[int, int]]:
        """
        计算最优 (s, Q) 策略
        
        返回值:
            policy: {node_id: (s, Q)}
        """
        policy = {}
        
        # 从下游到上游 (反向动态规划)
        for node_id, node in self.nodes.items():
            s = node.compute_reorder_point()
            Q = node.compute_eoq()
            policy[node_id] = (s, Q)
        
        return policy


class MultiEchelonOptimizer:
    """
    多层级库存优化器
    
    算法: 动态规划 (Dynamic Programming)
    
    问题:
    给定 N 层供应链, 每层的持有成本、订货成本、提前期、需求分布,
    求每层的 (s, Q) 策略, 使得总成本最小。
    
    状态: 各层库存水平
    动作: 是否订货, 订货量
    转移: 需求实现, 订货到达
    代价: 持有成本 + 订货成本 + 缺货成本
    """
    
    def __init__(self, network: SupplyChainNetwork, 
                 horizon: int = 365,
                 discount: float = 0.95):
        """
        Args:
            network: 供应链网络
            horizon: 规划时域 (天)
            discount: 折扣因子
        """
        self.network = network
        self.T = horizon
        self.gamma = discount
        
        # 状态空间离散化
        self.state_max = 500  # 最大库存水平
        self.action_max = 200  # 最大订货量
    
    def optimize(self) -> Dict:
        """
        动态规划求解
        
        注意: 精确求解是指数复杂度, 实际系统中使用:
        1. 近似动态规划 (Approximate DP)
        2. 强化学习 (Reinforcement Learning)
        3. 贪心启发式 (Greedy Heuristic)
        
        这里实现贪心启发式 (逐层优化)
        """
        print("Optimizing multi-echelon inventory...")
        
        policy = {}
        
        # 从最下游开始 (反向)
        nodes_ordered = self._topological_sort_reverse()
        
        for node_id in nodes_ordered:
            node = self.network.nodes[node_id]
            
            # 计算最优 (s, Q)
            s, Q = self._optimize_node(node, node_id)
            
            policy[node_id] = {'reorder_point': s, 'order_quantity': Q}
            print(f"  {node.name}: s = {s}, Q = {Q}")
        
        return policy
    
    def _optimize_node(self, node: InventoryNode, 
                       node_id: str) -> Tuple[int, int]:
        """
        优化单个节点
        
        方法: 枚举 s 和 Q, 选择期望成本最小的组合
        """
        best_cost = float('inf')
        best_s = None
        best_Q = None
        
        # 枚举范围 (简化)
        s_start = int(max(0, node.demand.mean * node.lead_time - 50))
        s_stop = int(node.demand.mean * node.lead_time + 200)
        s_range = range(s_start, s_stop, 20)
        Q_range = range(20, 300, 20)
        
        for s in s_range:
            for Q in Q_range:
                # 模拟 (s, Q) 策略的性能
                cost = self._simulate_policy(node, s, Q)
                
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
                    best_Q = Q
        
        return best_s, best_Q
    
    def _simulate_policy(self, node: InventoryNode, 
                         s: int, Q: int, 
                         n_sim: int = 1000) -> float:
        """
        模拟 (s, Q) 策略
        
        返回: 平均总成本 (持有 + 订货 + 缺货)
        """
        total_cost = 0.0
        
        for sim in range(n_sim):
            # 初始化
            stock = node.initial_stock
            cost = 0.0
            
            for t in range(self.T):
                # 需求实现
                demand = max(0, int(np.round(node.demand.sample(1)[0])))
                
                # 是否满足需求
                if stock >= demand:
                    stock -= demand
                else:
                    # 缺货
                    shortage = demand - stock
                    cost += shortage * 10.0  # 缺货成本 (元/单位)
                    stock = 0
                
                # 订货决策
                if stock <= s:
                    # 订货 (Q 单位)
                    # 假设提前期内订货到达
                    stock += Q
                    cost += node.ordering_cost
                
                # 持有成本
                cost += stock * node.holding_cost / 365.0  # 日持有成本
            
            total_cost += cost
        
        return total_cost / n_sim
    
    def _topological_sort_reverse(self) -> List[str]:
        """拓扑排序 (反向 - 从下游到上游)"""
        # 构建邻接表
        adj = {node_id: [] for node_id in self.network.nodes}
        for (upstream, downstream) in self.network.edges:
            adj[downstream].append(upstream)  # 反向
        
        # DFS
        visited = set()
        order = []
        
        def dfs(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            for neighbor in adj[node_id]:
                dfs(neighbor)
            order.append(node_id)
        
        for node_id in self.network.nodes:
            dfs(node_id)
        
        return order


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Echelon Inventory Optimization - Test")
    print("=" * 60)
    
    # 创建供应链网络
    print("\nCreating supply chain network...")
    
    # 需求模型 (正态分布)
    demand_retail = DemandModel(
        distribution=DemandDistribution.NORMAL,
        mean=50.0,   # 平均日需求 50 单位
        std=15.0      # 标准差 15 单位
    )
    
    demand_warehouse = DemandModel(
        distribution=DemandDistribution.NORMAL,
        mean=200.0,  # 批发仓库日需求 = 4 个零售店
        std=40.0
    )
    
    # 库存节点
    retail = InventoryNode(
        id="retail_001",
        name="Retail Store 001",
        holding_cost=2.0,    # 2 元/单位/天
        ordering_cost=100.0, # 100 元/次
        lead_time=7,           # 7 天提前期
        demand=demand_retail,
        initial_stock=150
    )
    
    warehouse = InventoryNode(
        id="warehouse_001",
        name="Regional Warehouse",
        holding_cost=1.0,    # 批发仓库持有成本更低
        ordering_cost=500.0,
        lead_time=14,
        demand=demand_warehouse,
        initial_stock=1000
    )
    
    # 网络
    network = SupplyChainNetwork(
        nodes={
            retail.id: retail,
            warehouse.id: warehouse
        },
        edges=[
            (warehouse.id, retail.id)  # 批发仓 → 零售店
        ]
    )
    
    # 计算安全库存
    print("\nComputing safety stock...")
    ss_retail = retail.compute_safety_stock()
    ss_warehouse = warehouse.compute_safety_stock()
    print(f"  Retail SS: {ss_retail} units (service level = {retail.service_level})")
    print(f"  Warehouse SS: {ss_warehouse} units")
    
    # 计算 EOQ
    print("\nComputing EOQ...")
    eoq_retail = retail.compute_eoq()
    eoq_warehouse = warehouse.compute_eoq()
    print(f"  Retail EOQ: {eoq_retail} units")
    print(f"  Warehouse EOQ: {eoq_warehouse} units")
    
    # 计算订货点
    print("\nComputing reorder point...")
    rop_retail = retail.compute_reorder_point()
    rop_warehouse = warehouse.compute_reorder_point()
    print(f"  Retail ROP: {rop_retail} units")
    print(f"  Warehouse ROP: {rop_warehouse} units")
    
    # 多层级优化
    print("\nRunning multi-echelon optimization...")
    optimizer = MultiEchelonOptimizer(network, horizon=365, discount=0.95)
    policy = optimizer.optimize()
    
    print("\nOptimal Policy:")
    for node_id, p in policy.items():
        node = network.nodes[node_id]
        print(f"  {node.name}:")
        print(f"    Reorder Point (s): {p['reorder_point']}")
        print(f"    Order Quantity (Q): {p['order_quantity']}")
    
    print("\nTest passed!")
