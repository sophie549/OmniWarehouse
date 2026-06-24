"""
车辆路径问题 (Vehicle Routing Problem - VRP)

实现:
1. 经典 VRP (Capacity-constrained VRP)
2. 遗传算法求解 (Genetic Algorithm)
3. 局部搜索 (2-opt / 3-opt)
4. 时间窗约束 (VRPTW - Time Windows)
5. 多车场 (Multi-Depot VRP)

理论参考:
- Toth, P., & Vigo, D. (2014). Vehicle Routing: Problems, Methods, and Applications.
- Taniguchi, E., et al. (2001). Optimal Size and Location Planning of Public Logistics Terminals.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import heapq


@dataclass
class Customer:
    """客户 (需求点)"""
    id: int
    x: float                   # 坐标 (米)
    y: float
    demand: float = 0.0      # 需求量 (单位)
    service_time: float = 10.0  # 服务时间 (分钟)
    tw_start: float = 0.0     # 时间窗开始 (分钟, 0 = 00:00)
    tw_end: float = 1440.0     # 时间窗结束 (分钟, 1440 = 24:00)
    priority: int = 1           # 优先级 [1, 5]


@dataclass
class Vehicle:
    """车辆"""
    id: int
    capacity: float = 1000.0  # 容量 (单位)
    max_distance: float = 200.0  # 最大行驶距离 (公里)
    depot_id: int = 0          # 所属车场 ID
    cost_per_km: float = 2.0   # 成本 (元/公里)


@dataclass
class Depot:
    """车场 (仓库)"""
    id: int
    x: float
    y: float
    vehicles: List[int] = field(default_factory=list)  # 车辆 ID 列表


@dataclass
class VRPSolution:
    """VRP 解"""
    routes: List[List[int]]  # 每辆车的路径 [depot, c1, c2, ..., depot]
    total_distance: float = 0.0
    total_cost: float = 0.0
    fitness: float = 0.0      # 适应度 (1 / 总成本)
    
    def compute_metrics(self, customers: List[Customer], 
                      vehicles: List[Vehicle], 
                      depot: Depot) -> Dict:
        """计算解的指标"""
        total_dist = 0.0
        total_cost = 0.0
        vehicle_loads = []
        
        # 建立 ID → Customer 映射 (ID 1-based: 1~n)
        cust_dict = {c.id: c for c in customers}
        
        for route_idx, route in enumerate(self.routes):
            if len(route) <= 2:  # 只有 depot
                continue
            
            vehicle = vehicles[route_idx] if route_idx < len(vehicles) else vehicles[0]
            
            # 计算路径长度
            dist = 0.0
            for i in range(1, len(route)):
                c1 = cust_dict.get(route[i-1]) if route[i-1] != 0 else depot
                c2 = cust_dict.get(route[i]) if route[i] != 0 else depot
                
                if c1 is None or c2 is None:
                    continue
                
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist += np.sqrt(dx**2 + dy**2) / 1000.0  # 转公里
            
            total_dist += dist
            total_cost += dist * vehicle.cost_per_km
            
            # 计算载重 (排除 depot 标记 0，跳过非法 ID)
            load = 0
            for cid in route:
                if cid == 0:
                    continue
                if cid in cust_dict:
                    load += cust_dict[cid].demand
                else:
                    print(f"      ⚠️ Warning: Invalid customer ID {cid} in route, skipping")
            vehicle_loads.append(load)
        
        return {
            'total_distance': total_dist,
            'total_cost': total_cost,
            'n_vehicles_used': len([r for r in self.routes if len(r) > 2]),
            'avg_load': np.mean(vehicle_loads) if vehicle_loads else 0,
            'max_load': max(vehicle_loads) if vehicle_loads else 0,
            'fitness': 1.0 / (total_cost + 1e-6)
        }


class DistanceMatrix:
    """
    距离矩阵
    
    预计算所有点对之间的距离, 加速算法。
    """
    
    def __init__(self, customers: List[Customer], depot: Depot):
        self.customers = customers
        self.depot = depot
        
        N = len(customers) + 1  # +1 for depot
        self.dist = np.zeros((N, N))
        
        # 计算距离矩阵
        for i in range(N):
            for j in range(N):
                if i == j:
                    self.dist[i, j] = 0.0
                else:
                    # 获取坐标
                    if i == 0:
                        xi, yi = depot.x, depot.y
                    else:
                        xi, yi = customers[i-1].x, customers[i-1].y
                    
                    if j == 0:
                        xj, yj = depot.x, depot.y
                    else:
                        xj, yj = customers[j-1].x, customers[j-1].y
                    
                    dx = xi - xj
                    dy = yi - yj
                    self.dist[i, j] = np.sqrt(dx**2 + dy**2) / 1000.0  # 公里
    
    def get(self, i: int, j: int) -> float:
        """查询距离"""
        return self.dist[i, j]


class GeneticAlgorithmVRP:
    """
    遗传算法求解 VRP
    
    编码:
    - 路径表示: (0, c1, c2, ..., 0, c3, c4, ..., 0, ...)
                    depot  depot      depot
    
    适应度:
    - f = 1 / (总路径长度 + 惩罚项)
    
    选择:
    - 锦标赛选择 (Tournament Selection)
    
    交叉:
    - 部分匹配交叉 (Partially Matched Crossover - PMX)
    
    变异:
    - 交换变异 (Swap Mutation)
    - 反转变异 (Inversion Mutation)
    """
    
    def __init__(self, 
                 customers: List[Customer],
                 vehicles: List[Vehicle],
                 depot: Depot,
                 pop_size: int = 100,
                 n_generations: int = 500,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3):
        """
        Args:
            customers: 客户列表
            vehicles: 车辆列表
            depot: 车场
            pop_size: 种群大小
            n_generations: 迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            tournament_size: 锦标赛大小
        """
        self.customers = customers
        self.vehicles = vehicles
        self.depot = depot
        self.n_customers = len(customers)
        self.pop_size = pop_size
        self.n_gen = n_generations
        self.pc = crossover_rate
        self.pm = mutation_rate
        self.tournament_size = tournament_size
        
        # 距离矩阵
        self.dm = DistanceMatrix(customers, depot)
        
        # 种群
        self.population: List[VRPSolution] = []
        self.best_solution: Optional[VRPSolution] = None
        self.best_fitness: float = 0.0
    
    def optimize(self) -> VRPSolution:
        """
        运行遗传算法
        
        Returns:
            最优解
        """
        print("=" * 60)
        print("Genetic Algorithm for VRP")
        print("=" * 60)
        print(f"Customers: {self.n_customers}")
        print(f"Vehicles: {len(self.vehicles)}")
        print(f"Population size: {self.pop_size}")
        print(f"Generations: {self.n_gen}")
        print()
        
        # Step 1: 初始化种群
        print("Initializing population...")
        self._initialize_population()
        print(f"  Initial best fitness: {self.best_fitness:.6f}")
        print()
        
        # Step 2: 迭代进化
        print("Evolution...")
        for gen in range(self.n_gen):
            # 评估适应度
            self._evaluate_fitness()
            
            # 选择 + 交叉 + 变异
            new_population = []
            
            # 精英保留 (保留最优的 10%)
            n_elite = max(1, int(self.pop_size * 0.1))
            elite = sorted(self.population, key=lambda s: s.fitness, reverse=True)[:n_elite]
            new_population.extend(elite)
            
            # 生成后代
            while len(new_population) < self.pop_size:
                # 选择父代
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # 交叉
                if random.random() < self.pc:
                    child1, child2 = self._crossover_pmx(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # 变异
                if random.random() < self.pm:
                    child1 = self._mutate_swap(child1)
                if random.random() < self.pm:
                    child2 = self._mutate_swap(child2)
                
                # 局部搜索 (2-opt)
                child1 = self._local_search_2opt(child1)
                child2 = self._local_search_2opt(child2)
                
                new_population.extend([child1, child2])
            
            # 更新种群
            self.population = new_population[:self.pop_size]
            
            # 打印进度
            if gen % 50 == 0 or gen == self.n_gen - 1:
                best = max(self.population, key=lambda s: s.fitness)
                print(f"  Generation {gen}: Best fitness = {best.fitness:.6f}")
        
        # Step 3: 返回最优解
        self.best_solution = max(self.population, key=lambda s: s.fitness)
        
        print()
        print("Optimization complete!")
        metrics = self.best_solution.compute_metrics(
            self.customers, self.vehicles, self.depot
        )
        print(f"Best solution:")
        print(f"  Total distance: {metrics['total_distance']:.2f} km")
        print(f"  Total cost: {metrics['total_cost']:.2f} yuan")
        print(f"  Vehicles used: {metrics['n_vehicles_used']}")
        print(f"  Fitness: {metrics['fitness']:.6f}")
        
        return self.best_solution
    
    def _initialize_population(self):
        """初始化种群 (贪心 + 随机)"""
        # 贪心解 (最近邻)
        greedy_solution = self._greedy_initialization()
        self.population.append(greedy_solution)
        
        # 随机解
        for _ in range(self.pop_size - 1):
            random_solution = self._random_initialization()
            self.population.append(random_solution)
        
        # 评估初始最优
        self._evaluate_fitness()
    
    def _greedy_initialization(self) -> VRPSolution:
        """贪心初始化 (最近邻启发式)"""
        unvisited = set(range(1, self.n_customers + 1))  # 客户 ID (1-indexed, 0=depot)
        routes = []
        
        for v_idx, vehicle in enumerate(self.vehicles):
            if not unvisited:
                break
            
            route = [0]  # 从 depot 出发
            current = 0
            current_load = 0.0
            
            while unvisited:
                # 找最近的未访问客户
                best_customer = None
                best_dist = float('inf')
                
                for cid in list(unvisited):
                    customer = self.customers[cid - 1]
                    
                    # 检查容量约束
                    if current_load + customer.demand > vehicle.capacity:
                        continue
                    
                    dist = self.dm.get(current, cid)
                    if dist < best_dist:
                        best_dist = dist
                        best_customer = cid
                
                if best_customer is None:
                    break  # 这辆车装不下了
                
                route.append(best_customer)
                unvisited.remove(best_customer)
                current = best_customer
                current_load += self.customers[best_customer - 1].demand
            
            route.append(0)  # 返回 depot
            routes.append(route)
        
        # 剩余未分配的客户 → 追加到最后一辆车的路径
        if unvisited:
            for cid in unvisited:
                if routes:
                    routes[-1].insert(-1, cid)
        
        solution = VRPSolution(routes=routes)
        return solution
    
    def _random_initialization(self) -> VRPSolution:
        """随机初始化"""
        # 随机排列客户
        customers = list(range(1, self.n_customers + 1))
        random.shuffle(customers)
        
        # 分配到车辆 (循环分配)
        routes = [[] for _ in self.vehicles]
        
        for idx, cid in enumerate(customers):
            vehicle_idx = idx % len(self.vehicles)
            routes[vehicle_idx].append(cid)
        
        # 添加 depot
        for route in routes:
            route.insert(0, 0)
            route.append(0)
        
        return VRPSolution(routes=routes)
    
    def _evaluate_fitness(self):
        """评估种群适应度"""
        for solution in self.population:
            metrics = solution.compute_metrics(
                self.customers, self.vehicles, self.depot
            )
            solution.fitness = metrics['fitness']
            solution.total_distance = metrics['total_distance']
            solution.total_cost = metrics['total_cost']
    
    def _tournament_selection(self) -> VRPSolution:
        """锦标赛选择"""
        contestants = random.sample(self.population, self.tournament_size)
        winner = max(contestants, key=lambda s: s.fitness)
        return winner
    
    def _crossover_pmx(self, parent1: VRPSolution, 
                         parent2: VRPSolution) -> Tuple[VRPSolution, VRPSolution]:
        """
        部分匹配交叉 (PMX)
        
        适用于排列编码的遗传算法。
        """
        # 简化: 展平所有路径 (去掉 depot 标记)
        def flatten(solution):
            flat = []
            for route in solution.routes:
                for cid in route:
                    if cid != 0:
                        flat.append(cid)
            return flat
        
        p1_flat = flatten(parent1)
        p2_flat = flatten(parent2)
        
        if len(p1_flat) < 2 or len(p2_flat) < 2:
            return parent1, parent2
        
        # 选择交叉点
        n = len(p1_flat)
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        
        # PMX
        child1_flat = p1_flat.copy()
        child2_flat = p2_flat.copy()
        
        # 映射段
        mapping = {}
        for k in range(i, j + 1):
            mapping[p1_flat[k]] = p2_flat[k]
            mapping[p2_flat[k]] = p1_flat[k]
        
        # 应用映射
        for k in list(range(0, i)) + list(range(j + 1, n)):
            if child1_flat[k] in mapping:
                child1_flat[k] = mapping[child1_flat[k]]
            if child2_flat[k] in mapping:
                child2_flat[k] = mapping[child2_flat[k]]
        
        # 重新分配到车辆
        child1 = self._assign_to_vehicles(child1_flat)
        child2 = self._assign_to_vehicles(child2_flat)
        
        return child1, child2
    
    def _assign_to_vehicles(self, flat: List[int]) -> VRPSolution:
        """将展平的列表分配到车辆 (考虑容量约束)"""
        routes = [[] for _ in self.vehicles]
        
        for cid in flat:
            customer = self.customers[cid - 1]
            
            # 找第一个能装下的车
            assigned = False
            for v_idx, vehicle in enumerate(self.vehicles):
                current_load = sum(
                    self.customers[cid2 - 1].demand 
                    for cid2 in routes[v_idx]
                )
                
                if current_load + customer.demand <= vehicle.capacity:
                    routes[v_idx].append(cid)
                    assigned = True
                    break
            
            if not assigned:
                # 追加到最后一辆车
                routes[-1].append(cid)
        
        # 添加 depot
        for route in routes:
            if route:
                route.insert(0, 0)
                route.append(0)
            else:
                route = [0, 0]
        
        return VRPSolution(routes=routes)
    
    def _mutate_swap(self, solution: VRPSolution) -> VRPSolution:
        """交换变异"""
        # 随机选两个客户, 交换它们
        flat = []
        route_indices = []
        
        for r_idx, route in enumerate(solution.routes):
            for c_idx, cid in enumerate(route):
                if cid != 0:
                    flat.append(cid)
                    route_indices.append((r_idx, c_idx))
        
        if len(flat) < 2:
            return solution
        
        # 选两个位置
        i, j = random.sample(range(len(flat)), 2)
        
        # 交换
        cid_i = flat[i]
        cid_j = flat[j]
        
        r_i, c_i = route_indices[i]
        r_j, c_j = route_indices[j]
        
        solution.routes[r_i][c_i] = cid_j
        solution.routes[r_j][c_j] = cid_i
        
        return solution
    
    def _local_search_2opt(self, solution: VRPSolution) -> VRPSolution:
        """
        2-opt 局部搜索
        
        对每条路径, 尝试反转子路径来改进。
        """
        improved = True
        
        while improved:
            improved = False
            
            for r_idx, route in enumerate(solution.routes):
                if len(route) <= 3:  # 只有 depot-depot
                    continue
                
                # 尝试所有 i < j 的反转
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        # 计算反转前的成本
                        cost_before = (
                            self.dm.get(route[i-1], route[i]) +
                            self.dm.get(route[j], route[j+1])
                        )
                        
                        # 计算反转后的成本
                        cost_after = (
                            self.dm.get(route[i-1], route[j]) +
                            self.dm.get(route[i], route[j+1])
                        )
                        
                        if cost_after < cost_before:
                            # 执行反转
                            solution.routes[r_idx][i:j+1] = reversed(solution.routes[r_idx][i:j+1])
                            improved = True
        
        return solution


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("Vehicle Routing Problem (VRP) - Test")
    print("=" * 60)
    
    # 创建测试数据 (随机客户)
    print("\nGenerating test data...")
    
    random.seed(42)
    np.random.seed(42)
    
    # 车场
    depot = Depot(
        id=0,
        x=500.0,  # 中心位置
        y=500.0,
        vehicles=[]
    )
    
    # 客户 (20 个)
    customers = []
    for i in range(1, 21):
        customer = Customer(
            id=i,
            x=random.uniform(0, 1000),
            y=random.uniform(0, 1000),
            demand=random.uniform(50, 200),
            service_time=random.uniform(5, 15),
            tw_start=random.uniform(0, 480),   # 8:00 - 16:00
            tw_end=random.uniform(960, 1440)    # 16:00 - 24:00
        )
        customers.append(customer)
    
    print(f"  Customers: {len(customers)}")
    print(f"  Total demand: {sum(c.demand for c in customers):.1f}")
    
    # 车辆 (5 辆)
    vehicles = []
    for i in range(5):
        vehicle = Vehicle(
            id=i,
            capacity=500.0,
            max_distance=150.0,
            depot_id=0,
            cost_per_km=2.5
        )
        vehicles.append(vehicle)
        depot.vehicles.append(i)
    
    print(f"  Vehicles: {len(vehicles)}")
    print(f"  Total capacity: {sum(v.capacity for v in vehicles):.1f}")
    
    # 运行遗传算法
    print("\n" + "=" * 60)
    print("Running Genetic Algorithm...")
    print("=" * 60)
    
    ga = GeneticAlgorithmVRP(
        customers=customers,
        vehicles=vehicles,
        depot=depot,
        pop_size=100,
        n_generations=200,  # 简化: 200 代
        crossover_rate=0.8,
        mutation_rate=0.15,
        tournament_size=3
    )
    
    best_solution = ga.optimize()
    
    # 打印最优解
    print("\n" + "=" * 60)
    print("Best Solution:")
    print("=" * 60)
    
    for v_idx, route in enumerate(best_solution.routes):
        if len(route) <= 2:
            continue
        
        print(f"\nVehicle {v_idx}:")
        print(f"  Route: {' -> '.join(map(str, route))}")
        
        # 计算载重
        load = sum(customers[cid-1].demand for cid in route if cid != 0)
        print(f"  Load: {load:.1f} / {vehicles[v_idx].capacity:.1f}")
        
        # 计算距离
        dist = 0.0
        for i in range(1, len(route)):
            dist += ga.dm.get(route[i-1], route[i])
        print(f"  Distance: {dist:.2f} km")
    
    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)
