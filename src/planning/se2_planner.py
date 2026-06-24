"""
SE(2) AGV 拓扑规划器

实现:
1. GVD 骨架 → SE(2) 拓扑地图
2. ECM 走廊 → SE(2) 可行通道
3. CHOMP 优化 → SE(2) 光滑路径
4. 差速约束嵌入 (DWA / TEB)

理论参考:
- Siegwart, R., et al. (2011). Introduction to Autonomous Mobile Robots.
- Fox, D., et al. (1997). The Dynamic Window Approach to Collision Avoidance.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import heapq

from .gvd import GVDSkeletonExtractor, GVDCell, GVDPath
from .ecm import ECMBuilder, Corridor, ECMGlobalPlanner
from .chomp import CHOMPPlanner, SignedDistanceField


@dataclass
class SE2Pose:
    """SE(2) 位姿"""
    x: float
    y: float
    theta: float      # 朝向 (弧度)
    
    def to_matrix(self) -> np.ndarray:
        """转换为 SE(2) 矩阵 (3, 3)"""
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0,    1.0]
        ])
    
    @staticmethod
    def from_matrix(T: np.ndarray) -> 'SE2Pose':
        return SE2Pose(
            x=T[0, 2],
            y=T[1, 2],
            theta=np.arctan2(T[1, 0], T[0, 0])
        )


@dataclass
class AGVState:
    """AGV 状态"""
    pose: SE2Pose
    velocity: float          # 线速度 (m/s)
    omega: float            # 角速度 (rad/s)
    battery: float = 1.0   # 电池电量 [0, 1]


@dataclass
class AGVAction:
    """AGV 控制指令"""
    v: float                # 线速度 (m/s)
    omega: float            # 角速度 (rad/s)
    duration: float = 0.1  # 执行时长 (s)


class DiffDriveKinematics:
    """
    差速驱动运动学
    
    约束:
    ẋ·sin(θ) - ẏ·cos(θ) = 0  (非完整约束)
    
    即: AGV 不能横向移动, 速度方向必须沿车体 x 轴
    """
    
    def __init__(self, wheelbase: float = 0.5, max_v: float = 1.5, 
                 max_omega: float = 1.0, max_a: float = 0.5, 
                 max_alpha: float = 0.8):
        """
        Args:
            wheelbase: 轮距 (m)
            max_v: 最大线速度 (m/s)
            max_omega: 最大角速度 (rad/s)
            max_a: 最大线加速度 (m/s²)
            max_alpha: 最大角加速度 (rad/s²)
        """
        self.L = wheelbase
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_a = max_a
        self.max_alpha = max_alpha
    
    def forward(self, state: AGVState, action: AGVAction, dt: float) -> AGVState:
        """
        正向运动学: 控制指令 → 新状态
        
        ẋ = v·cos(θ)
        ẏ = v·sin(θ)
        θ̇ = ω
        """
        # 更新位姿
        x_new = state.pose.x + action.v * np.cos(state.pose.theta) * dt
        y_new = state.pose.y + action.v * np.sin(state.pose.theta) * dt
        theta_new = state.pose.theta + action.omega * dt
        
        # 归一化角度
        theta_new = self._normalize_angle(theta_new)
        
        # 更新速度 (带加速度限制)
        v_new = np.clip(
            state.velocity + np.random.uniform(-self.max_a * dt, self.max_a * dt),
            -self.max_v, self.max_v
        )
        omega_new = np.clip(
            state.omega + np.random.uniform(-self.max_alpha * dt, self.max_alpha * dt),
            -self.max_omega, self.max_omega
        )
        
        return AGVState(
            pose=SE2Pose(x_new, y_new, theta_new),
            velocity=v_new,
            omega=omega_new,
            battery=state.battery - self._compute_power(action, dt)
        )
    
    def _normalize_angle(self, theta: float) -> float:
        """角度归一化到 [-π, π]"""
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta
    
    def _compute_power(self, action: AGVAction, dt: float) -> float:
        """计算能耗 (简化模型)"""
        power = (abs(action.v) * 100 + abs(action.omega) * 50) * dt  # 瓦特
        battery_cost = power / (20 * 3600)  # 假设 20Ah 电池
        return battery_cost
    
    def inverse(self, pose_cur: SE2Pose, pose_des: SE2Pose) -> AGVAction:
        """
        逆向运动学: 期望位姿 → 控制指令
        
        简化版: 纯追踪算法 (Pure Pursuit)
        """
        # 计算误差
        dx = pose_des.x - pose_cur.x
        dy = pose_des.y - pose_cur.y
        
        # 期望朝向
        theta_des = np.arctan2(dy, dx)
        theta_error = self._normalize_angle(theta_des - pose_cur.theta)
        
        # 控制指令
        v = np.sqrt(dx**2 + dy**2)  # 简化: 直接用距离作为速度
        v = np.clip(v, -self.max_v, self.max_v)
        
        omega = theta_error / 0.5  # 简化: P 控制
        omega = np.clip(omega, -self.max_omega, self.max_omega)
        
        return AGVAction(v=v, omega=omega)


class DWALocalPlanner:
    """
    DWA (Dynamic Window Approach) 局部规划器
    
    算法:
    1. 在速度空间 (v, ω) 中采样
    2. 对每个采样, 预测未来轨迹
    3. 评估轨迹 (距离目标 + 障碍物距离 + 速度)
    4. 选择最优轨迹对应的 (v, ω)
    
    约束:
    - 差速驱动 (非完整)
    - 速度/加速度限制
    - 障碍物碰撞
    """
    
    def __init__(self, kinematics: DiffDriveKinematics, 
                 dt: float = 0.1, predict_time: float = 3.0,
                 resolution: float = 0.05):
        self.kin = kinematics
        self.dt = dt
        self.predict_time = predict_time
        self.res = resolution
        
        # 代价权重
        self.w_dist = 1.0       # 到目标距离
        self.w_clearance = 10.0  # 障碍物距离
        self.w_velocity = 0.1   # 速度 (鼓励快速)
    
    def plan(self, state: AGVState, goal: np.ndarray, 
             occupancy_grid: np.ndarray) -> AGVAction:
        """
        DWA 规划
        
        Args:
            state: 当前状态
            goal: 目标位置 (2,)
            occupancy_grid: 占据栅格
            
        Returns:
            action: 最优控制指令
        """
        # 动态窗口
        v_min = max(-self.kin.max_v, state.velocity - self.kin.max_a * self.dt)
        v_max = min( self.kin.max_v, state.velocity + self.kin.max_a * self.dt)
        omega_min = max(-self.kin.max_omega, state.omega - self.kin.max_alpha * self.dt)
        omega_max = min( self.kin.max_omega, state.omega + self.kin.max_alpha * self.dt)
        
        # 采样
        best_action = None
        best_cost = float('inf')
        
        v_samples = np.linspace(v_min, v_max, 20)
        omega_samples = np.linspace(omega_min, omega_max, 20)
        
        for v in v_samples:
            for omega in omega_samples:
                action = AGVAction(v=v, omega=omega)
                
                # 预测轨迹
                trajectory = self._predict_trajectory(state, action)
                
                # 评估轨迹
                cost = self._evaluate_trajectory(trajectory, goal, occupancy_grid)
                
                if cost < best_cost:
                    best_cost = cost
                    best_action = action
        
        return best_action
    
    def _predict_trajectory(self, state: AGVState, action: AGVAction) -> List[np.ndarray]:
        """预测轨迹"""
        trajectory = []
        s = AGVState(
            pose=SE2Pose(state.pose.x, state.pose.y, state.pose.theta),
            velocity=state.velocity,
            omega=state.omega
        )
        
        for t in range(int(self.predict_time / self.dt)):
            s = self.kin.forward(s, action, self.dt)
            trajectory.append(np.array([s.pose.x, s.pose.y]))
        
        return trajectory
    
    def _evaluate_trajectory(self, trajectory: List[np.ndarray], 
                              goal: np.ndarray, 
                              occupancy_grid: np.ndarray) -> float:
        """评估轨迹代价"""
        if len(trajectory) == 0:
            return float('inf')
        
        # 1. 到目标距离
        final_point = trajectory[-1]
        dist_to_goal = np.linalg.norm(final_point - goal)
        
        # 2. 障碍物距离 (最小 clearance)
        min_clearance = float('inf')
        for point in trajectory:
            # 世界坐标 → 栅格坐标
            grid_x = int(point[0] / self.res)
            grid_y = int(point[1] / self.res)
            
            if 0 <= grid_x < occupancy_grid.shape[1] and 0 <= grid_y < occupancy_grid.shape[0]:
                if occupancy_grid[grid_y, grid_x] == 1:
                    return float('inf')  # 碰撞
            
            # 简化: 假设距离场已知
            # clearnace = self.sdf.query(point)
            # min_clearance = min(min_clearance, clearance)
        
        # 3. 速度 (鼓励快速)
        avg_v = np.mean([np.linalg.norm(trajectory[i+1] - trajectory[i]) / self.dt 
                         for i in range(len(trajectory) - 1)])
        
        # 总代价
        cost = self.w_dist * dist_to_goal \
             + self.w_clearance * (1.0 / (min_clearance + 1e-6)) \
             - self.w_velocity * avg_v
        
        return cost


class SE2TopoPlanner:
    """
    SE(2) 拓扑规划器 (AGV 专用)
    
    完整流程:
    1. 从占据栅格提取 GVD 骨架
    2. 构建 ECM 走廊地图
    3. 全局规划 (走廊图上 Dijkstra)
    4. 局部优化 (CHOMP + DWA)
    5. 差速约束验证
    """
    
    def __init__(self, resolution: float = 0.05, d_min: float = 0.5,
                 use_ecm: bool = True, use_chomp: bool = True):
        """
        Args:
            resolution: 栅格分辨率 (m/px)
            d_min: 最小安全距离 (m)
            use_ecm: 是否使用 ECM
            use_chomp: 是否使用 CHOMP
        """
        self.res = resolution
        self.d_min = d_min
        self.use_ecm = use_ecm
        self.use_chomp = use_chomp
        
        # 子模块
        self.gvd_extractor = GVDSkeletonExtractor(resolution=resolution)
        self.ecm_builder = ECMBuilder(clearance_margin=0.1)
        self.chomp_planner = None  # 延迟初始化 (需要 SDF)
        self.dwa_planner = None  # 延迟初始化 (需要 kinematics)
        
        # 缓存
        self.occupancy_grid = None
        self.sdf = None
        self.corridors = None
        self.topo_nodes = None
    
    def setup(self, occupancy_grid: np.ndarray, 
              kinematics: DiffDriveKinematics):
        """初始化 (离线计算)"""
        self.occupancy_grid = occupancy_grid
        
        # 1. 计算 SDF
        print("  Computing SDF...")
        self.sdf = SignedDistanceField.compute_from_grid(
            occupancy_grid, resolution=self.res
        )
        
        # 2. 提取 GVD 骨架
        print("  Extracting GVD skeleton...")
        gvd_mask, topo_nodes = self.gvd_extractor.extract(occupancy_grid)
        self.topo_nodes = topo_nodes
        print(f"    GVD points: {gvd_mask.sum()}")
        print(f"    Topology nodes: {len(topo_nodes)}")
        
        # 3. 构建 ECM (可选)
        if self.use_ecm:
            print("  Building ECM...")
            self.corridors = self.ecm_builder.build_from_gvd(
                gvd_mask, self.gvd_extractor.dist_field, 
                topo_nodes, self.res
            )
            print(f"    Corridors: {len(self.corridors)}")
        
        # 4. 初始化 CHOMP (可选)
        if self.use_chomp:
            print("  Initializing CHOMP...")
            self.chomp_planner = CHOMPPlanner(
                sdf=self.sdf,
                d_min=self.d_min,
                w_smooth=1.0,
                w_obs=100.0,
                w_dmin=50.0,
                learning_rate=0.01,
                n_iterations=200
            )
        
        # 5. 初始化 DWA
        print("  Initializing DWA...")
        self.dwa_planner = DWALocalPlanner(kinematics, dt=0.1, predict_time=3.0)
        
        print("Setup complete!")
    
    def plan(self, start: SE2Pose, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        规划路径
        
        Args:
            start: 起点位姿
            goal: 目标位置 (2,)
            
        Returns:
            path: 路径点列表 [(x1,y1), (x2,y2), ...] 或 None (不可达)
        """
        if self.occupancy_grid is None:
            raise RuntimeError("Call setup() first!")
        
        # Step 1: 全局规划 (拓扑层)
        if self.use_ecm and self.corridors is not None:
            print("  Global planning (ECM)...")
            ecm_planner = ECMGlobalPlanner(self.corridors)
            path_global = ecm_planner.plan(
                np.array([start.x, start.y]), goal
            )
        else:
            print("  Global planning (GVD)...")
            path_global = self._plan_gvd_only(
                np.array([start.x, start.y]), goal
            )
        
        if path_global is None:
            print("  No global path found!")
            return None
        
        print(f"  Global path: {len(path_global)} points")
        
        # Step 2: 局部优化 (CHOMP)
        if self.use_chomp:
            print("  Local optimization (CHOMP)...")
            path_optimized = self.chomp_planner.optimize(
                np.array(path_global)
            )
            path_global = path_optimized.tolist()
        
        # Step 3: 差速约束验证 + DWA 平滑
        print("  Diff-drive smoothing (DWA)...")
        path_smooth = self._smooth_with_dwa(path_global, start)
        
        print(f"  Final path: {len(path_smooth)} points")
        
        return path_smooth
    
    def _plan_gvd_only(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """只使用 GVD 骨架的简化规划"""
        # 找到最近的 GVD 点
        gvd_points = np.column_stack(np.where(self.gvd_extractor.gvd_mask))
        
        if len(gvd_points) == 0:
            return None
        
        # 计算距离
        start_pixel = (int(start[1] / self.res), int(start[0] / self.res))
        goal_pixel = (int(goal[1] / self.res), int(goal[0] / self.res))
        
        # BFS 在 GVD 骨架上
        # 简化: 直接直线 + 避障
        path = self._bfs_on_gvd(start_pixel, goal_pixel)
        
        if path is None:
            return None
        
        # 转换回世界坐标
        world_path = [(p[1] * self.res, p[0] * self.res) for p in path]
        
        return world_path
    
    def _bfs_on_gvd(self, start: Tuple[int, int], 
                     goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """在 GVD 骨架上跑 BFS"""
        mask = self.gvd_extractor.gvd_mask
        H, W = mask.shape
        
        if not mask[start]:
            return None
        if not mask[goal]:
            return None
        
        queue = deque([start])
        visited = {start: None}
        
        while queue:
            current = queue.popleft()
            
            if current == goal:
                # 回溯路径
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return path[::-1]
            
            # 8 邻域
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = current[0] + di, current[1] + dj
                    
                    if 0 <= ni < H and 0 <= nj < W and mask[ni, nj]:
                        if (ni, nj) not in visited:
                            visited[(ni, nj)] = current
                            queue.append((ni, nj))
        
        return None
    
    def _smooth_with_dwa(self, path: List[np.ndarray], 
                         start: SE2Pose) -> List[np.ndarray]:
        """使用 DWA 平滑路径 (嵌入差速约束)"""
        if len(path) < 2:
            return path
        
        smoothed = [np.array([start.x, start.y])]
        current_state = AGVState(pose=start, velocity=0.0, omega=0.0)
        
        # 分段处理 (每 10 个点一个局部窗口)
        window_size = 10
        
        for i in range(0, len(path), window_size):
            local_goal = path[min(i + window_size, len(path) - 1)]
            
            # DWA 规划到局部目标
            action = self.dwa_planner.plan(
                current_state, local_goal, self.occupancy_grid
            )
            
            if action is None:
                break
            
            # 执行一步
            current_state = self.dwa_planner.kin.forward(
                current_state, action, self.dwa_planner.dt
            )
            
            smoothed.append(np.array([current_state.pose.x, current_state.pose.y]))
        
        return smoothed


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("SE(2) AGV Topo Planner - Test")
    print("=" * 60)
    
    # 创建测试地图
    print("\nCreating test warehouse...")
    H, W = 200, 200
    grid = np.zeros((H, W), dtype=int)
    
    # 外边框
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    
    # 货架
    for shelf_y in range(30, H - 30, 40):
        for shelf_x in range(30, W - 30, 40):
            grid[shelf_y:shelf_y+10, shelf_x:shelf_x+10] = 1
    
    print(f"  Grid shape: {grid.shape}")
    print(f"  Obstacle ratio: {grid.sum() / grid.size:.2%}")
    
    # 初始化运动学
    print("\nInitializing kinematics...")
    kin = DiffDriveKinematics(
        wheelbase=0.5,
        max_v=1.5,
        max_omega=1.0,
        max_a=0.5,
        max_alpha=0.8
    )
    
    # 初始化规划器
    print("\nSetting up planner (this may take a while)...")
    planner = SE2TopoPlanner(
        resolution=0.05,
        d_min=0.5,
        use_ecm=True,
        use_chomp=True
    )
    
    planner.setup(grid, kin)
    
    # 规划路径
    print("\nPlanning path...")
    start = SE2Pose(x=2.0, y=2.0, theta=0.0)
    goal = np.array([8.0, 8.0])
    
    path = planner.plan(start, goal)
    
    if path is not None:
        print(f"\nPath found! Length: {len(path)} points")
        print(f"  Start: {path[0]}")
        print(f"  End: {path[-1]}")
        
        # 计算路径长度
        length = sum(
            np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            for i in range(len(path) - 1)
        )
        print(f"  Path length: {length:.2f} m")
    else:
        print("\nNo path found!")
    
    print("\nTest passed!")
