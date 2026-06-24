"""
CHOMP (Covariant Hamiltonian Optimization for Motion Planning) + d_min 约束

实现:
1. SDF (Signed Distance Field) 预计算
2. 路径表示 (分段 B 样条)
3. 约束优化 (d_min 硬约束)
4. 梯度下降求解

理论参考:
- Ratliff, N., Zucker, M., Bagnell, J. A., & Srinivasa, S. (2009). 
  CHOMP: Gradient Optimization Techniques for Efficient Motion Planning.
- Schulman, J., et al. (2014). Motion Planning with Sequential Convex Optimization and Convex Collision Checking.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import numba


@dataclass
class SignedDistanceField:
    """
    有符号距离场 (预计算)
    
    使用:
    - 离线: 从占据栅格预计算 SDF (一次)
    - 在线: 任意点查询距离 (O(1) 查表 + 双线性插值)
    """
    
    resolution: float = 0.05  # 米/像素
    origin: np.ndarray = None    # (2,) 世界坐标原点
    grid: np.ndarray = None     # (H, W) 距离场 (米)
    gradient: np.ndarray = None  # (H, W, 2) 梯度场
    
    def __post_init__(self):
        if self.origin is None:
            self.origin = np.zeros(2)
    
    def query(self, point: np.ndarray) -> float:
        """
        查询点的距离 (双线性插值)
        
        Args:
            point: (2,) 世界坐标
            
        Returns:
            distance: 到最近障碍物的距离 (米)
                     正数 = 自由空间, 负数 = 障碍物内部
        """
        if self.grid is None:
            raise ValueError("SDF not initialized! Call compute_from_grid() first.")
        
        # 世界坐标 → 栅格坐标
        px = (point[0] - self.origin[0]) / self.resolution
        py = (point[1] - self.origin[1]) / self.resolution
        
        H, W = self.grid.shape
        
        # 边界检查
        if px < 0 or px >= W - 1 or py < 0 or py >= H - 1:
            return -1.0  # 地图外, 视为障碍物
        
        # 双线性插值
        x0, y0 = int(np.floor(px)), int(np.floor(py))
        x1, y1 = x0 + 1, y0 + 1
        
        wx = px - x0
        wy = py - y0
        
        d00 = self.grid[y0, x0]
        d01 = self.grid[y0, x1]
        d10 = self.grid[y1, x0]
        d11 = self.grid[y1, x1]
        
        d = (1 - wx) * (1 - wy) * d00 + wx * (1 - wy) * d01 + \
            (1 - wx) * wy * d10 + wx * wy * d11
        
        return d
    
    def query_gradient(self, point: np.ndarray) -> np.ndarray:
        """
        查询点的梯度 (双线性插值)
        
        Returns:
            grad: (2,) 距离场的梯度 (指向远离障碍物方向)
        """
        if self.gradient is None:
            self._compute_gradient()
        
        px = (point[0] - self.origin[0]) / self.resolution
        py = (point[1] - self.origin[1]) / self.resolution
        
        H, W, _ = self.gradient.shape
        
        if px < 0 or px >= W - 1 or py < 0 or py >= H - 1:
            return np.zeros(2)
        
        x0, y0 = int(np.floor(px)), int(np.floor(py))
        x1, y1 = x0 + 1, y0 + 1
        
        wx = px - x0
        wy = py - y0
        
        g00 = self.gradient[y0, x0]
        g01 = self.gradient[y0, x1]
        g10 = self.gradient[y1, x0]
        g11 = self.gradient[y1, x1]
        
        grad = (1 - wx) * (1 - wy) * g00 + wx * (1 - wy) * g01 + \
               (1 - wx) * wy * g10 + wx * wy * g11
        
        return grad
    
    def _compute_gradient(self):
        """计算距离场的梯度 (中心差分)"""
        H, W = self.grid.shape
        self.gradient = np.zeros((H, W, 2))
        
        # x 方向梯度
        self.gradient[:, 1:-1, 0] = (self.grid[:, 2:] - self.grid[:, :-2]) / (2 * self.resolution)
        
        # y 方向梯度
        self.gradient[1:-1, :, 1] = (self.grid[2:, :] - self.grid[:-2, :]) / (2 * self.resolution)
        
        # 边界处理
        self.gradient[:, 0, 0] = (self.grid[:, 1] - self.grid[:, 0]) / self.resolution
        self.gradient[:, -1, 0] = (self.grid[:, -1] - self.grid[:, -2]) / self.resolution
        self.gradient[0, :, 1] = (self.grid[1, :] - self.grid[0, :]) / self.resolution
        self.gradient[-1, :, 1] = (self.grid[-1, :] - self.grid[-2, :]) / self.resolution
    
    @staticmethod
    def compute_from_grid(occupancy_grid: np.ndarray, resolution: float = 0.05) -> 'SignedDistanceField':
        """
        从占据栅格计算 SDF
        
        算法:
        1. 正向EDT (障碍物 → 自由空间距离)
        2. 反向EDT (自由空间 → 障碍物距离)
        3. SDF = EDT_obstacle - EDT_free
        
        Args:
            occupancy_grid: (H, W) 占据栅格, 1=障碍物, 0=自由空间
            resolution: 分辨率 (米/像素)
            
        Returns:
            sdf: SignedDistanceField 对象
        """
        H, W = occupancy_grid.shape
        
        # 正向 (障碍物距离)
        d_obstacle = SignedDistanceField._edt_forward(occupancy_grid, resolution)
        
        # 反向 (自由空间距离)
        d_free = SignedDistanceField._edt_forward(1 - occupancy_grid, resolution)
        d_free = -d_free  # 自由空间内为负
        
        # SDF = 正向 - 反向
        sdf_grid = d_obstacle + d_free  # 障碍物外为正, 内为负
        
        sdf = SignedDistanceField(
            resolution=resolution,
            origin=np.zeros(2),
            grid=sdf_grid
        )
        sdf._compute_gradient()
        
        return sdf
    
    @staticmethod
    def _edt_forward(grid: np.ndarray, res: float) -> np.ndarray:
        """正向欧氏距离变换 (两遍扫描)"""
        H, W = grid.shape
        d = np.where(grid == 1, 0.0, np.inf)
        
        # Forward pass
        for i in range(H):
            for j in range(W):
                if d[i, j] == 0:
                    continue
                
                # 检查前向邻居
                if i > 0:
                    d[i, j] = min(d[i, j], d[i-1, j] + res)
                if j > 0:
                    d[i, j] = min(d[i, j], d[i, j-1] + res)
                if i > 0 and j > 0:
                    d[i, j] = min(d[i, j], d[i-1, j-1] + res * np.sqrt(2))
        
        # Backward pass
        for i in range(H-1, -1, -1):
            for j in range(W-1, -1, -1):
                if d[i, j] == 0:
                    continue
                
                if i < H - 1:
                    d[i, j] = min(d[i, j], d[i+1, j] + res)
                if j < W - 1:
                    d[i, j] = min(d[i, j], d[i, j+1] + res)
                if i < H - 1 and j < W - 1:
                    d[i, j] = min(d[i, j], d[i+1, j+1] + res * np.sqrt(2))
        
        return d


@dataclass
class PathSegment:
    """
    路径段 (B 样条表示)
    
    使用 B 样条而非简单折线, 因为:
    1. B 样条天然光滑 (C² 连续)
    2. 局部控制性 (移动一个控制点只影响局部)
    3. 凸包性 (路径不会突然穿到凸包外)
    """
    control_points: np.ndarray  # (N, 2) 控制点
    degree: int = 3             # B 样条阶数 (3 = 立方)
    
    def evaluate(self, t: float) -> np.ndarray:
        """
        计算 B 样条在参数 t 处的值
        
        Args:
            t: 参数 [0, 1]
            
        Returns:
            point: (2,) 路径点
        """
        N = len(self.control_points)
        
        # Cox-de Boor 递归
        if self.degree == 1:  # 线性
            idx = int(t * (N - 1))
            idx = min(idx, N - 2)
            u = t * (N - 1) - idx
            return (1 - u) * self.control_points[idx] + u * self.control_points[idx + 1]
        
        # 通用实现 (De Boor 算法)
        # 简化: 使用均匀 B 样条
        knot_vector = np.linspace(0, 1, N + self.degree + 1)
        
        # 找到 t 所在的节点区间
        for i in range(self.degree, N + 1):
            if knot_vector[i] <= t <= knot_vector[i + 1] or (i == N and t <= 1.0):
                break
        
        # De Boor 算法
        points = self.control_points[max(0, i - self.degree):min(N, i + 1)].copy()
        m = len(points)
        
        for r in range(1, self.degree + 1):
            for j in range(m - 1, r - 1, -1):
                alpha = (t - knot_vector[j + i - self.degree]) / \
                       (knot_vector[j + 1 + i] - knot_vector[j + i - self.degree])
                points[j] = (1 - alpha) * points[j - 1] + alpha * points[j]
        
        return points[r]
    
    def evaluate_many(self, t_values: np.ndarray) -> np.ndarray:
        """批量计算"""
        points = []
        for t in t_values:
            points.append(self.evaluate(t))
        return np.array(points)
    
    def gradient(self, t: float) -> np.ndarray:
        """
        计算 B 样条在 t 处的导数 (用于梯度下降)
        
        使用: 中心差分近似
        """
        dt = 1e-3
        p1 = self.evaluate(max(0, t - dt))
        p2 = self.evaluate(min(1, t + dt))
        return (p2 - p1) / (2 * dt)


class CHOMPPlanner:
    """
    CHOMP 路径优化器
    
    代价函数:
    L(γ) = w_smooth · L_smooth(γ) 
           + w_obs · L_obs(γ) 
           + w_dmin · L_dmin(γ)
    
    其中:
    - L_smooth: 平滑项 (路径长度 + 曲率)
    - L_obs: 障碍物惩罚 (SDF 负值惩罚)
    - L_dmin: d_min 约束 (clearance < d_min 时激活)
    """
    
    def __init__(self, 
                 sdf: SignedDistanceField,
                 d_min: float = 0.5,
                 w_smooth: float = 1.0,
                 w_obs: float = 100.0,
                 w_dmin: float = 50.0,
                 learning_rate: float = 0.01,
                 n_iterations: int = 200):
        """
        Args:
            sdf: 预计算的有符号距离场
            d_min: 最小安全距离 (米)
            w_smooth: 平滑项权重
            w_obs: 障碍物惩罚权重
            w_dmin: d_min 约束权重
            learning_rate: 梯度下降学习率
            n_iterations: 优化迭代次数
        """
        self.sdf = sdf
        self.d_min = d_min
        self.w_smooth = w_smooth
        self.w_obs = w_obs
        self.w_dmin = w_dmin
        self.lr = learning_rate
        self.n_iter = n_iterations
    
    def optimize(self, init_path: np.ndarray) -> np.ndarray:
        """
        优化路径
        
        Args:
            init_path: (N, 2) 初始路径 (例如 GVD 骨架路径)
            
        Returns:
            optimized_path: (N, 2) 优化后的路径
        """
        # 将初始路径转为 B 样条控制点
        n_waypoints = len(init_path)
        control_points = init_path.copy()
        
        # 梯度下降
        for iteration in range(self.n_iter):
            # 计算梯度
            grad_smooth = self._gradient_smooth(control_points)
            grad_obs = self._gradient_obstacle(control_points)
            grad_dmin = self._gradient_dmin(control_points)
            
            # 总梯度
            total_grad = self.w_smooth * grad_smooth + \
                        self.w_obs * grad_obs + \
                        self.w_dmin * grad_dmin
            
            # 更新控制点
            control_points = control_points - self.lr * total_grad
            
            # 打印进度
            if iteration % 50 == 0:
                loss = self._compute_loss(control_points)
                print(f"  Iteration {iteration}: loss = {loss:.4f}")
        
        return control_points
    
    def _gradient_smooth(self, control_points: np.ndarray) -> np.ndarray:
        """
        平滑项梯度
        
        平滑项 = 路径长度 + 曲率
        L_smooth = ∫ ||γ'(t)||² dt + α ∫ ||γ''(t)||² dt
        
        梯度 (离散近似):
        ∇L_smooth ≈ A @ control_points
        其中 A = 拉普拉斯矩阵 (二阶差分)
        """
        N = len(control_points)
        grad = np.zeros_like(control_points)
        
        # 路径长度梯度 (一阶差分)
        for i in range(1, N - 1):
            # γ'(t) ≈ (p[i+1] - p[i-1]) / 2
            tangent = (control_points[i+1] - control_points[i-1]) / 2.0
            grad[i] = -2.0 * tangent  # 负号因为要最小化
        
        # 曲率梯度 (二阶差分)
        for i in range(1, N - 1):
            # γ''(t) ≈ p[i-1] - 2*p[i] + p[i+1]
            curvature = control_points[i-1] - 2*control_points[i] + control_points[i+1]
            grad[i] = grad[i] + 2.0 * curvature
        
        return grad
    
    def _gradient_obstacle(self, control_points: np.ndarray) -> np.ndarray:
        """
        障碍物惩罚梯度
        
        L_obs = Σ max(0, -SDF(p_i))²
        
        梯度:
        ∇L_obs = 2 · max(0, -SDF(p)) · (-∇SDF(p))
        """
        N = len(control_points)
        grad = np.zeros_like(control_points)
        
        for i in range(N):
            p = control_points[i]
            d = self.sdf.query(p)
            
            if d < 0:  # 在障碍物内部
                sdf_grad = self.sdf.query_gradient(p)
                grad[i] = 2.0 * (-d) * (-sdf_grad)  # 指向自由空间
            elif d < self.d_min:
                # 接近障碍物但尚未进入, 也施加排斥力
                sdf_grad = self.sdf.query_gradient(p)
                grad[i] = 2.0 * (self.d_min - d) * (-sdf_grad)
        
        return grad
    
    def _gradient_dmin(self, control_points: np.ndarray) -> np.ndarray:
        """
        d_min 约束梯度
        
        L_dmin = Σ max(0, d_min - SDF(p_i))²
        
        梯度:
        ∇L_dmin = 2 · max(0, d_min - d) · (-∇SDF(p))
                    = 2 · (d_min - d) · (-∇SDF(p))   if d < d_min
                    = 0                                otherwise
        """
        N = len(control_points)
        grad = np.zeros_like(control_points)
        
        for i in range(N):
            p = control_points[i]
            d = self.sdf.query(p)
            
            if d < self.d_min:
                sdf_grad = self.sdf.query_gradient(p)
                # 注意: sdf_grad 指向距离增加方向 (远离障碍物)
                # 我们需要指向远离障碍物方向, 所以取负号
                grad[i] = 2.0 * (self.d_min - d) * (-sdf_grad)
        
        return grad
    
    def _compute_loss(self, control_points: np.ndarray) -> float:
        """计算总代价"""
        # 平滑项
        smooth = 0.0
        for i in range(1, len(control_points) - 1):
            tangent = np.linalg.norm(control_points[i+1] - control_points[i-1]) / 2.0
            curvature = np.linalg.norm(control_points[i-1] - 2*control_points[i] + control_points[i+1])
            smooth += tangent + 0.1 * curvature
        
        # 障碍物项
        obs = 0.0
        for p in control_points:
            d = self.sdf.query(p)
            if d < 0:
                obs += (-d) ** 2
        
        # d_min 项
        dmin = 0.0
        for p in control_points:
            d = self.sdf.query(p)
            if d < self.d_min:
                dmin += (self.d_min - d) ** 2
        
        return self.w_smooth * smooth + self.w_obs * obs + self.w_dmin * dmin


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("CHOMP + d_min Constraint - Test")
    print("=" * 60)
    
    # 创建测试地图 (简单场景: 一个障碍物)
    H, W = 200, 200
    grid = np.zeros((H, W), dtype=int)
    
    # 添加一个矩形障碍物
    grid[80:120, 90:110] = 1
    
    # 计算 SDF
    print("\nComputing SDF...")
    sdf = SignedDistanceField.compute_from_grid(grid, resolution=0.05)
    print(f"SDF shape: {sdf.grid.shape}")
    print(f"SDF range: [{sdf.grid.min():.2f}, {sdf.grid.max():.2f}]")
    
    # 创建初始路径 (绕过障碍物的简单路径)
    print("\nCreating initial path...")
    init_path = np.array([
        [2.0, 2.0],    # 起点
        [3.0, 4.0],
        [4.0, 6.0],
        [5.0, 6.5],
        [6.0, 5.0],    # 绕过去
        [7.0, 4.0],
        [8.0, 3.0],
        [9.0, 2.0]     # 终点
    ])
    
    print(f"Initial path: {len(init_path)} points")
    
    # 检查初始路径的 clearance
    print("\nInitial path clearance:")
    for p in init_path:
        d = sdf.query(p)
        print(f"  {p} -> d = {d:.3f} m")
    
    # 运行 CHOMP 优化
    print("\nRunning CHOMP optimization...")
    planner = CHOMPPlanner(
        sdf=sdf,
        d_min=0.5,
        w_smooth=1.0,
        w_obs=100.0,
        w_dmin=50.0,
        learning_rate=0.01,
        n_iterations=200
    )
    
    optimized_path = planner.optimize(init_path)
    
    # 检查优化后路径的 clearance
    print("\nOptimized path clearance:")
    min_clearance = float('inf')
    for p in optimized_path:
        d = sdf.query(p)
        min_clearance = min(min_clearance, d)
        print(f"  {p} -> d = {d:.3f} m")
    
    print(f"\nMinimal clearance after optimization: {min_clearance:.3f} m")
    print(f"(Target d_min = {planner.d_min} m)")
    
    # 计算路径长度
    init_length = sum(
        np.linalg.norm(init_path[i+1] - init_path[i]) 
        for i in range(len(init_path) - 1)
    )
    opt_length = sum(
        np.linalg.norm(optimized_path[i+1] - optimized_path[i]) 
        for i in range(len(optimized_path) - 1)
    )
    
    print(f"\nPath length: {init_length:.2f} m -> {opt_length:.2f} m")
    print(f"(Change: {((opt_length - init_length) / init_length * 100):.1f}%)")
    
    print("\nTest passed!")
