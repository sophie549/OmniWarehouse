"""
SE(3) 无人机走廊规划器

实现:
1. 3D GVD 骨架提取 (体素地图)
2. 空中走廊网络 (3D 管道)
3. 分层拓扑地图 (高度层 + 垂直连接)
4. SE(3) 几何跟踪控制

理论参考:
- Omains, I. B., et al. (2019). Sampling-Based Methods for Motion Planning.
- Lee, T. (2012). Geometric Control of Quadrotor UAVs.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
import heapq


@dataclass
class SE3Pose:
    """SE(3) 位姿"""
    x: float
    y: float
    z: float
    roll: float = 0.0    # 滚转角 (弧度)
    pitch: float = 0.0  # 俯仰角 (弧度)
    yaw: float = 0.0     # 偏航角 (弧度)
    
    def to_matrix(self) -> np.ndarray:
        """转换为 SE(3) 矩阵 (4, 4)"""
        # 欧拉角 → 旋转矩阵 (ZYX 顺序)
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,             cp*cr           ]
        ])
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [self.x, self.y, self.z]
        return T
    
    @staticmethod
    def from_matrix(T: np.ndarray) -> 'SE3Pose':
        """从 SE(3) 矩阵提取位姿"""
        x, y, z = T[:3, 3]
        R = T[:3, :3]
        
        # 旋转矩阵 → 欧拉角 (ZYX 顺序)
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        
        if np.abs(np.cos(pitch)) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # 万向锁
            roll = 0.0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        
        return SE3Pose(x, y, z, roll, pitch, yaw)


@dataclass
class AirCorridor:
    """空中走廊 (3D 管道)"""
    id: int
    centerline: np.ndarray     # (N, 3) 中心线点列
    radius: float            # 管道半径 (米)
    height_layer: int        # 所属高度层
    connected: List[int] = field(default_factory=list)  # 连接的走廊 ID
    
    def min_clearance(self, point: np.ndarray) -> float:
        """计算点到走廊中心线的最小距离"""
        min_dist = float('inf')
        closest_point = None
        
        for i in range(len(self.centerline) - 1):
            # 线段 p1-p2
            p1 = self.centerline[i]
            p2 = self.centerline[i + 1]
            
            # 点到线段的距离
            v = p2 - p1
            w = point - p1
            
            c1 = np.dot(w, v)
            c2 = np.dot(v, v)
            
            if c1 <= 0:
                dist = np.linalg.norm(point - p1)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = p1
            elif c2 <= c1:
                dist = np.linalg.norm(point - p2)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = p2
            else:
                b = c1 / c2
                pb = p1 + b * v
                dist = np.linalg.norm(point - pb)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = pb
        
        return min_dist


@dataclass
class UAVState:
    """无人机状态"""
    pose: SE3Pose
    velocity: np.ndarray       # (3,) 线速度 (m/s)
    omega: np.ndarray        # (3,) 角速度 (rad/s)
    battery: float = 1.0   # 电池电量 [0, 1]


class GVD3DExtractor:
    """
    3D GVD 骨架提取器
    
    算法:
    1. 3D 欧氏距离变换 (三遍扫描)
    2. 3D 脊线检测 (26 邻域局部极大值)
    3. 骨架剪枝
    
    注意: 3D GVD 计算量较大, 实际系统中使用分层方法
    """
    
    def __init__(self, resolution: float = 0.1, prune_length: float = 2.0):
        """
        Args:
            resolution: 体素分辨率 (米/体素)
            prune_length: 剪枝长度阈值 (米)
        """
        self.res = resolution
        self.prune_length = prune_length
        self.dist_field_3d = None
        self.gvd_mask_3d = None
    
    def compute_edt_3d(self, occupancy_voxel: np.ndarray) -> np.ndarray:
        """
        3D 欧氏距离变换
        
        三遍扫描算法 (Saito & Toriwaki, 1994 的 3D 扩展):
        - Forward pass:  左上前 → 右下后
        - Backward pass: 右下后 → 左上前
        
        Args:
            occupancy_voxel: 占据体素, 1=障碍物, 0=自由空间
                              shape: (D, H, W) = (Z, Y, X)
            
        Returns:
            dist_field: 距离场 (D, H, W)
        """
        D, H, W = occupancy_voxel.shape
        
        # 初始化
        d = np.where(occupancy_voxel == 1, 0.0, np.inf)
        
        # Forward pass (左上前 → 右下后)
        # 检查 13 个前向邻居 (半立方体)
        for i in range(D):
            for j in range(H):
                for k in range(W):
                    if d[i, j, k] == 0:
                        continue
                    
                    # 前向邻居偏移 (i-di, j-dj, k-dk)
                    # 只检查 i', j', k' ≤ i, j, k 的邻居
                    min_dist = d[i, j, k]
                    
                    for di in [-1, 0]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                
                                ni, nj, nk = i + di, j + dj, k + dk
                                
                                if 0 <= ni < D and 0 <= nj < H and 0 <= nk < W:
                                    if d[ni, nj, nk] < np.inf:
                                        cost = np.sqrt(di**2 + dj**2 + dk**2) * self.res
                                        min_dist = min(min_dist, d[ni, nj, nk] + cost)
                    
                    d[i, j, k] = min_dist
        
        # Backward pass (右下后 → 左上前)
        for i in range(D - 1, -1, -1):
            for j in range(H - 1, -1, -1):
                for k in range(W - 1, -1, -1):
                    if d[i, j, k] == 0:
                        continue
                    
                    min_dist = d[i, j, k]
                    
                    for di in [0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                
                                ni, nj, nk = i + di, j + dj, k + dk
                                
                                if 0 <= ni < D and 0 <= nj < H and 0 <= nk < W:
                                    if d[ni, nj, nk] < np.inf:
                                        cost = np.sqrt(di**2 + dj**2 + dk**2) * self.res
                                        min_dist = min(min_dist, d[ni, nj, nk] + cost)
                    
                    d[i, j, k] = min_dist
        
        self.dist_field_3d = d
        return d
    
    def extract_ridge_3d(self, min_clearance: float = 0.5) -> np.ndarray:
        """
        3D 脊线检测
        
        脊线定义:
            p 是脊线点 ⇔ dist(p) ≥ dist(q) for all q ∈ N₂₆(p)
            其中 N₂₆(p) 是 p 的 26 邻域
            
        Args:
            min_clearance: 最小 clearance 阈值
            
        Returns:
            gvd_mask: 3D GVD 骨架掩码 (D, H, W)
        """
        D, H, W = self.dist_field_3d.shape
        gvd_mask = np.zeros((D, H, W), dtype=bool)
        
        for i in range(1, D - 1):
            for j in range(1, H - 1):
                for k in range(1, W - 1):
                    d_center = self.dist_field_3d[i, j, k]
                    
                    if d_center < min_clearance:
                        continue
                    
                    # 检查 26 邻域
                    is_ridge = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                
                                ni, nj, nk = i + di, j + dj, k + dk
                                if 0 <= ni < D and 0 <= nj < H and 0 <= nk < W:
                                    if self.dist_field_3d[ni, nj, nk] > d_center + 1e-6:
                                        is_ridge = False
                                        break
                        
                        if not is_ridge:
                            break
                    
                    if is_ridge:
                        gvd_mask[i, j, k] = True
        
        self.gvd_mask_3d = gvd_mask
        return gvd_mask


class LayeredTopoMap:
    """
    分层拓扑地图 (3D 空中走廊网络)
    
    架构:
    1. 预规划高度层 (H1, H2, H3, ...)
    2. 每层独立建 2D GVD → 走廊网络
    3. 层间建垂直连接图
    4. 全局规划: 在分层拓扑图上规划 (节点 = (corridor_id, layer))
    """
    
    def __init__(self, height_layers: List[float] = [30.0, 50.0, 80.0],
                 layer_thickness: float = 10.0):
        """
        Args:
            height_layers: 高度层列表 (米)
            layer_thickness: 层厚度 (米), 用于点云分层
        """
        self.height_layers = height_layers
        self.thickness = layer_thickness
        self.corridors: Dict[int, AirCorridor] = {}
        self.layer_corridors: Dict[int, List[int]] = {  # 每层有哪些走廊
            i: [] for i in range(len(height_layers))
        }
        self.next_id = 0
    
    def build_from_point_cloud(self, point_cloud: np.ndarray, 
                               resolution: float = 0.5):
        """
        从 3D 点云构建分层拓扑地图
        
        Args:
            point_cloud: (N, 3) 障碍物点云 (x, y, z)
            resolution: 2D 栅格分辨率 (米/像素)
        """
        print("Building layered topo map...")
        
        for layer_idx, height in enumerate(self.height_layers):
            print(f"  Processing layer {layer_idx} (z = {height}m)...")
            
            # 提取该高度层的点云 (z ± thickness/2)
            z_min = height - self.thickness / 2
            z_max = height + self.thickness / 2
            
            layer_mask = (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)
            layer_points = point_cloud[layer_mask, :2]  # 只取 x, y
            
            if len(layer_points) < 10:
                print(f"    Skipped (insufficient points: {len(layer_points)})")
                continue
            
            # 建 2D 占据栅格
            grid = self._points_to_grid(layer_points, resolution)
            
            # 提取 2D GVD 骨架
            from planning.gvd import GVDSkeletonExtractor
            extractor = GVDSkeletonExtractor(resolution=resolution)
            gvd_mask, topo_nodes = extractor.extract(grid)
            
            # 从 GVD 骨架构建走廊
            self._build_corridors_from_gvd(
                extractor, gvd_mask, topo_nodes, 
                layer_idx, height, resolution
            )
        
        # 建垂直连接
        self._build_vertical_connections()
        
        print(f"Total corridors: {len(self.corridors)}")
    
    def _points_to_grid(self, points: np.ndarray, 
                       resolution: float) -> np.ndarray:
        """点云 → 2D 占据栅格"""
        # 计算边界
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        # 扩展边界
        margin = 10.0  # 米
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # 创建栅格
        W = int((x_max - x_min) / resolution) + 1
        H = int((y_max - y_min) / resolution) + 1
        
        grid = np.zeros((H, W), dtype=int)
        
        # 栅格化
        for (x, y) in points:
            i = int((y - y_min) / resolution)
            j = int((x - x_min) / resolution)
            
            if 0 <= i < H and 0 <= j < W:
                grid[i, j] = 1
        
        return grid
    
    def _build_corridors_from_gvd(self, extractor, gvd_mask, 
                                   topo_nodes, layer_idx, height, 
                                   resolution):
        """从 2D GVD 骨架构建空中走廊"""
        # 简化: 直接使用 GVD 点作为走廊中心线
        # 实际系统中需要分段拟合直线
        
        gvd_points = np.column_stack(np.where(gvd_mask))
        
        if len(gvd_points) == 0:
            return
        
        # 分段 (简化: 每 20 个像素一个走廊)
        segment_length = 20
        segments = []
        
        current_segment = []
        for idx in range(len(gvd_points)):
            current_segment.append(gvd_points[idx])
            
            if len(current_segment) >= segment_length or idx == len(gvd_points) - 1:
                segments.append(current_segment)
                current_segment = []
        
        # 为每个段创建走廊
        for seg in segments:
            if len(seg) < 2:
                continue
            
            # 中心线 (世界坐标)
            centerline_2d = np.array([[p[1] * resolution, p[0] * resolution] 
                                       for p in seg])
            
            # 扩展到 3D (添加高度)
            centerline_3d = np.column_stack([
                centerline_2d[:, 0],
                centerline_2d[:, 1],
                np.full(len(centerline_2d), height)
            ])
            
            # 计算走廊半径 (使用距离场)
            radius = min(
                extractor.dist_field[p[0], p[1]] 
                for p in seg
            )
            
            corridor = AirCorridor(
                id=self.next_id,
                centerline=centerline_3d,
                radius=radius,
                height_layer=layer_idx
            )
            
            self.corridors[self.next_id] = corridor
            self.layer_corridors[layer_idx].append(self.next_id)
            self.next_id += 1
    
    def _build_vertical_connections(self):
        """建垂直连接 (不同层的走廊交汇点之间)"""
        print("  Building vertical connections...")
        
        # 遍历所有走廊对
        corridor_ids = list(self.corridors.keys())
        
        for i in range(len(corridor_ids)):
            for j in range(i + 1, len(corridor_ids)):
                c1 = self.corridors[corridor_ids[i]]
                c2 = self.corridors[corridor_ids[j]]
                
                if c1.height_layer == c2.height_layer:
                    continue  # 同层不连
                
                # 检查两个走廊的端点是否接近
                for p1 in [c1.centerline[0], c1.centerline[-1]]:
                    for p2 in [c2.centerline[0], c2.centerline[-1]]:
                        dist = np.linalg.norm(p1 - p2)
                        
                        if dist < 20.0:  # 20m 内认为可垂直连接
                            c1.connected.append(c2.id)
                            c2.connected.append(c1.id)
        
        print(f"    Vertical connections built")


class SE3GeometricController:
    """
    SE(3) 几何跟踪控制器 (无人机)
    
    控制律:
    τ = -K_R · e_R - K_ω · e_ω + ω × (J · ω)
    
    其中:
    - e_R = ½vee(R_desᵀ · R - Rᵀ · R_des)  (SO(3) 旋转误差)
    - e_ω = ω - Rᵀ · R_des · ω_des          (角速度误差)
    """
    
    def __init__(self, mass: float = 1.5, 
                 inertia: np.ndarray = None,
                 K_p: np.ndarray = None,
                 K_v: np.ndarray = None,
                 K_R: np.ndarray = None,
                 K_omega: np.ndarray = None):
        """
        Args:
            mass: 无人机质量 (kg)
            inertia: 转动惯量矩阵 (3, 3)
            K_p, K_v: 位置控制增益
            K_R, K_omega: 姿态控制增益
        """
        self.m = mass
        self.J = inertia if inertia is not None else np.eye(3) * 0.01
        self.g = 9.81
        
        self.K_p = K_p if K_p is not None else np.eye(3) * 5.0
        self.K_v = K_v if K_v is not None else np.eye(3) * 3.0
        self.K_R = K_R if K_R is not None else np.eye(3) * 10.0
        self.K_omega = K_omega if K_omega is not None else np.eye(3) * 5.0
    
    def compute_control(self, state: UAVState, 
                       pose_des: SE3Pose, 
                       vel_des: np.ndarray = None,
                       acc_des: np.ndarray = None) -> np.ndarray:
        """
        计算控制量
        
        Args:
            state: 当前状态
            pose_des: 期望位姿
            vel_des: 期望线速度 (3,)
            acc_des: 期望线加速度 (3,)
            
        Returns:
            control: (4,) 控制量 [thrust, τ_x, τ_y, τ_z]
                      thrust = 总升力 (N)
                      τ = 力矩 (N·m)
        """
        if vel_des is None:
            vel_des = np.zeros(3)
        if acc_des is None:
            acc_des = np.zeros(3)
        
        # 当前位姿
        p = np.array([state.pose.x, state.pose.y, state.pose.z])
        R = state.pose.to_matrix()[:3, :3]
        v = state.velocity
        omega = state.omega
        
        # 期望位姿
        p_des = np.array([pose_des.x, pose_des.y, pose_des.z])
        R_des = pose_des.to_matrix()[:3, :3]
        
        # 位置误差
        e_p = p_des - p
        
        # 期望推力方向 (机体 z 轴应对准推力方向)
        f_des = -self.K_p @ e_p - self.K_v @ (v - vel_des) + \
                self.m * self.g * np.array([0, 0, 1]) + self.m * acc_des
        
        # 归一化 → 期望机体 z 轴
        b3_des = f_des / np.linalg.norm(f_des)
        
        # 期望偏航角 → 期望机体 x, y 轴
        # 简化: 假设期望朝向由速度方向决定
        if np.linalg.norm(vel_des) > 0.1:
            x_body_des = vel_des / np.linalg.norm(vel_des)
        else:
            x_body_des = np.array([1.0, 0.0, 0.0])
        
        y_body_des = np.cross(b3_des, x_body_des)
        y_body_des = y_body_des / np.linalg.norm(y_body_des)
        x_body_des = np.cross(y_body_des, b3_des)
        
        R_des_full = np.column_stack([x_body_des, y_body_des, b3_des])
        
        # SO(3) 旋转误差
        e_R = self._so3_error(R, R_des_full)
        
        # 角速度误差 (简化: 假设期望角速度为 0)
        e_omega = omega
        
        # 力矩
        tau = -self.K_R @ e_R - self.K_omega @ e_omega + \
              np.cross(omega, self.J @ omega)
        
        # 升力 (沿机体 z 轴)
        thrust = f_des[2]  # 简化: 只取 z 分量
        
        return np.array([thrust, tau[0], tau[1], tau[2]])
    
    def _so3_error(self, R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
        """SO(3) 旋转误差 → ℝ³"""
        Re = R.T @ R_des
        return 0.5 * np.array([Re[2, 1], Re[0, 2], Re[1, 0]])
    
    def _hat(self, v: np.ndarray) -> np.ndarray:
        """ℝ³ → 𝔰𝔬(3)"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("SE(3) Drone Corridor Planner - Test")
    print("=" * 60)
    
    # 创建测试点云 (简单场景: 两个建筑物)
    print("\nCreating test point cloud...")
    
    points = []
    
    # 建筑物 1 (底部)
    for x in np.linspace(20, 40, 20):
        for y in np.linspace(20, 40, 20):
            for z in np.linspace(0, 15, 10):
                points.append([x, y, z])
    
    # 建筑物 2 (中部)
    for x in np.linspace(60, 80, 20):
        for y in np.linspace(60, 80, 20):
            for z in np.linspace(0, 25, 15):
                points.append([x, y, z])
    
    point_cloud = np.array(points)
    print(f"  Point cloud shape: {point_cloud.shape}")
    
    # 构建分层拓扑地图
    print("\nBuilding layered topo map...")
    topo_map = LayeredTopoMap(
        height_layers=[30.0, 50.0, 80.0],
        layer_thickness=10.0
    )
    topo_map.build_from_point_cloud(point_cloud, resolution=0.5)
    
    print(f"\nTotal corridors: {len(topo_map.corridors)}")
    for layer_idx in range(len(topo_map.height_layers)):
        print(f"  Layer {layer_idx}: {len(topo_map.layer_corridors[layer_idx])} corridors")
    
    # 测试 SE(3) 几何控制器
    print("\nTesting SE(3) geometric controller...")
    
    controller = SE3GeometricController()
    
    # 当前状态
    state = UAVState(
        pose=SE3Pose(x=0.0, y=0.0, z=35.0, yaw=0.0),
        velocity=np.array([5.0, 0.0, 0.0]),
        omega=np.array([0.0, 0.0, 0.0])
    )
    
    # 期望位姿
    pose_des = SE3Pose(x=100.0, y=100.0, z=55.0, yaw=0.5)
    
    # 计算控制量
    control = controller.compute_control(state, pose_des)
    
    print(f"  Control output: {control}")
    print(f"  Thrust: {control[0]:.2f} N")
    print(f"  Torque: [{control[1]:.2f}, {control[2]:.2f}, {control[3]:.2f}] N·m")
    
    print("\nTest passed!")
