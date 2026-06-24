"""
GVD (Generalized Voronoi Diagram) 骨架提取

实现:
1. 欧氏距离变换 (Euclidean Distance Transform)
2. 脊线检测 (Ridge Detection)
3. 骨架剪枝 (Pruning)
4. 拓扑节点提取 (Junction Extraction)

理论参考:
- Choset, H. M., & Burdick, J. (2000). Sensor-Based Exploration: The Hierarchical Generalized Voronoi Graph.
- O'Rourke, J. (1998). Computational Geometry in C.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq


@dataclass
class GVDCell:
    """GVD 骨架上的一个单元"""
    x: int
    y: int
    clearance: float        # 到最近障碍物的距离
    nearest_obstacles: Set[int]  # 最近的障碍物 ID 集合 (用于判断脊线)


@dataclass
class GVDPath:
    """GVD 骨架路径"""
    cells: List[GVDCell]
    length: float
    clearance_min: float   # 路径上的最小 clearance
    clearance_avg: float   # 路径上的平均 clearance


class GVDSkeletonExtractor:
    """
    GVD 骨架提取器
    
    算法流程:
    1. 欧氏距离变换 (EDT)
    2. 脊线检测 (局部极大值)
    3. 骨架剪枝 (移除死胡同)
    4. 拓扑节点提取 (分叉点)
    """
    
    def __init__(self, resolution: float = 0.05, prune_length: float = 1.0):
        """
        Args:
            resolution: 栅格分辨率 (米/像素)
            prune_length: 剪枝长度阈值 (米), 小于此值的死胡同被剪掉
        """
        self.res = resolution
        self.prune_length = prune_length
        self.dist_field = None      # 距离场
        self.gvd_mask = None      # GVD 骨架掩码
        self.obstacle_ids = None  # 每个像素的最近障碍物 ID
        
    def compute_edt(self, occupancy_grid: np.ndarray) -> np.ndarray:
        """
        欧氏距离变换 (Euclidean Distance Transform)
        
        两遍扫描算法 (Saito & Toriwaki, 1994):
        - Forward pass:  左上 → 右下
        - Backward pass: 右下 → 左上
        
        Args:
            occupancy_grid: 占据栅格, 1=障碍物, 0=自由空间
                        shape: (H, W)
                        
        Returns:
            dist_field: 距离场, 每个像素 = 到最近障碍物的欧氏距离 (米)
        """
        H, W = occupancy_grid.shape
        
        # 初始化: 障碍物=0, 自由空间=inf
        d = np.where(occupancy_grid == 1, 0.0, np.inf)
        self.obstacle_ids = np.where(occupancy_grid == 1, 0, -1).astype(int)
        
        # Forward pass (左上 → 右下)
        # 检查 8 个前向邻居 (3x3 半窗口)
        for i in range(H):
            for j in range(W):
                if d[i, j] == 0:  # 障碍物本身
                    continue
                
                # 邻居偏移 (前向)
                offsets = [
                    (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
                    ( 0, -1, 1.0)
                ]
                
                for di, dj, cost in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and d[ni, nj] < np.inf:
                        new_dist = d[ni, nj] + cost * self.res
                        if new_dist < d[i, j]:
                            d[i, j] = new_dist
                            self.obstacle_ids[i, j] = self.obstacle_ids[ni, nj]
        
        # Backward pass (右下 → 左上)
        for i in range(H - 1, -1, -1):
            for j in range(W - 1, -1, -1):
                if d[i, j] == 0:
                    continue
                
                # 邻居偏移 (后向)
                offsets = [
                    (1, -1, np.sqrt(2)), (1, 0, 1.0), (1, 1, np.sqrt(2)),
                    (0, 1, 1.0)
                ]
                
                for di, dj, cost in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and d[ni, nj] < np.inf:
                        new_dist = d[ni, nj] + cost * self.res
                        if new_dist < d[i, j]:
                            d[i, j] = new_dist
                            self.obstacle_ids[i, j] = self.obstacle_ids[ni, nj]
        
        self.dist_field = d
        return d
    
    def extract_ridge(self, min_clearance: float = 0.3) -> np.ndarray:
        """
        脊线检测: 提取距离场的局部极大值
        
        脊线定义:
            p 是脊线点 ⇔ dist(p) ≥ dist(q) for all q ∈ N(p)
            其中 N(p) 是 p 的 8 邻域
            
        脊线 = GVD 骨架 = 到至少两个障碍物等距的点的集合
        
        Args:
            min_clearance: 最小 clearance 阈值, 小于此值的脊线点被过滤
            
        Returns:
            gvd_mask: GVD 骨架掩码, True=骨架点
        """
        H, W = self.dist_field.shape
        gvd_mask = np.zeros((H, W), dtype=bool)
        
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                d_center = self.dist_field[i, j]
                
                # 过滤: clearance 太小
                if d_center < min_clearance:
                    continue
                
                # 检查 8 邻域
                neighbors = self.dist_field[i-1:i+2, j-1:j+2].flatten()
                
                # 局部极大值检测 (允许微小数值误差)
                if d_center >= np.max(neighbors) - 1e-6:
                    gvd_mask[i, j] = True
        
        self.gvd_mask = gvd_mask
        return gvd_mask
    
    def _check_multiple_obstacles(self, i: int, j: int, window: int = 2) -> bool:
        """
        检查点 (i,j) 是否到多个障碍物等距
        
        脊线的严格定义: 存在至少两个障碍物 O_a, O_b 使得
            dist(p, O_a) = dist(p, O_b) = min_k dist(p, O_k)
            
        实现: 检查局部窗口内的最近障碍物 ID 是否多样
        """
        H, W = self.obstacle_ids.shape
        ids = set()
        
        for di in range(-window, window + 1):
            for dj in range(-window, window + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    oid = self.obstacle_ids[ni, nj]
                    if oid >= 0:
                        ids.add(oid)
        
        return len(ids) >= 2
    
    def prune_skeleton(self, gvd_mask: np.ndarray) -> np.ndarray:
        """
        骨架剪枝: 移除死胡同 (短分支)
        
        算法:
        1. 找到所有度为 1 的端点
        2. 沿着边向内部走, 直到遇到分叉点 (度 ≥ 3)
        3. 如果这条路径长度 < prune_length → 剪掉整个分支
        4. 重复直到无分支可剪
        
        Args:
            gvd_mask: GVD 骨架掩码
            
        Returns:
            pruned_mask: 剪枝后的骨架掩码
        """
        pruned_mask = gvd_mask.copy()
        H, W = gvd_mask.shape
        
        changed = True
        while changed:
            changed = False
            
            # 找到所有端点 (度 = 1)
            endpoints = []
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    if not pruned_mask[i, j]:
                        continue
                    
                    # 计算度 (8 邻域内骨架点数量)
                    neighbors = pruned_mask[i-1:i+2, j-1:j+2].sum() - 1
                    
                    if neighbors == 1:  # 端点
                        endpoints.append((i, j))
            
            # 从每个端点开始剪枝
            for start_i, start_j in endpoints:
                if not pruned_mask[start_i, start_j]:  # 已被剪掉
                    continue
                
                # BFS 追踪这条分支
                path = [(start_i, start_j)]
                visited = {(start_i, start_j)}
                queue = [(start_i, start_j)]
                
                junction_found = False
                
                while queue:
                    ci, cj = queue.pop(0)
                    
                    # 检查邻居
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = ci + di, cj + dj
                            
                            if (0 <= ni < H and 0 <= nj < W and 
                                pruned_mask[ni, nj] and (ni, nj) not in visited):
                                
                                # 检查是否是分叉点
                                n_neighbors = pruned_mask[ni-1:ni+2, nj-1:nj+2].sum() - 1
                                
                                if n_neighbors >= 3:  # 分叉点
                                    junction_found = True
                                    break
                                
                                visited.add((ni, nj))
                                path.append((ni, nj))
                                queue.append((ni, nj))
                        
                        if junction_found:
                            break
                    if junction_found:
                        break
                
                # 判断是否需要剪枝
                if not junction_found:
                    path_length = len(path) * self.res
                    
                    if path_length < self.prune_length:
                        # 剪掉这个分支
                        for (pi, pj) in path:
                            pruned_mask[pi, pj] = False
                        changed = True
        
        return pruned_mask
    
    def extract_topo_nodes(self, gvd_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        提取拓扑节点 (分叉点)
        
        分叉点定义: GVD 骨架上的点的度 ≥ 3
        (度 = 8 邻域内骨架点数量)
        
        Args:
            gvd_mask: GVD 骨架掩码
            
        Returns:
            nodes: 拓扑节点列表 [(i1, j1), (i2, j2), ...]
        """
        H, W = gvd_mask.shape
        nodes = []
        
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if not gvd_mask[i, j]:
                    continue
                
                # 计算度
                neighbors = gvd_mask[i-1:i+2, j-1:j+2].sum() - 1
                
                if neighbors >= 3:  # 分叉点
                    nodes.append((i, j))
        
        return nodes
    
    def extract_gvd_paths(self, gvd_mask: np.ndarray) -> List[GVDPath]:
        """
        提取 GVD 骨架上的路径 (分叉点之间的线段)
        
        Args:
            gvd_mask: GVD 骨架掩码
            
        Returns:
            paths: GVD 路径列表
        """
        H, W = gvd_mask.shape
        visited = np.zeros((H, W), dtype=bool)
        paths = []
        
        # 找到所有分叉点和端点
        special_points = set()
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if not gvd_mask[i, j]:
                    continue
                neighbors = gvd_mask[i-1:i+2, j-1:j+2].sum() - 1
                if neighbors != 2:  # 非直线点 (分叉点或端点)
                    special_points.add((i, j))
        
        # 从每个分叉点/端点开始, 追踪到下一个分叉点/端点
        for start_i, start_j in special_points:
            if visited[start_i, start_j]:
                continue
            
            # BFS 追踪
            path_cells = [GVDCell(start_i, start_j, 
                                self.dist_field[start_i, start_j], 
                                {self.obstacle_ids[start_i, start_j]})]
            visited[start_i, start_j] = True
            
            # 两个方向都追踪
            for direction in [1, -1]:
                ci, cj = start_i, start_j
                while True:
                    # 找下一个骨架点
                    found_next = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = ci + di * direction, cj + dj * direction
                            
                            if (0 <= ni < H and 0 <= nj < W and 
                                gvd_mask[ni, nj] and not visited[ni, nj]):
                                
                                # 检查是否是分叉点
                                n_neighbors = gvd_mask[ni-1:ni+2, nj-1:nj+2].sum() - 1
                                
                                path_cells.append(GVDCell(ni, nj, 
                                                       self.dist_field[ni, nj],
                                                       {self.obstacle_ids[ni, nj]}))
                                visited[ni, nj] = True
                                
                                if n_neighbors >= 3:  # 到达分叉点, 停止
                                    found_next = True
                                    break
                                
                                ci, cj = ni, nj
                                found_next = True
                                break
                        
                        if found_next and n_neighbors >= 3:
                            break
                    
                    if not found_next:
                        break
            
            # 计算路径属性
            length = 0.0
            clearance_min = float('inf')
            clearance_sum = 0.0
            
            for cell in path_cells:
                clearance_min = min(clearance_min, cell.clearance)
                clearance_sum += cell.clearance
            
            for k in range(1, len(path_cells)):
                di = path_cells[k].x - path_cells[k-1].x
                dj = path_cells[k].y - path_cells[k-1].y
                length += np.sqrt(di*di + dj*dj) * self.res
            
            paths.append(GVDPath(
                cells=path_cells,
                length=length,
                clearance_min=clearance_min,
                clearance_avg=clearance_sum / len(path_cells)
            ))
        
        return paths
    
    def extract(self, occupancy_grid: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        完整 GVD 骨架提取流程
        
        Args:
            occupancy_grid: 占据栅格
            
        Returns:
            gvd_mask: GVD 骨架掩码
            topo_nodes: 拓扑节点列表
        """
        # Step 1: 距离变换
        self.compute_edt(occupancy_grid)
        
        # Step 2: 脊线检测
        gvd_mask = self.extract_ridge(min_clearance=0.3)
        
        # Step 3: 骨架剪枝
        gvd_mask = self.prune_skeleton(gvd_mask)
        
        # Step 4: 提取拓扑节点
        topo_nodes = self.extract_topo_nodes(gvd_mask)
        
        return gvd_mask, topo_nodes


# ============ 测试代码 ============

def create_test_warehouse(width: int = 200, height: int = 200) -> np.ndarray:
    """
    创建测试仓库栅格地图
    
    布局:
    - 外边框: 障碍物
    - 内部: 货架 (矩形障碍物)
    - 通道: 自由空间
    """
    grid = np.zeros((height, width), dtype=int)
    
    # 外边框
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    
    # 货架 (每隔 20 像素一个)
    for shelf_y in range(30, height - 30, 40):
        for shelf_x in range(30, width - 30, 40):
            grid[shelf_y:shelf_y+10, shelf_x:shelf_x+10] = 1
    
    return grid


if __name__ == "__main__":
    # 测试 GVD 骨架提取
    print("=" * 60)
    print("GVD Skeleton Extractor - Test")
    print("=" * 60)
    
    # 创建测试地图
    grid = create_test_warehouse(200, 200)
    print(f"Grid shape: {grid.shape}")
    print(f"Obstacle ratio: {grid.sum() / grid.size:.2%}")
    
    # 提取 GVD 骨架
    extractor = GVDSkeletonExtractor(resolution=0.05, prune_length=1.0)
    gvd_mask, topo_nodes = extractor.extract(grid)
    
    print(f"GVD skeleton points: {gvd_mask.sum()}")
    print(f"Topology nodes: {len(topo_nodes)}")
    
    # 可视化 (保存为文本)
    H, W = grid.shape
    viz = np.where(grid == 1, '#', '.').astype(str)
    for (i, j) in topo_nodes:
        if 0 <= i < H and 0 <= j < W:
            viz[i, j] = 'O'
    
    gvd_i, gvd_j = np.where(gvd_mask)
    for i, j in zip(gvd_i, gvd_j):
        if viz[i, j] == '.':
            viz[i, j] = '*'
    
    print("\nVisualization (sample):")
    print("  '#' = obstacle, '*' = GVD skeleton, 'O' = topology node")
    print()
    
    # 只打印中间区域
    for i in range(80, 120):
        print("  " + ''.join(viz[i, 80:120]))
    
    print("\nTest passed!")
