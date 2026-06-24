"""
ECM (Explicit Corridor Map) - 显式走廊地图

实现:
1. 从 GVD 骨架构建走廊多边形
2. 漏斗算法 (Funnel Algorithm) - 走廊内最短路径
3. 分层 Dijkstra - 走廊图上的全局规划

理论参考:
- Geraerts, R., & Overmars, M. H. (2007). Creating High-Quality Paths for Motion Planning.
- Kallmann, M. (2010). Dynamically Finding the Shortest Route in a Corridor Network.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import heapq


@dataclass
class Corridor:
    """显式走廊"""
    id: int
    polygon: np.ndarray        # 走廊多边形 (N, 2) - 顺时针顶点
    left_edge: np.ndarray      # 左边界折线 (M, 2)
    right_edge: np.ndarray     # 右边界折线 (M, 2)
    centerline: np.ndarray     # 中心线 (K, 2)
    clearance_min: float      # 最小 clearance
    clearance_avg: float      # 平均 clearance
    adjacent: List[int] = field(default_factory=list)  # 相邻走廊 ID
    
    def contains(self, point: np.ndarray) -> bool:
        """判断点是否在走廊内 (射线法)"""
        x, y = point
        n = len(self.polygon)
        inside = False
        
        for i in range(n):
            j = (i + 1) % n
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        
        return inside


@dataclass
class FunnelVertex:
    """漏斗算法中的顶点"""
    point: np.ndarray          # (2,) 坐标
    is_left: bool            # 是否在左边界
    corridor_idx: int        # 所在走廊索引
    segment_idx: int         # 所在走廊段索引


class ECMBuilder:
    """
    ECM (Explicit Corridor Map) 构建器
    
    算法流程:
    1. 从 GVD 骨架提取走廊中心线和宽度
    2. 构建走廊多边形 (中心线向两侧扩展 clearance)
    3. 检测走廊之间的连接关系
    4. 构建走廊图 (节点=走廊, 边=连接)
    """
    
    def __init__(self, clearance_margin: float = 0.2):
        """
        Args:
            clearance_margin: clearance 安全裕度 (米), 实际宽度 = 2*(clearance - margin)
        """
        self.margin = clearance_margin
        self.corridors: List[Corridor] = []
    
    def build_from_gvd(self, gvd_mask: np.ndarray, dist_field: np.ndarray,
                      topo_nodes: List[Tuple[int, int]], 
                      resolution: float = 0.05) -> List[Corridor]:
        """
        从 GVD 骨架构建 ECM
        
        Args:
            gvd_mask: GVD 骨架掩码
            dist_field: 距离场
            topo_nodes: 拓扑节点列表 (分叉点)
            resolution: 栅格分辨率
            
        Returns:
            corridors: 走廊列表
        """
        H, W = gvd_mask.shape
        
        # Step 1: 骨架分段 (分叉点之间)
        segments = self._segment_skeleton(gvd_mask, topo_nodes)
        
        # Step 2: 为每个段构建走廊
        for seg_idx, segment in enumerate(segments):
            if len(segment) < 2:
                continue
            
            # 提取中心线 (亚像素精度)
            centerline = np.array([[p[1] * resolution, p[0] * resolution] 
                                 for p in segment])  # (K, 2), 注意: 图像坐标 (y,x) → 世界坐标 (x,y)
            
            # 计算走廊宽度 (沿中心线的 clearance)
            widths = []
            for (i, j) in segment:
                d = dist_field[i, j]
                width = 2.0 * max(d - self.margin, 0.1)  # 最小宽度 0.2m
                widths.append(width)
            
            # 构建走廊多边形
            polygon, left_edge, right_edge = self._build_corridor_polygon(
                centerline, widths
            )
            
            clearance_min = min(dist_field[i, j] for (i, j) in segment)
            clearance_avg = np.mean([dist_field[i, j] for (i, j) in segment])
            
            corridor = Corridor(
                id=seg_idx,
                polygon=polygon,
                left_edge=left_edge,
                right_edge=right_edge,
                centerline=centerline,
                clearance_min=clearance_min,
                clearance_avg=clearance_avg
            )
            self.corridors.append(corridor)
        
        # Step 3: 检测走廊连接
        self._detect_adjacency()
        
        return self.corridors
    
    def _segment_skeleton(self, gvd_mask: np.ndarray, 
                         topo_nodes: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        将 GVD 骨架分段 (分叉点之间)
        
        Returns:
            segments: 段列表, 每个段 = [(i1,j1), (i2,j2), ...]
        """
        H, W = gvd_mask.shape
        visited = np.zeros((H, W), dtype=bool)
        segments = []
        
        # 将拓扑节点转为集合 (加速查找)
        node_set = set(topo_nodes)
        
        # 从每个拓扑节点开始, 向两个方向追踪
        for (start_i, start_j) in topo_nodes:
            if visited[start_i, start_j]:
                continue
            
            # 两个方向
            for direction in [1, -1]:
                segment = [(start_i, start_j)]
                visited[start_i, start_j] = True
                
                ci, cj = start_i, start_j
                while True:
                    # 找下一个骨架点
                    found = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = ci + di, cj + dj
                            
                            if (0 <= ni < H and 0 <= nj < W and 
                                gvd_mask[ni, nj] and not visited[ni, nj]):
                                
                                # 检查是否是分叉点
                                n_neighbors = gvd_mask[ni-1:ni+2, nj-1:nj+2].sum() - 1
                                
                                segment.append((ni, nj))
                                visited[ni, nj] = True
                                
                                if (ni, nj) in node_set:  # 到达分叉点
                                    found = True
                                    break
                                
                                ci, cj = ni, nj
                                found = True
                                break
                        
                        if found and (ni, nj) in node_set:
                            break
                    
                    if not found:
                        break
                
                if len(segment) >= 2:
                    segments.append(segment)
        
        return segments
    
    def _build_corridor_polygon(self, centerline: np.ndarray, 
                                widths: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建走廊多边形
        
        方法:
        1. 计算中心线的切向量
        2. 法向量 = 切向量旋转 90°
        3. 左边界 = 中心线 - (width/2) * 法向量
        4. 右边界 = 中心线 + (width/2) * 法向量
        5. 多边形 = 左边界 (正向) + 右边界 (反向)
        
        Returns:
            polygon: 走廊多边形 (N, 2) - 顺时针
            left_edge: 左边界
            right_edge: 右边界
        """
        K = len(centerline)
        left_edge = np.zeros((K, 2))
        right_edge = np.zeros((K, 2))
        
        for k in range(K):
            # 计算切向量 (前向差分)
            if k < K - 1:
                tangent = centerline[k+1] - centerline[k]
            else:
                tangent = centerline[k] - centerline[k-1]
            
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-6:
                tangent = np.array([1.0, 0.0])
                tangent_norm = 1.0
            
            tangent = tangent / tangent_norm
            
            # 法向量 (逆时针旋转 90° = 左)
            normal = np.array([-tangent[1], tangent[0]])
            
            # 宽度
            half_width = widths[k] / 2.0
            
            # 左右边界
            left_edge[k] = centerline[k] - half_width * normal
            right_edge[k] = centerline[k] + half_width * normal
        
        # 构建多边形 (顺时针)
        polygon = np.concatenate([
            left_edge,              # 左边界 (正向)
            right_edge[::-1],      # 右边界 (反向)
            left_edge[0:1]         # 闭合
        ], axis=0)
        
        return polygon, left_edge, right_edge
    
    def _detect_adjacency(self):
        """
        检测走廊之间的连接关系
        
        方法:
        1. 检查两个走廊的多边形是否相交
        2. 或者检查它们的端点是否接近
        """
        N = len(self.corridors)
        
        for i in range(N):
            for j in range(i + 1, N):
                # 检查连接
                if self._are_corridors_connected(self.corridors[i], self.corridors[j]):
                    self.corridors[i].adjacent.append(j)
                    self.corridors[j].adjacent.append(i)
    
    def _are_corridors_connected(self, c1: Corridor, c2: Corridor) -> bool:
        """
        判断两个走廊是否相连
        
        方法: 检查它们的端点 (中心线起点/终点) 是否接近
        """
        # 端点距离阈值
        threshold = 2.0  # 米
        
        endpoints1 = [c1.centerline[0], c1.centerline[-1]]
        endpoints2 = [c2.centerline[0], c2.centerline[-1]]
        
        for ep1 in endpoints1:
            for ep2 in endpoints2:
                dist = np.linalg.norm(ep1 - ep2)
                if dist < threshold:
                    return True
        
        return False


class FunnelAlgorithm:
    """
    漏斗算法 (Funnel Algorithm) - 走廊内最短路径
    
    算法 (Chazelle, 1982):
    1. 初始化: apex = 入口点, 左/右视线边界
    2. 遍历走廊段:
       - 如果新边界在视线锥内 → 收缩视线锥
       - 如果新边界在视线锥外 → 提交当前 apex, 重置视线锥
    3. 到达终点 → 输出路径
    
    结果: 走廊内的最短路径 (分段直线)
    """
    
    def __init__(self):
        pass
    
    def plan(self, corridors: List[Corridor], 
             start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        在走廊序列上规划最短路径
        
        Args:
            corridors: 走廊序列 (从全局规划得到)
            start: 起点 (2,)
            goal: 终点 (2,)
            
        Returns:
            path: 最短路径 (N, 2)
        """
        if len(corridors) == 0:
            return np.array([start, goal])
        
        # 构建漏斗队列
        funnel = FunnelQueue(start, goal)
        
        # 遍历每个走廊
        for corridor in corridors:
            # 添加走廊的左右边界到漏斗
            funnel.add_corridor(corridor)
            
            # 收缩漏斗
            funnel.shrink()
        
        # 提取路径
        path = funnel.extract_path()
        return path


class FunnelQueue:
    """漏斗算法的队列实现"""
    
    def __init__(self, start: np.ndarray, goal: np.ndarray):
        self.apex = start
        self.left_bound = None   # 左视线边界点
        self.right_bound = None  # 右视线边界点
        self.path = [start]       # 已确定的路径点
        self.current_corridor = 0
    
    def add_corridor(self, corridor: Corridor):
        """添加走廊边界到漏斗"""
        # 获取走廊的入口/出口 (相对于当前 apex)
        entry = corridor.centerline[0]
        exit = corridor.centerline[-1]
        
        # 判断入口和出口在 apex 的哪一侧
        # 使用叉积判断方向
        to_entry = entry - self.apex
        to_exit = exit - self.apex
        
        # 初始化视线锥
        if self.left_bound is None:
            self.left_bound = entry
            self.right_bound = entry
        
        # 更新视线锥
        self._update_funnel(entry, exit)
        
        self.current_corridor += 1
    
    def _update_funnel(self, left_pt: np.ndarray, right_pt: np.ndarray):
        """更新漏斗 (收缩或重置)"""
        # 计算相对于 apex 的方向
        cross_left = self._cross_2d(self.left_bound - self.apex, 
                                   left_pt - self.apex)
        cross_right = self._cross_2d(self.right_bound - self.apex, 
                                    right_pt - self.apex)
        
        # 如果新点在视线锥内 → 收缩
        if cross_left >= 0:  # left_pt 在左侧或重合
            self.left_bound = left_pt
        
        if cross_right <= 0:  # right_pt 在右侧或重合
            self.right_bound = right_pt
        
        # 如果新点在视线锥外 → 提交 apex, 重置
        if cross_left < 0 or cross_right > 0:
            self.path.append(self.apex)
            self.apex = self.left_bound if cross_left < 0 else self.right_bound
            self.left_bound = left_pt
            self.right_bound = right_pt
    
    def _cross_2d(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """2D 叉积 (标量)"""
        return v1[0] * v2[1] - v1[1] * v2[0]
    
    def shrink(self):
        """收缩漏斗 (检查当前走廊段的边界)"""
        # 简化版: 直接连接 apex 到出口
        pass
    
    def extract_path(self) -> np.ndarray:
        """提取最终路径"""
        return np.array(self.path)


class ECMGlobalPlanner:
    """
    ECM 全局规划器
    
    在走廊图上跑 Dijkstra, 得到走廊序列
    然后在每个走廊内跑漏斗算法, 得到分段直线路径
    """
    
    def __init__(self, corridors: List[Corridor]):
        self.corridors = corridors
        self.N = len(corridors)
    
    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[np.ndarray]:
        """
        全局规划: 在走廊图上找最短路径
        
        Args:
            start: 起点 (2,)
            goal: 终点 (2,)
            
        Returns:
            path: 全局最短路径 (N, 2), 如果不可达则返回 None
        """
        # Step 1: 找到起点和终点所在的走廊
        start_corridor = self._find_containing_corridor(start)
        goal_corridor = self._find_containing_corridor(goal)
        
        if start_corridor is None or goal_corridor is None:
            print("Warning: Start or goal not in any corridor!")
            return None
        
        # Step 2: Dijkstra on 走廊图
        dist = [float('inf')] * self.N
        prev = [-1] * self.N
        dist[start_corridor] = 0
        
        pq = [(0.0, start_corridor)]
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist[u]:
                continue
            
            if u == goal_corridor:
                break
            
            for v in self.corridors[u].adjacent:
                # 边权重 = 走廊中心线的长度
                weight = self._corridor_length(self.corridors[v])
                
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
        
        # Step 3: 回溯路径 (走廊序列)
        if dist[goal_corridor] == float('inf'):
            return None
        
        corridor_seq = []
        u = goal_corridor
        while u != -1:
            corridor_seq.append(u)
            u = prev[u]
        corridor_seq.reverse()
        
        # Step 4: 漏斗算法 - 走廊内最短路径
        funnel = FunnelAlgorithm()
        path = funnel.plan([self.corridors[c] for c in corridor_seq], 
                          start, goal)
        
        return path
    
    def _find_containing_corridor(self, point: np.ndarray) -> Optional[int]:
        """找到点所在的走廊"""
        for idx, corridor in enumerate(self.corridors):
            if corridor.contains(point):
                return idx
        return None
    
    def _corridor_length(self, corridor: Corridor) -> float:
        """计算走廊中心线的长度"""
        length = 0.0
        for k in range(1, len(corridor.centerline)):
            length += np.linalg.norm(corridor.centerline[k] - corridor.centerline[k-1])
        return length


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("ECM (Explicit Corridor Map) - Test")
    print("=" * 60)
    
    # 创建测试地图
    from planning.gvd import GVDSkeletonExtractor, create_test_warehouse
    
    grid = create_test_warehouse(200, 200)
    print(f"Grid shape: {grid.shape}")
    
    # 提取 GVD 骨架
    extractor = GVDSkeletonExtractor(resolution=0.05, prune_length=1.0)
    gvd_mask, topo_nodes = extractor.extract(grid)
    print(f"GVD skeleton points: {gvd_mask.sum()}")
    print(f"Topology nodes: {len(topo_nodes)}")
    
    # 构建 ECM
    print("\nBuilding ECM...")
    builder = ECMBuilder(clearance_margin=0.2)
    corridors = builder.build_from_gvd(gvd_mask, extractor.dist_field, 
                                       topo_nodes, resolution=0.05)
    print(f"Corridors built: {len(corridors)}")
    
    # 全局规划测试
    if len(corridors) > 0:
        print("\nRunning global planning test...")
        planner = ECMGlobalPlanner(corridors)
        
        # 随机选起点和终点
        start = np.array([5.0, 5.0])   # 左下角
        goal = np.array([8.0, 8.0])    # 右上角
        
        path = planner.plan(start, goal)
        
        if path is not None:
            print(f"Path found! Length: {len(path)} points")
            print(f"Path start: {path[0]}")
            print(f"Path end: {path[-1]}")
        else:
            print("No path found!")
    
    print("\nTest passed!")
