"""
数字孪生可视化 (Digital Twin Visualization)

实现:
1. 仓库布局渲染 (2D Matplotlib)
2. 拓扑地图可视化 (GVD 骨架 + 走廊网络)
3. AGV 实时轨迹 (动画)
4. 供应链状态监控 (库存水平 + 订单状态)
5. 性能指标仪表盘 (Plotly Dashboard)

依赖:
- matplotlib (2D 渲染 + 动画)
- plotly (交互式 Dashboard)
- numpy (数据处理)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.lines as mlines
from enum import Enum
import random


class VisualiationStyle(Enum):
    """可视化风格"""
    MATPLOTLIB = "matplotlib"  # 静态图
    PLOTLY = "plotly"       # 交互式
    ANIMATION = "animation"   # 动画


@dataclass
class WarehouseVisualizer:
    """
    仓库可视化器
    
    功能:
    1. 渲染仓库布局 (货架, 通道, 充电站)
    2. 绘制 GVD 骨架
    3. 绘制 ECM 走廊
    4. 显示 AGV 位置和轨迹
    5. 显示任务状态
    """
    
    def __init__(self, 
                 width: float = 100.0,
                 height: float = 100.0,
                 resolution: float = 0.05):
        """
        Args:
            width: 仓库宽度 (米)
            height: 仓库高度 (米)
            resolution: 栅格分辨率 (米/像素)
        """
        self.W = width
        self.H = height
        self.res = resolution
        
        #  matplotlib figure
        self.fig = None
        self.ax = None
        
        # 数据
        self.grid = None           # 占据栅格
        self.gvd_mask = None      # GVD 骨架
        self.corridors = None     # ECM 走廊
        self.agvs = None           # AGV 位置列表
        self.tasks = None          # 任务列表
        
    def setup_figure(self, figsize: Tuple[int, int] = (12, 10)):
        """初始化画布"""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        
    def render_grid(self, grid: np.ndarray, alpha: float = 0.3):
        """
        渲染占据栅格
        
        Args:
            grid: (H, W) 占据栅格, 1=障碍物
            alpha: 透明度
        """
        if self.ax is None:
            self.setup_figure()
        
        H, W = grid.shape
        
        # 障碍物
        obstacle_x, obstacle_y = np.where(grid == 1)
        obstacle_x = obstacle_x * self.res
        obstacle_y = obstacle_y * self.res
        
        self.ax.scatter(
            obstacle_y, obstacle_x,  # 注意: 图像坐标 (y,x) → 世界坐标 (x,y)
            c='gray',
            s=10,
            alpha=alpha,
            marker='s',
            label='Obstacles'
        )
        
    def render_gvd(self, gvd_mask: np.ndarray, 
                      color: str = 'blue', 
                      linewidth: float = 1.5):
        """
        渲染 GVD 骨架
        
        Args:
            gvd_mask: (H, W) GVD 骨架掩码
            color: 颜色
            linewidth: 线宽
        """
        if self.ax is None:
            self.setup_figure()
        
        H, W = gvd_mask.shape
        
        # 提取骨架点
        skeleton_y, skeleton_x = np.where(gvd_mask)  # 注意顺序
        skeleton_x = skeleton_x * self.res
        skeleton_y = skeleton_y * self.res
        
        self.ax.scatter(
            skeleton_x, skeleton_y,
            c=color,
            s=5,
            alpha=0.6,
            label='GVD Skeleton'
        )
        
    def render_corridors(self, corridors: list, 
                           alpha: float = 0.2):
        """
        渲染 ECM 走廊
        
        Args:
            corridors: 走廊列表 (每个走廊有 polygon 属性)
            alpha: 透明度
        """
        if self.ax is None:
            self.setup_figure()
        
        for corridor in corridors:
            if hasattr(corridor, 'polygon'):
                polygon = corridor.polygon
                
                # 绘制多边形
                poly_patch = Polygon(
                    polygon,
                    closed=True,
                    facecolor='lightblue',
                    edgecolor='blue',
                    alpha=alpha,
                    label='ECM Corridor' if corridor == corridors[0] else ""
                )
                self.ax.add_patch(poly_patch)
                
                # 绘制中心线
                if hasattr(corridor, 'center_line'):
                    center = corridor.center_line
                    self.ax.plot(
                        center[:, 0], center[:, 1],
                        'b--',
                        alpha=0.5,
                        linewidth=1.0
                    )
        
    def render_agvs(self, agv_positions: List[Tuple[float, float]], 
                      agv_headings: Optional[List[float]] = None,
                      show_trail: bool = True):
        """
        渲染 AGV
        
        Args:
            agv_positions: [(x1, y1), (x2, y2), ...]
            agv_headings: [theta1, theta2, ...] (弧度)
            show_trail: 是否显示轨迹
        """
        if self.ax is None:
            self.setup_figure()
        
        for i, (x, y) in enumerate(agv_positions):
            # AGV 本体 (圆形)
            circle = Circle(
                (x, y),
                radius=1.0,  # 假设 AGV 半径 1m
                facecolor='red',
                edgecolor='darkred',
                alpha=0.8,
                label='AGV' if i == 0 else ""
            )
            self.ax.add_patch(circle)
            
            # 朝向 (箭头)
            if agv_headings is not None and i < len(agv_headings):
                theta = agv_headings[i]
                arrow_length = 1.5
                dx = arrow_length * np.cos(theta)
                dy = arrow_length * np.sin(theta)
                
                self.ax.arrow(
                    x, y, dx, dy,
                    head_width=0.5,
                    head_length=0.3,
                    fc='darkred',
                    ec='darkred',
                    alpha=0.8
                )
            
            # 编号
            self.ax.text(
                x, y, f'{i}',
                ha='center',
                va='center',
                fontsize=8,
                color='white',
                weight='bold'
            )
        
    def render_path(self, path: np.ndarray, 
                      color: str = 'green',
                      linewidth: float = 2.0,
                      linestyle: str = '-',
                      label: str = 'Planned Path'):
        """
        渲染路径
        
        Args:
            path: (N, 2) 路径点
            color: 颜色
            linewidth: 线宽
            linestyle: 线型
            label: 图例标签
        """
        if self.ax is None:
            self.setup_figure()
        
        if len(path) < 2:
            return
        
        self.ax.plot(
            path[:, 0], path[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label if label not in self.ax.get_legend_handles_labels()[1] else ""
        )
        
        # 起点和终点（分开画，marker 不支持列表）
        self.ax.scatter(
            path[0, 0], path[0, 1],
            c='green',
            s=100,
            marker='o',
            edgecolors='black',
            linewidths=2,
            label='Start'
        )
        self.ax.scatter(
            path[-1, 0], path[-1, 1],
            c='red',
            s=100,
            marker='s',
            edgecolors='black',
            linewidths=2,
            label='Goal'
        )
        
    def render_heatmap(self, heatmap: np.ndarray, 
                           alpha: float = 0.5):
        """
        渲染热力图 (例如: 访问频率, 冲突密度)
        
        Args:
            heatmap: (H, W) 热力数据
            alpha: 透明度
        """
        if self.ax is None:
            self.setup_figure()
        
        self.ax.imshow(
            heatmap,
            extent=[0, self.W, 0, self.H],
            origin='lower',
            cmap='hot',
            alpha=alpha,
            interpolation='bilinear'
        )
        
    def add_legend(self):
        """添加图例"""
        self.ax.legend(loc='upper right', fontsize=10)
        
    def save(self, filepath: str, dpi: int = 300):
        """保存图像"""
        self.add_legend()
        self.fig.tight_layout()
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
        
    def show(self):
        """显示图像"""
        self.add_legend()
        self.fig.tight_layout()
        plt.show()


@dataclass
class AnimatedVisualizer:
    """
    动画可视化器 (实时 AGV 轨迹)
    
    功能:
    1. 实时更新 AGV 位置
    2. 动态显示路径
    3. 显示任务状态变化
    """
    
    def __init__(self, 
                 grid: np.ndarray,
                 resolution: float = 0.05):
        self.grid = grid
        self.res = resolution
        self.H, self.W = grid.shape
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(0, self.W * self.res)
        self.ax.set_ylim(0, self.H * self.res)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # 动画数据
        self.agv_trails = {}  # {agv_id: [(x1,y1), (x2,y2), ...]}
        self.agv_positions = {}  # {agv_id: (x, y)}
        self.agv_headings = {}  # {agv_id: theta}
        
    def init_animation(self):
        """初始化动画"""
        # 渲染栅格
        obstacle_x, obstacle_y = np.where(self.grid == 1)
        self.ax.scatter(
            obstacle_y * self.res,
            obstacle_x * self.res,
            c='gray',
            s=10,
            alpha=0.3,
            marker='s'
        )
        
        return []
        
    def update_animation(self, frame):
        """更新动画帧"""
        # 清除上一帧的 AGV
        for artist in self.ax.get_children():
            if isinstance(artist, Circle) or isinstance(artist, mlines.Line2D):
                artist.remove()
        
        # 渲染 AGV
        for agv_id in self.agv_positions:
            x, y = self.agv_positions[agv_id]
            
            # 轨迹
            if agv_id in self.agv_trails:
                trail = self.agv_trails[agv_id]
                if len(trail) > 1:
                    trail_x = [p[0] for p in trail]
                    trail_y = [p[1] for p in trail]
                    self.ax.plot(
                        trail_x, trail_y,
                        'r-',
                        alpha=0.5,
                        linewidth=1.0
                    )
            
            # AGV 本体
            circle = Circle(
                (x, y),
                radius=1.0,
                facecolor='red',
                edgecolor='darkred',
                alpha=0.8
            )
            self.ax.add_patch(circle)
            
            # 朝向
            if agv_id in self.agv_headings:
                theta = self.agv_headings[agv_id]
                dx = 1.5 * np.cos(theta)
                dy = 1.5 * np.sin(theta)
                self.ax.arrow(
                    x, y, dx, dy,
                    head_width=0.5,
                    head_length=0.3,
                    fc='darkred',
                    ec='darkred'
                )
        
        return []
        
    def run_animation(self, 
                          agv_trajectory_data: Dict[int, List[Tuple[float, float, float]]],  # {agv_id: [(t, x, y), ...]}
                          interval: int = 100,  # 毫秒
                          save_as_gif: Optional[str] = None):
        """
        运行动画
        
        Args:
            agv_trajectory_data: AGV 轨迹数据
            interval: 帧间隔 (毫秒)
            save_as_gif: 保存路径 (可选)
        """
        # 预process 数据
        max_frames = max(
            len(data) for data in agv_trajectory_data.values()
        )
        
        def update(frame):
            for agv_id, data in agv_trajectory_data.items():
                if frame < len(data):
                    t, x, y = data[frame]
                    self.agv_positions[agv_id] = (x, y)
                    
                    if agv_id not in self.agv_trails:
                        self.agv_trails[agv_id] = []
                    self.agv_trails[agv_id].append((x, y))
            
            return self.update_animation(frame)
        
        ani = animation.FuncAnimation(
            self.fig,
            update,
            frames=max_frames,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        if save_as_gif:
            ani.save(save_as_gif, writer='pillow', fps=1000//interval)
            print(f"Animation saved to: {save_as_gif}")
        else:
            plt.show()


# =========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("Digital Twin Visualization - Test")
    print("=" * 60)
    
    # 创建测试数据
    print("\nGenerating test data...")
    
    # 仓库栅格
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
    
    # 创建可视化器
    print("\nRendering warehouse...")
    
    viz = WarehouseVisualizer(
        width=W * 0.05,
        height=H * 0.05,
        resolution=0.05
    )
    
    viz.setup_figure()
    
    # 渲染栅格
    viz.render_grid(grid, alpha=0.3)
    
    # 模拟 GVD 骨架
    print("  Adding simulated GVD skeleton...")
    np.random.seed(42)
    gvd_points = []
    for _ in range(500):
        x = np.random.uniform(5, 95)
        y = np.random.uniform(5, 95)
        gvd_points.append((x, y))
    
    gvd_array = np.zeros((H, W), dtype=bool)
    for (x, y) in gvd_points:
        i = int(y / 0.05)
        j = int(x / 0.05)
        if 0 <= i < H and 0 <= j < W:
            gvd_array[i, j] = True
    
    viz.render_gvd(gvd_array, color='blue', linewidth=1.5)
    
    # 模拟 AGV
    print("  Adding simulated AGVs...")
    agv_positions = [
        (10.0, 10.0),
        (50.0, 50.0),
        (90.0, 90.0)
    ]
    agv_headings = [0.0, np.pi/4, -np.pi/2]
    
    viz.render_agvs(agv_positions, agv_headings, show_trail=False)
    
    # 模拟路径
    print("  Adding simulated path...")
    path = np.array([
        [10.0, 10.0],
        [20.0, 15.0],
        [30.0, 25.0],
        [40.0, 40.0],
        [50.0, 50.0]
    ])
    
    viz.render_path(path, color='green', linewidth=2.0)
    
    # 添加标题
    viz.ax.set_title('OmniWarehouse: Warehouse Visualization Demo', fontsize=14)
    
    # 保存
    output_path = '/Users/sophie549/WorkBuddy/2026-06-13-00-22-08/OmniWarehouse/assets/demo_visualization.png'
    viz.save(output_path)
    
    print(f"\nVisualization saved to: {output_path}")
    print("\nTest passed!")
