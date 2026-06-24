"""
AGV 路径动画 - 生成 AGV 沿路径移动的 GIF
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'Helvetica Neue', 'DejaVu Sans']
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

from planning.gvd import GVDSkeletonExtractor
from planning.ecm import ECMBuilder, ECMGlobalPlanner


def generate_warehouse_grid(H=200, W=200):
    """生成仓库栅格"""
    grid = np.zeros((H, W), dtype=int)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    for sy in range(30, H - 30, 40):
        for sx in range(30, W - 30, 40):
            grid[sy:sy+10, sx:sx+10] = 1
    return grid


def create_animation(output_path='assets/agv_animation.gif', fps=10, duration=5):
    """
    创建 AGV 路径动画
    
    Args:
        output_path: 输出 GIF 路径
        fps: 帧率
        duration: 动画时长（秒）
    """
    # ========== 1. 生成仓库和路径 ==========
    H, W = 200, 200
    res = 0.05
    grid = generate_warehouse_grid(H, W)
    
    # 计算 GVD 和 ECM
    extractor = GVDSkeletonExtractor(resolution=res, prune_length=1.0)
    extractor.compute_edt(grid)
    gvd_mask = extractor.extract_ridge(min_clearance=0.1)
    gvd_mask = extractor.prune_skeleton(gvd_mask)
    topo_nodes = extractor.extract_topo_nodes(gvd_mask)
    
    builder = ECMBuilder(clearance_margin=0.2)
    corridors = builder.build_from_gvd(gvd_mask, extractor.dist_field, topo_nodes, resolution=res)
    
    planner = ECMGlobalPlanner(corridors)
    
    # 规划路径
    start = np.array([2.0, 2.0])
    goal = np.array([8.0, 8.0])
    path = planner.plan(start, goal)
    
    if path is None:
        # 手动创建路径
        path = np.array([
            [2.0, 2.0],
            [5.0, 2.0],
            [5.0, 5.0],
            [8.0, 5.0],
            [8.0, 8.0],
        ])
    
    print(f"Path: {len(path)} points")
    
    # ========== 2. 设置动画参数 ==========
    total_frames = fps * duration
    n_path = len(path)
    
    # 插值路径（让动画更平滑）
    from scipy.interpolate import interp1d
    t = np.linspace(0, 1, n_path)
    t_smooth = np.linspace(0, 1, total_frames)
    
    interp_x = interp1d(t, path[:, 0], kind='linear')
    interp_y = interp1d(t, path[:, 1], kind='linear')
    
    agv_x = interp_x(t_smooth)
    agv_y = interp_y(t_smooth)
    
    # 计算 AGV 朝向（速度方向）
    agv_theta = np.zeros(total_frames)
    for i in range(1, total_frames):
        dx = agv_x[i] - agv_x[i-1]
        dy = agv_y[i] - agv_y[i-1]
        agv_theta[i] = np.arctan2(dy, dx)
    
    print(f"Frames: {total_frames}, AGV positions computed")
    
    # ========== 3. 创建动画 ==========
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制静态元素
    cmap = plt.cm.colors.ListedColormap(['white', 'dimgray'])
    ax.imshow(grid.T, origin='lower', extent=[0, W*res, 0, H*res],
              cmap=cmap, aspect='equal', alpha=0.6)
    
    # 绘制 GVD 骨架
    skeleton_y, skeleton_x = np.where(gvd_mask)
    ax.scatter(skeleton_x * res, skeleton_y * res,
               c='dodgerblue', s=2, alpha=0.3, label='GVD')
    
    # 绘制完整路径（灰色背景）
    ax.plot(path[:, 0], path[:, 1], 'gray', lw=2, alpha=0.5, label='Full Path')
    
    # 绘制起点和终点
    ax.scatter(start[0], start[1], c='green', s=150, marker='o', 
               edgecolors='black', lw=2, zorder=20, label='Start')
    ax.scatter(goal[0], goal[1], c='red', s=150, marker='*', 
               edgecolors='black', lw=2, zorder=20, label='Goal')
    
    # 动态元素：AGV 位置
    agv_circle = Circle((agv_x[0], agv_y[0]), radius=0.3, 
                        fc='#E74C3C', ec='black', lw=2, zorder=10)
    ax.add_patch(agv_circle)
    
    # 动态元素：已行驶路径
    traveled_path, = ax.plot([], [], '#27AE60', lw=3, alpha=0.8, label='Traveled')
    
    # 动态元素：AGV 朝向箭头
    arrow = ax.arrow(agv_x[0], agv_y[0], 
                     0.5*np.cos(agv_theta[0]), 0.5*np.sin(agv_theta[0]),
                     head_width=0.2, head_length=0.12,
                     fc='white', ec='white', zorder=11)
    
    # 文本信息
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlim(0, W * res)
    ax.set_ylim(0, H * res)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('OmniWarehouse: AGV Path Animation', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # ========== 4. 动画更新函数 ==========
    def update(frame):
        # 更新 AGV 位置
        agv_circle.center = (agv_x[frame], agv_y[frame])
        
        # 更新已行驶路径
        traveled_path.set_data(agv_x[:frame+1], agv_y[:frame+1])
        
        # 更新朝向箭头
        nonlocal arrow
        arrow.remove()
        arrow = ax.arrow(agv_x[frame], agv_y[frame],
                         0.5*np.cos(agv_theta[frame]), 0.5*np.sin(agv_theta[frame]),
                         head_width=0.2, head_length=0.12,
                         fc='white', ec='white', zorder=11)
        
        # 更新文本信息
        progress = frame / total_frames * 100
        dist_traveled = sum(np.sqrt((agv_x[i+1]-agv_x[i])**2 + (agv_y[i+1]-agv_y[i])**2) 
                           for i in range(frame))
        info_text.set_text(f'Frame: {frame}/{total_frames}\n'
                          f'Progress: {progress:.1f}%\n'
                          f'Distance: {dist_traveled:.2f}m\n'
                          f'Position: ({agv_x[frame]:.2f}, {agv_y[frame]:.2f})')
        
        return agv_circle, traveled_path, arrow, info_text
    
    # ========== 5. 生成动画 ==========
    print(f"Generating animation with {total_frames} frames...")
    
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000/fps, blit=False, repeat=True
    )
    
    # 保存为 GIF
    print(f"Saving to {output_path}...")
    ani.save(output_path, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)
    
    print(f"\n✅ Animation saved to: {output_path}")
    print(f"   - Frames: {total_frames}")
    print(f"   - FPS: {fps}")
    print(f"   - Duration: {duration}s")


if __name__ == "__main__":
    import os
    os.makedirs('assets', exist_ok=True)
    
    # 检查 scipy 是否可用
    try:
        from scipy.interpolate import interp1d
    except ImportError:
        print("Installing scipy for path interpolation...")
        import subprocess
        subprocess.run(['pip', 'install', 'scipy'], check=True)
    
    create_animation()
    print("\nDone!")
