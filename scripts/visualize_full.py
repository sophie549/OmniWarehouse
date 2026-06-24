"""
增强可视化 - 使用真实 GVD 算法和 ECM 走廊
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'Helvetica Neue', 'DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon
import matplotlib.lines as mlines

from planning.gvd import GVDSkeletonExtractor, create_test_warehouse
from planning.ecm import ECMBuilder, ECMGlobalPlanner


def generate_warehouse_grid(H=200, W=200):
    """生成真实的仓库栅格"""
    grid = np.zeros((H, W), dtype=int)
    
    # 外墙
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    
    # 货架 (每 40 像素一个，大小 10x10 像素)
    # 货架中心间距 2.0 米 (40 * 0.05)
    # 货架大小 0.5 米 (10 * 0.05)
    for sy in range(30, H - 30, 40):
        for sx in range(30, W - 30, 40):
            grid[sy:sy+10, sx:sx+10] = 1
    
    return grid


def visualize_full_pipeline(output_path='assets/demo_visualization_full.png'):
    """完整可视化：GVD + ECM + AGV"""
    
    # ========== 1. 生成仓库栅格 ==========
    H, W = 200, 200
    res = 0.05
    grid = generate_warehouse_grid(H, W)
    
    print(f"[1/5] Warehouse grid generated: {H}x{W}, {grid.sum()} obstacles")
    
    # ========== 2. 计算真实 GVD 骨架 ==========
    extractor = GVDSkeletonExtractor(resolution=res, prune_length=1.0)
    # 先单独调用，降低 min_clearance
    extractor.compute_edt(grid)
    print(f"  Distance field max: {extractor.dist_field.max():.3f}m")
    
    # 降低 min_clearance 到 0.1m (2 个像素)
    gvd_mask = extractor.extract_ridge(min_clearance=0.1)
    gvd_mask = extractor.prune_skeleton(gvd_mask)
    topo_nodes = extractor.extract_topo_nodes(gvd_mask)
    
    n_gvd = gvd_mask.sum()
    n_nodes = len(topo_nodes)
    print(f"[2/5] GVD skeleton extracted: {n_gvd} skeleton points, {n_nodes} topology nodes")
    
    # ========== 3. 构建 ECM 走廊 ==========
    builder = ECMBuilder(clearance_margin=0.2)
    corridors = builder.build_from_gvd(gvd_mask, extractor.dist_field, topo_nodes, resolution=res)
    
    print(f"[3/5] ECM corridors built: {len(corridors)} corridors")
    
    # ========== 4. 规划路径 ==========
    planner = ECMGlobalPlanner(corridors)
    
    # 起点和终点 (世界坐标)
    start = np.array([2.0, 2.0])
    goal = np.array([8.0, 8.0])
    
    path = planner.plan(start, goal)
    path_len = None
    if path is not None:
        path_len = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
        print(f"[4/5] Path found! {len(path)} points, length={path_len:.2f}m")
    else:
        print(f"[4/5] No path found, using manual path")
        path = np.array([
            [2.0, 2.0],
            [5.0, 2.0],
            [5.0, 5.0],
            [8.0, 5.0],
            [8.0, 8.0],
        ])
        path_len = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
    
    # ========== 5. 可视化 ==========
    print(f"[5/5] Rendering visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 5.1 渲染仓库栅格
    cmap = plt.cm.colors.ListedColormap(['white', 'dimgray'])
    ax.imshow(grid.T, origin='lower', extent=[0, W*res, 0, H*res],
              cmap=cmap, aspect='equal', alpha=0.6)
    
    # 5.2 渲染 ECM 走廊 (半透明多边形)
    for i, corr in enumerate(corridors):
        poly = corr.polygon
        # 转换到世界坐标
        poly_world = poly * res
        
        # 绘制走廊多边形
        patch = MplPolygon(
            poly_world,
            closed=True,
            facecolor='lightblue',
            edgecolor='blue',
            alpha=0.15,
            linewidth=0.5,
            label='ECM Corridor' if i == 0 else ""
        )
        ax.add_patch(patch)
        
        # 绘制中心线
        center = corr.centerline
        if len(center) > 1:
            center_world = center * res
            ax.plot(center_world[:, 0], center_world[:, 1],
                    'b--', alpha=0.3, linewidth=0.8,
                    label='Centerline' if i == 0 else "")
    
    # 5.3 渲染 GVD 骨架
    skeleton_y, skeleton_x = np.where(gvd_mask)
    skeleton_world_x = skeleton_x * res
    skeleton_world_y = skeleton_y * res
    ax.scatter(skeleton_world_x, skeleton_world_y,
               c='dodgerblue', s=2, alpha=0.4, label='GVD Skeleton')
    
    # 5.4 渲染拓扑节点
    for (j, i) in topo_nodes:
        ax.scatter(i * res, j * res,
                   c='blue', s=30, alpha=0.8, marker='x', zorder=8)
    
    # 5.5 渲染 AGV (3 台，在通道上)
    agv_positions = [
        (2.0, 2.0),
        (5.0, 5.0),
        (8.0, 8.0),
    ]
    agv_headings = [0.0, 0.785, -1.57]
    
    for idx, ((x, y), theta) in enumerate(zip(agv_positions, agv_headings)):
        # 检查是否在障碍物上
        gi, gj = int(y / res), int(x / res)
        if 0 <= gi < H and 0 <= gj < W and grid[gi, gj] == 0:
            color = '#E74C3C'
        else:
            color = 'orange'
            print(f"  WARNING: AGV {idx} at ({x}, {y}) is on obstacle!")
        
        ax.add_patch(Circle((x, y), radius=0.3, 
                            fc=color, ec='#922B21', lw=2, alpha=0.9, zorder=10))
        ax.arrow(x, y, 0.5*np.cos(theta), 0.5*np.sin(theta),
                 head_width=0.2, head_length=0.12,
                 fc='white', ec='white', zorder=11)
        ax.text(x, y, str(idx), ha='center', va='center',
                fontsize=9, color='white', weight='bold', zorder=12)
    
    # 5.6 渲染规划路径
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1],
                color='#27AE60', lw=3.0, label='CHOMP Path', zorder=6)
        ax.scatter(path[0, 0], path[0, 1],
                   c='#2ECC71', s=120, marker='o',
                   edgecolors='black', lw=2, zorder=20, label='Start')
        ax.scatter(path[-1, 0], path[-1, 1],
                   c='#F39C12', s=120, marker='*',
                   edgecolors='black', lw=2, zorder=20, label='Goal')
    
    # 5.7 标题和图例
    ax.set_xlim(0, W * res)
    ax.set_ylim(0, H * res)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('OmniWarehouse: Topological Path Planning\n(GVD Skeleton + ECM Corridors + AGV)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n✅ Visualization saved to: {output_path}")
    print(f"   - GVD skeleton points: {n_gvd}")
    print(f"   - ECM corridors: {len(corridors)}")
    print(f"   - Topology nodes: {n_nodes}")
    print(f"   - Path length: {path_len:.2f}m" if path is not None else "   - No path")


if __name__ == "__main__":
    import os
    os.makedirs('assets', exist_ok=True)
    generate_warehouse_grid()
    visualize_full_pipeline()
    print("\nDone!")
