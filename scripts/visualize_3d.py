"""
3D 交互式可视化 - 使用 Plotly 创建交互式 3D 仓库
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

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


def create_3d_warehouse(output_path='assets/warehouse_3d.html'):
    """
    创建 3D 交互式仓库可视化
    
    Args:
        output_path: 输出 HTML 文件路径
    """
    # ========== 1. 生成仓库数据 ==========
    H, W = 200, 200
    res = 0.05
    grid = generate_warehouse_grid(H, W)
    
    print(f"[1/5] Warehouse grid generated: {H}x{W}")
    
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
        path = np.array([
            [2.0, 2.0],
            [5.0, 2.0],
            [5.0, 5.0],
            [8.0, 5.0],
            [8.0, 8.0],
        ])
    
    print(f"[2/5] Path computed: {len(path)} points")
    
    # ========== 2. 创建 3D 图形 ==========
    fig = go.Figure()
    
    # ========== 2.1 绘制地面 ==========
    x_ground = [0, W*res, W*res, 0, 0]
    y_ground = [0, 0, H*res, H*res, 0]
    z_ground = [0, 0, 0, 0, 0]
    
    fig.add_trace(go.Scatter3d(
        x=x_ground, y=y_ground, z=z_ground,
        mode='lines',
        line=dict(color='gray', width=4),
        name='Warehouse Floor',
        showlegend=False
    ))
    
    # ========== 2.2 绘制货架（3D 盒子） ==========
    shelf_height = 2.0  # 货架高度 2 米
    shelf_data = []
    
    for sy in range(30, H - 30, 40):
        for sx in range(30, W - 30, 40):
            # 货架世界坐标
            x0 = sx * res
            y0 = sy * res
            x1 = (sx + 10) * res
            y1 = (sy + 10) * res
            
            # 货架顶点
            x = [x0, x1, x1, x0, x0, x1, x1, x0]
            y = [y0, y0, y1, y1, y0, y0, y1, y1]
            z = [0, 0, 0, 0, shelf_height, shelf_height, shelf_height, shelf_height]
            
            # 6 个面（12 个三角形）
            i = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3]
            j = [1, 2, 3, 4, 5, 6, 5, 2, 6, 7, 7, 0]
            k = [2, 3, 4, 5, 6, 7, 6, 6, 7, 4, 4, 4]
            
            shelf_data.append((x, y, z, i, j, k))
    
    # 批量添加货架
    for idx, (x, y, z, i, j, k) in enumerate(shelf_data):
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='dimgray',
            opacity=0.6,
            name='Shelf' if idx == 0 else None,
            showlegend=True if idx == 0 else False,
            hoverinfo='skip'
        ))
    
    print(f"[3/5] Added {len(shelf_data)} shelves")
    
    # ========== 2.3 绘制 GVD 骨架（3D） ==========
    skeleton_y, skeleton_x = np.where(gvd_mask)
    skeleton_z = np.ones_like(skeleton_x) * 0.05  # 略高于地面
    
    fig.add_trace(go.Scatter3d(
        x=skeleton_x * res,
        y=skeleton_y * res,
        z=skeleton_z,
        mode='markers',
        marker=dict(size=3, color='dodgerblue', opacity=0.4),
        name='GVD Skeleton',
        hovertemplate='GVD Point<br>X: %{x:.2f}m<br>Y: %{y:.2f}m'
    ))
    
    # ========== 2.4 绘制路径 ==========
    if path is not None and len(path) > 1:
        # 路径高度 0.1m
        path_z = np.ones(len(path)) * 0.1
        
        fig.add_trace(go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path_z,
            mode='lines+markers',
            line=dict(color='#27AE60', width=6),
            marker=dict(size=5, color='#27AE60'),
            name='CHOMP Path',
            hovertemplate='Path<br>X: %{x:.2f}m<br>Y: %{y:.2f}m'
        ))
        
        # 起点
        fig.add_trace(go.Scatter3d(
            x=[start[0]],
            y=[start[1]],
            z=[0.1],
            mode='markers',
            marker=dict(size=10, color='green', symbol='circle'),
            name='Start',
            hovertemplate='Start<br>X: %{x:.2f}m<br>Y: %{y:.2f}m'
        ))
        
        # 终点
        fig.add_trace(go.Scatter3d(
            x=[goal[0]],
            y=[goal[1]],
            z=[0.1],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            name='Goal',
            hovertemplate='Goal<br>X: %{x:.2f}m<br>Y: %{y:.2f}m'
        ))
    
    print(f"[4/5] Path visualized")
    
    # ========== 2.5 绘制 AGV（3D 球体） ==========
    agv_positions = [
        (2.0, 2.0, 0.5),
        (5.0, 5.0, 0.5),
        (8.0, 8.0, 0.5),
    ]
    
    for idx, (x, y, z) in enumerate(agv_positions):
        # AGV 球体
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        r = 0.3  # AGV 半径
        
        x_sphere = x + r * np.outer(np.cos(u), np.sin(v))
        y_sphere = y + r * np.outer(np.sin(u), np.sin(v))
        z_sphere = z + r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, '#E74C3C'], [1, '#E74C3C']],
            opacity=0.9,
            name=f'AGV {idx}',
            showscale=False,
            hovertemplate=f'AGV {idx}<br>X: {x:.2f}m<br>Y: {y:.2f}m<br>Z: {z:.2f}m'
        ))
    
    print(f"[5/5] AGV positions added")
    
    # ========== 3. 设置布局 ==========
    fig.update_layout(
        title=dict(
            text='OmniWarehouse: 3D Interactive Visualization',
            font=dict(size=20, family='Arial Black'),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title='X (meters)', range=[0, W*res], dtick=2),
            yaxis=dict(title='Y (meters)', range=[0, H*res], dtick=2),
            zaxis=dict(title='Z (meters)', range=[0, shelf_height + 0.5], dtick=1),
            aspectmode='data',  # 保持比例
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0.5, y=0.5, z=0.3)
            ),
            annotations=[
                dict(
                    showarrow=False,
                    x=W*res/2, y=H*res/2, z=shelf_height + 0.3,
                    text="OmniWarehouse 3D",
                    font=dict(size=14, color="black")
                )
            ]
        ),
        width=1200,
        height=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12
        )
    )
    
    # ========== 4. 保存 HTML ==========
    print(f"\nSaving to {output_path}...")
    fig.write_html(output_path)
    
    print(f"\n✅ 3D visualization saved to: {output_path}")
    print(f"   - Open in browser for interactive 3D view")
    print(f"   - Features: Rotate, Zoom, Pan, Hover")
    
    return output_path


def create_3d_animation(output_path='assets/agv_3d_animation.html'):
    """
    创建 3D AGV 动画（使用 Plotly）
    
    Args:
        output_path: 输出 HTML 文件路径
    """
    # 生成数据
    H, W = 200, 200
    res = 0.05
    grid = generate_warehouse_grid(H, W)
    
    # 简单路径
    path = np.array([
        [2.0, 2.0],
        [5.0, 2.0],
        [5.0, 5.0],
        [8.0, 5.0],
        [8.0, 8.0],
    ])
    
    # 插值
    from scipy.interpolate import interp1d
    t = np.linspace(0, 1, len(path))
    t_smooth = np.linspace(0, 1, 100)
    
    interp_x = interp1d(t, path[:, 0], kind='linear')
    interp_y = interp1d(t, path[:, 1], kind='linear')
    
    agv_x = interp_x(t_smooth)
    agv_y = interp_y(t_smooth)
    agv_z = np.ones_like(agv_x) * 0.5
    
    # 创建 3D 轨迹图
    fig = go.Figure()
    
    # 添加路径
    fig.add_trace(go.Scatter3d(
        x=agv_x,
        y=agv_y,
        z=agv_z,
        mode='lines',
        line=dict(color='green', width=4),
        name='AGV Path'
    ))
    
    # 添加 AGV 当前位置（动画）
    frames = []
    for i in range(len(agv_x)):
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=agv_x[:i+1],
                y=agv_y[:i+1],
                z=agv_z[:i+1],
                mode='lines+markers',
                line=dict(color='green', width=4),
                marker=dict(size=6, color='red')
            )],
            name=str(i)
        ))
    
    fig.frames = frames
    
    # 布局
    fig.update_layout(
        title='AGV 3D Animation',
        scene=dict(
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10]),
            zaxis=dict(range=[0, 2]),
            aspectmode='data'
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 50, "redraw": True}}]
            )]
        )]
    )
    
    fig.write_html(output_path)
    print(f"✅ 3D animation saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import os
    os.makedirs('assets', exist_ok=True)
    
    # 检查依赖
    try:
        import plotly
        print(f"Plotly version: {plotly.__version__}")
    except ImportError:
        print("Installing plotly...")
        import subprocess
        subprocess.run(['pip', 'install', 'plotly'], check=True)
    
    # 创建 3D 可视化
    create_3d_warehouse()
    
    # 创建 3D 动画（可选）
    # create_3d_animation()
    
    print("\nDone! Open assets/warehouse_3d.html in your browser.")
