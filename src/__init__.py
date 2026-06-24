"""
OmniWarehouse: 下一代仓储-供应链联合优化平台

核心模块:
- planning: 拓扑路径规划 (GVD, ECM, CHOMP, SE(2)/SE(3))
- supply_chain: 供应链优化 (库存, 交叉 docking, VRP, 预测)
- coordination: 多智能体协同 (MAPPO, 冲突消解, 电池管理)
- simulation: 仿真环境 (Isaac Sim/MuJoCo, ROS2)
- models: VLA 模型 (VLAPick, 视觉-语言-动作)
- visualization: 数字孪生可视化 (Dashboard, 3D 渲染)
"""

__version__ = "1.0.0"
__author__ = "OmniWarehouse Team"
__email__ = "omniwarehouse@example.com"

# 模块导入
from .planning import *
from .supply_chain import *
from .coordination import *
from .simulation import *
from .models import *
from .visualization import *

__all__ = [
    # 规划
    "SE2TopoPlanner",
    "SE3GeometricController", 
    "GVDvskeletonExtractor",
    "ECMGlobalPlanner",
    "CHOMPPlanner",
    
    # 供应链
    "MultiEchelonOptimizer",
    "CrossDockingOptimizer",
    "GeneticAlgorithmVRP",
    "DemandForecaster",
    
    # 协同
    "MAPPOAgent",
    "ConflictResolver",
    "BatteryManager",
    
    # 仿真
    "WarehouseEnv",
    "AGVDynamics",
    
    # 模型
    "VLAPickModel",
    "TopoNavigationPolicy",
]

def get_version():
    """返回版本号"""
    return __version__

def run_demo():
    """运行演示"""
    print("=" * 60)
    print(f"OmniWarehouse v{__version__} - Integrated Demo")
    print("=" * 60)
    print()
    print("This demo showcases the integration of:")
    print("  1. Topology-based path planning (GVD, ECM, CHOMP)")
    print("  2. Supply chain optimization (inventory, VRP, cross-docking)")
    print("  3. Multi-AGV MARL coordination (MAPPO)")
    print("  4. SE(2)/SE(3) motion planning")
    print()
    print("Run individual module tests to see detailed outputs.")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
