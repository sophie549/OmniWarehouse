"""
供应链优化模块 (Supply Chain Module)

包含:
- inventory: 多层级库存优化
- cross_docking: 交叉 docking 优化
- vrp: 车辆路径问题 (遗传算法)
- forecasting: 需求预测 (Transformer)
"""

from .inventory import MultiEchelonOptimizer, InventoryNode, DemandModel, DemandDistribution, SupplyChainNetwork
from .cross_docking import CrossDockingOptimizer, Product, ProductType, DockDoor, Truck, CrossDockFacility
from .vrp import GeneticAlgorithmVRP, Customer, Vehicle, Depot, VRPSolution, DistanceMatrix

# torch is optional (for forecasting module)
try:
    from .forecasting import DemandForecaster, TimeSeries, TransformerForecaster, ForecastingModelType
    __all__ = [
        # 库存
        'MultiEchelonOptimizer',
        'InventoryNode',
        'DemandModel',
        'DemandDistribution',
        'SupplyChainNetwork',
        
        # 交叉 docking
        'CrossDockingOptimizer',
        'Product',
        'ProductType',
        'DockDoor',
        'Truck',
        'CrossDockFacility',
        
        # VRP
        'GeneticAlgorithmVRP',
        'Customer',
        'Vehicle',
        'Depot',
        'VRPSolution',
        'DistanceMatrix',
        
        # 预测
        'DemandForecaster',
        'TimeSeries',
        'TransformerForecaster',
        'ForecastingModelType',
    ]
except ImportError:
    # torch not installed, forecasting unavailable
    __all__ = [
        # 库存
        'MultiEchelonOptimizer',
        'InventoryNode',
        'DemandModel',
        'DemandDistribution',
        'SupplyChainNetwork',
        
        # 交叉 docking
        'CrossDockingOptimizer',
        'Product',
        'ProductType',
        'DockDoor',
        'Truck',
        'CrossDockFacility',
        
        # VRP
        'GeneticAlgorithmVRP',
        'Customer',
        'Vehicle',
        'Depot',
        'VRPSolution',
        'DistanceMatrix',
    ]
