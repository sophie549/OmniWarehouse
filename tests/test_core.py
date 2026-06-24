"""
OmniWarehouse 单元测试

测试各核心模块的基本功能
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================
# 测试 1: GVD 骨架提取
# ============================
class TestGVD:
    def test_create_test_warehouse(self):
        from planning.gvd import create_test_warehouse
        grid = create_test_warehouse(width=100, height=100)
        assert grid.shape == (100, 100)
        assert grid.dtype == bool
    
    def test_gvd_extract(self):
        from planning.gvd import GVDSkeletonExtractor, create_test_warehouse
        grid = create_test_warehouse(width=100, height=100)
        extractor = GVDSkeletonExtractor(resolution=0.05, prune_length=1.0)
        gvd_mask, topo_nodes = extractor.extract(grid)
        assert gvd_mask.shape == grid.shape
        assert isinstance(topo_nodes, list)


# ============================
# 测试 2: SE(2) 规划器
# ============================
class TestSE2Planner:
    def test_se2_pose(self):
        from planning.se2_planner import SE2Pose
        pose = SE2Pose(x=1.0, y=2.0, theta=0.5)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.theta == 0.5
    
    def test_diff_drive_kinematics(self):
        from planning.se2_planner import DiffDriveKinematics, SE2Pose
        kin = DiffDriveKinematics(wheel_radius=0.1, wheel_base=0.5)
        pose = SE2Pose(x=0.0, y=0.0, theta=0.0)
        v = 1.0
        omega = 0.1
        dt = 0.1
        new_pose = kin.step(pose, v, omega, dt)
        assert new_pose.x > 0.0
        assert new_pose.y == 0.0  # 直线行驶


# ============================
# 测试 3: 供应链优化
# ============================
class TestSupplyChain:
    def test_demand_model(self):
        from supply_chain.inventory import DemandModel, DemandDistribution
        demand = DemandModel(
            distribution=DemandDistribution.NORMAL,
            mean=50.0,
            std=15.0
        )
        samples = [demand.sample() for _ in range(100)]
        assert len(samples) == 100
    
    def test_inventory_node(self):
        from supply_chain.inventory import InventoryNode, DemandModel, DemandDistribution
        demand = DemandModel(
            distribution=DemandDistribution.NORMAL,
            mean=50.0,
            std=15.0
        )
        node = InventoryNode(
            id="test_001",
            name="Test Node",
            holding_cost=2.0,
            ordering_cost=100.0,
            lead_time=7,
            demand=demand
        )
        assert node.id == "test_001"
        assert node.holding_cost == 2.0
    
    def test_vrp_customer(self):
        from supply_chain.vrp import Customer
        customer = Customer(id=1, x=100.0, y=200.0, demand=50.0)
        assert customer.id == 1
        assert customer.demand == 50.0


# ============================
# 测试 4: 协同调度
# ============================
class TestCoordination:
    def test_agv_agent(self):
        from coordination.mapo import AGVAgent, AGVRole
        import numpy as np
        agv = AGVAgent(
            id=0,
            role=AGVRole.PICKER,
            state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        )
        assert agv.id == 0
        assert agv.role == AGVRole.PICKER
        assert agv.state.shape == (6,)
    
    def test_task(self):
        from coordination.mapo import Task
        task = Task(
            id=0,
            type="pick",
            pickup=(0.0, 0.0),
            dropoff=(10.0, 10.0),
            priority=3
        )
        assert task.id == 0
        assert task.type == "pick"
    
    def test_conflict_resolver(self):
        from coordination.mapo import ConflictResolver
        resolver = ConflictResolver(safety_distance=1.5)
        assert resolver.safety_distance == 1.5


# ============================
# 测试 5: SDF (有符号距离场)
# ============================
class TestSDF:
    def test_sdf_compute(self):
        from planning.chomp import SignedDistanceField
        import numpy as np
        # 创建一个简单的栅格 (中间有障碍物)
        grid = np.zeros((50, 50), dtype=bool)
        grid[20:30, 20:30] = True  # 障碍物
        sdf = SignedDistanceField.compute_from_grid(grid, resolution=0.1)
        assert sdf is not None
    
    def test_sdf_query(self):
        from planning.chomp import SignedDistanceField
        import numpy as np
        grid = np.zeros((50, 50), dtype=bool)
        sdf = SignedDistanceField.compute_from_grid(grid, resolution=0.1)
        d = sdf.query(np.array([0.5, 0.5]))
        assert isinstance(d, (int, float, np.floating))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
