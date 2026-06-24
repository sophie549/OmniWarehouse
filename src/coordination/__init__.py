"""
多 AGV 协同调度模块 (Multi-AGV Coordination Module)

包含:
- mapo: MAPPO 多智能体强化学习协同
"""

from .mapo import (
    AGVRole,
    Task,
    AGVAgent,
    ConflictResolver,
    BatteryManager,
)

# torch-dependent classes (only available if torch is installed)
try:
    from .mapo import (
        GraphConvLayer,
        ActorNetwork,
        CriticNetwork,
        MAPPOAgent,
    )
except ImportError:
    # torch not available
    pass

__all__ = [
    'AGVRole',
    'Task',
    'AGVAgent',
    'ConflictResolver',
    'BatteryManager',
]

# add torch-dependent classes to __all__ if available
try:
    from .mapo import GraphConvLayer  # just to check
    __all__.extend([
        'GraphConvLayer',
        'ActorNetwork',
        'CriticNetwork',
        'MAPPOAgent',
    ])
except ImportError:
    pass
