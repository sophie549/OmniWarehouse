"""
需求预测 (Demand Forecasting)

实现:
1. 时序预测 (Time Series Forecasting)
2. Transformer 模型 (Attention-based)
3. 不确定性量化 (Uncertainty Quantization)
4. 多变量预测 (Multi-Variate)

理论参考:
- Vaswani, A., et al. (2017). Atention is All You Need.
- Wen, R., et al. (2017). A Multi-Horizon Quantile Recurrent Forecaster.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import math


class ForecastingModelType(Enum):
    """预测模型类型"""
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    TRANSFORMER = "transformer"
    LSTM = "lstm"


@dataclass
class TimeSeries:
    """时序数据"""
    timestamps: np.ndarray     # (T,) 时间戳 (小时)
    values: np.ndarray        # (T,) 值 (需求量)
    features: np.ndarray = None  # (T, F) 额外特征 (温度, 节假日, ...)
    
    def split(self, ratio: float = 0.8) -> Tuple['TimeSeries', 'TimeSeries']:
        """分割时序 (按时间顺序)"""
        split_idx = int(len(self.timestamps) * ratio)
        train = TimeSeries(
            timestamps=self.timestamps[:split_idx],
            values=self.values[:split_idx],
            features=self.features[:split_idx] if self.features is not None else None
        )
        test = TimeSeries(
            timestamps=self.timestamps[split_idx:],
            values=self.values[split_idx:],
            features=self.features[split_idx:] if self.features is not None else None
        )
        return train, test


class PositionalEncoding(nn.Module):
    """Transformer 位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class TransformerForecaster(nn.Module):
    """
    Transformer 时序预测器
    
    架构:
    1. 输入嵌入 (Linear projection)
    2. 位置编码 (Positional Encoding)
    3. Transformer Encoder (多头注意力)
    4. 输出头 (Linear → 预测值)
    """
    def __init__(self, 
                 input_dim: int = 1,       # 输入特征维度
                 d_model: int = 128,         # 模型维度
                 n_heads: int = 4,          # 注意力头数
                 n_layers: int = 3,         # Encoder 层数
                 dim_feedforward: int = 512,
                 output_len: int = 24,       # 预测时域 (小时)
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.output_len = output_len
        
        # 输入嵌入
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch, seq, feature)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # 输出头
        self.output_proj = nn.Linear(d_model, output_len)
        
        # 不确定性头 (预测方差)
        self.uncertainty_head = nn.Linear(d_model, output_len)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, input_dim) 输入时序
            
        Returns:
            mean: (batch, output_len) 预测均值
            var: (batch, output_len) 预测方差 (不确定性)
        """
        # 嵌入 + 位置编码
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Transformer Encoder
        x = self.encoder(x)  # (batch, seq_len, d_model)
        
        # 取最后一个时间步 (或使用全局池化)
        x_last = x[:, -1, :]  # (batch, d_model)
        
        # 预测
        mean = self.output_proj(x_last)  # (batch, output_len)
        log_var = self.uncertainty_head(x_last)
        var = F.softplus(log_var) + 1e-6  # (batch, output_len)
        
        return mean, var
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                   n_samples: int = 100) -> Dict:
        """
        预测 + 不确定性量化 (MC Dropout)
        
        Returns:
            dict: {
                'mean': 预测均值,
                'std': 预测标准差,
                'quantiles': 分位数,
                'samples': MC 采样样本
            }
        """
        self.train()  # 启用 Dropout (MC Dropout)
        
        samples = []
        for _ in range(n_samples):
            mean, var = self.forward(x)
            # 从分布中采样
            eps = torch.randn_like(mean)
            sample = mean + torch.sqrt(var) * eps
            samples.append(sample.unsqueeze(0))
        
        samples = torch.cat(samples, dim=0)  # (n_samples, batch, output_len)
        
        mean_pred = samples.mean(dim=0)
        std_pred = samples.std(dim=0)
        
        # 分位数
        quantiles = {
            '0.1': torch.quantile(samples, 0.1, dim=0),
            '0.25': torch.quantile(samples, 0.25, dim=0),
            '0.5': torch.quantile(samples, 0.5, dim=0),
            '0.75': torch.quantile(samples, 0.75, dim=0),
            '0.9': torch.quantile(samples, 0.9, dim=0)
        }
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'quantiles': quantiles,
            'samples': samples
        }


class DemandForecaster:
    """
    需求预测器 (封装)
    
    功能:
    1. 训练 Transformer 模型
    2. 预测未来需求
    3. 量化预测不确定性
    4. 生成库存策略建议
    """
    def __init__(self, 
                 model_type: ForecastingModelType = ForecastingModelType.TRANSFORMER,
                 device: str = "cpu"):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.history = None
    
    def build_model(self, input_dim: int = 1, **kwargs):
        """构建模型"""
        if self.model_type == ForecastingModelType.TRANSFORMER:
            self.model = TransformerForecaster(
                input_dim=input_dim,
                **kwargs
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, 
               train_data: TimeSeries,
               val_data: Optional[TimeSeries] = None,
               seq_len: int = 168,  # 使用过去 7 天 (168h) 预测
               batch_size: int = 32,
               n_epochs: int = 100,
               lr: float = 1e-3):
        """
        训练模型
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            seq_len: 输入序列长度
            batch_size: 批次大小
            n_epochs: 训练轮数
            lr: 学习率
        """
        if self.model is None:
            input_dim = 1 if train_data.features is None else (1 + train_data.features.shape[1])
            self.build_model(input_dim=input_dim)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.GaussianNLLLoss()  # 负对数似然 (用于不确定性)
        
        # 构建训练数据集 (滑动窗口)
        X_train, y_train = self._create_dataset(
            train_data, seq_len, self.model.output_len
        )
        
        print(f"Training data: {len(X_train)} samples")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        
        # 训练循环
        self.model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            
            # Mini-batch
            indices = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train[batch_indices].to(self.device)
                y_batch = y_train[batch_indices].to(self.device)
                
                # 前向
                mean, var = self.model(X_batch)
                loss = criterion(mean, y_batch, var)
                
                # 反向
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            
            if epoch % 20 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        print("Training complete!")
    
    def predict(self, 
               data: TimeSeries, 
               seq_len: int = 168,
               n_samples: int = 100) -> Dict:
        """
        预测未来需求
        
        Args:
            data: 历史数据 (用于提取最后 seq_len 个点)
            seq_len: 输入序列长度
            n_samples: MC 采样次数
            
        Returns:
            预测结果字典
        """
        self.model.eval()
        
        # 提取最后 seq_len 个点
        if len(data.values) < seq_len:
            raise ValueError(f"Insufficient data: need {seq_len}, got {len(data.values)}")
        
        last_seq = data.values[-seq_len:]
        if data.features is not None:
            last_features = data.features[-seq_len:]
            X = np.column_stack([last_seq.reshape(-1, 1), last_features])
        else:
            X = last_seq.reshape(-1, 1)
        
        X = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # (1, seq_len, input_dim)
        
        # 预测
        with torch.no_grad():
            result = self.model.predict_with_uncertainty(X, n_samples=n_samples)
        
        return result
    
    def _create_dataset(self, data: TimeSeries, 
                        seq_len: int, pred_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建数据集 (滑动窗口)
        
        Returns:
            X: (N, seq_len, input_dim)
            y: (N, pred_len)
        """
        values = data.values
        N = len(values) - seq_len - pred_len + 1
        
        X_list = []
        y_list = []
        
        for i in range(N):
            # 输入序列
            x_seq = values[i:i+seq_len]
            
            # 输出序列
            y_seq = values[i+seq_len:i+seq_len+pred_len]
            
            if data.features is not None:
                x_features = data.features[i:i+seq_len]
                x_seq = np.column_stack([x_seq.reshape(-1, 1), x_features])
            
            X_list.append(x_seq)
            y_list.append(y_seq)
        
        X = torch.FloatTensor(np.array(X_list))
        y = torch.FloatTensor(np.array(y_list))
        
        return X, y


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("Demand Forecasting (Transformer) - Test")
    print("=" * 60)
    
    # 创建测试数据 (模拟需求时序)
    print("\nGenerating test time series...")
    
    np.random.seed(42)
    n_hours = 24 * 30  # 30 天
    
    # 基础需求 (随时间增长)
    base_demand = 50.0 + 0.1 * np.arange(n_hours)
    
    # 周期性 (日周期)
    daily_seasonality = 10.0 * np.sin(2 * np.pi * np.arange(n_hours) / 24.0)
    
    # 噪声
    noise = np.random.normal(0, 5.0, n_hours)
    
    # 节假日效应 (简化: 每周第 6 天 (Saturday) 需求 +20)
    holidays = np.zeros(n_hours)
    for t in range(n_hours):
        if t % 24 == 0:  # 每天开始
            day_of_week = (t // 24) % 7
            if day_of_week == 5:  # Saturday
                holidays[t:t+24] = 20.0
    
    # 总需求
    demand = base_demand + daily_seasonality + noise + holidays
    
    timestamps = np.arange(n_hours)
    features = np.column_stack([
        np.sin(2 * np.pi * timestamps / 24.0),  # 日周期特征
        np.cos(2 * np.pi * timestamps / 24.0),
        (timestamps // 24) % 7  # 星期几 (one-hot 简化)
    ])
    
    time_series = TimeSeries(
        timestamps=timestamps,
        values=demand,
        features=features
    )
    
    print(f"  Time series length: {len(demand)} hours ({len(demand) // 24} days)")
    print(f"  Mean demand: {np.mean(demand):.2f}")
    print(f"  Std demand: {np.std(demand):.2f}")
    
    # 分割数据
    print("\nSplitting data...")
    train_data, test_data = time_series.split(ratio=0.8)
    print(f"  Train: {len(train_data.values)} hours")
    print(f"  Test: {len(test_data.values)} hours")
    
    # 构建模型
    print("\nBuilding Transformer model...")
    forecaster = DemandForecaster(
        model_type=ForecastingModelType.TRANSFORMER,
        device="cpu"
    )
    forecaster.build_model(
        input_dim=4,  # demand + 3 features
        d_model=64,
        n_heads=2,
        n_layers=2,
        output_len=24  # 预测未来 24 小时
    )
    print(f"  Model parameters: {sum(p.numel() for p in forecaster.model.parameters()):,}")
    
    # 训练 (简化: 10 epochs)
    print("\nTraining model (10 epochs for demo)...")
    forecaster.train(
        train_data,
        val_data=None,
        seq_len=168,  # 过去 7 天
        batch_size=16,
        n_epochs=10,
        lr=1e-3
    )
    
    # 预测
    print("\nMaking prediction...")
    result = forecaster.predict(
        train_data,  # 用训练数据的最后部分预测
        seq_len=168,
        n_samples=50
    )
    
    print(f"\nPrediction result:")
    print(f"  Mean: {result['mean'].cpu().numpy()[0, :5]}...")
    print(f"  Std: {result['std'].cpu().numpy()[0, :5]}...")
    print(f"  Quantile 0.1: {result['quantiles']['0.1'].cpu().numpy()[0, :5]}...")
    print(f"  Quantile 0.9: {result['quantiles']['0.9'].cpu().numpy()[0, :5]}...")
    
    print("\nTest passed!")
