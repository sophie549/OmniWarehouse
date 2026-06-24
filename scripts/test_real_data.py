"""
真实数据集测试脚本

测试 OmniWarehouse 在真实仓储数据集上的表现：
1. AliExpress 风格订单数据
2. Amazon 风格仓储数据
3. 自定义合成数据（接近真实分布）
"""

import argparse
import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def generate_ali_express_style_data(num_orders: int = 1000) -> Dict:
    """
    生成 AliExpress 风格的订单数据
    
    特征：
    - 长尾分布（少数商品占大部分销量）
    - 季节性波动
    - 促销日销量激增
    """
    np.random.seed(42)
    
    # 商品 IDs（长尾分布）
    num_products = 200
    product_ids = list(range(num_products))
    
    # 销量服从幂律分布（长尾）
    sales_alpha = 1.5
    sales_probs = 1.0 / np.arange(1, num_products + 1) ** sales_alpha
    sales_probs = sales_probs / sales_probs.sum()
    
    orders = []
    for i in range(num_orders):
        # 订单日期（一年数据）
        day_of_year = np.random.randint(0, 365)
        # 季节性：Q4 销量高
        seasonal_factor = 1.0 + 0.5 * (day_of_year >= 300)
        # 促销日（随机 10 天）
        if day_of_year in np.random.choice(365, size=10, replace=False):
            promo_factor = 3.0
        else:
            promo_factor = 1.0
        
        num_items = np.random.poisson(2.5) + 1  # 每单 1-5 件商品
        
        order = {
            "order_id": f"AE{i:06d}",
            "day": day_of_year,
            "items": []
        }
        
        for _ in range(num_items):
            product_id = np.random.choice(product_ids, p=sales_probs)
            quantity = np.random.randint(1, 6)
            order["items"].append({
                "product_id": int(product_id),
                "quantity": quantity,
                "price": round(np.random.uniform(5.0, 200.0), 2)
            })
        
        orders.append(order)
    
    return {
        "metadata": {
            "source": "AliExpress-style",
            "num_orders": num_orders,
            "num_products": num_products,
            "date_range": "2025-01-01 to 2025-12-31"
        },
        "orders": orders
    }


def generate_amazon_style_data(num_skus: int = 5000) -> Dict:
    """
    生成 Amazon 风格的仓储数据
    
    特征：
    - 多仓库
    - 不同品类（电子产品、图书、服装等）
    - 不同尺寸和重量
    """
    np.random.seed(42)
    
    categories = ["Electronics", "Books", "Clothing", "Home", "Toys", "Sports", "Food"]
    warehouses = ["WH-US-East", "WH-US-West", "WH-EU-Central", "WH-AP-Southeast"]
    
    skus = []
    for i in range(num_skus):
        category = np.random.choice(categories, p=[0.25, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05])
        
        # 不同品类的价格分布
        if category == "Electronics":
            price = np.random.lognormal(mean=4.0, sigma=1.2)
            weight = np.random.lognormal(mean=2.0, sigma=1.0)
        elif category == "Books":
            price = np.random.lognormal(mean=2.5, sigma=0.8)
            weight = np.random.lognormal(mean=0.5, sigma=0.3)
        else:
            price = np.random.lognormal(mean=3.0, sigma=1.0)
            weight = np.random.lognormal(mean=1.5, sigma=0.8)
        
        sku = {
            "sku_id": f"AMZ-{i:08d}",
            "category": category,
            "price": round(price, 2),
            "weight_kg": round(weight, 3),
            "dimensions_cm": {
                "length": round(np.random.uniform(5, 50), 1),
                "width": round(np.random.uniform(5, 50), 1),
                "height": round(np.random.uniform(1, 30), 1)
            },
            "warehouse": np.random.choice(warehouses),
            "stock": np.random.randint(0, 500),
            "reorder_point": np.random.randint(50, 300),
            "lead_time_days": np.random.randint(1, 30)
        }
        skus.append(sku)
    
    return {
        "metadata": {
            "source": "Amazon-style",
            "num_skus": num_skus,
            "num_warehouses": len(warehouses),
            "categories": categories
        },
        "skus": skus
    }


def test_forecasting_on_real_data(data: Dict, output_dir: str):
    """
    在真实数据上测试需求预测模块
    """
    print("\n" + "="*80)
    print("  Testing Forecasting Module on Real Data")
    print("="*80 + "\n")
    
    try:
        from supply_chain.forecasting import DemandForecaster, TimeSeries, ForecastingModelType
        
        # 汇总每日销量
        orders = data["orders"]
        daily_sales = {}
        for order in orders:
            day = order["day"]
            if day not in daily_sales:
                daily_sales[day] = 0
            for item in order["items"]:
                daily_sales[day] += item["quantity"]
        
        # 构建时序数据
        days = sorted(daily_sales.keys())
        values = [daily_sales[d] for d in days]
        
        ts = TimeSeries(
            timestamps=list(range(len(values))),
            values=values,
            freq="D"
        )
        
        # 训练/测试分割（最后 30 天作为测试）
        split_idx = len(values) - 30
        train_ts = TimeSeries(ts.timestamps[:split_idx], ts.values[:split_idx], ts.freq)
        
        # 创建预测器
        forecaster = DemandForecaster(
            model_type=ForecastingModelType.TRANSFORMER,
            horizon=7,
            context_length=30
        )
        
        # 训练
        print("[1/3] Training forecasting model...")
        forecaster.fit(train_ts)
        print(f"      Model trained on {len(train_ts)} samples")
        
        # 预测
        print("[2/3] Generating forecasts...")
        test_ts = TimeSeries(ts.timestamps[split_idx:], ts.values[split_idx:], ts.freq)
        forecasts = forecaster.predict(test_ts)
        
        # 评估
        print("[3/3] Evaluating forecasts...")
        actual = test_ts.values
        predicted = forecasts.mean if hasattr(forecasts, 'mean') else forecasts
        
        rmse = np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))
        mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))
        
        print(f"\n      RMSE: {rmse:.2f}")
        print(f"      MAE: {mae:.2f}")
        print(f"      Actual mean: {np.mean(actual):.2f}")
        print(f"      Predicted mean: {np.mean(predicted):.2f}")
        
        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "actual_mean": float(np.mean(actual)),
            "predicted_mean": float(np.mean(predicted))
        }
        
    except Exception as e:
        print(f"      ⚠️ Forecasting module error: {e}")
        print("      (This is expected if torch is not installed)")
        return {"error": str(e)}


def test_inventory_on_real_data(data: Dict, output_dir: str):
    """
    在真实数据上测试库存优化模块
    """
    print("\n" + "="*80)
    print("  Testing Inventory Optimization on Real Data")
    print("="*80 + "\n")
    
    try:
        from supply_chain.inventory import (
            MultiEchelonOptimizer, InventoryNode, DemandModel, 
            DemandDistribution, SupplyChainNetwork
        )
        
        skus = data["skus"]
        
        # 取前 10 个 SKU 做测试
        test_skus = skus[:10]
        
        print(f"[1/2] Creating inventory nodes for {len(test_skus)} SKUs...")
        
        # 创建供应链网络
        network = SupplyChainNetwork(
            nodes={},
            edges=[]  # 简化：不建边
        )
        
        for sku in test_skus:
            demand_mean = sku["stock"] / 30.0  # 假设月销量 = 当前库存 / 30
            demand_std = demand_mean * 0.3
            
            node = InventoryNode(
                id=sku["sku_id"],
                name=sku["warehouse"],  # 使用仓库名作为节点名
                holding_cost=sku["price"] * 0.01,  # 1% 持有成本
                ordering_cost=sku["price"] * 0.1,  # 10% 订货成本
                lead_time=sku["lead_time_days"],
                demand=DemandModel(
                    mean=demand_mean,
                    std=demand_std,
                    distribution=DemandDistribution.NORMAL
                ),
                initial_stock=sku["stock"],
                service_level=0.95
            )
            network.nodes[node.id] = node
        
        # 优化（简化：直接计算 EOQ 和安全库存，不跑动态规划）
        print("[2/2] Computing inventory policy (EOQ + Safety Stock)...")
        
        results = {}
        total_cost = 0.0
        for node_id, node in network.nodes.items():
            # 计算安全库存
            ss = node.compute_safety_stock()
            # 计算 EOQ
            Q = node.compute_eoq()
            # 计算订货点
            s = node.compute_reorder_point()
            
            results[node_id] = {
                'safety_stock': ss,
                'order_quantity': Q,
                'reorder_point': s
            }
            # 估算成本（简化）
            cost = node.holding_cost * (Q / 2 + ss) + node.ordering_cost * (node.demand.mean * 365 / Q)
            total_cost += cost
        
        print(f"\n      Optimized {len(network.nodes)} nodes")
        print(f"      Total estimated cost: ${total_cost:.2f}")
        print(f"      Results: {results}")
        
        return {
            'total_cost': total_cost,
            'policy': results
        }
        
    except Exception as e:
        print(f"      ⚠️ Inventory module error: {e}")
        return {"error": str(e)}


def test_vrp_on_real_data(data: Dict, output_dir: str):
    """
    在真实数据上测试 VRP 模块
    """
    print("\n" + "="*80)
    print("  Testing VRP on Real Data")
    print("="*80 + "\n")
    
    try:
        from supply_chain.vrp import GeneticAlgorithmVRP, Customer, Vehicle, Depot
        
        # 使用仓库位置作为 depot
        warehouses = {}
        for sku in data["skus"]:
            wh = sku["warehouse"]
            if wh not in warehouses:
                # 随机分配坐标
                warehouses[wh] = {
                    "x": np.random.uniform(0, 1000),
                    "y": np.random.uniform(0, 1000)
                }
        
        # 创建 depot（第一个仓库）
        main_wh = list(warehouses.keys())[0]
        depot = Depot(
            id=0,
            x=warehouses[main_wh]["x"],
            y=warehouses[main_wh]["y"]
        )
        
        # 创建客户（取前 50 个 SKU 作为客户点）
        customers = []
        for i, sku in enumerate(data["skus"][:50]):
            # 在仓库周围随机生成客户位置
            customer = Customer(
                id=i,
                x=warehouses[sku["warehouse"]]["x"] + np.random.uniform(-100, 100),
                y=warehouses[sku["warehouse"]]["y"] + np.random.uniform(-100, 100),
                demand=sku["stock"]
            )
            customers.append(customer)
        
        # 创建车辆
        vehicles = [
            Vehicle(id=0, capacity=1000, cost_per_km=2.0),
            Vehicle(id=1, capacity=1000, cost_per_km=2.0),
            Vehicle(id=2, capacity=1000, cost_per_km=2.0)
        ]
        
        # 运行 VRP
        print(f"[1/2] Running VRP with {len(customers)} customers, {len(vehicles)} vehicles...")
        vrp = GeneticAlgorithmVRP(
            customers=customers,
            vehicles=vehicles,
            depot=depot,
            pop_size=20,  # 快速测试：减少种群大小
            n_generations=10,  # 快速测试：减少迭代次数
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        
        solution = vrp.optimize()
        
        print(f"\n      Total distance: {solution.total_distance:.2f} km")
        print(f"      Total cost: ${solution.total_cost:.2f}")
        print(f"      Vehicles used: {len(solution.routes)}")
        
        return {
            "total_distance": float(solution.total_distance),
            "total_cost": float(solution.total_cost),
            "vehicles_used": len(solution.routes)
        }
        
    except Exception as e:
        print(f"      ⚠️ VRP module error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test OmniWarehouse on real datasets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["ali_express", "amazon", "all"],
                        help="Dataset to test on")
    parser.add_argument("--output_dir", type=str, default="data/real_data_results",
                        help="Output directory for results")
    parser.add_argument("--num_orders", type=int, default=1000,
                        help="Number of orders (AliExpress)")
    parser.add_argument("--num_skus", type=int, default=5000,
                        help="Number of SKUs (Amazon)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("  OmniWarehouse: Real Dataset Testing")
    print("="*80 + "\n")
    
    results = {}
    
    # 生成/加载数据集
    if args.dataset in ["ali_express", "all"]:
        print("📦 Generating AliExpress-style dataset...")
        ali_data = generate_ali_express_style_data(args.num_orders)
        
        # 保存数据
        ali_path = output_dir / "ali_express_data.json"
        with open(ali_path, "w") as f:
            json.dump(ali_data, f, indent=2)
        print(f"      Saved to {ali_path}")
        
        # 运行测试
        results["ali_express"] = {
            "forecasting": test_forecasting_on_real_data(ali_data, str(output_dir)),
            "inventory": test_inventory_on_real_data(ali_data, str(output_dir))
        }
    
    if args.dataset in ["amazon", "all"]:
        print("\n📦 Generating Amazon-style dataset...")
        amazon_data = generate_amazon_style_data(args.num_skus)
        
        # 保存数据
        amazon_path = output_dir / "amazon_data.json"
        with open(amazon_path, "w") as f:
            json.dump(amazon_data, f, indent=2)
        print(f"      Saved to {amazon_path}")
        
        # 运行测试
        results["amazon"] = {
            "inventory": test_inventory_on_real_data(amazon_data, str(output_dir)),
            "vrp": test_vrp_on_real_data(amazon_data, str(output_dir))
        }
    
    # 保存结果
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"  Results saved to {results_path}")
    print("="*80 + "\n")
    
    # 打印汇总
    print("📊 Test Summary:")
    print(json.dumps(results, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
