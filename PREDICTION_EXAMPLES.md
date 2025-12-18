# THRB预测功能使用示例

## ✨ 新功能

`predict.py`现已支持**全部6个模型**的预测，包括GNN！

---

## 🚀 命令行使用

### 1. 使用最佳模型预测单个化合物

```bash
python predict.py --smiles "CCOc1ccc(C2NC(=O)NC2=O)cc1"
```

### 2. 使用GNN模型预测

```bash
python predict.py --model gnn --smiles "Cc1ccc(O)cc1"
```

### 3. 使用XGBoost模型预测

```bash
python predict.py --model xgboost --smiles "c1ccccc1"
```

### 4. 比较所有模型的预测结果

```bash
python predict.py --compare --smiles "Cc1ccc(O)cc1"
```

### 5. 从文件批量预测（使用GNN）

```bash
python predict.py --model gnn --input example_compounds.csv --output gnn_predictions.csv
```

### 6. 指定SMILES列名

```bash
python predict.py --input mydata.csv --smiles-column compound_smiles --output results.csv
```

---

## 💻 Python API使用

### 1. 使用特定模型预测

```python
from predict import THRBPredictor

# 使用GNN模型
gnn_predictor = THRBPredictor(model_name='gnn')
result = gnn_predictor.predict_single('CCO')

print(f"预测: {result['activity']}")
print(f"活性概率: {result['probability_active']:.4f}")

# 使用XGBoost模型
xgb_predictor = THRBPredictor(model_name='xgboost')
result = xgb_predictor.predict_single('CCO')
```

### 2. 批量预测

```python
# 使用GNN批量预测（更高效）
gnn_predictor = THRBPredictor(model_name='gnn')

smiles_list = [
    'CCO',
    'c1ccccc1',
    'CC(=O)O'
]

results = gnn_predictor.predict_batch(smiles_list)

for res in results:
    print(f"{res['smiles']}: {res['activity']} ({res['probability_active']:.3f})")
```

### 3. 比较多个模型

```python
from predict import compare_models

# 比较所有模型对同一化合物的预测
smiles = "Cc1ccc(O)cc1"
compare_df = compare_models(smiles)

# 结果会显示每个模型的预测结果
```

### 4. 从文件预测

```python
from predict import THRBPredictor

# 使用GNN从文件预测
predictor = THRBPredictor(model_name='gnn')
results_df = predictor.predict_from_file(
    input_file='compounds.csv',
    smiles_column='smiles',
    output_file='gnn_predictions.csv'
)

print(results_df.head())
```

---

## 📊 可用的模型

| 模型名称 | model_name | 特点 | 推荐场景 |
|---------|-----------|------|---------|
| 最佳模型 | `'best'` | 自动选择 | 通用 |
| Random Forest | `'random_forest'` | 稳定、可解释 | 需要特征重要性 |
| XGBoost | `'xgboost'` | 高性能 | 性能优先 |
| Gradient Boosting | `'gradient_boosting'` | 集成学习 | 稳定预测 |
| SVM | `'svm'` | 核方法 | 小数据集 |
| Logistic Regression | `'logistic_regression'` | 简单快速 | 快速预测 |
| **THRB GNN** ⭐ | `'gnn'` | 直接学习分子图 | **最高精度** |

---

## 🎯 选择模型的建议

### 场景1: 需要最高精度
```python
predictor = THRBPredictor(model_name='gnn')
```

### 场景2: 需要快速预测
```python
predictor = THRBPredictor(model_name='logistic_regression')
```

### 场景3: 需要可解释性
```python
predictor = THRBPredictor(model_name='random_forest')
```

### 场景4: 平衡性能和速度
```python
predictor = THRBPredictor(model_name='xgboost')
```

### 场景5: 不确定选哪个
```python
predictor = THRBPredictor(model_name='best')  # 使用训练时的最佳模型
```

---

## 📝 输出格式

预测结果包含以下字段：

```python
{
    'smiles': 'CCO',
    'valid': True,
    'prediction': 0,
    'activity': 'Inactive',
    'model': 'gnn',
    'probability_inactive': 0.8234,
    'probability_active': 0.1766,
    'confidence': 0.8234,
    'molecular_weight': 46.07,
    'logp': -0.07,
    'h_bond_donors': 1,
    'h_bond_acceptors': 1
}
```

---

## ⚡ 性能对比

| 模型 | 单个预测 | 批量预测(100个) | 准确性 |
|------|---------|----------------|--------|
| Logistic Regression | 最快 | 最快 | 中等 |
| Random Forest | 快 | 快 | 好 |
| XGBoost | 中等 | 中等 | 好 |
| GNN | 慢 | 中等* | **最好** |

*注：GNN支持批量预测优化，批量时效率大幅提升

---

## 🔧 完整示例

```python
from predict import THRBPredictor, compare_models
import pandas as pd

# 1. 创建预测器
print("=" * 60)
print("使用GNN模型预测")
print("=" * 60)

predictor = THRBPredictor(model_name='gnn')

# 2. 预测单个化合物
test_smiles = "Cc1ccc(O)cc1"
result = predictor.predict_single(test_smiles)

print(f"\n单个预测:")
print(f"  SMILES: {result['smiles']}")
print(f"  预测: {result['activity']}")
print(f"  活性概率: {result['probability_active']:.4f}")

# 3. 批量预测
smiles_list = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "Cc1ccc(O)cc1"
]

print(f"\n批量预测 {len(smiles_list)} 个化合物...")
results = predictor.predict_batch(smiles_list)

results_df = pd.DataFrame(results)
print(results_df[['smiles', 'activity', 'probability_active']])

# 4. 多模型对比
print("\n多模型对比:")
compare_models("Cc1ccc(O)cc1")

# 5. 保存结果
results_df.to_csv('my_predictions.csv', index=False)
print(f"\n✅ 结果已保存到 my_predictions.csv")
```

---

## 📞 故障排除

### 问题1: GNN模型不可用

**错误**: `提示: GNN模型依赖未安装，GNN预测功能不可用`

**解决**:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
python test_gnn.py  # 测试安装
```

### 问题2: 模型文件不存在

**错误**: `模型文件不存在: models/model_thrb_gnn.pkl`

**解决**:
```bash
python model_training.py  # 训练所有模型包括GNN
```

### 问题3: SMILES无效

**返回**: `{'valid': False, 'error': 'Invalid SMILES'}`

**解决**: 检查SMILES格式是否正确

---

## 🎓 高级用法

### 自定义模型路径

```python
predictor = THRBPredictor(
    model_path='path/to/custom_model.pkl',
    scaler_path='path/to/custom_scaler.pkl'
)
```

### 获取分子结构图

```python
from predict import THRBPredictor

predictor = THRBPredictor(model_name='gnn')
img = predictor.visualize_molecule('CCO', output_path='molecule.png')
```

---

**更新时间**: 2025-12-07  
**版本**: 2.0 (支持GNN)

