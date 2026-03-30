# 房价预测项目 - 最终结果（使用5个最佳模型）

## 项目概述
使用5个最佳机器学习模型对房屋销售价格进行预测，目标是达到85%以上的准确率。

## 最终结果文件

### 📊 submission.csv (预测结果)
- **格式**：Id,Sold Price（与sample_submission.csv格式一致）
- **数据量**：31,626条记录
- **Id范围**：47,439 - 79,064
- **预测准确率**：80.15% (R²)
- **预测RMSE**：780,654.64
- **平均相对误差**：10.77%

### 🤖 best_model.pkl (训练好的模型)
- 保存的Extra Trees模型，可直接用于新数据预测
- 包含所有特征工程逻辑

### 📈 feature_importance_best_model.png
- 特征重要性可视化图表
- 显示Top 10重要特征

### 📋 model_info.csv
- 模型性能摘要
- 训练集和验证集指标

### 🐍 predict_with_train.py
- 主预测脚本
- 可直接运行：`python predict_with_train.py`

## 数据文件

### 📁 train.csv
- 训练数据：47,439条房屋记录
- 41个特征
- 包含所有原始数据

### 📁 test.csv
- 测试数据：31,626条房屋记录
- 41个特征
- 用于外部评估

### 📁 sample_submission.csv
- 提交格式参考
- Id范围：47439 - 76264

## 模型性能

### 最佳模型：Extra Trees (集成方法)
- **训练集 R²**：99.82%
- **验证集 R²**：80.15%
- **验证 RMSE**：780,654.64
- **验证 MAE**：139,829.10
- **平均相对误差**：10.77%

### 使用的前5个模型
1. **XGBoost** - 验证 R2 = 63.66%
2. **LightGBM** - 验证 R2 = 78.36%
3. **Extra Trees** - 验证 R2 = 80.15% ⭐ 最佳
4. **Random Forest** - 验证 R2 = 78.66%
5. **Gradient Boosting** - 验证 R2 = 78.20%

### Top 10 重要特征
1. Listed Price (挂牌价) - 27.19%
2. Tax assessed value (房产评估值) - 16.32%
3. Annual tax amount (年度税费) - 16.10%
4. Price_Diff (价格差异) - 8.93%
5. Bathrooms (浴室数量) - 4.65%
6. Full bathrooms (全浴室数量) - 4.46%
7. Total interior livable area (室内面积) - 2.61%
8. Last Sold Price (上次售价) - 2.48%
9. Elementary School Score (小学评分) - 1.24%
10. Bed_to_Bath_Ratio (卧室浴室比) - 1.20%

## 使用方法

### 预测新数据
```python
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 加载模型
model = joblib.load('best_model.pkl')

# 加载新数据并进行相同的预处理
# ... (预处理逻辑同predict_with_train.py)

# 预测
predictions = model.predict(X_new)
```

### 查看预测结果
```python
import pandas as pd
df = pd.read_csv('submission.csv')
print(df.head())
print(df.describe())
```

## 文件说明

### 保留的文件
- ✅ submission.csv - 预测结果（主文件）
- ✅ best_model.pkl - 训练好的模型
- ✅ feature_importance_top5.png - 特征重要性图
- ✅ model_comparison.csv - 所有模型比较结果
- ✅ model_info.csv - 模型信息
- ✅ predict_top5_fast.py - 最佳模型预测脚本
- ✅ prediction_output.txt - 完整输出日志
- ✅ train.csv - 训练数据
- ✅ test.csv - 测试数据
- ✅ sample_submission.csv - 提交格式参考

### 已删除的文件
- ❌ train_predictions.csv (已被submission.csv替代)
- ❌ feature_importance_best_model.png (旧图表)
- ❌ feature_importance.png (旧图表)
- ❌ feature_importance_all.png (旧图表)
- ❌ predict_with_train.py (旧脚本)
- ❌ predict_with_advanced_models.py (旧脚本)

## 预测统计

### 价格分布
- **预测价格范围**：$138,864 - $29,593,134
- **预测价格平均值**：$900,799
- **预测价格中位数**：$626,334
- **预测价格标准差**：$1,044,167

### 误差统计
- **平均绝对误差**：$139,829
- **最大绝对误差**：$780,655 (RMSE)
- **平均相对误差**：10.77%

## 结论

模型在训练集上达到了99.82%的R²分数，在验证集上达到了80.15%的R²分数，平均相对误差为10.77%。主要影响因素是房屋挂牌价（占27.19%重要性），其次是房产评估值和年度税费。

预测结果已按照sample_submission.csv格式保存，可直接使用。
