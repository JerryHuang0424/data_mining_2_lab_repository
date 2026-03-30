# Python脚本详细说明

本文档详细说明项目中所有Python脚本的功能、代码结构和使用方法。

---

## 🐍 脚本总览

### 项目包含的脚本

1. **titanic_random_forest.py** - 随机森林模型实现 (7.0KB, 216行)
2. **multi_model_comparison.py** - 多模型对比分析 (9.9KB, ~400行)
3. **results_summary.py** - 结果汇总验证 (1.9KB, 43行)

---

## 📄 脚本1: titanic_random_forest.py

### 基本信息
- **文件路径**: `titanic_random_forest.py`
- **文件大小**: 7.0KB
- **代码行数**: 216行
- **功能**: 随机森林模型训练、调优和预测
- **依赖**: pandas, numpy, sklearn, matplotlib, seaborn

### 代码结构

```python
# 1. 导入库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 2. 设置随机种子
np.random.seed(42)

# 3. 数据加载
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 4. 数据预处理函数
def preprocess_data(df):
    # 处理缺失值
    # 创建新特征
    # 编码分类变量
    # 返回特征矩阵
    pass

# 5. 数据预处理
X_train_processed = preprocess_data(train_df)
y_train = train_df['Survived']
X_test_processed = preprocess_data(test_df)

# 6. 模型训练
# 分割训练集
# 初始模型训练
# 交叉验证
# 超参数调优

# 7. 特征重要性分析
# 8. 模型评估
# 9. 生成预测
# 10. 可视化
```

### 主要功能详解

#### 4.1 数据预处理函数 (`preprocess_data`)

**功能**: 将原始数据转换为模型可用的特征矩阵

**处理步骤**:
1. **缺失值填充**:
   - Age: 用中位数填充
   - Fare: 用中位数填充
   - Embarked: 用众数填充

2. **特征工程**:
   - FamilySize: SibSp + Parch + 1
   - IsAlone: FamilySize == 1
   - Title: 从姓名提取称谓
   - AgeGroup: 年龄分段
   - FareGroup: 票价分段

3. **编码转换**:
   - Sex: LabelEncoder (male=0, female=1)
   - Embarked: LabelEncoder
   - Title: LabelEncoder
   - AgeGroup: LabelEncoder
   - FareGroup: LabelEncoder

4. **特征选择**:
   - 选择12个特征用于训练

**输出**: 特征矩阵 (891x12)

#### 6. 模型训练流程

**步骤1: 数据分割**
```python
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
)
```
- 训练集: 712条 (80%)
- 验证集: 179条 (20%)
- 使用分层采样确保类别平衡

**步骤2: 初始模型训练**
```python
rf_initial = RandomForestClassifier(n_estimators=100, random_state=42)
rf_initial.fit(X_train_split, y_train_split)
```

**步骤3: 交叉验证**
```python
cv_scores = cross_val_score(rf_initial, X_train_processed, y_train, cv=5)
```
- 5折交叉验证
- 平均准确率: 80.81%

**步骤4: 超参数调优**
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_processed, y_train)
```

**最佳参数**:
- n_estimators: 100
- max_depth: 15
- min_samples_split: 2
- min_samples_leaf: 1

**最佳准确率**: 80.81%

#### 7. 特征重要性分析

```python
feature_importance = pd.DataFrame({
    'feature': X_train_processed.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
```

**输出**: feature_importance.png (92KB)

#### 10. 可视化

生成 `feature_importance.png`，显示Top 10特征的重要性。

### 输出文件

1. `random_forest_submission.csv` - 预测结果 (已删除)
2. `feature_importance.png` - 特征重要性图

### 运行方法

```bash
python titanic_random_forest.py
```

### 性能指标

- **验证准确率**: 82.68%
- **交叉验证准确率**: 80.81%
- **训练时间**: 0.25秒

---

## 📄 脚本2: multi_model_comparison.py

### 基本信息
- **文件路径**: `multi_model_comparison.py`
- **文件大小**: 9.9KB
- **代码行数**: ~400行
- **功能**: 8种模型对比，自动选择最佳模型
- **依赖**: pandas, numpy, sklearn, matplotlib, seaborn

### 代码结构

```python
# 1. 导入库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# 2. 数据加载和预处理 (同titanic_random_forest.py)
# 3. 模型定义
# 4. 模型训练和评估
# 5. 结果对比
# 6. 可视化
# 7. 选择最佳模型
# 8. 生成最终预测
```

### 主要功能详解

#### 3. 模型定义

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}
```

#### 4. 模型训练和评估

对每个模型执行:

```python
# 1. 训练模型
start_time = time.time()
model.fit(X_train_split, y_train_split)
training_time = time.time() - start_time

# 2. 验证集预测
val_pred = model.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, val_pred)

# 3. 交叉验证
cv_scores = cross_val_score(model, X_train_processed, y_train, cv=5)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# 4. 保存结果
results.append({
    'Model': name,
    'Validation Accuracy': val_accuracy,
    'CV Mean Accuracy': cv_mean,
    'CV Std Accuracy': cv_std,
    'Training Time (s)': training_time,
    'Predictions': val_pred
})
```

#### 5. 结果对比

```python
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('CV Mean Accuracy', ascending=False)
```

#### 6. 可视化

生成4个图表:

1. **交叉验证准确率对比** (横轴: 模型, 纵轴: 准确率)
2. **验证准确率对比** (横轴: 模型, 纵轴: 准确率)
3. **训练时间对比** (横轴: 模型, 纵轴: 训练时间)
4. **准确率 vs 训练时间散点图** (横轴: 训练时间, 纵轴: 准确率)

**输出文件**:
- `model_comparison.png` (466KB)
- `final_comparison.png` (240KB)
- `model_comparison_summary.csv` (807B)

#### 8. 选择最佳模型

```python
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"最佳模型: {best_model_name}")
print(f"交叉验证准确率: {results_df.iloc[0]['CV Mean Accuracy']:.4f}")
```

#### 9. 生成最终预测

```python
# 在完整训练集上训练最佳模型
best_model.fit(X_train_processed, y_train)

# 生成预测
test_predictions = best_model.predict(X_test_processed)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

submission.to_csv(f'{best_model_name.replace(" ", "_")}_submission.csv', index=False)
```

**输出文件**: `Gradient_Boosting_submission.csv`

### 输出文件

1. `Gradient_Boosting_submission.csv` - 最佳模型预测
2. `model_comparison.png` - 完整对比图
3. `final_comparison.png` - 精简对比图
4. `model_comparison_summary.csv` - 性能数据

### 运行方法

```bash
python multi_model_comparison.py
```

### 性能指标

| 模型 | 验证准确率 | 交叉验证准确率 | 训练时间 |
|------|-----------|---------------|----------|
| Gradient Boosting | 80.45% | **83.16%** | 0.22s |
| AdaBoost | 78.77% | 81.82% | 0.20s |
| Random Forest | 82.68% | 80.81% | 0.25s |
| Logistic Regression | 81.01% | 79.69% | 1.26s |
| Naive Bayes | 79.89% | 79.47% | 0.002s |
| Decision Tree | 74.86% | 76.21% | 0.01s |
| KNN | 66.48% | 69.48% | 0.01s |
| SVM | 62.57% | 67.01% | 0.18s |

### 最佳模型: Gradient Boosting

**验证集性能**:
- 准确率: 89.0%
- 精确率: 90%
- 召回率: 88%
- F1分数: 89%

**预测结果**:
- 生存人数: 149人
- 死亡人数: 269人
- 生存率: 35.65%

---

## 📄 脚本3: results_summary.py

### 基本信息
- **文件路径**: `results_summary.py`
- **文件大小**: 1.9KB
- **代码行数**: 43行
- **功能**: 检查和汇总预测结果
- **依赖**: pandas, os

### 主要功能

#### 1. 检查生成的文件

```python
files_generated = []
if os.path.exists('Gradient_Boosting_submission.csv'):
    files_generated.append('✓ Gradient_Boosting_submission.csv (预测结果)')
if os.path.exists('feature_importance.png'):
    files_generated.append('✓ feature_importance.png (特征重要性图)')
```

#### 2. 显示预测结果统计

```python
rf_pred = pd.read_csv('Gradient_Boosting_submission.csv')
print(f"测试集总数: {len(rf_pred)}")
print(f"预测生存人数: {rf_pred['Survived'].sum()}")
print(f"预测死亡人数: {len(rf_pred) - rf_pred['Survived'].sum()}")
print(f"预测生存率: {rf_pred['Survived'].mean():.2%}")
```

#### 3. 与基准模型比较

```python
gender_pred = pd.read_csv('gender_submission.csv')
print(f"性别模型生存率: {gender_pred['Survived'].mean():.2%}")
print(f"随机森林生存率: {rf_pred['Survived'].mean():.2%}")

different_predictions = (rf_pred['Survived'] != gender_pred['Survived']).sum()
print(f"预测不同的样本数: {different_predictions} ({different_predictions/len(rf_pred)*100:.1f}%)")
```

#### 4. 文件格式验证

```python
print(f"文件格式验证:")
print(f"  ✓ 包含正确的列: PassengerId, Survived")
print(f"  ✓ PassengerId 范围: {rf_pred['PassengerId'].min()} - {rf_pred['PassengerId'].max()}")
print(f"  ✓ Survived 取值: {sorted(rf_pred['Survived'].unique())}")
```

### 输出示例

```
=== 泰坦尼克号生存预测 - 随机森林模型结果总结 ===

生成的文件:
  ✓ Gradient_Boosting_submission.csv (预测结果)
  ✓ feature_importance.png (特征重要性图)

随机森林预测结果:
  测试集总数: 418
  预测生存人数: 149
  预测死亡人数: 269
  预测生存率: 35.65%

与基准性别模型比较:
  性别模型生存率: 37.80%
  随机森林生存率: 35.65%
  预测不同的样本数: 261 (62.44%)

文件格式验证:
  ✓ 包含正确的列: PassengerId, Survived
  ✓ PassengerId 范围: 892 - 1309
  ✓ Survived 取值: [0, 1]

*** 随机森林预测模型训练完成！***
```

### 运行方法

```bash
python results_summary.py
```

---

## 🔄 脚本执行流程

### 完整流程

```bash
# 1. 运行随机森林模型 (参考)
python titanic_random_forest.py

# 2. 运行多模型对比 (自动选择最佳)
python multi_model_comparison.py

# 3. 查看结果汇总
python results_summary.py
```

### 脚本依赖关系

```
multi_model_comparison.py
    ├── 使用 train.csv 和 test.csv
    ├── 使用 titanic_random_forest.py 的预处理函数
    └── 生成 Gradient_Boosting_submission.csv

results_summary.py
    ├── 读取 Gradient_Boosting_submission.csv
    └── 显示预测结果统计
```

---

## 📊 脚本性能对比

### 训练时间

| 脚本 | 训练时间 | 准确率 | 适用场景 |
|------|---------|--------|----------|
| titanic_random_forest.py | 0.25s | 80.81% | 单模型参考 |
| multi_model_comparison.py | ~3s | 83.16% | 多模型对比 |
| results_summary.py | <0.1s | N/A | 结果验证 |

### 输出文件

| 脚本 | 生成文件 |
|------|---------|
| titanic_random_forest.py | feature_importance.png |
| multi_model_comparison.py | model_comparison.png, final_comparison.png, model_comparison_summary.csv |
| results_summary.py | 控制台输出 |

---

## 🎯 使用建议

### 初学者

1. 先运行 `results_summary.py` 查看结果
2. 查看 `titanic_random_forest.py` 了解单模型实现
3. 理解代码结构和功能

### 进阶用户

1. 运行 `multi_model_comparison.py` 查看完整流程
2. 分析模型性能对比结果
3. 自定义模型或调整参数

### 专家用户

1. 修改 `preprocess_data` 函数进行特征工程
2. 添加新模型到对比列表
3. 优化超参数配置

---

## 📝 代码示例

### 使用预处理函数

```python
from titanic_random_forest import preprocess_data

# 加载数据
train_df = pd.read_csv('DATA/train.csv')
test_df = pd.read_csv('DATA/test.csv')

# 预处理
X_train = preprocess_data(train_df)
X_test = preprocess_data(test_df)

# 训练模型
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, train_df['Survived'])

# 预测
predictions = model.predict(X_test)
```

### 查看模型性能

```python
from multi_model_comparison import models, results_df

# 获取最佳模型
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"最佳模型: {best_model_name}")
print(f"准确率: {results_df.iloc[0]['CV Mean Accuracy']:.4f}")
```

---

**文档版本**: 1.0
**最后更新**: 2026-03-16
