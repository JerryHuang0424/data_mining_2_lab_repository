# 文件详细说明

本文档详细说明项目中每个文件的功能、内容和用途。

---

## 📁 项目文件总览

### 📊 数据文件 (DATA/)

#### `train.csv`
- **路径**: `DATA/train.csv`
- **大小**: 60KB
- **行数**: 891行
- **列数**: 12列
- **用途**: 训练数据集
- **内容**:
  - PassengerId: 乘客ID
  - Survived: 生存状态 (0=死亡, 1=生存)
  - Pclass: 客舱等级 (1=头等舱, 2=二等舱, 3=三等舱)
  - Name: 姓名
  - Sex: 性别
  - Age: 年龄
  - SibSp: 兄弟姐妹/配偶数量
  - Parch: 父母/子女数量
  - Ticket: 票号
  - Fare: 票价
  - Cabin: 客舱号
  - Embarked: 登船港口 (C=Cherbourg, Q=Queenstown, S=Southampton)

**关键信息**:
- 训练集样本: 891条
- 目标变量: Survived (二分类)
- 特征数量: 11个特征 (不含目标变量)

---

#### `test.csv`
- **路径**: `DATA/test.csv`
- **大小**: 28KB
- **行数**: 418行
- **列数**: 11列 (不含Survived)
- **用途**: 测试数据集
- **内容**: 与train.csv相同，但缺少Survived列

**关键信息**:
- 测试集样本: 418条
- 用于生成预测结果

---

#### `gender_submission.csv`
- **路径**: `SUBMISSIONS/gender_submission.csv`
- **大小**: 3.2KB
- **行数**: 418行
- **列数**: 2列
- **用途**: 基准提交文件
- **内容**:
  - PassengerId: 乘客ID
  - Survived: 预测结果 (所有女性=1, 所有男性=0)

**说明**: 这是一个简单的基准模型，仅基于性别预测生存。

---

#### `Gradient_Boosting_submission.csv`
- **路径**: `SUBMISSIONS/Gradient_Boosting_submission.csv`
- **大小**: 3.2KB
- **行数**: 418行
- **列数**: 2列
- **用途**: 最佳模型预测结果
- **内容**:
  - PassengerId: 乘客ID
  - Survived: 预测生存状态 (0或1)

**预测统计**:
- 预测生存人数: 149人
- 预测死亡人数: 269人
- 预测生存率: 35.65%

**性能指标**:
- 交叉验证准确率: 83.16%
- 验证准确率: 89.0%

---

### 🖼️ 可视化文件 (VISUALIZATIONS/)

#### `feature_importance.png`
- **路径**: `VISUALIZATIONS/feature_importance.png`
- **大小**: 92KB
- **类型**: PNG图片
- **用途**: 展示随机森林模型的特征重要性
- **内容**: 横轴为重要性分数，纵轴为特征名称

**特征重要性排序**:
1. Sex (性别) - 最重要
2. Pclass (客舱等级)
3. Fare (票价)
4. Age (年龄)
5. Title (称谓)
6. FamilySize (家庭规模)
7. Embarked (登船港口)
8. IsAlone (是否独自一人)
9. SibSp (兄弟姐妹/配偶数量)
10. Parch (父母/子女数量)
11. AgeGroup (年龄段)
12. FareGroup (票价段)

---

#### `model_comparison.png`
- **路径**: `VISUALIZATIONS/model_comparison.png`
- **大小**: 466KB
- **类型**: PNG图片
- **用途**: 完整的8模型性能对比可视化
- **包含图表**:
  1. 交叉验证准确率对比
  2. 验证准确率对比
  3. 训练时间对比
  4. 准确率 vs 训练时间散点图

**对比模型**:
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- Naive Bayes
- AdaBoost
- Gradient Boosting

---

#### `final_comparison.png`
- **路径**: `VISUALIZATIONS/final_comparison.png`
- **大小**: 240KB
- **类型**: PNG图片
- **用途**: 精简版模型性能对比
- **内容**: 仅展示交叉验证和验证准确率的对比

---

### 🐍 Python脚本 (SCRIPTS/)

#### `titanic_random_forest.py`
- **路径**: `SCRIPTS/titanic_random_forest.py`
- **大小**: 7.0KB
- **行数**: 216行
- **用途**: 随机森林模型训练和预测
- **主要功能**:
  1. 数据加载和探索
  2. 数据预处理（缺失值填充、特征工程）
  3. 随机森林模型训练
  4. 超参数调优（GridSearchCV）
  5. 特征重要性分析
  6. 模型评估
  7. 生成预测结果
  8. 可视化结果

**关键特性**:
- 使用5折交叉验证
- 网格搜索优化超参数
- 生成feature_importance.png

**输出文件**:
- `random_forest_submission.csv` (已删除，被Gradient Boosting替代)

---

#### `multi_model_comparison.py`
- **路径**: `SCRIPTS/multi_model_comparison.py`
- **大小**: 9.9KB
- **行数**: ~400行
- **用途**: 多模型性能对比分析
- **主要功能**:
  1. 数据加载和预处理
  2. 8种不同模型的训练
  3. 模型性能评估（验证集和交叉验证）
  4. 性能对比分析
  5. 生成可视化图表
  6. 自动选择最佳模型
  7. 使用最佳模型生成最终预测

**对比的模型**:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. KNN (K-Nearest Neighbors)
5. SVM (Support Vector Machine)
6. Naive Bayes (GaussianNB)
7. Gradient Boosting
8. AdaBoost

**输出文件**:
- `Gradient_Boosting_submission.csv` - 最佳模型预测
- `model_comparison.png` - 性能对比图
- `model_comparison_summary.csv` - 数据汇总
- `final_comparison.png` - 精简对比图

**性能指标**:
- 验证准确率
- 交叉验证准确率
- 交叉验证标准差
- 训练时间

---

#### `results_summary.py`
- **路径**: `SCRIPTS/results_summary.py`
- **大小**: 1.9KB
- **行数**: 43行
- **用途**: 结果汇总和验证
- **主要功能**:
  1. 检查生成的文件
  2. 显示预测结果统计
  3. 与基准模型比较
  4. 验证文件格式

**检查内容**:
- 提交文件是否存在
- 预测结果统计
- 与gender_submission.csv的对比
- 文件格式验证

---

### 📊 结果文件 (RESULTS/)

#### `model_comparison_summary.csv`
- **路径**: `RESULTS/model_comparison_summary.csv`
- **大小**: 807B
- **行数**: 9行 (含表头)
- **列数**: 5列
- **用途**: 模型性能数据汇总
- **内容**:
  - Model: 模型名称
  - Validation Accuracy: 验证准确率
  - CV Mean Accuracy: 交叉验证平均准确率
  - CV Std Accuracy: 交叉验证标准差
  - Training Time (s): 训练时间(秒)

**数据示例**:
```csv
Model,Validation Accuracy,CV Mean Accuracy,CV Std Accuracy,Training Time (s)
Gradient Boosting,0.8044692737430168,0.8316489862532169,0.021616072355367656,0.21901631355285645
AdaBoost,0.7877094972067039,0.8182097796748478,0.017646621912252962,0.20199346542358398
Random Forest,0.8268156424581006,0.8080911430544221,0.03353027527314701,0.24598240852355957
...
```

---

### 📄 文档文件 (DOCUMENTATION/)

#### `README.md`
- **路径**: `README.md`
- **大小**: 5.7KB
- **用途**: 项目总览和快速开始指南
- **内容**:
  - 项目概述
  - 文件结构
  - 快速开始
  - 主要成果
  - 技术栈
  - 相关文档链接

---

#### `MODEL_COMPARISON_REPORT.md`
- **路径**: `MODEL_COMPARISON_REPORT.md`
- **大小**: 5.7KB
- **用途**: 完整的模型对比分析报告
- **内容**:
  - 执行摘要
  - 数据概览
  - 模型性能对比
  - 最佳模型详细分析
  - 模型对比可视化
  - 与其他模型比较
  - 最终建议
  - 结论

**主要章节**:
1. 数据概览
2. 模型性能对比
3. 最佳模型详细分析
4. 模型对比可视化
5. 与其他模型比较
6. 最终建议
7. 结论

---

## 🗑️ 已删除文件

#### `random_forest_submission.csv`
- **删除原因**: 被Gradient Boosting模型替代
- **替代文件**: `Gradient_Boosting_submission.csv`
- **理由**: Gradient Boosting性能更好（83.16% vs 80.81%）

---

## 📋 文件使用建议

### 新手用户
1. 查看 `README.md` 了解项目概览
2. 查看 `MODEL_COMPARISON_REPORT.md` 了解详细分析
3. 运行 `multi_model_comparison.py` 查看完整流程

### 进阶用户
1. 查看 `titanic_random_forest.py` 了解单模型实现
2. 查看 `multi_model_comparison.py` 了解多模型对比
3. 自定义模型或调整参数

### 提交预测结果
1. 使用 `Gradient_Boosting_submission.csv`
2. 格式符合Kaggle要求
3. 可直接提交

---

**文档版本**: 1.0
**最后更新**: 2026-03-16
