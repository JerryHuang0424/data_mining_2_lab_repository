# 泰坦尼克号生存预测 - 项目文档

## 📋 项目概述

本项目使用多种机器学习算法对泰坦尼克号乘客生存进行预测，并通过对比分析确定最佳模型。

**项目目标**:
- 比较多种机器学习模型的性能
- 识别表现最佳的模型
- 生成最终的预测结果

**数据集**: Titanic Dataset
**完成时间**: 2026-03-16

---

## 🗂️ 文件结构

```
lab_01/
├── 📁 DATA/                    # 原始数据文件
│   ├── train.csv              # 训练数据
│   └── test.csv               # 测试数据
│
├── 📁 SUBMISSIONS/            # 预测结果
│   ├── gender_submission.csv  # 基准模型（性别规则）
│   └── Gradient_Boosting_submission.csv  # 最佳模型
│
├── 📁 VISUALIZATIONS/         # 可视化图表
│   ├── feature_importance.png     # 特征重要性
│   ├── model_comparison.png       # 模型对比
│   └── final_comparison.png       # 最终对比
│
├── 📁 SCRIPTS/                # Python脚本
│   ├── titanic_random_forest.py   # 随机森林模型
│   ├── multi_model_comparison.py  # 多模型对比
│   └── results_summary.py         # 结果汇总
│
├── 📁 DOCUMENTATION/          # 文档
│   ├── README.md              # 项目总览
│   ├── FILE_DOCUMENTATION.md   # 文件详细说明
│   ├── DATA_DOCUMENTATION.md   # 数据说明
│   ├── SCRIPT_DOCUMENTATION.md # 脚本说明
│   └── MODEL_COMPARISON_REPORT.md  # 模型对比报告
│
├── 📊 RESULTS/                # 结果汇总
│   └── model_comparison_summary.csv  # 性能数据
│
├── train.csv                  # 训练数据（根目录）
├── test.csv                   # 测试数据（根目录）
├── MODEL_COMPARISON_REPORT.md # 模型对比报告
└── model_comparison_summary.csv  # 性能数据
```

---

## 🚀 快速开始

### 运行最佳模型

```bash
# 运行多模型对比（自动选择最佳模型）
python multi_model_comparison.py
```

### 运行随机森林模型（参考）

```bash
python titanic_random_forest.py
```

### 查看结果汇总

```bash
python results_summary.py
```

---

## 📊 主要成果

### 最佳模型：Gradient Boosting

| 指标 | 数值 |
|------|------|
| **交叉验证准确率** | 83.16% |
| **验证准确率** | 89.0% |
| **预测生存人数** | 149人 |
| **预测死亡人数** | 269人 |
| **预测生存率** | 35.65% |

### 模型性能排名

1. Gradient Boosting (83.16%)
2. AdaBoost (81.82%)
3. Random Forest (80.81%)
4. Logistic Regression (79.69%)
5. Naive Bayes (79.47%)

---

## 📈 可视化结果

### 模型对比图表
- `model_comparison.png` - 完整的8模型性能对比
- `final_comparison.png` - 精简版性能对比
- `feature_importance.png` - 随机森林特征重要性

### 数据分布图表
- 各模型的准确率、训练时间、性能权衡等可视化

---

## 📝 使用说明

### 数据文件

- `train.csv`: 包含891条训练记录
- `test.csv`: 包含418条测试记录
- `gender_submission.csv`: 基准提交文件（所有女性预测为生存）

### 预测结果

- `Gradient_Boosting_submission.csv`: 最佳模型的预测结果
- 格式: PassengerId, Survived
- 可直接用于Kaggle提交

---

## 🔧 技术栈

- **Python 3.x**
- **Scikit-learn** - 机器学习库
- **Pandas** - 数据处理
- **NumPy** - 数值计算
- **Matplotlib/Seaborn** - 可视化

---

## 📚 相关文档

详细文档请查看:
- `FILE_DOCUMENTATION.md` - 各文件详细说明
- `DATA_DOCUMENTATION.md` - 数据集说明
- `SCRIPT_DOCUMENTATION.md` - 脚本功能说明
- `MODEL_COMPARISON_REPORT.md` - 完整分析报告

---

## 🎯 下一步建议

1. **超参数调优**: 优化Gradient Boosting的超参数
2. **特征工程**: 尝试更多特征组合
3. **集成学习**: 结合多个模型的预测结果
4. **交叉验证**: 使用更严格的验证策略

---

## 📞 联系方式

如有问题，请查看详细文档或运行相关脚本。

---

**最后更新**: 2026-03-16
**版本**: 1.0
