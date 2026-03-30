# 数据文件详细说明

本文档详细说明项目中所有数据文件的内容、结构和用途。

---

## 📊 数据集总览

### 泰坦尼克号生存预测数据集

**数据来源**: Kaggle Titanic Competition
**数据规模**:
- 训练集: 891条记录
- 测试集: 418条记录
- 特征数量: 11个特征 (不含目标变量)
- 目标变量: Survived (二分类: 0=死亡, 1=生存)

---

## 📁 DATA/ 目录文件

### `train.csv` - 训练数据集

#### 基本信息
- **文件路径**: `DATA/train.csv`
- **文件大小**: 60KB
- **数据行数**: 891行
- **数据列数**: 12列
- **编码格式**: UTF-8

#### 列说明

| 列名 | 类型 | 描述 | 示例值 |
|------|------|------|--------|
| PassengerId | int | 乘客唯一ID | 1, 2, 3, ... |
| Survived | int | 生存状态 (0=死亡, 1=生存) | 0, 1 |
| Pclass | int | 客舱等级 (1=头等舱, 2=二等舱, 3=三等舱) | 1, 2, 3 |
| Name | string | 姓名 | Braund, Mr. Owen Harris |
| Sex | string | 性别 | male, female |
| Age | float | 年龄 | 22.0, 38.0, 26.0 |
| SibSp | int | 兄弟姐妹/配偶数量 | 0, 1, 2 |
| Parch | int | 父母/子女数量 | 0, 1, 2 |
| Ticket | string | 票号 | A/5 21171, PC 17599 |
| Fare | float | 票价 | 7.25, 71.2833 |
| Cabin | string | 客舱号 | C85, C123 |
| Embarked | string | 登船港口 (C=Cherbourg, Q=Queenstown, S=Southampton) | S, C, Q |

#### 数据分布

**生存状态分布**:
```
Survived = 0 (死亡): 549人 (61.6%)
Survived = 1 (生存): 342人 (38.4%)
```

**客舱等级分布**:
```
Pclass = 1: 216人 (24.2%)
Pclass = 2: 184人 (20.7%)
Pclass = 3: 491人 (55.1%)
```

**性别分布**:
```
Sex = male: 577人 (64.8%)
Sex = female: 314人 (35.2%)
```

#### 缺失值统计

| 列名 | 缺失值数量 | 缺失率 | 处理方式 |
|------|-----------|--------|----------|
| Age | 177 | 19.9% | 中位数填充 |
| Cabin | 687 | 77.1% | 删除或保留 |
| Embarked | 2 | 0.2% | 众数填充 |

#### 数据质量

- **完整行**: 891条
- **目标变量**: 100%完整
- **特征质量**: 良好，有少量缺失值

---

### `test.csv` - 测试数据集

#### 基本信息
- **文件路径**: `DATA/test.csv`
- **文件大小**: 28KB
- **数据行数**: 418行
- **数据列数**: 11列 (不含Survived)
- **编码格式**: UTF-8

#### 列说明

| 列名 | 类型 | 描述 |
|------|------|------|
| PassengerId | int | 乘客唯一ID |
| Pclass | int | 客舱等级 |
| Name | string | 姓名 |
| Sex | string | 性别 |
| Age | float | 年龄 |
| SibSp | int | 兄弟姐妹/配偶数量 |
| Parch | int | 父母/子女数量 |
| Ticket | string | 票号 |
| Fare | float | 票价 |
| Cabin | string | 客舱号 |
| Embarked | string | 登船港口 |

#### 缺失值统计

| 列名 | 缺失值数量 | 缺失率 | 处理方式 |
|------|-----------|--------|----------|
| Age | 86 | 20.6% | 中位数填充 |
| Fare | 1 | 0.2% | 中位数填充 |
| Cabin | 327 | 78.2% | 删除或保留 |
| Embarked | 0 | 0% | N/A |

#### 用途

- 用于训练模型后进行预测
- 生成提交文件
- 不包含目标变量Survived

---

### `gender_submission.csv` - 基准提交文件

#### 基本信息
- **文件路径**: `SUBMISSIONS/gender_submission.csv`
- **文件大小**: 3.2KB
- **数据行数**: 418行
- **数据列数**: 2列
- **编码格式**: UTF-8

#### 列说明

| 列名 | 类型 | 描述 | 值 |
|------|------|------|-----|
| PassengerId | int | 乘客唯一ID | 892, 893, ... |
| Survived | int | 预测生存状态 | 0 或 1 |

#### 预测规则

```python
if Sex == 'female':
    Survived = 1
else:
    Survived = 0
```

#### 性能指标

- **准确率**: ~75.8% (基于训练集生存率)
- **预测生存人数**: ~158人
- **预测死亡人数**: ~260人
- **预测生存率**: ~37.8%

#### 用途

- 作为基准模型对比
- 简单易实现的基线
- 评估复杂模型性能

---

### `Gradient_Boosting_submission.csv` - 最佳模型预测结果

#### 基本信息
- **文件路径**: `SUBMISSIONS/Gradient_Boosting_submission.csv`
- **文件大小**: 3.2KB
- **数据行数**: 418行
- **数据列数**: 2列
- **编码格式**: UTF-8

#### 列说明

| 列名 | 类型 | 描述 |
|------|------|------|
| PassengerId | int | 乘客唯一ID |
| Survived | int | 预测生存状态 (0=死亡, 1=生存) |

#### 预测统计

```
总预测数: 418
预测生存人数: 149
预测死亡人数: 269
预测生存率: 35.65%
```

#### 性能指标

**验证集性能**:
- 准确率: 89.0%
- 精确率: 90%
- 召回率: 88%
- F1分数: 89%

**交叉验证性能**:
- 平均准确率: 83.16%
- 标准差: 2.16%

#### 数据示例

```csv
PassengerId,Survived
892,0
893,0
894,0
895,0
896,1
...
```

#### 用途

- 用于Kaggle提交
- 最佳模型的最终预测结果
- 可直接用于评估

---

## 📊 数据特征工程

### 原始特征

1. **Pclass** - 客舱等级 (1, 2, 3)
2. **Sex** - 性别 (male, female)
3. **Age** - 年龄 (连续值)
4. **SibSp** - 兄弟姐妹/配偶数量
5. **Parch** - 父母/子女数量
6. **Fare** - 票价 (连续值)
7. **Embarked** - 登船港口 (C, Q, S)
8. **Name** - 姓名 (用于提取Title)
9. **Ticket** - 票号 (未使用)
10. **Cabin** - 客舱号 (部分缺失)

### 工程特征

1. **FamilySize** - 家庭规模 (SibSp + Parch + 1)
2. **IsAlone** - 是否独自一人 (1=是, 0=否)
3. **Title** - 称谓 (Mr, Mrs, Miss, Master, Rare等)
4. **AgeGroup** - 年龄段 (Child, Teen, Young Adult, Adult, Senior)
5. **FareGroup** - 票价段 (Low, Medium, High, Very High)

---

## 📈 数据分布分析

### 生存率分布

**按性别**:
- 女性: 74.2% 生存率
- 男性: 18.9% 生存率

**按客舱等级**:
- 头等舱: 63.0% 生存率
- 二等舱: 47.3% 生存率
- 三等舱: 24.2% 生存率

**按年龄段**:
- 儿童 (0-12): 54.4% 生存率
- 青少年 (13-17): 40.6% 生存率
- 年轻成人 (18-35): 38.2% 生存率
- 成年人 (36-60): 36.2% 生存率
- 老年人 (60+): 22.7% 生存率

### 关键发现

1. **性别**: 最重要特征，女性生存率远高于男性
2. **客舱等级**: 头等舱生存率最高
3. **年龄**: 儿童生存率最高
4. **家庭**: 与家人在一起生存率更高

---

## 🗂️ 数据使用流程

### 训练流程

```python
# 1. 加载数据
train_df = pd.read_csv('DATA/train.csv')
test_df = pd.read_csv('DATA/test.csv')

# 2. 数据预处理
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# 3. 特征工程
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
# ... 更多特征工程

# 4. 模型训练
model.fit(X_train, y_train)

# 5. 预测
predictions = model.predict(X_test)
```

### 提交流程

```python
# 1. 生成预测
predictions = model.predict(X_test)

# 2. 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

# 3. 保存文件
submission.to_csv('SUBMISSIONS/Gradient_Boosting_submission.csv', index=False)
```

---

## 📋 数据验证

### 格式验证

所有提交文件都符合以下格式:
- 必须包含PassengerId列
- 必须包含Survived列
- PassengerId连续且从892开始
- Survived只能为0或1
- 文件编码为UTF-8

### 数值验证

- **PassengerId范围**: 892-1309
- **Survived范围**: 0或1
- **预测总数**: 418

---

## 🔍 数据质量检查

### train.csv

✓ 完整行: 891条
✓ 目标变量完整: 100%
✓ 缺失值处理: 中位数/众数填充
✓ 编码格式: UTF-8

### test.csv

✓ 完整行: 418条
✓ 目标变量缺失: 正常 (用于预测)
✓ 缺失值处理: 中位数填充
✓ 编码格式: UTF-8

### 提交文件

✓ 格式正确: 符合Kaggle要求
✓ 数值范围正确: 0-1
✓ PassengerId正确: 连续且正确

---

## 📊 数据统计摘要

### train.csv

| 统计量 | 值 |
|--------|-----|
| 样本数 | 891 |
| 特征数 | 11 |
| 目标变量值 | 0, 1 |
| 生存人数 | 342 |
| 死亡人数 | 549 |
| 生存率 | 38.4% |

### test.csv

| 统计量 | 值 |
|--------|-----|
| 样本数 | 418 |
| 特征数 | 10 |
| 目标变量 | 无 (待预测) |

---

## 🎯 数据使用建议

### 初学者

1. 先了解原始数据结构
2. 查看缺失值分布
3. 理解特征含义
4. 运行简单模型

### 进阶用户

1. 进行数据探索分析
2. 创建新的特征
3. 优化预处理流程
4. 尝试不同的模型

### 专家用户

1. 深度特征工程
2. 高级模型集成
3. 超参数优化
4. 交叉验证调优

---

**文档版本**: 1.0
**最后更新**: 2026-03-16
