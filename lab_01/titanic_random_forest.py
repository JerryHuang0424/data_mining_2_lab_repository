import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保结果可重现
np.random.seed(42)

print("=== 泰坦尼克号生存预测 - 随机森林模型 ===\n")

# 1. 数据加载
print("1. 加载数据...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 2. 数据探索
print("\n2. 数据探索...")
print("\n训练集基本信息:")
print(train_df.info())

print("\n缺失值统计:")
print(train_df.isnull().sum())

print("\n生存率统计:")
print(train_df['Survived'].value_counts())
print(f"整体生存率: {train_df['Survived'].mean():.2%}")

# 3. 数据预处理函数
def preprocess_data(df):
    """数据预处理函数"""
    # 复制数据避免修改原始数据
    data = df.copy()

    # 处理年龄缺失值 - 用中位数填充
    data['Age'].fillna(data['Age'].median(), inplace=True)

    # 处理票价缺失值 - 用中位数填充
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # 处理登船港口缺失值 - 用众数填充
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # 创建新特征
    # 家庭大小
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # 是否独自一人
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # 从姓名中提取称谓
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    # 年龄段分组
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100],
                             labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

    # 票价段分组
    data['FareGroup'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

    # 编码分类变量
    le = LabelEncoder()

    # 性别编码
    data['Sex'] = le.fit_transform(data['Sex'])

    # 登船港口编码
    data['Embarked'] = le.fit_transform(data['Embarked'])

    # 称谓编码
    data['Title'] = le.fit_transform(data['Title'])

    # 年龄段编码
    data['AgeGroup'] = le.fit_transform(data['AgeGroup'])

    # 票价段编码
    data['FareGroup'] = le.fit_transform(data['FareGroup'])

    # 选择特征
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
               'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']

    return data[features]

print("\n3. 数据预处理...")
# 预处理训练数据
X_train_processed = preprocess_data(train_df)
y_train = train_df['Survived']

# 预处理测试数据
X_test_processed = preprocess_data(test_df)

print("预处理后的特征:")
print(X_train_processed.columns.tolist())
print(f"训练特征形状: {X_train_processed.shape}")
print(f"测试特征形状: {X_test_processed.shape}")

# 检查预处理后的缺失值
print(f"\n训练集缺失值: {X_train_processed.isnull().sum().sum()}")
print(f"测试集缺失值: {X_test_processed.isnull().sum().sum()}")

print("\n数据预处理完成！")

# 4. 模型训练
print("\n4. 随机森林模型训练...")

# 分割训练集用于验证
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 初始随机森林模型
rf_initial = RandomForestClassifier(n_estimators=100, random_state=42)
rf_initial.fit(X_train_split, y_train_split)

# 初始模型验证
val_pred = rf_initial.predict(X_val_split)
print(f"初始模型验证准确率: {accuracy_score(y_val_split, val_pred):.4f}")

# 交叉验证评估
cv_scores = cross_val_score(rf_initial, X_train_processed, y_train, cv=5)
print(f"5折交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 超参数调优
print("\n进行超参数调优...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 网格搜索
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_processed, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数的模型
best_rf = grid_search.best_estimator_

# 5. 特征重要性分析
print("\n5. 特征重要性分析...")
feature_importance = pd.DataFrame({
    'feature': X_train_processed.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("特征重要性排序:")
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# 6. 模型评估
print("\n6. 模型评估...")
# 在验证集上的表现
val_pred_best = best_rf.predict(X_val_split)
print(f"最佳模型验证准确率: {accuracy_score(y_val_split, val_pred_best):.4f}")

print("\n验证集分类报告:")
print(classification_report(y_val_split, val_pred_best))

# 7. 生成预测结果
print("\n7. 生成测试集预测...")
test_predictions = best_rf.predict(X_test_processed)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

# 保存结果
submission.to_csv('random_forest_submission.csv', index=False)
print(f"预测结果已保存到: random_forest_submission.csv")

# 显示预测结果统计
print(f"\n预测结果统计:")
print(f"预测生存人数: {test_predictions.sum()}")
print(f"预测死亡人数: {len(test_predictions) - test_predictions.sum()}")
print(f"预测生存率: {test_predictions.mean():.2%}")

# 8. 可视化结果（可选）
try:
    # 特征重要性图
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("特征重要性图已保存为: feature_importance.png")
except Exception as e:
    print(f"无法生成可视化图表: {e}")

print("\n=== 随机森林预测完成！ ===")