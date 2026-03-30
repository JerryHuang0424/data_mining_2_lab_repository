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

# 设置随机种子以确保结果可重现
np.random.seed(42)

print("=" * 80)
print("泰坦尼克号生存预测 - 多模型性能对比分析")
print("=" * 80)

# 1. 数据加载
print("\n1. 加载数据...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 2. 数据预处理函数
def preprocess_data(df):
    """数据预处理函数"""
    data = df.copy()

    # 处理缺失值
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # 创建新特征
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
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
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Embarked'] = le.fit_transform(data['Embarked'])
    data['Title'] = le.fit_transform(data['Title'])
    data['AgeGroup'] = le.fit_transform(data['AgeGroup'])
    data['FareGroup'] = le.fit_transform(data['FareGroup'])

    # 选择特征
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
               'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']

    return data[features]

print("\n2. 数据预处理...")
X_train_processed = preprocess_data(train_df)
y_train = train_df['Survived']
X_test_processed = preprocess_data(test_df)

# 分割训练集用于验证
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"训练集: {X_train_split.shape}")
print(f"验证集: {X_val_split.shape}")
print(f"测试集: {X_test_processed.shape}")

# 3. 模型定义和性能比较
print("\n3. 模型定义和性能比较...")

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

results = []

for name, model in models.items():
    print(f"\n训练 {name} 模型...")

    # 训练模型
    start_time = time.time()
    model.fit(X_train_split, y_train_split)
    training_time = time.time() - start_time

    # 验证集预测
    val_pred = model.predict(X_val_split)
    val_accuracy = accuracy_score(y_val_split, val_pred)

    # 交叉验证
    cv_scores = cross_val_score(model, X_train_processed, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # 保存结果
    results.append({
        'Model': name,
        'Validation Accuracy': val_accuracy,
        'CV Mean Accuracy': cv_mean,
        'CV Std Accuracy': cv_std,
        'Training Time (s)': training_time,
        'Predictions': val_pred
    })

    print(f"  验证准确率: {val_accuracy:.4f}")
    print(f"  5折交叉验证: {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"  训练时间: {training_time:.2f}秒")

# 4. 结果总结
print("\n4. 模型性能对比总结...")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('CV Mean Accuracy', ascending=False)

print("\n模型性能排名 (按交叉验证准确率):")
print("=" * 80)
print(results_df[['Model', 'Validation Accuracy', 'CV Mean Accuracy', 'CV Std Accuracy', 'Training Time (s)']].to_string(index=False))
print("=" * 80)

# 5. 可视化对比
print("\n5. 生成可视化对比图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 交叉验证准确率对比
ax1 = axes[0, 0]
bars = ax1.barh(results_df['Model'], results_df['CV Mean Accuracy'])
ax1.set_xlabel('Mean Cross-Validation Accuracy')
ax1.set_title('Model Performance Comparison (Cross-Validation)')
ax1.set_xlim(0.6, 1.0)
for i, (bar, val) in enumerate(zip(bars, results_df['CV Mean Accuracy'])):
    ax1.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)

# 验证准确率对比
ax2 = axes[0, 1]
bars = ax2.barh(results_df['Model'], results_df['Validation Accuracy'])
ax2.set_xlabel('Validation Accuracy')
ax2.set_title('Model Performance Comparison (Validation Set)')
ax2.set_xlim(0.6, 1.0)
for i, (bar, val) in enumerate(zip(bars, results_df['Validation Accuracy'])):
    ax2.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)

# 训练时间对比
ax3 = axes[1, 0]
bars = ax3.barh(results_df['Model'], results_df['Training Time (s)'])
ax3.set_xlabel('Training Time (seconds)')
ax3.set_title('Training Time Comparison')
for i, (bar, val) in enumerate(zip(bars, results_df['Training Time (s)'])):
    ax3.text(val + 0.01, i, f'{val:.2f}s', va='center', fontsize=9)

# 准确率 vs 训练时间散点图
ax4 = axes[1, 1]
scatter = ax4.scatter(results_df['Training Time (s)'], results_df['CV Mean Accuracy'],
                     s=100, alpha=0.6, c=range(len(results_df)), cmap='viridis')
for i, row in results_df.iterrows():
    ax4.annotate(row['Model'], (row['Training Time (s)'], row['CV Mean Accuracy']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax4.set_xlabel('Training Time (seconds)')
ax4.set_ylabel('CV Mean Accuracy')
ax4.set_title('Accuracy vs Training Time')
plt.colorbar(scatter, ax=ax4, label='Model Index')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("模型对比图已保存为: model_comparison.png")

# 6. 选择最佳模型
print("\n6. 选择最佳模型...")

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n最佳模型: {best_model_name}")
print(f"交叉验证准确率: {results_df.iloc[0]['CV Mean Accuracy']:.4f}")
print(f"验证准确率: {results_df.iloc[0]['Validation Accuracy']:.4f}")

# 7. 使用最佳模型进行预测
print("\n7. 使用最佳模型生成测试集预测...")

# 在整个训练集上训练最佳模型
print(f"在完整训练集上训练 {best_model_name}...")
best_model.fit(X_train_processed, y_train)

# 生成预测
test_predictions = best_model.predict(X_test_processed)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

# 保存结果
submission_file = f'{best_model_name.replace(" ", "_")}_submission.csv'
submission.to_csv(submission_file, index=False)

print(f"\n最佳模型预测结果已保存到: {submission_file}")

# 显示预测结果统计
print(f"\n预测结果统计:")
print(f"预测生存人数: {test_predictions.sum()}")
print(f"预测死亡人数: {len(test_predictions) - test_predictions.sum()}")
print(f"预测生存率: {test_predictions.mean():.2%}")

# 8. 详细评估最佳模型
print("\n8. 最佳模型详细评估...")

# 在验证集上的详细评估
val_pred_best = best_model.predict(X_val_split)
print(f"\n验证集分类报告:")
print(classification_report(y_val_split, val_pred_best))

print(f"\n混淆矩阵:")
cm = confusion_matrix(y_val_split, val_pred_best)
cm_df = pd.DataFrame(cm,
                    index=['Actually Died', 'Actually Survived'],
                    columns=['Predicted Died', 'Predicted Survived'])
print(cm_df)

# 9. 保存结果对比
print("\n9. 保存结果对比...")

comparison_df = pd.DataFrame({
    'Model': results_df['Model'],
    'Validation Accuracy': results_df['Validation Accuracy'],
    'CV Mean Accuracy': results_df['CV Mean Accuracy'],
    'CV Std Accuracy': results_df['CV Std Accuracy'],
    'Training Time (s)': results_df['Training Time (s)']
})

comparison_df.to_csv('model_comparison_summary.csv', index=False)
print("模型对比总结已保存为: model_comparison_summary.csv")

# 10. 最终结论
print("\n" + "=" * 80)
print("最终结论")
print("=" * 80)
print(f"最佳模型: {best_model_name}")
print(f"交叉验证准确率: {results_df.iloc[0]['CV Mean Accuracy']:.4f}")
print(f"验证准确率: {results_df.iloc[0]['Validation Accuracy']:.4f}")
print(f"预测结果已保存到: {submission_file}")
print("=" * 80)

print("\n*** 多模型性能对比分析完成！***")
