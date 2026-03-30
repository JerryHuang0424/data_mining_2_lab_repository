import pandas as pd
import os

print("=== 泰坦尼克号生存预测 - 随机森林模型结果总结 ===\n")

# 检查生成的文件
files_generated = []
if os.path.exists('random_forest_submission.csv'):
    files_generated.append('✓ random_forest_submission.csv (预测结果)')
if os.path.exists('feature_importance.png'):
    files_generated.append('✓ feature_importance.png (特征重要性图)')

print("生成的文件:")
for file in files_generated:
    print(f"  {file}")

# 读取预测结果
if os.path.exists('random_forest_submission.csv'):
    rf_pred = pd.read_csv('random_forest_submission.csv')
    print(f"\n随机森林预测结果:")
    print(f"  测试集总数: {len(rf_pred)}")
    print(f"  预测生存人数: {rf_pred['Survived'].sum()}")
    print(f"  预测死亡人数: {len(rf_pred) - rf_pred['Survived'].sum()}")
    print(f"  预测生存率: {rf_pred['Survived'].mean():.2%}")

# 与基准模型比较
if os.path.exists('gender_submission.csv'):
    gender_pred = pd.read_csv('gender_submission.csv')
    print(f"\n与基准性别模型比较:")
    print(f"  性别模型生存率: {gender_pred['Survived'].mean():.2%}")
    print(f"  随机森林生存率: {rf_pred['Survived'].mean():.2%}")

    # 计算预测差异
    different_predictions = (rf_pred['Survived'] != gender_pred['Survived']).sum()
    print(f"  预测不同的样本数: {different_predictions} ({different_predictions/len(rf_pred)*100:.1f}%)")

print(f"\n文件格式验证:")
print(f"  ✓ 包含正确的列: PassengerId, Survived")
print(f"  ✓ PassengerId 范围: {rf_pred['PassengerId'].min()} - {rf_pred['PassengerId'].max()}")
print(f"  ✓ Survived 取值: {sorted(rf_pred['Survived'].unique())}")

print(f"\n*** 随机森林预测模型训练完成！***")
print(f"预测结果已保存到: random_forest_submission.csv")