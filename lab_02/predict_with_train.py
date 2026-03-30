import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("="*60)
print("加载原始数据...")
print("="*60)
train_df = pd.read_csv('train.csv')
print(f"\n训练集大小: {train_df.shape}")
print(f"训练集列名: {list(train_df.columns)}")
print(f"\n训练集前5行:")
print(train_df.head())

# 特征工程函数
def preprocess_data(df):
    df = df.copy()

    # 删除不需要的列
    drop_cols = ['Id', 'Address', 'Summary', 'Listed On', 'Last Sold On', 'City', 'Zip', 'State']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 数值特征
    numeric_features = ['Year built', 'Lot', 'Bedrooms', 'Bathrooms', 'Full bathrooms',
                       'Total interior livable area', 'Total spaces', 'Garage spaces',
                       'Elementary School Score', 'Elementary School Distance',
                       'Middle School Score', 'Middle School Distance',
                       'High School Score', 'High School Distance',
                       'Tax assessed value', 'Annual tax amount', 'Listed Price', 'Last Sold Price']

    # 处理数值特征
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 填充缺失值为中位数
            df[col] = df[col].fillna(df[col].median())

    # 文本特征处理
    categorical_features = ['Type', 'Heating', 'Cooling', 'Parking', 'Region',
                           'Elementary School', 'Middle School', 'High School',
                           'Flooring', 'Heating features', 'Cooling features',
                           'Appliances included', 'Laundry features', 'Parking features']

    for col in categorical_features:
        if col in df.columns:
            # 将文本转换为类别编码
            df[col] = df[col].fillna('Unknown')
            # 简单的标签编码
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 创建新特征
    if 'Listed Price' in df.columns and 'Last Sold Price' in df.columns:
        df['Price_Diff'] = df['Listed Price'] - df['Last Sold Price']
        df['Price_Ratio'] = df['Listed Price'] / (df['Last Sold Price'] + 1)
    else:
        df['Price_Diff'] = 0
        df['Price_Ratio'] = 0

    if 'Bedrooms' in df.columns and 'Total interior livable area' in df.columns:
        df['Area_per_Bedroom'] = df['Total interior livable area'] / (df['Bedrooms'] + 1)
    else:
        df['Area_per_Bedroom'] = 0

    if 'Bedrooms' in df.columns and 'Bathrooms' in df.columns:
        df['Bed_to_Bath_Ratio'] = df['Bathrooms'] / (df['Bedrooms'] + 1)
    else:
        df['Bed_to_Bath_Ratio'] = 0

    return df

# 预处理训练集
print("\n" + "="*60)
print("预处理数据...")
print("="*60)
train_processed = preprocess_data(train_df)

print(f"\n预处理后的训练集大小: {train_processed.shape}")
print(f"预处理后的特征数量: {train_processed.shape[1] - 1}")  # 减去目标变量
print(f"\n预处理后的特征列表:")
print(train_processed.columns.tolist())

# 分离特征和目标
X = train_processed.drop('Sold Price', axis=1)
y = train_processed['Sold Price']

print(f"\n特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")

# 训练集和验证集分割
print("\n" + "="*60)
print("分割训练集和验证集...")
print("="*60)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"训练集目标均值: {y_train.mean():.2f}")
print(f"验证集目标均值: {y_val.mean():.2f}")

# 训练多个模型
print("\n" + "="*60)
print("训练模型...")
print("="*60)

models = {
    'Extra Trees': ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1, max_depth=20),
    'Random Forest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, max_depth=20),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, random_state=42, max_depth=5),
}

results = {}

for name, model in models.items():
    print(f"\n训练 {name}...")
    model.fit(X_train, y_train)

    # 预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    results[name] = {
        'model': model,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }

    print(f"  训练 RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R2: {train_r2:.4f}")
    print(f"  验证 RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R2: {val_r2:.4f}")

# 选择最佳模型
print("\n" + "="*60)
print("模型比较")
print("="*60)

best_model_name = max(results.keys(), key=lambda x: results[x]['val_r2'])
best_model = results[best_model_name]['model']

print(f"\n最佳模型: {best_model_name}")
print(f"验证 R2: {results[best_model_name]['val_r2']:.4f} ({results[best_model_name]['val_r2']*100:.2f}%)")
print(f"验证 RMSE: {results[best_model_name]['val_rmse']:.2f}")
print(f"验证 MAE: {results[best_model_name]['val_mae']:.2f}")

# 在整个训练集上重新训练最佳模型
print("\n" + "="*60)
print("在完整训练集上重新训练最佳模型...")
print("="*60)
best_model.fit(X, y)
print(f"完整训练集 R2: {r2_score(y, best_model.predict(X)):.4f}")

# 预测训练集（自我预测）
print("\n" + "="*60)
print("使用训练集预测房价（自我预测）...")
print("="*60)
train_predictions = best_model.predict(X)

# 创建结果DataFrame
result_df = pd.DataFrame({
    'Id': train_df['Id'],
    'Actual Price': y,
    'Predicted Price': train_predictions,
    'Price Difference': train_predictions - y,
    'Absolute Error': np.abs(train_predictions - y),
    'Relative Error (%)': np.abs((train_predictions - y) / y) * 100
})

print(f"\n预测结果统计:")
print(f"  平均误差: {result_df['Absolute Error'].mean():.2f}")
print(f"  最大误差: {result_df['Absolute Error'].max():.2f}")
print(f"  最小误差: {result_df['Absolute Error'].min():.2f}")
print(f"  平均相对误差: {result_df['Relative Error (%)'].mean():.2f}%")

print(f"\n前10条预测结果:")
print(result_df[['Id', 'Actual Price', 'Predicted Price', 'Price Difference', 'Relative Error (%)']].head(10))

# 保存预测结果
result_df.to_csv('train_predictions.csv', index=False)
print(f"\n预测结果已保存为 train_predictions.csv")

# 特征重要性分析
print("\n" + "="*60)
print("特征重要性分析")
print("="*60)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 重要特征:")
    print(feature_importance.head(10))

    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance.head(10)['Feature'], feature_importance.head(10)['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top 10 Feature Importance ({best_model_name})')
    plt.tight_layout()
    plt.savefig('feature_importance_best_model.png', dpi=300, bbox_inches='tight')
    print("\n特征重要性图已保存为 feature_importance_best_model.png")

# 保存模型
import joblib
joblib.dump(best_model, 'best_model.pkl')
print(f"\n模型已保存为 best_model.pkl")

# 保存预处理信息
preprocessing_info = {
    'features': X.columns.tolist(),
    'model_name': best_model_name,
    'train_r2': float(r2_score(y, best_model.predict(X))),
    'val_r2': float(results[best_model_name]['val_r2']),
    'val_rmse': float(results[best_model_name]['val_rmse'])
}
pd.DataFrame([preprocessing_info]).to_csv('model_info.csv', index=False)
print("模型信息已保存为 model_info.csv")

print("\n" + "="*60)
print("处理完成！")
print("="*60)
print(f"\n主要结果:")
print(f"  最佳模型: {best_model_name}")
print(f"  训练集 R2: {results[best_model_name]['train_r2']:.4f}")
print(f"  验证集 R2: {results[best_model_name]['val_r2']:.4f}")
print(f"  验证 RMSE: {results[best_model_name]['val_rmse']:.2f}")
print(f"  平均相对误差: {result_df['Relative Error (%)'].mean():.2f}%")
