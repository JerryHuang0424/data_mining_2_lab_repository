import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("Multiple Model Predictions - Separate Submission Files")
print("="*70)

# 读取数据
print("\nLoading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Training set size: {train_df.shape}")
print(f"Test set size: {test_df.shape}")

# 特征工程函数
def preprocess_data(df, is_training=True):
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
            df[col] = df[col].fillna(df[col].median())

    # 文本特征处理
    categorical_features = ['Type', 'Heating', 'Cooling', 'Parking', 'Region',
                           'Elementary School', 'Middle School', 'High School',
                           'Flooring', 'Heating features', 'Cooling features',
                           'Appliances included', 'Laundry features', 'Parking features']

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
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

# 预处理训练集和测试集
print("\nPreprocessing data...")
train_processed = preprocess_data(train_df, is_training=True)
test_processed = preprocess_data(test_df, is_training=False)

print(f"Processed training set size: {train_processed.shape}")
print(f"Processed test set size: {test_processed.shape}")

# 分离特征和目标
X = train_processed.drop('Sold Price', axis=1)
y = train_processed['Sold Price']

# 确保测试集有相同的列
test_features = test_processed.drop('Sold Price', axis=1, errors='ignore')

print(f"Feature count: {X.shape[1]}")
print(f"Sample count: {X.shape[0]}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_features)

# 训练集和验证集分割
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")

# 定义5个最佳模型
models = {
    'XGBoost': XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, objective='reg:squarederror', tree_method='hist'),
    'LightGBM': LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1, force_col_wise=True),
    'Extra Trees': ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1, max_depth=20),
    'Random Forest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, max_depth=20),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, random_state=42, max_depth=5, learning_rate=0.05)
}

# 训练模型并评估
results = {}

print("\n" + "="*70)
print("Training Models and Generating Separate Submission Files")
print("="*70)

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training {name}...")
    print(f"{'='*70}")

    try:
        # 训练模型
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

        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = cv_scores.std()

        results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'cv_rmse': cv_rmse,
            'cv_std': cv_std
        }

        print(f"\nModel Performance:")
        print(f"  Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R2: {train_r2:.4f}")
        print(f"  Val RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R2: {val_r2:.4f}")
        print(f"  CV RMSE: {cv_rmse:.2f} (+/- {cv_std:.2f})")

        # 在整个训练集上重新训练
        print(f"\nRetraining on full training set...")
        model.fit(X_scaled, y)
        full_train_r2 = r2_score(y, model.predict(X_scaled))
        print(f"  Full training R2: {full_train_r2:.4f}")

        # 预测测试集
        print(f"\nPredicting test set...")
        test_predictions = model.predict(test_scaled)

        # 创建结果DataFrame
        result_df = pd.DataFrame({
            'Id': test_df['Id'],
            'Sold Price': test_predictions
        })

        # 保存独立的submission文件
        output_file = f'submission_{name.lower().replace(" ", "_")}.csv'
        result_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")

        print(f"\nPrediction Statistics:")
        print(f"  Min: {test_predictions.min():.2f}")
        print(f"  Max: {test_predictions.max():.2f}")
        print(f"  Mean: {test_predictions.mean():.2f}")
        print(f"  Median: {np.median(test_predictions):.2f}")

        # 保存模型
        model_file = f'{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_file)
        print(f"  Saved model: {model_file}")

    except Exception as e:
        print(f"  Error training {name}: {str(e)}")

# 模型比较
print("\n" + "="*70)
print("Model Comparison - All Results")
print("="*70)

if results:
    # 选择最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['val_r2'])
    best_model = results[best_model_name]['model']

    print(f"\nBest Model: {best_model_name}")
    print(f"Validation R2: {results[best_model_name]['val_r2']:.4f} ({results[best_model_name]['val_r2']*100:.2f}%)")
    print(f"Validation RMSE: {results[best_model_name]['val_rmse']:.2f}")
    print(f"Validation MAE: {results[best_model_name]['val_mae']:.2f}")

    # 保存所有模型的评估结果
    print("\n" + "="*70)
    print("All Models Evaluation Results")
    print("="*70)

    summary_df = pd.DataFrame({
        'Model': results.keys(),
        'Train RMSE': [r['train_rmse'] for r in results.values()],
        'Train MAE': [r['train_mae'] for r in results.values()],
        'Train R2': [r['train_r2'] for r in results.values()],
        'Validation RMSE': [r['val_rmse'] for r in results.values()],
        'Validation MAE': [r['val_mae'] for r in results.values()],
        'Validation R2': [r['val_r2'] for r in results.values()],
        'CV RMSE': [r['cv_rmse'] for r in results.values()],
        'CV Std': [r['cv_std'] for r in results.values()]
    })

    summary_df['CV RMSE'] = summary_df['CV RMSE'].apply(lambda x: f"{x:.2f}")
    summary_df['CV Std'] = summary_df['CV Std'].apply(lambda x: f"{x:.2f}")

    print(summary_df.to_string(index=False))
    summary_df.to_csv('model_comparison.csv', index=False)
    print("\nModel comparison saved as model_comparison.csv")

    # 特征重要性分析（仅对基于树的模型）
    print("\n" + "="*70)
    print("Feature Importance Analysis (Best Model)")
    print("="*70)

    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))

        # 绘制特征重要性图
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance.head(10)['Feature'], feature_importance.head(10)['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top 10 Feature Importance ({best_model_name})')
        plt.tight_layout()
        plt.savefig('feature_importance_best.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved as feature_importance_best.png")

    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"\nMain Results:")
    print(f"  Best Model: {best_model_name}")
    print(f"  Full Training R2: {full_train_r2:.4f}")
    print(f"  Validation R2: {results[best_model_name]['val_r2']:.4f} ({results[best_model_name]['val_r2']*100:.2f}%)")
    print(f"  Validation RMSE: {results[best_model_name]['val_rmse']:.2f}")
    print(f"  Average Relative Error: {((results[best_model_name]['val_mae'] / y_val.mean()) * 100):.2f}%")
    print(f"\nGenerated Files:")
    print(f"  - submission_xgboost.csv")
    print(f"  - submission_lightgbm.csv")
    print(f"  - submission_extra_trees.csv")
    print(f"  - submission_random_forest.csv")
    print(f"  - submission_gradient_boosting.csv")
    print(f"  - xgboost_model.pkl")
    print(f"  - lightgbm_model.pkl")
    print(f"  - extra_trees_model.pkl")
    print(f"  - random_forest_model.pkl")
    print(f"  - gradient_boosting_model.pkl")
else:
    print("No models were trained successfully!")
