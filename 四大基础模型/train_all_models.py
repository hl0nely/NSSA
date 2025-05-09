# train_all_models.py - 一键化训练所有模型
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
import pickle
import time
import os
import traceback

# 导入各种模型
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=默认, 1=不显示INFO, 2=不显示INFO/WARNING, 3=不显示所有

# 结果存储目录
RESULTS_DIR = 'model_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===== 数据可视化和比较函数 =====

def add_performance_table(y_test, y_pred, y_probs, model_name):
    """创建并保存详细的性能指标表格"""
    # 计算各项指标
    metrics = {
        '准确率 (Accuracy)': accuracy_score(y_test, y_pred),
        '精确率 (Precision)': precision_score(y_test, y_pred),
        '召回率 (Recall)': recall_score(y_test, y_pred),
        'F1分数': f1_score(y_test, y_pred),
        'AUC分数': roc_auc_score(y_test, y_probs),
        '正样本数量': sum(y_test == 1),
        '负样本数量': sum(y_test == 0),
        '预测正例': sum(y_pred == 1),
        '预测负例': sum(y_pred == 0)
    }

    # 转换为DataFrame并保存
    df_metrics = pd.DataFrame(list(metrics.items()), columns=['指标', '值'])
    df_metrics['值'] = df_metrics['值'].round(4)

    # 保存为CSV
    df_metrics.to_csv(f'{RESULTS_DIR}/{model_name}_performance_metrics.csv', index=False)

    # 可视化为条形图
    plt.figure(figsize=(10, 6))
    plt.bar(df_metrics['指标'][:5], df_metrics['值'][:5], color='teal')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{model_name}模型性能指标')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{model_name}_metrics_bar.png')

    return df_metrics


def analyze_predictions(y_test, y_pred, y_probs, model_name):
    """详细分析预测结果并生成可视化"""
    # 1. 错误分析
    false_pos = np.where((y_pred == 1) & (y_test == 0))[0]
    false_neg = np.where((y_pred == 0) & (y_test == 1))[0]

    error_analysis = {
        '正确预测': sum(y_pred == y_test),
        '错误预测': sum(y_pred != y_test),
        '假阳性(FP)': len(false_pos),
        '假阴性(FN)': len(false_neg),
        '假阳性比例': len(false_pos) / max(sum(y_test == 0), 1),
        '假阴性比例': len(false_neg) / max(sum(y_test == 1), 1),
    }

    # 创建DataFrame
    df_errors = pd.DataFrame(list(error_analysis.items()),
                             columns=['错误类型', '数量/比例'])

    # 保存为CSV
    df_errors.to_csv(f'{RESULTS_DIR}/{model_name}_error_analysis.csv', index=False)

    # 2. 预测概率分布柱状图
    plt.figure(figsize=(12, 6))

    # 为正确和错误预测创建不同的直方图
    correct_probs = y_probs[y_pred == y_test]
    wrong_probs = y_probs[y_pred != y_test]

    bins = np.linspace(0, 1, 21)  # 20个区间，从0到1

    plt.hist(correct_probs, bins=bins, alpha=0.5, label='正确预测', color='green')
    plt.hist(wrong_probs, bins=bins, alpha=0.5, label='错误预测', color='red')

    plt.xlabel('预测概率')
    plt.ylabel('样本数量')
    plt.title(f'{model_name}预测置信度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{RESULTS_DIR}/{model_name}_prediction_confidence.png')

    # 3. 概率区间的准确率
    interval_accuracy = []
    intervals = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0

    for i in range(len(intervals) - 1):
        lower = intervals[i]
        upper = intervals[i + 1]
        mask = (y_probs >= lower) & (y_probs < upper)

        if sum(mask) > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            interval_accuracy.append((f"{lower:.1f}-{upper:.1f}", acc, sum(mask)))

    df_intervals = pd.DataFrame(interval_accuracy,
                                columns=['概率区间', '准确率', '样本数'])

    # 保存为CSV
    df_intervals.to_csv(f'{RESULTS_DIR}/{model_name}_interval_accuracy.csv', index=False)

    # 可视化概率区间准确率
    plt.figure(figsize=(12, 6))
    plt.bar(df_intervals['概率区间'], df_intervals['准确率'], color='purple')

    # 添加样本数量标签
    for i, (_, row) in enumerate(df_intervals.iterrows()):
        plt.text(i, row['准确率'] + 0.02, f"n={row['样本数']}",
                 ha='center', va='bottom', fontsize=9)

    plt.ylim(0, 1.1)
    plt.title(f'{model_name}不同概率区间的准确率')
    plt.ylabel('准确率')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{model_name}_interval_accuracy.png')

    return df_errors, df_intervals


def find_optimal_threshold(y_true, y_scores, model_name):
    """寻找最佳阈值，平衡精确率和召回率"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # 计算F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # 计算精确率与召回率的几何平均，对极端值更敏感
    geo_mean = np.sqrt(precision * recall)

    # 找到F1和几何平均最大值对应的索引
    f1_idx = np.argmax(f1_scores)
    geo_idx = np.argmax(geo_mean)

    # 考虑两者，偏向于平衡模型
    best_idx = geo_idx  # 使用几何平均作为最终标准
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # 可视化阈值选择
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b--', label='精确率')
    plt.plot(thresholds, recall[:-1], 'g-', label='召回率')
    plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1分数')
    plt.plot(thresholds, geo_mean[:-1], 'c-', label='几何平均')
    plt.axvline(x=best_threshold, color='purple', linestyle='--',
                label=f'最佳阈值 ({best_threshold:.3f})')
    plt.xlabel('阈值')
    plt.ylabel('分数')
    plt.title(f'{model_name}阈值优化')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{RESULTS_DIR}/{model_name}_threshold_optimization.png')

    return best_threshold


def compare_models(models_results, save_path=f'{RESULTS_DIR}/model_comparison.csv'):
    """比较多个模型的性能指标"""
    # 创建比较DataFrame
    df_comparison = pd.DataFrame(columns=['模型', '准确率', 'AUC', '精确率', '召回率', 'F1分数'])

    for i, (model_name, metrics) in enumerate(models_results.items()):
        if metrics[0] is not None:  # 只比较成功训练的模型
            df_comparison.loc[i] = [model_name] + [metrics[2], metrics[3], metrics[4], metrics[5], metrics[6]]

    # 按准确率降序排序
    df_comparison = df_comparison.sort_values('准确率', ascending=False).reset_index(drop=True)

    # 保存为CSV
    df_comparison.to_csv(save_path, index=False)

    # 创建比较条形图
    metrics = ['准确率', 'AUC', '精确率', '召回率', 'F1分数']
    plt.figure(figsize=(15, 10))

    x = np.arange(len(metrics))
    width = 0.15
    n_models = len(df_comparison)

    for i, row in df_comparison.iterrows():
        offset = width * (i - n_models / 2 + 0.5)
        values = row[1:].values.astype(float)
        plt.bar(x + offset, values, width, label=row['模型'])

    plt.title('模型性能比较')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/model_comparison.png')

    return df_comparison


# ===== 特征工程函数 =====

def create_enhanced_features(df):
    """创建增强特征"""
    # 创建副本避免修改原始数据
    df_enhanced = df.copy()

    # 1. 流量相关特征
    df_enhanced['bytes_per_packet_src'] = df['sbytes'] / (df['spkts'] + 1e-8)
    df_enhanced['bytes_per_packet_dst'] = df['dbytes'] / (df['dpkts'] + 1e-8)
    df_enhanced['packet_ratio'] = df['spkts'] / (df['dpkts'] + 1e-8)
    df_enhanced['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1e-8)

    # 2. 时间相关特征
    df_enhanced['log_duration'] = np.log1p(df['dur'])
    df_enhanced['is_short_conn'] = (df['dur'] < 0.1).astype(int)
    df_enhanced['packets_per_second'] = (df['spkts'] + df['dpkts']) / (df['dur'] + 1e-8)

    # 3. 其他相关特征
    df_enhanced['tcp_window_ratio'] = df['swin'] / (df['dwin'] + 1e-8)
    df_enhanced['tcp_param_ratio'] = df['stcpb'] / (df['dtcpb'] + 1e-8)
    df_enhanced['connection_density'] = df['ct_srv_src'] + df['ct_dst_ltm'] + df['ct_src_ltm'] + df['ct_dst_src_ltm']
    df_enhanced['load_ratio'] = df['sload'] / (df['dload'] + 1e-8)
    df_enhanced['loss_ratio'] = (df['sloss'] + df['dloss']) / (df['spkts'] + df['dpkts'] + 1e-8)

    # 4. 简单协议标记
    if 'proto' in df.columns:
        df_enhanced['is_tcp'] = (df['proto'] == 'tcp').astype(int)
        df_enhanced['is_udp'] = (df['proto'] == 'udp').astype(int)

    # 5. 简单服务标记
    if 'service' in df.columns:
        df_enhanced['is_http'] = (df['service'] == 'http').astype(int)
        df_enhanced['is_dns'] = (df['service'] == 'dns').astype(int)

    # 计算相关性
    if 'label' in df.columns:
        try:
            numeric_cols = df_enhanced.select_dtypes(include=['number']).columns
            if pd.api.types.is_numeric_dtype(df['label']):
                correlations = df_enhanced[numeric_cols].corrwith(df['label']).abs().sort_values(ascending=False)
                print("与标签最相关的前10个特征:")
                print(correlations.head(10))
        except Exception as e:
            print(f"计算相关性时出错 (不影响特征创建): {e}")

    return df_enhanced


# ===== XGBoost模型函数 =====

def preprocess_data_for_tree_models(train_df, test_df, num_features=30, model_type='xgboost'):
    """为树模型预处理数据，包括特征选择和标准化"""
    # 移除不需要的列
    drop_cols = ['id', 'attack_cat']
    X_train = train_df.drop(drop_cols + ['label'], axis=1, errors='ignore')
    y_train = train_df['label']
    X_test = test_df.drop(drop_cols + ['label'], axis=1, errors='ignore')
    y_test = test_df['label']

    # 处理类别特征
    cat_features = ['proto', 'service', 'state']
    X_train_encoded = pd.get_dummies(X_train, columns=cat_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_features, drop_first=True)

    # 确保训练集和测试集有相同的特征
    train_cols = X_train_encoded.columns
    test_cols = X_test_encoded.columns

    # 找出训练集有而测试集没有的特征
    missing_cols = set(train_cols) - set(test_cols)
    for col in missing_cols:
        X_test_encoded[col] = 0

    # 确保列顺序一致
    X_test_encoded = X_test_encoded[train_cols]

    # 缺失值处理
    X_train_encoded = X_train_encoded.fillna(0)
    X_test_encoded = X_test_encoded.fillna(0)

    # 标准化数值特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # 根据模型类型选择特征重要性计算器
    if model_type == 'xgboost':
        selector = xgb.XGBClassifier(n_estimators=100, random_state=42)
    elif model_type == 'lightgbm':
        selector = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    else:  # gbdt
        selector = GradientBoostingClassifier(n_estimators=100, random_state=42)

    selector.fit(X_train_scaled, y_train)

    # 获取特征重要性并排序
    importance = selector.feature_importances_
    indices = np.argsort(importance)[::-1]

    # 打印最重要的特征
    feature_names = X_train_encoded.columns
    print(f"{model_type.upper()}特征排名:")
    for i, idx in enumerate(indices[:10]):
        print(f"{i + 1}. {feature_names[idx]} ({importance[idx]:.4f})")

    # 选择前num_features个最重要的特征
    top_indices = indices[:num_features]
    X_train_selected = X_train_scaled[:, top_indices]
    X_test_selected = X_test_scaled[:, top_indices]

    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    plt.title(f'{model_type.upper()}特征重要性')
    plt.bar(range(10), importance[indices[:10]], align='center')
    plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{model_type}_feature_importance.png')

    return X_train_selected, y_train, X_test_selected, y_test, [feature_names[i] for i in top_indices]


def balance_data(X, y, sampling_strategy=0.6):
    """使用SMOTE平衡数据"""
    print(f"原始标签分布: {np.bincount(y)}")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"重采样后的标签分布: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """训练和评估XGBoost模型"""
    # 确保输入是numpy数组
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 分割验证集用于阈值优化(避免数据泄露)
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 设置XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'n_estimators': 100,
        'random_state': 42
    }

    # 创建XGBoost分类器
    model = xgb.XGBClassifier(**params)

    try:
        # 使用交叉验证获取更可靠的概率估计
        print("执行交叉验证训练...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_probs = np.zeros(len(X_test))

        # 记录每次交叉验证的训练和验证性能
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_main, y_train_main)):
            # 正确使用数组切片
            X_cv_train, X_cv_val = X_train_main[train_idx], X_train_main[val_idx]
            y_cv_train, y_cv_val = y_train_main[train_idx], y_train_main[val_idx]

            # 创建评估集
            eval_set = [(X_cv_train, y_cv_train), (X_cv_val, y_cv_val)]

            # 训练模型
            model.fit(
                X_cv_train, y_cv_train,
                eval_set=eval_set,
                verbose=False
            )

            # 记录训练过程
            cv_results.append(model.evals_result())

            # 累加预测概率
            y_probs += model.predict_proba(X_test)[:, 1] / cv.n_splits

        # 训练最终模型
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X_train, y_train)

        # 在验证集上寻找最佳阈值(避免数据泄露)
        val_probs = final_model.predict_proba(X_val)[:, 1]
        best_threshold = find_optimal_threshold(y_val, val_probs, "XGBoost")
        print(f"最佳阈值: {best_threshold:.4f}")

        # 使用最佳阈值进行测试集预测
        test_probs = final_model.predict_proba(X_test)[:, 1]
        y_pred = (test_probs >= best_threshold).astype(int)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, test_probs)

        # 打印详细结果
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 添加数值化结果分析
        metrics_df = add_performance_table(y_test, y_pred, test_probs, "XGBoost")
        error_df, interval_df = analyze_predictions(y_test, y_pred, test_probs, "XGBoost")

        print("\n性能指标表:")
        print(metrics_df)
        print("\n错误分析:")
        print(error_df)
        print("\n概率区间准确率:")
        print(interval_df)

        # 可视化混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '攻击'],
                    yticklabels=['正常', '攻击'])
        plt.xlabel('预测')
        plt.ylabel('实际')
        plt.title('XGBoost混淆矩阵')
        plt.savefig(f'{RESULTS_DIR}/xgboost_confusion_matrix.png')

        # 可视化预测概率分布
        plt.figure(figsize=(10, 6))
        plt.hist(test_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
        plt.hist(test_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
        plt.axvline(x=best_threshold, color='red', linestyle='--',
                    label=f'阈值 ({best_threshold:.3f})')
        plt.xlabel('预测概率')
        plt.ylabel('样本数量')
        plt.title('XGBoost预测概率分布')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/xgboost_probability_distribution.png')

        # ROC曲线可视化
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('XGBoost ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/xgboost_roc_curve.png')

        # 保存模型
        with open(f'{RESULTS_DIR}/unsw_nb15_xgboost_model.pkl', 'wb') as f:
            pickle.dump({
                'model': final_model,
                'threshold': best_threshold,
                'selected_features': None,  # 根据实际情况添加
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }, f)

        return final_model, best_threshold, accuracy, auc, precision, recall, f1, cv_results

    except Exception as e:
        print(f"XGBoost模型训练失败: {str(e)}")
        traceback.print_exc()  # 打印详细的堆栈跟踪
        return None, 0.5, 0, 0, 0, 0, 0, None


# ===== LightGBM模型函数 =====

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """训练并评估LightGBM模型，使用交叉验证"""
    # 确保输入是numpy数组
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 分割验证集用于阈值优化(避免数据泄露)
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    try:
        # 创建LightGBM分类器 - 修改参数避免冗余警告
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            num_leaves=31,
            # 只使用LightGBM原生参数，不使用别名
            feature_fraction=0.9,  # 控制特征采样比例
            bagging_fraction=0.8,  # 控制数据采样比例
            bagging_freq=5,  # 控制bagging频率
            is_unbalance=True,  # 处理不平衡数据
            subsample_for_bin=200000,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1  # 减少冗余警告
        )

        # 使用交叉验证获取更可靠的概率估计
        print("执行交叉验证训练...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_probs = np.zeros(len(X_test))

        # 记录每次交叉验证的训练结果
        eval_results = []

        for train_idx, val_idx in cv.split(X_train_main, y_train_main):
            # 正确使用数组切片
            X_cv_train, X_cv_val = X_train_main[train_idx], X_train_main[val_idx]
            y_cv_train, y_cv_val = y_train_main[train_idx], y_train_main[val_idx]

            # 训练模型并记录结果
            evals_result = {}
            model.fit(
                X_cv_train, y_cv_train,
                eval_set=[(X_cv_train, y_cv_train), (X_cv_val, y_cv_val)],
                eval_names=['train', 'valid'],  # 显式命名验证集
                callbacks=[
                    lgb.record_evaluation(evals_result),
                    lgb.early_stopping(stopping_rounds=20, verbose=False)
                ]
            )

            eval_results.append(evals_result)

            # 累加预测概率
            y_probs += model.predict_proba(X_test)[:, 1] / cv.n_splits

        # 训练最终模型
        final_model = lgb.LGBMClassifier(**model.get_params())
        final_model.fit(X_train, y_train)

        # 在验证集上寻找最佳阈值(避免数据泄露)
        val_probs = final_model.predict_proba(X_val)[:, 1]
        best_threshold = find_optimal_threshold(y_val, val_probs, "LightGBM")
        print(f"最佳阈值: {best_threshold:.4f}")

        # 使用最佳阈值进行测试集预测
        test_probs = final_model.predict_proba(X_test)[:, 1]
        y_pred = (test_probs >= best_threshold).astype(int)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, test_probs)

        # 打印详细结果
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 添加数值化结果分析
        metrics_df = add_performance_table(y_test, y_pred, test_probs, "LightGBM")
        error_df, interval_df = analyze_predictions(y_test, y_pred, test_probs, "LightGBM")

        print("\n性能指标表:")
        print(metrics_df)
        print("\n错误分析:")
        print(error_df)
        print("\n概率区间准确率:")
        print(interval_df)

        # 可视化混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '攻击'],
                    yticklabels=['正常', '攻击'])
        plt.xlabel('预测')
        plt.ylabel('实际')
        plt.title('LightGBM混淆矩阵')
        plt.savefig(f'{RESULTS_DIR}/lightgbm_confusion_matrix.png')

        # 可视化预测概率分布
        plt.figure(figsize=(10, 6))
        plt.hist(test_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
        plt.hist(test_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
        plt.axvline(x=best_threshold, color='red', linestyle='--',
                    label=f'阈值 ({best_threshold:.3f})')
        plt.xlabel('预测概率')
        plt.ylabel('样本数量')
        plt.title('LightGBM预测概率分布')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/lightgbm_probability_distribution.png')

        # ROC曲线可视化
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('LightGBM ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/lightgbm_roc_curve.png')

        # 可视化学习曲线 - 修复键错误问题
        plt.figure(figsize=(10, 6))

        print("评估结果键:", list(eval_results[0].keys()))

        # 动态确定正确的键名
        try:
            # 尝试不同可能的键名
            if 'train' in eval_results[0]:
                train_key = 'train'
                valid_key = 'valid'
            elif 'training' in eval_results[0]:
                train_key = 'training'
                valid_key = 'valid_0'
            else:
                # 通过列表推导找出合适的键
                all_keys = list(eval_results[0].keys())
                train_key = [k for k in all_keys if 'train' in k.lower()][0]
                valid_key = [k for k in all_keys if 'valid' in k.lower()][0]

            # 获取可用的指标
            available_metrics = list(eval_results[0][train_key].keys())
            print(f"可用的评估指标: {available_metrics}")

            if available_metrics:
                metric_name = available_metrics[0]

                # 绘制学习曲线
                train_metric = eval_results[0][train_key][metric_name]
                valid_metric = eval_results[0][valid_key][metric_name]

                epochs = len(train_metric)
                x_axis = range(0, epochs)

                plt.plot(x_axis, train_metric, label=f'训练 {metric_name}')
                plt.plot(x_axis, valid_metric, label=f'验证 {metric_name}')

        except (KeyError, IndexError) as e:
            print(f"绘制学习曲线时出错: {e}")
            print("可用的键:", eval_results[0].keys())

        plt.legend()
        plt.title('LightGBM学习曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('指标')
        plt.grid()
        plt.savefig(f'{RESULTS_DIR}/lightgbm_learning_curve.png')

        # 预测结果散点图
        max_points = 1000
        if len(y_test) > max_points:
            idx = np.random.choice(len(y_test), max_points, replace=False)
            y_test_sample = y_test[idx]
            y_pred_sample = y_pred[idx]

            # 确保长度一致
            indices = np.arange(len(y_test_sample))

            plt.figure(figsize=(10, 6))
            plt.scatter(indices, y_test_sample, color='blue', alpha=0.5, label='实际值')
            plt.scatter(indices, y_pred_sample, color='red', alpha=0.5, label='预测值')
        else:
            indices = np.arange(len(y_test))

            plt.figure(figsize=(10, 6))
            plt.scatter(indices, y_test, color='blue', alpha=0.5, label='实际值')
            plt.scatter(indices, y_pred, color='red', alpha=0.5, label='预测值')

        plt.xlabel('样本索引')
        plt.ylabel('标签值')
        plt.title('LightGBM预测结果可视化')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/lightgbm_prediction_results.png')

        # 保存模型
        with open(f'{RESULTS_DIR}/unsw_nb15_lightgbm_model.pkl', 'wb') as f:
            pickle.dump({
                'model': final_model,
                'threshold': best_threshold,
                'selected_features': None,  # 根据实际情况添加
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }, f)

        return final_model, best_threshold, accuracy, auc, precision, recall, f1, eval_results

    except Exception as e:
        print(f"LightGBM模型训练失败: {str(e)}")
        traceback.print_exc()  # 打印详细的堆栈跟踪
        return None, 0.5, 0, 0, 0, 0, 0, None


# ===== GBDT模型函数 =====

def train_gbdt_model(X_train, y_train, X_test, y_test):
    """训练并评估GBDT模型，使用交叉验证，并添加早停机制"""
    # 确保输入是numpy数组
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    try:
        # 创建验证集以便实现早停和阈值优化
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # 优化参数 - 使用warm_start实现早停
        model = GradientBoostingClassifier(
            n_estimators=50,  # 初始迭代次数较少
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.7,  # 减小以提高速度
            warm_start=True,  # 启用warm_start实现早停
            random_state=42,
            verbose=0  # 减少输出
        )

        # 实现早停
        max_estimators = 200  # 最大迭代次数
        patience = 10  # 容忍多少次迭代没有改进
        min_delta = 0.001  # 最小改进量

        best_val_score = 0
        best_n_estimators = 50
        no_improve_count = 0
        val_scores = []

        print("使用warm_start和早停训练GBDT...")
        for n_est in range(50, max_estimators + 1, 10):  # 每次增加10个estimator
            model.n_estimators = n_est
            model.fit(X_train_main, y_train_main)

            val_pred = model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_pred)
            val_scores.append(val_score)

            print(f"  Estimators: {n_est}, 验证AUC: {val_score:.4f}")

            # 检查是否有改进
            if val_score > best_val_score + min_delta:
                best_val_score = val_score
                best_n_estimators = n_est
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 如果连续多次没有改进，则早停
            if no_improve_count >= patience:
                print(f"早停触发! 最佳estimators数量: {best_n_estimators}")
                break

        # 使用最佳参数重新训练最终模型
        print(f"使用最佳参数训练最终模型 (n_estimators={best_n_estimators})...")
        final_model = GradientBoostingClassifier(
            n_estimators=best_n_estimators,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.7,
            random_state=42,
            verbose=0
        )

        # 使用交叉验证获取更可靠的概率估计
        print("执行交叉验证训练...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_probs = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_main, y_train_main)):
            # 正确使用数组切片
            X_cv_train, X_cv_val = X_train_main[train_idx], X_train_main[val_idx]
            y_cv_train, y_cv_val = y_train_main[train_idx], y_train_main[val_idx]

            # 训练模型
            cv_model = GradientBoostingClassifier(
                n_estimators=best_n_estimators,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.7,
                random_state=42,
                verbose=0
            )
            cv_model.fit(X_cv_train, y_cv_train)

            # 累加预测概率
            y_probs += cv_model.predict_proba(X_test)[:, 1] / cv.n_splits

        # 训练最终模型
        final_model.fit(X_train, y_train)

        # 在验证集上寻找最佳阈值(避免数据泄露)
        val_probs = final_model.predict_proba(X_val)[:, 1]
        best_threshold = find_optimal_threshold(y_val, val_probs, "GBDT")
        print(f"最佳阈值: {best_threshold:.4f}")

        # 使用最佳阈值进行测试集预测
        test_probs = final_model.predict_proba(X_test)[:, 1]
        y_pred = (test_probs >= best_threshold).astype(int)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, test_probs)

        # 打印详细结果
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 添加数值化结果分析
        metrics_df = add_performance_table(y_test, y_pred, test_probs, "GBDT")
        error_df, interval_df = analyze_predictions(y_test, y_pred, test_probs, "GBDT")

        print("\n性能指标表:")
        print(metrics_df)
        print("\n错误分析:")
        print(error_df)
        print("\n概率区间准确率:")
        print(interval_df)

        # 可视化混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '攻击'],
                    yticklabels=['正常', '攻击'])
        plt.xlabel('预测')
        plt.ylabel('实际')
        plt.title('GBDT混淆矩阵')
        plt.savefig(f'{RESULTS_DIR}/gbdt_confusion_matrix.png')

        # 可视化预测概率分布
        plt.figure(figsize=(10, 6))
        plt.hist(test_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
        plt.hist(test_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
        plt.axvline(x=best_threshold, color='red', linestyle='--',
                    label=f'阈值 ({best_threshold:.3f})')
        plt.xlabel('预测概率')
        plt.ylabel('样本数量')
        plt.title('GBDT预测概率分布')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/gbdt_probability_distribution.png')

        # ROC曲线可视化
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('GBDT ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/gbdt_roc_curve.png')

        # 学习曲线可视化
        plt.figure(figsize=(10, 6))

        # 使用早停阶段收集的验证分数
        plt.plot(range(50, 50 + len(val_scores) * 10, 10), val_scores, 'r-', label='验证集AUC')
        plt.axvline(x=best_n_estimators, color='purple', linestyle='--',
                    label=f'最佳estimators: {best_n_estimators}')

        plt.title('GBDT早停迭代过程')
        plt.xlabel('Estimators数量')
        plt.ylabel('验证集AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/gbdt_early_stopping.png')

        # 训练最终模型用于正式部署
        final_model.fit(X_train, y_train)

        # 预测结果散点图 - 修复长度不匹配问题
        try:
            # 限制最多显示1000个样本点
            max_points = 1000
            if len(y_test) > max_points:
                idx = np.random.choice(len(y_test), max_points, replace=False)
                y_test_sample = y_test[idx]
                y_pred_sample = y_pred[idx]

                # 使用索引作为X轴
                indices = np.arange(len(y_test_sample))

                # 确保所有数组长度相同
                min_len = min(len(indices), len(y_test_sample), len(y_pred_sample))
                indices = indices[:min_len]
                y_test_sample = y_test_sample[:min_len]
                y_pred_sample = y_pred_sample[:min_len]

                plt.figure(figsize=(10, 6))
                plt.scatter(indices, y_test_sample, color='blue', alpha=0.5, label='实际值')
                plt.scatter(indices, y_pred_sample, color='red', alpha=0.5, label='预测值')
            else:
                indices = np.arange(len(y_test))

                plt.figure(figsize=(10, 6))
                plt.scatter(indices, y_test, color='blue', alpha=0.5, label='实际值')
                plt.scatter(indices, y_pred, color='red', alpha=0.5, label='预测值')

            plt.xlabel('样本索引')
            plt.ylabel('标签值')
            plt.title('GBDT预测结果可视化')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{RESULTS_DIR}/gbdt_prediction_results.png')
        except Exception as e:
            print(f"绘制预测结果散点图时出错 (不影响模型): {e}")

        # 保存模型
        with open(f'{RESULTS_DIR}/unsw_nb15_gbdt_model.pkl', 'wb') as f:
            pickle.dump({
                'model': final_model,
                'threshold': best_threshold,
                'selected_features': None,  # 根据实际情况添加
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }, f)

        return final_model, best_threshold, accuracy, auc, precision, recall, f1, val_scores

    except Exception as e:
        print(f"GBDT模型训练失败: {str(e)}")
        traceback.print_exc()  # 打印详细的堆栈跟踪
        return None, 0.5, 0, 0, 0, 0, 0, None


# ===== LSTM模型函数 =====

def preprocess_data_for_lstm(train_df, test_df):
    """特别为LSTM准备数据，包括序列化处理"""
    # 移除不需要的列
    drop_cols = ['id', 'attack_cat']
    X_train = train_df.drop(drop_cols + ['label'], axis=1, errors='ignore')
    y_train = train_df['label']
    X_test = test_df.drop(drop_cols + ['label'], axis=1, errors='ignore')
    y_test = test_df['label']

    # 处理类别特征 - 对于LSTM，我们可以使用标签编码而不是独热编码
    cat_features = ['proto', 'service', 'state']

    # 为每个类别特征创建标签编码器
    encoders = {}
    for feature in cat_features:
        if feature in X_train.columns:
            # 合并训练集和测试集中的所有可能类别
            all_categories = set(X_train[feature].unique()).union(set(X_test[feature].unique()))
            print(f"特征 {feature} 中的类别数量: {len(all_categories)}")

            le = LabelEncoder()
            le.fit(list(all_categories))  # 使用所有可能类别拟合编码器

            X_train[feature] = le.transform(X_train[feature])
            X_test[feature] = le.transform(X_test[feature])

            encoders[feature] = le

    # 缺失值处理
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # 标准化数值特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 对于LSTM，我们需要重塑数据为3D格式 [samples, time_steps, features]
    # 这里我们使用一个时间步，但在实际中可能需要更多
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    print(f"LSTM输入形状: {X_train_lstm.shape}")

    return X_train_lstm, y_train, X_test_lstm, y_test, X_train.columns.tolist(), scaler


def balance_data_for_lstm(X, y, sampling_strategy=0.6):
    """为LSTM准备平衡数据集"""
    print(f"原始标签分布: {np.bincount(y)}")

    # 对于LSTM数据，我们需要先将3D数据展平，应用SMOTE，然后再重塑回3D
    n_samples, time_steps, n_features = X.shape
    X_reshaped = X.reshape(n_samples, time_steps * n_features)

    # 应用SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)

    # 重塑回3D
    X_resampled = X_resampled.reshape(X_resampled.shape[0], time_steps, n_features)

    print(f"重采样后的标签分布: {np.bincount(y_resampled)}")
    print(f"重采样后LSTM输入形状: {X_resampled.shape}")

    return X_resampled, y_resampled


def create_lstm_model(input_shape):
    """创建LSTM模型"""
    model = Sequential([
        # 双向LSTM层
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),

        # 第二个LSTM层
        Bidirectional(LSTM(64)),
        Dropout(0.3),

        # 全连接层
        Dense(32, activation='relu'),
        Dropout(0.2),

        # 输出层
        Dense(1, activation='sigmoid')
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # 打印模型摘要
    model.summary()

    return model


def train_lstm_model(X_train, y_train, X_test, y_test):
    """训练并评估LSTM模型"""
    # 确保输入是numpy数组
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    try:
        # 从训练集中拆分出验证集
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # 创建模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_lstm_model(input_shape)

        # 设置回调函数
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
            ModelCheckpoint(f'{RESULTS_DIR}/best_lstm_model.keras', monitor='val_auc', save_best_only=True, mode='max')
        ]

        # 训练模型
        print("训练LSTM模型...")
        history = model.fit(
            X_train_main, y_train_main,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )

        # 加载最佳模型
        model.load_weights(f'{RESULTS_DIR}/best_lstm_model.keras')

        # 在验证集上寻找最佳阈值(避免数据泄露)
        val_probs = model.predict(X_val).flatten()
        best_threshold = find_optimal_threshold(y_val, val_probs, "LSTM")
        print(f"最佳阈值: {best_threshold:.4f}")

        # 使用最佳阈值进行测试集预测
        test_probs = model.predict(X_test).flatten()
        y_pred = (test_probs >= best_threshold).astype(int)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, test_probs)

        # 打印详细结果
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 添加数值化结果分析
        metrics_df = add_performance_table(y_test, y_pred, test_probs, "LSTM")
        error_df, interval_df = analyze_predictions(y_test, y_pred, test_probs, "LSTM")

        print("\n性能指标表:")
        print(metrics_df)
        print("\n错误分析:")
        print(error_df)
        print("\n概率区间准确率:")
        print(interval_df)

        # 打印混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(cm)

        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '攻击'],
                    yticklabels=['正常', '攻击'])
        plt.xlabel('预测')
        plt.ylabel('实际')
        plt.title('LSTM混淆矩阵')
        plt.savefig(f'{RESULTS_DIR}/lstm_confusion_matrix.png')

        # 可视化训练历史
        plt.figure(figsize=(12, 5))

        # 绘制精度
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='训练集准确率')
        plt.plot(history.history['val_accuracy'], label='验证集准确率')
        plt.title('LSTM模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)

        # 绘制损失
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='训练集损失')
        plt.plot(history.history['val_loss'], label='验证集损失')
        plt.title('LSTM模型损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/lstm_training_history.png')

        # 可视化预测概率分布
        plt.figure(figsize=(10, 6))
        plt.hist(test_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
        plt.hist(test_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
        plt.axvline(x=best_threshold, color='red', linestyle='--',
                    label=f'阈值 ({best_threshold:.3f})')
        plt.xlabel('预测概率')
        plt.ylabel('样本数量')
        plt.title('LSTM预测概率分布')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/lstm_probability_distribution.png')

        # ROC曲线可视化
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('LSTM ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/lstm_roc_curve.png')

        # 预测结果散点图
        try:
            max_points = 1000
            if len(y_test) > max_points:
                idx = np.random.choice(len(y_test), max_points, replace=False)
                y_test_sample = y_test[idx]
                y_pred_sample = y_pred[idx]

                # 确保长度一致
                indices = np.arange(len(y_test_sample))

                plt.figure(figsize=(10, 6))
                plt.scatter(indices, y_test_sample, color='blue', alpha=0.5, label='实际值')
                plt.scatter(indices, y_pred_sample, color='red', alpha=0.5, label='预测值')
            else:
                indices = np.arange(len(y_test))

                plt.figure(figsize=(10, 6))
                plt.scatter(indices, y_test, color='blue', alpha=0.5, label='实际值')
                plt.scatter(indices, y_pred, color='red', alpha=0.5, label='预测值')

            plt.xlabel('样本索引')
            plt.ylabel('标签值')
            plt.title('LSTM预测结果可视化')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{RESULTS_DIR}/lstm_prediction_results.png')
        except Exception as e:
            print(f"绘制预测结果散点图时出错 (不影响模型): {e}")

        # 绘制AUC曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['auc'], label='训练集AUC')
        plt.plot(history.history['val_auc'], label='验证集AUC')
        plt.title('LSTM模型AUC曲线')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{RESULTS_DIR}/lstm_auc_curve.png')

        # 保存模型和相关信息
        model.save(f'{RESULTS_DIR}/lstm_model.h5')

        # 保存阈值和评估指标
        results = {
            'threshold': best_threshold,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        with open(f'{RESULTS_DIR}/lstm_model_info.pkl', 'wb') as f:
            pickle.dump(results, f)

        return model, best_threshold, accuracy, auc, precision, recall, f1, history.history

    except Exception as e:
        print(f"LSTM模型训练失败: {str(e)}")
        traceback.print_exc()  # 打印详细的堆栈跟踪
        return None, 0.5, 0, 0, 0, 0, 0, None


# ===== 主函数 =====

def main():
    # 记录总开始时间
    total_start_time = time.time()

    # 1. 加载数据
    print("=== 加载UNSW-NB15数据集 ===")
    try:
        train_data = pd.read_csv("./data/UNSW_NB15_training-set.csv")
        test_data = pd.read_csv("./data/UNSW_NB15_testing-set.csv")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保数据文件位于 ./data/ 目录下")
        return

    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")

    # 2. 数据探索和分析
    print("\n=== 分析标签分布 ===")
    print(f"训练集标签分布: {train_data['label'].value_counts()}")
    print(f"测试集标签分布: {test_data['label'].value_counts()}")

    # 3. 特征工程
    print("\n=== 创建增强特征 ===")
    train_enhanced = create_enhanced_features(train_data)
    test_enhanced = create_enhanced_features(test_data)

    # 存储所有模型结果
    models_results = {}

    # ===== XGBoost模型训练 =====
    try:
        print("\n\n======== 训练XGBoost模型 ========")
        start_time = time.time()

        # 预处理数据
        print("\n=== 预处理数据和选择特征 ===")
        X_train, y_train, X_test, y_test, selected_features = preprocess_data_for_tree_models(
            train_enhanced, test_enhanced, num_features=30, model_type='xgboost'
        )

        # 平衡数据
        print("\n=== 平衡训练数据 ===")
        X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

        # 训练和评估模型
        print("\n=== 训练和评估XGBoost模型 ===")
        xgb_model, xgb_threshold, xgb_accuracy, xgb_auc, xgb_precision, xgb_recall, xgb_f1, xgb_cv = train_xgboost_model(
            X_train_balanced, y_train_balanced, X_test, y_test
        )

        # 记录结果
        if xgb_model is not None:
            models_results['XGBoost'] = (
            xgb_model, xgb_threshold, xgb_accuracy, xgb_auc, xgb_precision, xgb_recall, xgb_f1)

            # 记录训练时间
            xgb_time = time.time() - start_time
            hours, remainder = divmod(xgb_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\nXGBoost训练完成，用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    except Exception as e:
        print(f"XGBoost模型训练失败: {e}")
        traceback.print_exc()

    # ===== LightGBM模型训练 =====
    try:
        print("\n\n======== 训练LightGBM模型 ========")
        start_time = time.time()

        # 预处理数据
        print("\n=== 预处理数据和选择特征 ===")
        X_train, y_train, X_test, y_test, selected_features = preprocess_data_for_tree_models(
            train_enhanced, test_enhanced, num_features=30, model_type='lightgbm'
        )

        # 平衡数据
        print("\n=== 平衡训练数据 ===")
        X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

        # 训练和评估模型
        print("\n=== 训练和评估LightGBM模型 ===")
        lgb_model, lgb_threshold, lgb_accuracy, lgb_auc, lgb_precision, lgb_recall, lgb_f1, lgb_cv = train_lightgbm_model(
            X_train_balanced, y_train_balanced, X_test, y_test
        )

        # 记录结果
        if lgb_model is not None:
            models_results['LightGBM'] = (
            lgb_model, lgb_threshold, lgb_accuracy, lgb_auc, lgb_precision, lgb_recall, lgb_f1)

            # 记录训练时间
            lgb_time = time.time() - start_time
            hours, remainder = divmod(lgb_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\nLightGBM训练完成，用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    except Exception as e:
        print(f"LightGBM模型训练失败: {e}")
        traceback.print_exc()

    # ===== GBDT模型训练 =====
    try:
        print("\n\n======== 训练GBDT模型 ========")
        start_time = time.time()

        # 预处理数据
        print("\n=== 预处理数据和选择特征 ===")
        X_train, y_train, X_test, y_test, selected_features = preprocess_data_for_tree_models(
            train_enhanced, test_enhanced, num_features=30, model_type='gbdt'
        )

        # 平衡数据
        print("\n=== 平衡训练数据 ===")
        X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

        # 训练和评估模型
        print("\n=== 训练和评估GBDT模型 ===")
        gbdt_model, gbdt_threshold, gbdt_accuracy, gbdt_auc, gbdt_precision, gbdt_recall, gbdt_f1, gbdt_cv = train_gbdt_model(
            X_train_balanced, y_train_balanced, X_test, y_test
        )

        # 记录结果
        if gbdt_model is not None:
            models_results['GBDT'] = (
            gbdt_model, gbdt_threshold, gbdt_accuracy, gbdt_auc, gbdt_precision, gbdt_recall, gbdt_f1)

            # 记录训练时间
            gbdt_time = time.time() - start_time
            hours, remainder = divmod(gbdt_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\nGBDT训练完成，用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    except Exception as e:
        print(f"GBDT模型训练失败: {e}")
        traceback.print_exc()

    # ===== LSTM模型训练 =====
    try:
        print("\n\n======== 训练LSTM模型 ========")
        start_time = time.time()

        # 设置GPU内存增长，避免占用全部GPU内存
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # 限制GPU内存使用，让TensorFlow按需分配内存
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"发现 {len(gpus)} 个GPU设备，已启用内存增长模式")
            except RuntimeError as e:
                print(f"GPU配置失败: {e}")
        else:
            print("未检测到GPU设备，将使用CPU训练")

        # 预处理数据
        print("\n=== 预处理数据 ===")
        X_train, y_train, X_test, y_test, feature_names, scaler = preprocess_data_for_lstm(
            train_enhanced, test_enhanced
        )

        # 平衡数据
        print("\n=== 平衡训练数据 ===")
        X_train_balanced, y_train_balanced = balance_data_for_lstm(X_train, y_train)

        # 训练和评估模型
        print("\n=== 训练和评估LSTM模型 ===")
        lstm_model, lstm_threshold, lstm_accuracy, lstm_auc, lstm_precision, lstm_recall, lstm_f1, lstm_hist = train_lstm_model(
            X_train_balanced, y_train_balanced, X_test, y_test
        )

        # 记录结果
        if lstm_model is not None:
            models_results['LSTM'] = (
            lstm_model, lstm_threshold, lstm_accuracy, lstm_auc, lstm_precision, lstm_recall, lstm_f1)

            # 记录训练时间
            lstm_time = time.time() - start_time
            hours, remainder = divmod(lstm_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\nLSTM训练完成，用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    except Exception as e:
        print(f"LSTM模型训练失败: {e}")
        traceback.print_exc()

    # ===== 模型对比 =====
    if models_results:
        print("\n\n======== 模型性能比较 ========")
        comparison_df = compare_models(models_results)
        print("\n模型性能比较表:")
        print(comparison_df)
    else:
        print("\n所有模型训练失败，无法进行比较")

    # 记录总训练时间
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n\n所有模型训练完成，总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    print(f"结果保存在 {RESULTS_DIR} 目录下")


if __name__ == "__main__":
    main()