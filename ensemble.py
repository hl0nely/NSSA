# ensemble_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm.callback import early_stopping

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import joblib
import pickle
import os
import time
import warnings
import traceback

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


class FeatureSelectedMetaLearner:
    def __init__(self, selector, model):
        self.selector = selector
        self.model = model

    def predict_proba(self, X):
        X_selected = X[:, self.selector.get_support()]
        return self.model.predict_proba(X_selected)

    def predict(self, X):
        X_selected = X[:, self.selector.get_support()]
        return self.model.predict(X_selected)


class WeightedMetaLearner:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], 2))
        for i, model in enumerate(self.models):
            probas += self.weights[i] * model.predict_proba(X)
        return probas

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class StackedMetaLearner:
    """堆叠元学习器，结合两层模型"""

    def __init__(self, level1, level2):
        self.level1 = level1
        self.level2 = level2

    def predict_proba(self, X):
        # 第一层预测
        level1_preds = self.level1.predict_proba(X)[:, 1].reshape(-1, 1)
        # 组合原始特征和第一层预测
        enhanced_X = np.hstack([X, level1_preds])
        # 第二层预测
        return self.level2.predict_proba(enhanced_X)


class WeightedEnsemble:
    """加权集成模型"""
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        for model, weight in zip(self.models, self.weights):
            probs += weight * model.predict_proba(X)
        return probs


class EnhancedClusterer:
    """增强的聚类器，处理降维和分类聚类"""

    def __init__(self, pca, kmeans_pos, kmeans_neg, n_pos_clusters):
        self.pca = pca
        self.kmeans_pos = kmeans_pos
        self.kmeans_neg = kmeans_neg
        self.n_pos_clusters = n_pos_clusters

    def predict(self, X):
        # 降维
        X_reduced = self.pca.transform(X)

        # 计算到每个正负簇中心的距离
        pos_distances = self.kmeans_pos.transform(X_reduced)
        neg_distances = self.kmeans_neg.transform(X_reduced)

        # 合并距离，找到最近的簇
        all_distances = np.hstack([pos_distances, neg_distances])
        closest_cluster = np.argmin(all_distances, axis=1)

        # 调整负样本簇的编号
        mask = closest_cluster >= self.n_pos_clusters
        closest_cluster[mask] = closest_cluster[mask] + self.n_pos_clusters

        return closest_cluster


# ===================================
# 数据处理和基础功能函数
# ===================================

def create_enhanced_features(df):
    """创建增强特征"""
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

    # 4. 简单协议标记   其实有130多种协议，但是只需要识别攻击所以直接尝试多类型
    if 'proto' in df.columns:
        df_enhanced['is_tcp'] = (df['proto'] == 'tcp').astype(int)
        df_enhanced['is_udp'] = (df['proto'] == 'udp').astype(int)

    # 5. 简单服务标记   不需要其他，因为攻击类型会有识别
    if 'service' in df.columns:
        df_enhanced['is_http'] = (df['service'] == 'http').astype(int)
        df_enhanced['is_dns'] = (df['service'] == 'dns').astype(int)

    return df_enhanced


def preprocess_data(train_df, test_df, num_features=30):
    """预处理数据，包括特征选择和标准化"""
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

    return X_train_scaled, y_train, X_test_scaled, y_test, X_train_encoded.columns.tolist(), scaler


def reshape_for_lstm(X):
    return X.reshape(X.shape[0], 1, X.shape[1])   #LSTM是时序模型，默认要求输入具有时间步维度，还原三维


def find_optimal_threshold(y_true, y_scores, model_name="Ensemble"):
    """寻找最佳阈值，平衡精确率和召回率"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # 计算F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # 计算精确率与召回率的几何平均，对极端值更敏感
    geo_mean = np.sqrt(precision * recall)

    # 找到F1和几何平均最大值对应的索引
    f1_idx = np.argmax(f1_scores)
    geo_idx = np.argmax(geo_mean)


    best_idx = geo_idx  # 使用几何平均作为最终标准
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # 可视化阈值选择 - 确保命名包含"validation"表明这是在验证集上优化的
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b--', label='精确率')
    plt.plot(thresholds, recall[:-1], 'g-', label='召回率')
    plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1分数')
    plt.plot(thresholds, geo_mean[:-1], 'c-', label='几何平均')
    plt.axvline(x=best_threshold, color='purple', linestyle='--',
                label=f'最佳阈值 ({best_threshold:.3f})')
    plt.xlabel('阈值')
    plt.ylabel('分数')
    plt.title(f'{model_name}验证集阈值优化')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name.lower()}_validation_threshold_optimization.png')

    return best_threshold


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

    return model


# ===================================
# 加载和准备基础模型
# ===================================

def prepare_data_and_models():
    """准备数据并加载或训练基础模型"""
    # 加载数据
    train_data = pd.read_csv("../AAAA/最终版本/data/UNSW_NB15_training-set.csv")
    test_data = pd.read_csv("../AAAA/最终版本/data/UNSW_NB15_testing-set.csv")

    # 创建增强特征
    train_enhanced = create_enhanced_features(train_data)
    test_enhanced = create_enhanced_features(test_data)

    print("训练增强版基础模型...")

    X_train_full, y_train, X_test_full, y_test, feature_names_full, scaler = preprocess_data(
        train_enhanced, test_enhanced, num_features=None  # 不限制特征数量
    )

    print(f"预处理后特征数量: {X_train_full.shape[1]}")

    # 为LSTM准备数据
    X_train_lstm = reshape_for_lstm(X_train_full)
    X_test_lstm = reshape_for_lstm(X_test_full)

    # 创建验证集
    X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
        X_train_full, y_train, test_size=0.3, random_state=42, stratify=y_train
    )

    X_val_lstm_meta = reshape_for_lstm(X_val_meta)

    base_models = {}

    # 1. 增强XGBoost模型
    print("训练增强版XGBoost...")
    num_features = 30
    # 计算特征与目标变量的相关性
    correlations = np.array([np.abs(np.corrcoef(X_train_full[:, i], y_train)[0, 1])
                             for i in range(X_train_full.shape[1])])

    # 额外添加一些流量相关特征
    flow_features = []
    if feature_names_full is not None:
        flow_features = np.array([i for i, name in enumerate(feature_names_full)
                                  if 'bytes' in str(name) or 'pkts' in str(name) or 'load' in str(name)])

    # 组合相关性高的特征和流量特征
    if len(flow_features) > 0:
        feature_indices = np.unique(np.concatenate([
            np.argsort(-correlations)[:num_features],
            flow_features
        ]))
    else:
        feature_indices = np.argsort(-correlations)[:num_features]

    X_train_xgb = X_train_full[:, feature_indices]
    X_test_xgb = X_test_full[:, feature_indices]
    X_val_xgb = X_val_meta[:, feature_indices]

    print(f"XGBoost选择了{len(feature_indices)}个特征")

    # 增强配置 - 新版本XGBoost支持
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        use_label_encoder=False,  # 避免新版本警告
        eval_metric='auc'  # 新版本支持在构造函数中设置
    )

    # 使用早停机制 - 新版本支持
    xgb_model.fit(
        X_train_xgb, y_train,
        eval_set=[(X_val_xgb, y_val_meta)],
        early_stopping_rounds=20,
        verbose=False
    )

    base_models['XGBoost'] = xgb_model

    # 2. 增强LightGBM
    print("训练增强版LightGBM...")
    num_features = 40
    # 使用不同的特征选择方法
    # 结合相关性和方差选择
    variances = np.var(X_train_full, axis=0)
    var_corr_score = correlations * np.sqrt(variances)  # 同时考虑相关性和变异性
    feature_indices = np.argsort(-var_corr_score)[:num_features]

    X_train_lgb = X_train_full[:, feature_indices]
    X_test_lgb = X_test_full[:, feature_indices]
    X_val_lgb = X_val_meta[:, feature_indices]

    print(f"LightGBM选择了{len(feature_indices)}个特征")

    # 优化的LightGBM配置
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=127,  # 更合理，与max_depth=8一致
        max_depth=8,  # 降低深度
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,  # 增加正则化
        min_split_gain=0.01,  # 设置最小增益
        min_child_samples=20,  # 每个叶节点最小样本数
        random_state=42
    )

    # 简化训练过程，不使用早停机制
    lgb_model.fit(X_train_lgb, y_train)

    base_models['LightGBM'] = lgb_model

    # 3. 增强GBDT
    print("训练增强版GBDT...")
    num_features = 35
    # 特征分类
    time_features = []
    conn_features = []

    if feature_names_full is not None:
        time_features = [i for i, name in enumerate(feature_names_full)
                         if 'dur' in str(name) or 'time' in str(name)]
        conn_features = [i for i, name in enumerate(feature_names_full)
                         if 'ct_' in str(name) or 'srv' in str(name)]

    # 与高相关性特征结合
    feature_list = [np.argsort(-correlations)[:25]]  # 基于相关性的25个特征
    if len(time_features) > 0:
        feature_list.append(time_features)  # 所有时间特征
    if len(conn_features) > 0:
        feature_list.append(conn_features)  # 所有连接特征

    feature_indices = np.unique(np.concatenate(feature_list))

    X_train_gbdt = X_train_full[:, feature_indices]
    X_test_gbdt = X_test_full[:, feature_indices]
    X_val_gbdt = X_val_meta[:, feature_indices]

    print(f"GBDT选择了{len(feature_indices)}个特征")

    # 增强配置
    gbdt_model = GradientBoostingClassifier(
        n_estimators=250,
        max_depth=7,
        learning_rate=0.05,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        max_features=0.8,
        random_state=42
    )

    gbdt_model.fit(X_train_gbdt, y_train)
    base_models['GBDT'] = gbdt_model

    # 4. 增强LSTM - TensorFlow 2.x兼容
    print("训练增强版LSTM模型...")
    # 更复杂的LSTM架构
    lstm_model = Sequential([
        # 双向LSTM层
        Bidirectional(LSTM(164, return_sequences=True), input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dropout(0.4),

        Bidirectional(LSTM(96, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64)),
        Dropout(0.3),

        # 更深的全连接层
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    # 使用Adam优化器
    optimizer = Adam(learning_rate=0.001)

    lstm_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # 早停和学习率调整
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=3,
        min_lr=0.0001,
        mode='max'
    )

    # 分割验证集
    X_train_lstm_model, X_val_lstm_model, y_train_lstm, y_val_lstm = train_test_split(
        X_train_lstm, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 训练LSTM - TensorFlow 2.x写法，使用回调函数
    lstm_model.fit(
        X_train_lstm_model, y_train_lstm,
        epochs=10,  # 减少轮次，加速训练
        batch_size=128,
        validation_data=(X_val_lstm_model, y_val_lstm),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    base_models['LSTM'] = lstm_model

    # 为了集成模型，保存各个模型的特征集
    X_feature_sets = {
        'XGBoost': (X_train_xgb, X_test_xgb, X_val_xgb),
        'LightGBM': (X_train_lgb, X_test_lgb, X_val_lgb),
        'GBDT': (X_train_gbdt, X_test_gbdt, X_val_gbdt),
        'LSTM': (X_train_lstm, X_test_lstm, X_val_lstm_meta)
    }

    # 生成预测
    base_test_preds = {}
    for name, model in base_models.items():
        if name == 'LSTM':
            base_test_preds[name] = model.predict(X_feature_sets[name][1]).flatten()
        else:
            base_test_preds[name] = model.predict_proba(X_feature_sets[name][1])[:, 1]

    # 验证集预测
    base_val_preds = {}
    for name, model in base_models.items():
        if name == 'LSTM':
            base_val_preds[name] = model.predict(X_feature_sets[name][2]).flatten()
        else:
            base_val_preds[name] = model.predict_proba(X_feature_sets[name][2])[:, 1]

    return (X_train_full, y_train, X_test_full, y_test, X_train_meta, X_val_meta,
            y_train_meta, y_val_meta, X_train_lstm, X_test_lstm, X_val_lstm_meta,
            base_models, base_test_preds, base_val_preds, X_feature_sets)


# ===================================
# 集成学习框架   有bug  2025.4.10
# ===================================

def create_feature_space_partitions(X_train, y_train, n_clusters=8):
    """创建优化的特征空间分区"""
    print("创建优化的特征空间分区...")

    # 1. 先做特征降维，避免高维空间的聚类问题
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(20, X_train.shape[1]))
    X_reduced = pca.fit_transform(X_train)

    # 2. 使用更好的聚类方法 - 考虑类别分布
    from sklearn.cluster import KMeans

    # 创建阳性和阴性样本的单独聚类
    pos_idx = (y_train == 1)
    neg_idx = (y_train == 0)

    X_pos = X_reduced[pos_idx]
    X_neg = X_reduced[neg_idx]

    # 根据类别比例分配簇数
    pos_ratio = np.mean(pos_idx)
    n_pos_clusters = max(1, int(n_clusters * pos_ratio))
    n_neg_clusters = n_clusters - n_pos_clusters

    print(f"创建 {n_pos_clusters} 个正样本簇和 {n_neg_clusters} 个负样本簇")

    # 正样本和负样本各自聚类
    kmeans_pos = KMeans(n_clusters=n_pos_clusters, random_state=42)
    kmeans_neg = KMeans(n_clusters=n_neg_clusters, random_state=42)

    pos_clusters = kmeans_pos.fit_predict(X_pos)
    neg_clusters = kmeans_neg.fit_predict(X_neg)

    # 创建全局聚类标签
    all_clusters = np.zeros(len(y_train), dtype=int)
    all_clusters[pos_idx] = pos_clusters
    all_clusters[neg_idx] = neg_clusters + n_pos_clusters

    # 使用全局定义的类
    kmeans = EnhancedClusterer(pca, kmeans_pos, kmeans_neg, n_pos_clusters)

    # 分析每个簇的特性
    cluster_stats = {}
    for i in range(n_clusters):
        mask = (all_clusters == i)
        if np.sum(mask) == 0:
            continue

        cluster_stats[i] = {
            'size': np.sum(mask),
            'positive_ratio': np.mean(y_train[mask]),
            'feature_means': np.mean(X_train[mask], axis=0)
        }
        print(f"簇 {i}: 包含 {cluster_stats[i]['size']} 个样本, "
              f"攻击比例: {cluster_stats[i]['positive_ratio']:.2f}")

    return kmeans, cluster_stats


def extract_meta_features(X, base_preds):
    """增强版元特征提取"""
    # 基础模型预测概率
    model_probs = np.column_stack([base_preds[name] for name in base_preds.keys()])

    # 模型间的一致性/不一致性
    model_std = np.std(model_probs, axis=1, keepdims=True)
    mean_probs = np.mean(model_probs, axis=1, keepdims=True)
    max_deviation = np.max(np.abs(model_probs - mean_probs), axis=1, keepdims=True)

    # 样本难度特征
    sample_difficulty = 1.0 - 2.0 * np.abs(mean_probs - 0.5)

    n_models = model_probs.shape[1]
    binary_preds = (model_probs >= 0.5).astype(int)
    agreement_ratio = np.sum(binary_preds, axis=1, keepdims=True) / n_models


    model_correlations = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            # 两个模型预测之间的相似度
            sim = np.abs(model_probs[:, i] - model_probs[:, j]).reshape(-1, 1)
            model_correlations.append(sim)

    model_correlation_features = np.hstack(model_correlations) if model_correlations else np.zeros((len(X), 1))


    confidence_features = np.abs(model_probs - 0.5) * 2  # 转换为0-1范围的置信度


    max_confidence = np.max(confidence_features, axis=1, keepdims=True)
    min_confidence = np.min(confidence_features, axis=1, keepdims=True)


    # 连接所有元特征
    meta_features = np.hstack([
        model_probs,  # 基础模型预测
        model_std,  # 模型间不确定性
        max_deviation,  # 最大偏差
        sample_difficulty,  # 样本难度
        agreement_ratio,  # 新增: 预测一致性比率
        model_correlation_features,  # 新增: 模型间相关性
        confidence_features,  # 新增: 各模型置信度
        max_confidence,  # 新增: 最高置信度
        min_confidence,  # 新增: 最低置信度
    ])

    return meta_features


def train_meta_learner(X_val, y_val, base_preds_val, meta_model_type='advanced'):
    """高级元学习器训练，针对性能表现进行优化"""
    print(f"训练优化的元学习器 (类型: {meta_model_type})...")

    # 准备基础元特征
    meta_features = extract_meta_features(X_val, base_preds_val)

    if meta_model_type == 'advanced':
        # 使用更复杂的随机森林配置
        meta_learner = RandomForestClassifier(
            n_estimators=300,  # 增加树的数量
            max_depth=10,  # 增加深度
            min_samples_split=5,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',  # 处理不平衡数据
            bootstrap=True,
            random_state=42,
            n_jobs=-1  # 使用所有CPU加速
        )

        # 如果特征数量很多，可以考虑特征选择
        if meta_features.shape[1] > 20:
            from sklearn.feature_selection import SelectFromModel

            # 初始特征选择
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                threshold='median'
            )
            selector.fit(meta_features, y_val)

            # 获取选择的特征
            selected_indices = selector.get_support()
            selected_features = meta_features[:, selected_indices]
            print(f"元特征数量从 {meta_features.shape[1]} 减少到 {selected_features.shape[1]}")

            # 使用选择的特征训练
            meta_learner.fit(selected_features, y_val)

            # 创建带有特征选择的完整元学习器
            class FeatureSelectedMetaLearner:
                def __init__(self, selector, model):
                    self.selector = selector
                    self.model = model

                def predict_proba(self, X):
                    X_selected = X[:, self.selector.get_support()]
                    return self.model.predict_proba(X_selected)

                def predict(self, X):
                    X_selected = X[:, self.selector.get_support()]
                    return self.model.predict(X_selected)

            meta_learner = FeatureSelectedMetaLearner(selector, meta_learner)
        else:
            # 直接使用所有特征训练
            meta_learner.fit(meta_features, y_val)

    elif meta_model_type == 'weighted':
        # 使用加权集成多个元学习器
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier

        # 训练多个基础元学习器
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)

        rf.fit(meta_features, y_val)
        gb.fit(meta_features, y_val)
        lr.fit(meta_features, y_val)

        # 在验证集上评估以确定权重
        rf_proba = rf.predict_proba(meta_features)[:, 1]
        gb_proba = gb.predict_proba(meta_features)[:, 1]
        lr_proba = lr.predict_proba(meta_features)[:, 1]

        rf_auc = roc_auc_score(y_val, rf_proba)
        gb_auc = roc_auc_score(y_val, gb_proba)
        lr_auc = roc_auc_score(y_val, lr_proba)

        # 基于AUC的相对性能计算权重
        total_auc = rf_auc + gb_auc + lr_auc
        weights = [rf_auc / total_auc, gb_auc / total_auc, lr_auc / total_auc]

        print(f"元学习器权重: RF={weights[0]:.2f}, GB={weights[1]:.2f}, LR={weights[2]:.2f}")

        # 创建组合元学习器
        class WeightedMetaLearner:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights

            def predict_proba(self, X):
                probas = np.zeros((X.shape[0], 2))
                for i, model in enumerate(self.models):
                    probas += self.weights[i] * model.predict_proba(X)
                return probas

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        meta_learner = WeightedMetaLearner([rf, gb, lr], weights)

    else:
        # 默认使用标准随机森林
        meta_learner = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        meta_learner.fit(meta_features, y_val)

    # 评估元学习器在验证集上的性能
    if hasattr(meta_learner, 'predict'):
        y_pred = meta_learner.predict(meta_features)
    else:
        y_pred = (meta_learner.predict_proba(meta_features)[:, 1] >= 0.5).astype(int)

    accuracy = accuracy_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, meta_learner.predict_proba(meta_features)[:, 1])
    print(f"元学习器验证集性能 - 准确率: {accuracy:.4f}, AUC: {auc_score:.4f}")

    return meta_learner
def compute_dynamic_weights(X, cluster_id, base_preds, difficulty_threshold=0.2):
    """增强版动态权重计算"""

    meta_features = extract_meta_features(X, base_preds)


    model_probs = np.column_stack([base_preds[name] for name in base_preds.keys()])

    # 获取样本难度 (0 = 容易, 1 = 困难)
    mean_probs = np.mean(model_probs, axis=1)
    sample_difficulty = 1.0 - 2.0 * np.abs(mean_probs - 0.5)


    n_samples = len(X)
    n_models = len(base_preds)
    weights = np.ones((n_samples, n_models)) / n_models


    cluster_weight_bias = {
        0: [0.35, 0.30, 0.30, 0.05],  # 偏向树模型
        1: [0.45, 0.25, 0.25, 0.05],  # 强调XGBoost
        2: [0.25, 0.40, 0.30, 0.05],  # 强调LightGBM
        3: [0.25, 0.25, 0.45, 0.05],  # 强调GBDT
        4: [0.30, 0.30, 0.30, 0.10],  # 平衡权重
    }

    # 获取当前簇的权重偏好
    cluster_bias = np.array(cluster_weight_bias.get(cluster_id, [0.25, 0.25, 0.25, 0.25]))

    # 改进: 更复杂的样本类型识别
    for i in range(n_samples):
        # 极端预测样本 (所有模型都非常确定)
        all_confident = True
        for j in range(n_models):
            if model_probs[i, j] > 0.2 and model_probs[i, j] < 0.8:
                all_confident = False
                break

        if all_confident:

            for j in range(n_models):
                conf = abs(model_probs[i, j] - 0.5) * 2  # 转换为0-1置信度
                weights[i, j] = conf ** 2  # 平方强调高置信度模型

        elif sample_difficulty[i] > difficulty_threshold:  # 难样本
            for j in range(n_models):
                conf = abs(model_probs[i, j] - 0.5) * 2
                # 反转置信度: 对于难样本，过于自信的模型可能不可靠
                inv_conf = 1.0 - conf * 0.5  # 减弱反转效果
                weights[i, j] = inv_conf

        else:  # 普通样本
            weights[i] = cluster_bias.copy()

            # 额外考虑: 如果簇内预测冲突严重，减少簇偏好影响
            std_within_cluster = np.std(model_probs[i])
            if std_within_cluster > 0.25:  # 高冲突
                # 减弱簇偏好，增加独立判断
                weights[i] = weights[i] * 0.5 + 0.5 / n_models

            for j in range(n_models):
                if model_probs[i, j] > 0.9 or model_probs[i, j] < 0.1:
                    weights[i, j] *= 2.0

        # 确保权重归一化
        weights[i] = weights[i] / np.sum(weights[i])

    return weights


def ensemble_predict(X, X_lstm, base_models, base_preds, meta_learner, kmeans, mode='dynamic'):
    """使用集成模型进行预测"""
    if mode == 'simple':
        print("使用简单平均集成...")
        final_probs = np.mean(list(base_preds.values()), axis=0)

    elif mode == 'weighted':
        print("使用固定权重集成...")
        weights = [0.25, 0.28, 0.32, 0.15]  # XGB, LGB, GBDT, LSTM
        final_probs = np.zeros(len(X))
        for i, (name, preds) in enumerate(base_preds.items()):
            final_probs += weights[i] * preds

    elif mode == 'meta':
        print("使用元学习器集成...")
        meta_features = extract_meta_features(X, base_preds)
        final_probs = meta_learner.predict_proba(meta_features)[:, 1]

    elif mode == 'dynamic':
        print("使用动态权重集成...")

        clusters = kmeans.predict(X)
        final_probs = np.zeros(len(X))
        model_probs = np.column_stack([base_preds[name] for name in base_preds.keys()])

        for cluster_id in np.unique(clusters):
            cluster_idx = np.where(clusters == cluster_id)[0]

            if len(cluster_idx) == 0:
                continue

            print(f"处理簇 {cluster_id}，包含 {len(cluster_idx)} 个样本")

            # 为当前簇的样本计算动态权重
            X_cluster = X[cluster_idx]

            # 为当前簇提取基础模型预测
            cluster_preds = {name: preds[cluster_idx] for name, preds in base_preds.items()}

            # 计算动态权重 - 不使用标签信息，只使用特征和预测
            weights = compute_dynamic_weights(X_cluster, cluster_id, cluster_preds)

            # 使用动态权重组合预测
            for i, idx in enumerate(cluster_idx):
                sample_weights = weights[i]
                final_probs[idx] = np.sum(model_probs[idx] * sample_weights)

    else:
        raise ValueError(f"不支持的集成模式: {mode}")

    return final_probs


def adaptive_ensemble_predict(X, X_lstm, base_models, base_preds, meta_learner, kmeans):
    """增强版自适应模型选择集成策略 - 向量化优化版"""
    print("使用增强版自适应模型选择集成...")

    # 准备基础数据
    meta_features = extract_meta_features(X, base_preds)
    model_probs = np.column_stack([base_preds[name] for name in base_preds.keys()])
    model_names = list(base_preds.keys())

    # 批量计算样本特性指标
    mean_probs = np.mean(model_probs, axis=1)
    sample_difficulty = 1.0 - 2.0 * np.abs(mean_probs - 0.5)  # 0=简单，1=困难
    model_std = np.std(model_probs, axis=1)  # 模型间一致性，越低越一致

    # 确定样本所在特征空间区域
    clusters = kmeans.predict(X)

    # 为每个簇定义最佳专家模型 - 基于先验知识或验证集性能
    cluster_experts = {
        0: 'XGBoost',
        1: 'LightGBM',
        2: 'GBDT',
        3: 'XGBoost',
        4: 'LightGBM',
        5: 'Meta',
        6: 'Meta',
        7: 'Ensemble'
    }

    # 初始化最终概率和决策记录
    final_probs = np.zeros(len(X))
    decision_record = np.empty(len(X), dtype=object)

    # 策略1: 批量处理高一致性样本
    high_consistency_mask = (model_std < 0.05)
    final_probs[high_consistency_mask] = mean_probs[high_consistency_mask]
    decision_record[high_consistency_mask] = "高一致性-平均"

    # 策略2: 批量处理困难样本
    difficult_mask = (~high_consistency_mask) & (sample_difficulty > 0.45)
    if np.any(difficult_mask):
        difficult_indices = np.where(difficult_mask)[0]
        meta_preds = meta_learner.predict_proba(meta_features[difficult_indices])[:, 1]

        # 区分元学习器确定和不确定的预测
        uncertain_meta = (0.4 < meta_preds) & (meta_preds < 0.6)

        # 两步赋值
        final_probs[difficult_indices[uncertain_meta]] = 0.55
        final_probs[difficult_indices[~uncertain_meta]] = meta_preds[~uncertain_meta]
        decision_record[difficult_indices] = "高难度-元学习器"

    # 策略3: 处理剩余样本
    remaining_mask = (~high_consistency_mask) & (~difficult_mask)
    remaining_indices = np.where(remaining_mask)[0]

    # 按簇处理剩余样本
    for cluster_id in np.unique(clusters[remaining_indices]):
        cluster_mask = remaining_mask & (clusters == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        expert = cluster_experts.get(cluster_id, 'Ensemble')

        if expert == 'Meta':
            # 批量计算元学习器预测
            meta_preds = meta_learner.predict_proba(meta_features[cluster_indices])[:, 1]
            final_probs[cluster_indices] = meta_preds
            decision_record[cluster_indices] = f"簇{cluster_id}-元学习器"

        elif expert == 'Ensemble':
            # 批量计算加权集成
            weights = np.array([0.3, 0.3, 0.25, 0.15])
            final_probs[cluster_indices] = np.sum(model_probs[cluster_indices] * weights, axis=1)
            decision_record[cluster_indices] = f"簇{cluster_id}-加权集成"

        elif expert in model_names:
            expert_idx = model_names.index(expert)

            # 获取专家预测及置信度
            expert_probs = model_probs[cluster_indices, expert_idx]
            expert_conf = np.abs(expert_probs - 0.5) * 2

            # 区分高置信度和低置信度情况
            high_conf_mask = expert_conf > 0.7
            high_conf_indices = cluster_indices[high_conf_mask]
            low_conf_indices = cluster_indices[~high_conf_mask]

            # 高置信度情况直接使用专家预测
            final_probs[high_conf_indices] = expert_probs[high_conf_mask]
            decision_record[high_conf_indices] = f"簇{cluster_id}-高置信专家{expert}"

            # 低置信度情况需结合次优模型 - 这部分仍需逐样本处理
            for i in low_conf_indices:
                # 找出次优模型
                other_probs = np.delete(model_probs[i], expert_idx)
                other_names = [name for j, name in enumerate(model_names) if j != expert_idx]
                other_conf = np.abs(other_probs - 0.5) * 2
                best_other_idx = np.argmax(other_conf)

                # 加权组合专家和次优模型
                final_probs[i] = 0.7 * model_probs[i, expert_idx] + 0.3 * other_probs[best_other_idx]
                decision_record[i] = f"簇{cluster_id}-专家{expert}+辅助{other_names[best_other_idx]}"
        else:
            # 默认使用平均值
            final_probs[cluster_indices] = mean_probs[cluster_indices]
            decision_record[cluster_indices] = "默认-平均"

    # 统计决策类型
    decision_counts = {}
    for decision in np.unique(decision_record):
        decision_counts[decision] = np.sum(decision_record == decision)

    print("决策路径统计:")
    for decision, count in sorted(decision_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {decision}: {count} 样本 ({count / len(X) * 100:.1f}%)")

    return final_probs


def hard_voting_ensemble(X, X_lstm, base_models, base_preds, meta_learner, kmeans):
    print("使用硬投票集成策略...")

    model_probs = np.column_stack([base_preds[name] for name in base_preds.keys()])
    model_names = list(base_preds.keys())

    # 精确率和召回率的权衡
    thresholds = {
        'XGBoost': 0.45,  # XGBoost通常精确度高
        'LightGBM': 0.48,  # LightGBM整体平衡
        'GBDT': 0.48,  # GBDT类似LightGBM
        'LSTM': 0.52
    }

    model_votes = {}
    for name, probs in base_preds.items():
        threshold = thresholds.get(name, 0.5)
        model_votes[name] = (probs >= threshold).astype(int)

    # 计算每个样本的投票总数
    vote_counts = np.zeros(len(X))
    for votes in model_votes.values():
        vote_counts += votes

    # 基于投票数计算最终概率
    final_probs = np.zeros(len(X))

    # 根据投票统计计算概率
    # 4票：100% 置信度攻击
    mask_4_votes = (vote_counts == 4)
    final_probs[mask_4_votes] = 0.99

    # 3票：90% 置信度攻击
    mask_3_votes = (vote_counts == 3)
    final_probs[mask_3_votes] = 0.90

    # 2票：特殊处理 - 检查哪两个模型投了票
    mask_2_votes = (vote_counts == 2)
    for i in np.where(mask_2_votes)[0]:
        # 检查XGBoost和LightGBM（两个最强模型）
        xgb_vote = model_votes['XGBoost'][i]
        lgb_vote = model_votes['LightGBM'][i]

        if xgb_vote == 1 and lgb_vote == 1:
            # XGBoost和LightGBM一致认为是攻击
            final_probs[i] = 0.85
        else:
            # 其他两个模型组合认为是攻击
            final_probs[i] = 0.70

    # 1票：检查是哪个模型投的票
    mask_1_vote = (vote_counts == 1)
    for i in np.where(mask_1_vote)[0]:
        # 找出哪个模型投了票
        voting_model = None
        for name, votes in model_votes.items():
            if votes[i] == 1:
                voting_model = name
                break

        # 根据模型的可靠性给不同的置信度
        if voting_model == 'XGBoost':
            final_probs[i] = 0.60  # XGBoost比较可靠
        elif voting_model == 'LightGBM':
            final_probs[i] = 0.58  # LightGBM次之
        else:
            final_probs[i] = 0.55  # 其他模型再次之

    # 0票：高度确信不是攻击
    mask_0_votes = (vote_counts == 0)
    final_probs[mask_0_votes] = 0.01

    # 记录各投票类别的样本数量
    vote_stats = {
        "4票": np.sum(mask_4_votes),
        "3票": np.sum(mask_3_votes),
        "2票": np.sum(mask_2_votes),
        "1票": np.sum(mask_1_vote),
        "0票": np.sum(mask_0_votes)
    }

    print("投票统计:")
    for vote, count in vote_stats.items():
        print(f"  {vote}: {count} 样本 ({count / len(X) * 100:.1f}%)")

    return final_probs


def meta_adaptive_ensemble(X, X_lstm, base_models, base_preds, meta_learner, kmeans):
    """向量化优化版的元-自适应混合模型集成"""
    print("使用元-自适应混合模型集成...")
    start_time = time.time()

    # 准备基础数据
    meta_features = extract_meta_features(X, base_preds)
    model_probs = np.column_stack([base_preds[name] for name in base_preds.keys()])
    model_names = list(base_preds.keys())
    n_samples = len(X)
    n_models = len(model_names)

    # 计算样本特性
    mean_probs = np.mean(model_probs, axis=1)
    model_std = np.std(model_probs, axis=1)
    sample_difficulty = 1.0 - 2.0 * np.abs(mean_probs - 0.5)

    # 获取模型权重
    model_weights = {
        'XGBoost': 0.30,
        'LightGBM': 0.30,
        'GBDT': 0.25,
        'LSTM': 0.15
    }
    default_weights = np.array([model_weights.get(name, 0.25) for name in model_names])

    # 获取簇分配
    clusters = kmeans.predict(X)

    # 初始化结果数组和决策类型
    final_probs = np.zeros(n_samples)
    decision_type = np.empty(n_samples, dtype=object)

    # 计算所有模型的置信度
    confidence_scores = np.abs(model_probs - 0.5) * 2

    # 定义样本分类掩码
    high_consistency_mask = (model_std < 0.05)
    high_difficulty_mask = (~high_consistency_mask) & (sample_difficulty > 0.45)
    remaining_mask = ~(high_consistency_mask | high_difficulty_mask)

    # 1. 处理高一致性样本
    final_probs[high_consistency_mask] = mean_probs[high_consistency_mask]
    decision_type[high_consistency_mask] = "高一致性-平均"

    # 2. 处理高难度样本
    if np.any(high_difficulty_mask):
        # 获取高难度样本索引
        high_diff_indices = np.where(high_difficulty_mask)[0]

        # 计算元学习器预测 - 只计算一次
        meta_preds = meta_learner.predict_proba(meta_features)[:, 1]

        # 找出每个高难度样本的最佳模型
        best_model_indices = np.argmax(confidence_scores[high_diff_indices], axis=1)

        # 计算元学习器在高难度样本上的置信度
        meta_conf_high_diff = np.abs(meta_preds[high_diff_indices] - 0.5) * 2

        # 生成权重向量 - 只为高难度样本计算
        meta_weights = np.clip(0.7 + meta_conf_high_diff * 0.2, 0.7, 0.9)
        expert_weights = 1.0 - meta_weights

        # 应用权重
        for i, idx in enumerate(high_diff_indices):
            best_idx = best_model_indices[i]
            final_probs[idx] = (meta_weights[i] * meta_preds[idx] +
                                expert_weights[i] * model_probs[idx, best_idx])

        decision_type[high_difficulty_mask] = "高难度-元学习器"

    # 3. 处理其余样本
    if np.any(remaining_mask):
        # 簇专家映射
        cluster_experts = {
            0: 'XGBoost',
            1: 'LightGBM',
            2: 'GBDT',
            3: 'XGBoost',
            4: 'LightGBM',
        }

        # 按簇分组处理
        for cluster_id in np.unique(clusters[remaining_mask]):
            # 获取当前簇的掩码
            cluster_mask = (clusters == cluster_id) & remaining_mask
            if not np.any(cluster_mask):
                continue

            # 获取簇内样本索引
            cluster_indices = np.where(cluster_mask)[0]
            expert_name = cluster_experts.get(cluster_id, None)

            if expert_name in model_names:
                # 获取专家模型索引
                expert_idx = model_names.index(expert_name)

                # 获取专家模型在簇内样本上的置信度
                expert_conf = confidence_scores[cluster_indices, expert_idx]

                # 区分高置信度和低置信度样本
                high_conf = expert_conf > 0.7

                # 处理高置信度样本
                high_conf_indices = cluster_indices[high_conf]
                if len(high_conf_indices) > 0:
                    final_probs[high_conf_indices] = model_probs[high_conf_indices, expert_idx]
                    decision_type[high_conf_indices] = f"簇{cluster_id}-专家{expert_name}"

                # 处理低置信度样本
                low_conf_indices = cluster_indices[~high_conf]
                if len(low_conf_indices) > 0:
                    final_probs[low_conf_indices] = (0.6 * model_probs[low_conf_indices, expert_idx] +
                                                     0.4 * mean_probs[low_conf_indices])
                    decision_type[low_conf_indices] = f"簇{cluster_id}-专家结合平均"
            else:
                # 使用加权平均
                final_probs[cluster_indices] = np.sum(model_probs[cluster_indices] * default_weights, axis=1)
                decision_type[cluster_indices] = "加权平均"

    # 确保所有样本都有决策类型
    if np.any(decision_type == None):
        undefined_indices = np.where(decision_type == None)[0]
        final_probs[undefined_indices] = mean_probs[undefined_indices]
        decision_type[undefined_indices] = "未定义-使用平均值"

    # 统计决策类型
    unique_decisions, counts = np.unique(decision_type, return_counts=True)
    decision_counts = dict(zip(unique_decisions, counts))

    print("决策类型统计:")
    for decision, count in sorted(decision_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {decision}: {count} 样本 ({count / n_samples * 100:.1f}%)")

    elapsed_time = time.time() - start_time
    print(f"自适应集成完成，耗时: {elapsed_time:.2f} 秒")

    return final_probs


def meta_adaptive_hybrid_ensemble(X, X_lstm, base_models, base_preds, meta_learner, kmeans):
    """向量化优化版元-自适应混合集成"""
    print("使用元-自适应混合集成策略...")
    start_time = time.time()

    # 准备基础数据
    meta_features = extract_meta_features(X, base_preds)
    model_probs = np.column_stack([base_preds[name] for name in base_preds.keys()])
    model_names = list(base_preds.keys())
    n_samples = len(X)

    # 预计算所有样本特性
    mean_probs = np.mean(model_probs, axis=1)
    model_std = np.std(model_probs, axis=1)
    sample_difficulty = 1.0 - 2.0 * np.abs(mean_probs - 0.5)

    # 计算所有模型的置信度
    model_confidence = np.abs(model_probs - 0.5) * 2

    # 获取簇分配
    clusters = kmeans.predict(X)

    # 初始化结果数组和决策类型
    final_probs = np.zeros(n_samples)
    decision_type = np.empty(n_samples, dtype=object)

    # 创建样本分类掩码
    high_consistency = model_std < 0.05
    high_difficulty = (~high_consistency) & (sample_difficulty > 0.4)
    medium_difficulty = (~high_consistency) & (~high_difficulty) & (sample_difficulty > 0.2)
    low_difficulty = (~high_consistency) & (~high_difficulty) & (~medium_difficulty)

    # 一次性计算元学习器预测 - 避免重复计算
    meta_preds = meta_learner.predict_proba(meta_features)[:, 1]

    # 1. 处理高一致性样本
    final_probs[high_consistency] = mean_probs[high_consistency]
    decision_type[high_consistency] = "高一致性-平均值"

    # 2. 处理高难度样本
    if np.any(high_difficulty):
        # 获取高难度样本索引
        high_diff_indices = np.where(high_difficulty)[0]

        # 找出每个高难度样本的最佳模型
        best_models = np.argmax(model_confidence, axis=1)

        # 计算元学习器在高难度样本上的置信度
        meta_conf = np.abs(meta_preds[high_diff_indices] - 0.5) * 2

        # 动态权重计算 - 只为高难度样本计算
        meta_weights = np.minimum(0.8, 0.5 + meta_conf * 0.3)
        model_weights = 1.0 - meta_weights

        # 应用权重
        for i, idx in enumerate(high_diff_indices):
            best_idx = best_models[idx]
            model_name = model_names[best_idx]
            final_probs[idx] = meta_weights[i] * meta_preds[idx] + model_weights[i] * model_probs[idx, best_idx]
            decision_type[idx] = f"高难度-元学习器+{model_name}"

    # 3. 处理中等难度样本
    if np.any(medium_difficulty):
        # 定义簇专家
        cluster_experts = {
            0: 'XGBoost', 1: 'LightGBM', 2: 'GBDT', 3: 'XGBoost', 4: 'LightGBM'
        }

        # 处理各个簇
        for cluster_id in np.unique(clusters):
            # 获取当前簇的中等难度样本
            cluster_medium_mask = (clusters == cluster_id) & medium_difficulty
            if not np.any(cluster_medium_mask):
                continue

            # 获取簇内样本索引
            cluster_indices = np.where(cluster_medium_mask)[0]
            expert = cluster_experts.get(cluster_id)

            if expert in model_names:
                expert_idx = model_names.index(expert)

                # 计算专家模型在簇内样本上的置信度
                expert_scores = model_confidence[cluster_indices, expert_idx]

                # 区分高置信度和低置信度样本
                high_conf = expert_scores > 0.7

                # 处理高置信度样本
                high_conf_indices = cluster_indices[high_conf]
                if len(high_conf_indices) > 0:
                    final_probs[high_conf_indices] = (0.7 * model_probs[high_conf_indices, expert_idx] +
                                                      0.3 * mean_probs[high_conf_indices])
                    decision_type[high_conf_indices] = f"中难度-高置信专家{expert}"

                # 处理低置信度样本
                low_conf_indices = cluster_indices[~high_conf]
                if len(low_conf_indices) > 0:
                    final_probs[low_conf_indices] = (0.5 * model_probs[low_conf_indices, expert_idx] +
                                                     0.5 * mean_probs[low_conf_indices])
                    decision_type[low_conf_indices] = f"中难度-低置信专家{expert}"
            else:
                # 使用默认权重
                weights = np.array([0.3, 0.3, 0.25, 0.15])
                if len(model_names) != len(weights):
                    weights = np.ones(len(model_names)) / len(model_names)

                # 应用默认权重
                final_probs[cluster_indices] = np.sum(model_probs[cluster_indices] * weights, axis=1)
                decision_type[cluster_indices] = "中难度-加权平均"

    # 4. 处理低难度样本
    if np.any(low_difficulty):
        # 获取低难度样本索引
        low_diff_indices = np.where(low_difficulty)[0]

        # 找出每个低难度样本置信度最高的模型
        best_model_indices = np.argmax(model_confidence[low_diff_indices], axis=1)

        # 应用最佳模型预测
        for i, idx in enumerate(low_diff_indices):
            best_idx = best_model_indices[i]
            model_name = model_names[best_idx]
            final_probs[idx] = model_probs[idx, best_idx]
            decision_type[idx] = f"低难度-{model_name}"

    # 确保所有样本都有决策类型和预测值
    if np.any(decision_type == None):
        undefined_indices = np.where(decision_type == None)[0]
        final_probs[undefined_indices] = mean_probs[undefined_indices]
        decision_type[undefined_indices] = "未定义-使用平均值"

    # 统计决策类型
    unique_decisions, counts = np.unique(decision_type, return_counts=True)
    decision_counts = dict(zip(unique_decisions, counts))

    print("决策类型统计:")
    for decision, count in sorted(decision_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {decision}: {count} 样本 ({count / n_samples * 100:.1f}%)")

    elapsed_time = time.time() - start_time
    print(f"混合集成完成，耗时: {elapsed_time:.2f} 秒")

    return final_probs

def track_decision_path(sample_idx, y_true, base_models, base_preds, final_pred, X_feature_sets, threshold=0.5):
    """跟踪单个样本的决策路径"""
    # 收集所有基础模型的预测
    predictions = {}
    confidence = {}

    for name, model in base_models.items():
        # 直接使用已有的预测结果，而不是重新预测
        prob = base_preds[name][sample_idx]
        pred = 1 if prob >= threshold else 0
        predictions[name] = pred
        confidence[name] = prob

    # 计算每个模型的贡献
    contributions = {}
    final_binary = 1 if final_pred >= threshold else 0

    for name, pred in predictions.items():
        # 如果模型预测与最终预测一致，则有正贡献
        if pred == final_binary:
            contributions[name] = confidence[name] if final_binary == 1 else 1 - confidence[name]
        else:
            contributions[name] = -(confidence[name] if final_binary == 0 else 1 - confidence[name])

    # 归一化贡献
    total = sum(abs(c) for c in contributions.values())
    for name in contributions:
        contributions[name] /= total if total > 0 else 1

    # 输出解释
    explanation = {
        'true_label': y_true[sample_idx],
        'final_prediction': final_binary,
        'confidence': final_pred,
        'model_predictions': predictions,
        'model_confidence': confidence,
        'model_contributions': contributions
    }

    return explanation

# ===================================
# 评估和可视化函数
# ===================================

def evaluate_ensemble(y_test, y_pred, y_probs, base_preds):
    """评估集成模型性能，只使用提供的预测结果，不进行任何优化"""
    # 基本评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    # 与基础模型比较 - 使用固定阈值0.5，不重新优化
    base_metrics = {}
    for name, probs in base_preds.items():
        base_preds_binary = (probs >= 0.5).astype(int)
        base_metrics[name] = {
            'accuracy': accuracy_score(y_test, base_preds_binary),
            'precision': precision_score(y_test, base_preds_binary),
            'recall': recall_score(y_test, base_preds_binary),
            'f1': f1_score(y_test, base_preds_binary),
            'auc': roc_auc_score(y_test, probs)
        }

    ensemble_errors = (y_pred != y_test)
    base_errors = {name: ((p >= 0.5).astype(int) != y_test) for name, p in base_preds.items()}

    # 集成模型独特贡献 - 所有基础模型错误但集成模型正确的情况
    unique_correct = np.zeros_like(ensemble_errors, dtype=bool)
    for i in range(len(y_test)):
        if not ensemble_errors[i] and all(err[i] for err in base_errors.values()):
            unique_correct[i] = True

    return {
        'ensemble_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        },
        'base_metrics': base_metrics,
        'unique_contribution': {
            'count': np.sum(unique_correct),
            'percentage': np.mean(unique_correct) * 100,
            'indices': np.where(unique_correct)[0]
        }
    }


def visualize_ensemble_performance(results, base_preds, ensemble_probs, y_test):
    """创建集成模型性能可视化"""
    # 1. 模型性能比较柱状图
    plt.figure(figsize=(12, 6))
    models = list(results['base_metrics'].keys()) + ['Ensemble']
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    x = np.arange(len(metrics))
    width = 0.15

    for i, model in enumerate(models[:-1]):
        values = [results['base_metrics'][model][m] for m in metrics]
        plt.bar(x + (i - 2) * width, values, width, label=model)

    # 添加集成模型结果
    ensemble_values = [results['ensemble_metrics'][m] for m in metrics]
    plt.bar(x + 2 * width, ensemble_values, width, label='Ensemble', color='red')

    plt.xlabel('评估指标')
    plt.ylabel('得分')
    plt.title('集成模型与基础模型性能比较')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ensemble_performance_comparison.png')

    # 2. 模型预测分布对比
    plt.figure(figsize=(15, 10))

    for i, (name, probs) in enumerate(base_preds.items()):
        plt.subplot(3, 2, i + 1)
        plt.hist(probs[y_test == 0], bins=20, alpha=0.5, label='正常流量', color='green')
        plt.hist(probs[y_test == 1], bins=20, alpha=0.5, label='攻击流量', color='red')
        plt.title(f'{name} 预测概率分布')
        plt.xlabel('预测概率')
        plt.ylabel('样本数量')
        plt.legend()

    plt.subplot(3, 2, 5)
    plt.hist(ensemble_probs[y_test == 0], bins=20, alpha=0.5, label='正常流量', color='green')
    plt.hist(ensemble_probs[y_test == 1], bins=20, alpha=0.5, label='攻击流量', color='red')
    plt.title('集成模型预测概率分布')
    plt.xlabel('预测概率')
    plt.ylabel('样本数量')
    plt.legend()

    plt.tight_layout()
    plt.savefig('prediction_distributions.png')

    # 3. ROC曲线比较
    plt.figure(figsize=(10, 8))

    # 绘制基础模型ROC曲线
    for name, probs in base_preds.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.4f})')

    # 绘制集成模型ROC曲线
    fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
    auc_score = roc_auc_score(y_test, ensemble_probs)
    plt.plot(fpr, tpr, lw=3, label=f'Ensemble (AUC = {auc_score:.4f})', color='red')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线比较')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('ensemble_roc_comparison.png')


def visualize_decision_contributions(y_test, base_models, base_preds, ensemble_probs, threshold=0.5, num_samples=10):
    """可视化决策贡献度"""
    sample_indices = np.random.choice(len(y_test), size=num_samples, replace=False)
    contribution_matrix = np.zeros((len(sample_indices), len(base_models)))

    for i, sample_idx in enumerate(sample_indices):
        explanation = track_decision_path(
            sample_idx, y_test, base_models,
            base_preds, ensemble_probs[sample_idx], threshold
        )

        for j, model_name in enumerate(base_models.keys()):
            contribution_matrix[i, j] = explanation['model_contributions'][model_name]

        if i < 3:  # 只打印前3个样本的详细解释
            true_label = explanation['true_label']
            pred_label = explanation['final_prediction']
            confidence = explanation['confidence']

            print(f"\n样本 {sample_idx} 的决策解释:")
            print(f"真实标签: {'攻击' if true_label == 1 else '正常'}")
            print(f"预测标签: {'攻击' if pred_label == 1 else '正常'} (置信度: {confidence:.4f})")
            print("各模型贡献:")

            for model, contrib in explanation['model_contributions'].items():
                direction = "正向" if contrib > 0 else "负向"
                print(f"  {model}: {abs(contrib):.4f} ({direction}贡献)")

    # 可视化贡献矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(contribution_matrix, cmap='coolwarm', center=0,
                xticklabels=list(base_models.keys()),
                yticklabels=[f'样本 {i}' for i in sample_indices])
    plt.title('模型对样本预测的贡献度')
    plt.tight_layout()
    plt.savefig('decision_contribution_heatmap.png')


def save_ensemble_model(base_models, meta_learner, kmeans, best_threshold):
    """保存集成模型的所有组件"""
    try:
        # 创建保存目录（如果不存在）
        save_dir = './saved_models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"创建目录: {save_dir}")

        # 保存路径
        ensemble_path = os.path.join(save_dir, 'ensemble_components.pkl')

        # 只保存必要的组件
        ensemble_components = {
            'meta_learner': meta_learner,
            'kmeans': kmeans,
            'threshold': best_threshold
        }

        # 保存meta-learner和kmeans
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_components, f)
        print(f"集成模型组件已保存到: {ensemble_path}")

    except Exception as e:
        print(f"保存模型时出错: {e}")
        print("尝试使用joblib保存...")
        try:
            import joblib
            meta_path = os.path.join(save_dir, 'meta_learner.joblib')
            kmeans_path = os.path.join(save_dir, 'kmeans.joblib')
            threshold_path = os.path.join(save_dir, 'threshold.txt')

            joblib.dump(meta_learner, meta_path)
            joblib.dump(kmeans, kmeans_path)
            with open(threshold_path, 'w') as f:
                f.write(str(best_threshold))
            print(f"使用joblib保存成功，文件保存在: {save_dir}")

        except Exception as e2:
            print(f"使用joblib保存也失败: {e2}")
            print("无法保存模型，但可以继续使用当前会话中的模型")


def main():
    print("\n======== 动态多层次集成学习框架(hl0nely) ========")
    start_time = time.time()

    # 1. 数据准备和模型加载/训练
    print("\n1. 准备数据和基础模型...")
    (X_train, y_train, X_test, y_test, X_train_meta, X_val_meta, y_train_meta, y_val_meta,
     X_train_lstm, X_test_lstm, X_val_lstm_meta, base_models, base_test_preds, base_val_preds,
     X_feature_sets) = prepare_data_and_models()

    # 2. 创建特征空间分区
    print("\n2. 创建特征空间分区...")
    kmeans, cluster_stats = create_feature_space_partitions(X_train, y_train, n_clusters=5)

    # 3. 训练元学习器
    print("\n3. 训练元学习器...")
    meta_learner = train_meta_learner(X_val_meta, y_val_meta, base_val_preds, meta_model_type='advanced')

    # 4. 在验证集上执行集成预测，优化阈值
    print("\n4. 在验证集上执行集成预测，优化阈值...")

    # 原有的集成策略预测   暂存，策略有时候会偏弱  2025.4.11
    simple_val_probs = ensemble_predict(
        X_val_meta, X_val_lstm_meta, base_models, base_val_preds, meta_learner, kmeans, mode='simple'
    )

    weighted_val_probs = ensemble_predict(
        X_val_meta, X_val_lstm_meta, base_models, base_val_preds, meta_learner, kmeans, mode='weighted'
    )

    meta_val_probs = ensemble_predict(
        X_val_meta, X_val_lstm_meta, base_models, base_val_preds, meta_learner, kmeans, mode='meta'
    )

    dynamic_val_probs = ensemble_predict(
        X_val_meta, X_val_lstm_meta, base_models, base_val_preds, meta_learner, kmeans, mode='dynamic'
    )

    adaptive_val_probs = adaptive_ensemble_predict(
        X_val_meta, X_val_lstm_meta, base_models, base_val_preds, meta_learner, kmeans
    )

    # 新增集成策略预测   2025.4.13  硬投票决策作为第一要务
    hard_voting_val_probs = hard_voting_ensemble(
        X_val_meta, X_val_lstm_meta, base_models, base_val_preds, meta_learner, kmeans
    )

    meta_adaptive_val_probs = meta_adaptive_ensemble(
        X_val_meta, X_val_lstm_meta, base_models, base_val_preds, meta_learner, kmeans
    )

    # 5. 在验证集上优化阈值
    print("\n5. 优化决策阈值...")
    simple_threshold = find_optimal_threshold(y_val_meta, simple_val_probs, "Simple_Ensemble")
    weighted_threshold = find_optimal_threshold(y_val_meta, weighted_val_probs, "Weighted_Ensemble")
    meta_threshold = find_optimal_threshold(y_val_meta, meta_val_probs, "Meta_Ensemble")
    dynamic_threshold = find_optimal_threshold(y_val_meta, dynamic_val_probs, "Dynamic_Ensemble")
    adaptive_threshold = find_optimal_threshold(y_val_meta, adaptive_val_probs, "Adaptive_Ensemble")

    # 新增策略的阈值优化   2025.4.13
    hard_voting_threshold = find_optimal_threshold(y_val_meta, hard_voting_val_probs, "Hard_Voting_Ensemble")
    meta_adaptive_threshold = find_optimal_threshold(y_val_meta, meta_adaptive_val_probs, "Meta_Adaptive_Ensemble")

    # 计算并输出验证集F1分数
    hard_voting_val_preds = (hard_voting_val_probs >= hard_voting_threshold).astype(int)
    meta_adaptive_val_preds = (meta_adaptive_val_probs >= meta_adaptive_threshold).astype(int)

    hard_voting_val_f1 = f1_score(y_val_meta, hard_voting_val_preds)
    meta_adaptive_val_f1 = f1_score(y_val_meta, meta_adaptive_val_preds)

    print(f"硬投票最佳阈值: {hard_voting_threshold:.4f}, 验证集F1: {hard_voting_val_f1:.4f}")
    print(f"元-自适应最佳阈值: {meta_adaptive_threshold:.4f}, 验证集F1: {meta_adaptive_val_f1:.4f}")

    # 6. 在测试集上执行集成预测
    print("\n6. 在测试集上执行集成预测...")

    # 原有策略
    simple_probs = ensemble_predict(
        X_test, X_test_lstm, base_models, base_test_preds, meta_learner, kmeans, mode='simple'
    )
    weighted_probs = ensemble_predict(
        X_test, X_test_lstm, base_models, base_test_preds, meta_learner, kmeans, mode='weighted'
    )
    meta_probs = ensemble_predict(
        X_test, X_test_lstm, base_models, base_test_preds, meta_learner, kmeans, mode='meta'
    )
    dynamic_probs = ensemble_predict(
        X_test, X_test_lstm, base_models, base_test_preds, meta_learner, kmeans, mode='dynamic'
    )

    adaptive_probs = adaptive_ensemble_predict(
        X_test, X_test_lstm, base_models, base_test_preds, meta_learner, kmeans
    )

    # 新增策略
    hard_voting_probs = hard_voting_ensemble(
        X_test, X_test_lstm, base_models, base_test_preds, meta_learner, kmeans
    )
    meta_adaptive_probs = meta_adaptive_ensemble(
        X_test, X_test_lstm, base_models, base_test_preds, meta_learner, kmeans
    )

    # 应用在验证集上优化的阈值到测试集预测
    simple_preds = (simple_probs >= simple_threshold).astype(int)
    weighted_preds = (weighted_probs >= weighted_threshold).astype(int)
    meta_preds = (meta_probs >= meta_threshold).astype(int)
    dynamic_preds = (dynamic_probs >= dynamic_threshold).astype(int)
    adaptive_preds = (adaptive_probs >= adaptive_threshold).astype(int)
    hard_voting_preds = (hard_voting_probs >= hard_voting_threshold).astype(int)
    meta_adaptive_preds = (meta_adaptive_probs >= meta_adaptive_threshold).astype(int)

    # 7. 评估性能
    print("\n7. 评估集成模型性能...")
    simple_results = evaluate_ensemble(y_test, simple_preds, simple_probs, base_test_preds)
    weighted_results = evaluate_ensemble(y_test, weighted_preds, weighted_probs, base_test_preds)
    meta_results = evaluate_ensemble(y_test, meta_preds, meta_probs, base_test_preds)
    dynamic_results = evaluate_ensemble(y_test, dynamic_preds, dynamic_probs, base_test_preds)
    adaptive_results = evaluate_ensemble(y_test, adaptive_preds, adaptive_probs, base_test_preds)
    hard_voting_results = evaluate_ensemble(y_test, hard_voting_preds, hard_voting_probs, base_test_preds)
    meta_adaptive_results = evaluate_ensemble(y_test, meta_adaptive_preds, meta_adaptive_probs, base_test_preds)

    # 添加新的高级混合策略
    print("\n添加元-自适应混合策略...")
    # 在验证集上执行
    hybrid_val_probs = meta_adaptive_hybrid_ensemble(
        X_val_meta, X_val_lstm_meta, base_models, base_val_preds, meta_learner, kmeans
    )

    # 优化阈值
    hybrid_threshold = find_optimal_threshold(y_val_meta, hybrid_val_probs, "Hybrid_Ensemble")

    # 在测试集上执行
    hybrid_probs = meta_adaptive_hybrid_ensemble(
        X_test, X_test_lstm, base_models, base_test_preds, meta_learner, kmeans
    )

    # 应用阈值
    hybrid_preds = (hybrid_probs >= hybrid_threshold).astype(int)

    # 评估新策略
    hybrid_results = evaluate_ensemble(y_test, hybrid_preds, hybrid_probs, base_test_preds)

    # 8. 保存集成模型
    print("\n8. 保存集成模型...")
    try:
        save_ensemble_model(base_models, meta_learner, kmeans, meta_adaptive_threshold)
    except Exception as e:
        print(f"保存模型时出错: {e}")
        print("继续执行不保存模型")

    # 9. 可视化结果
    print("\n9. 生成可视化结果...")
    visualize_ensemble_performance(meta_adaptive_results, base_test_preds, meta_adaptive_probs, y_test)

    try:
        visualize_decision_contributions(y_test, base_models, base_test_preds, meta_adaptive_probs)
    except Exception as e:
        print(f"生成决策贡献可视化时出错: {e}")

    # 10. 输出性能比较总结
    print("\n======== 集成策略性能比较 ========")
    strategies = {
        'Simple Average': simple_results['ensemble_metrics'],
        'Fixed Weights': weighted_results['ensemble_metrics'],
        'Meta-Learner': meta_results['ensemble_metrics'],
        'Dynamic DMLEF': dynamic_results['ensemble_metrics'],
        'Adaptive Model': adaptive_results['ensemble_metrics'],
        'Hard Voting': hard_voting_results['ensemble_metrics'],
        'Meta-Adaptive': meta_adaptive_results['ensemble_metrics'],
        'Meta-Adaptive-Hybrid': hybrid_results['ensemble_metrics']  # 添加新策略
    }

    df_strategies = pd.DataFrame({
        '集成策略': list(strategies.keys()),
        '准确率': [strat['accuracy'] for strat in strategies.values()],
        'AUC': [strat['auc'] for strat in strategies.values()],
        '精确率': [strat['precision'] for strat in strategies.values()],
        '召回率': [strat['recall'] for strat in strategies.values()],
        'F1分数': [strat['f1'] for strat in strategies.values()]
    })

    print(df_strategies)
    df_strategies.to_csv('ensemble_strategies_comparison.csv', index=False)

    # 找出性能最好的策略
    best_strategy_idx = np.argmax([strat['f1'] for strat in strategies.values()])
    best_strategy_name = list(strategies.keys())[best_strategy_idx]

    print(f"\n最佳集成策略: {best_strategy_name}")

    # 选择最佳策略的结果作为DMLEF最终结果   动态
    if best_strategy_name == 'Simple Average':  # 基础模型预测概率直接平均
        dmlef_preds = simple_preds
        dmlef_probs = simple_probs
        dmlef_results = simple_results
    elif best_strategy_name == 'Fixed Weights':  # 预设权重加权平均（如XGBoost 0.3, LSTM 0.15）
        dmlef_preds = weighted_preds
        dmlef_probs = weighted_probs
        dmlef_results = weighted_results
    elif best_strategy_name == 'Meta-Learner':
        dmlef_preds = meta_preds
        dmlef_probs = meta_probs
        dmlef_results = meta_results
    elif best_strategy_name == 'Dynamic DMLEF':  # 基于特征空间分区的动态权重
        dmlef_preds = dynamic_preds
        dmlef_probs = dynamic_probs
        dmlef_results = dynamic_results
    elif best_strategy_name == 'Adaptive Model':
        dmlef_preds = adaptive_preds
        dmlef_probs = adaptive_probs
        dmlef_results = adaptive_results
    elif best_strategy_name == 'Hard Voting':  # 多数表决，直接硬投票
        dmlef_preds = hard_voting_preds
        dmlef_probs = hard_voting_probs
        dmlef_results = hard_voting_results
    elif best_strategy_name == 'Meta-Adaptive':
        dmlef_preds = meta_adaptive_preds
        dmlef_probs = meta_adaptive_probs
        dmlef_results = meta_adaptive_results
    elif best_strategy_name == 'Meta-Adaptive-Hybrid':  # 添加新策略处理
        dmlef_preds = hybrid_preds
        dmlef_probs = hybrid_probs
        dmlef_results = hybrid_results


    # 11. 输出基础模型与最佳集成模型比较
    print("\n======== 基础模型与DMLEF比较 ========")
    models_performance = {
        'XGBoost': dmlef_results['base_metrics']['XGBoost'],
        'LightGBM': dmlef_results['base_metrics']['LightGBM'],
        'GBDT': dmlef_results['base_metrics']['GBDT'],
        'LSTM': dmlef_results['base_metrics']['LSTM'],
        'DMLEF': dmlef_results['ensemble_metrics']
    }

    df_comparison = pd.DataFrame({
        '模型': list(models_performance.keys()),
        '准确率': [model['accuracy'] for model in models_performance.values()],
        'AUC': [model['auc'] for model in models_performance.values()],
        '精确率': [model['precision'] for model in models_performance.values()],
        '召回率': [model['recall'] for model in models_performance.values()],
        'F1分数': [model['f1'] for model in models_performance.values()]
    })

    print(df_comparison)
    df_comparison.to_csv('model_comparison.csv', index=False)

    # 12. 独特贡献分析
    unique_correct = dmlef_results['unique_contribution']['count']
    print(f"\n集成模型独特正确预测: {unique_correct} 个样本")
    print(f"占总测试集的 {dmlef_results['unique_contribution']['percentage']:.2f}%")

    # 13. 运行时间
    execution_time = time.time() - start_time
    print(f"\n总执行时间: {execution_time:.2f} 秒")


if __name__ == "__main__":
    main()