import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- 配置区域 ---
INPUT_CSV = r'D:\yanjiushen\波浪能预测\WavePowerPrediction-master\data_15分钟叠加功率.csv'

N_PAST_DAYS = 3
N_FUTURE_DAYS = 1
POINTS_PER_DAY = 24
# --- 配置结束 ---
import numpy as np
from metrics.visualization_metrics import visualization
# 加载 .npy 文件
generated_data = np.load('Z_train.npy')

# 打印数组内容
print(generated_data)
def create_full_sequences(data, n_past_steps, n_future_steps):
    """
    创建完整序列样本：每个样本包含过去 + 未来两个部分。
    例如：输入过去3天（288个点）+ 未来1天（96个点） → 共384个时间步。
    """
    Z = []
    for i in range(len(data) - n_past_steps - n_future_steps + 1):
        past = data[i: i + n_past_steps]
        future = data[i + n_past_steps: i + n_past_steps + n_future_steps]
        full_seq = np.concatenate([past, future], axis=0)
        Z.append(full_seq)
    return np.array(Z)

try:
    print(f"正在加载数据: {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    power_data = df.iloc[:, 0].values.reshape(-1, 1)
    print("数据加载成功！")

    print("正在进行Min-Max归一化...")
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled_data = scaler.fit_transform(power_data)
    #joblib.dump(scaler, 'power_scaler.pkl')
    print("归一化完成！")

    # 计算步长
    past_steps = N_PAST_DAYS * POINTS_PER_DAY
    future_steps = N_FUTURE_DAYS * POINTS_PER_DAY

    print(f"正在创建完整时间序列样本（过去{past_steps}点 + 未来{future_steps}点）...")
    Z = create_full_sequences( power_data , past_steps, future_steps)
    print("样本创建成功！")

    # 重新reshape为天数 × 每天96点
    Z = Z.reshape((Z.shape[0],POINTS_PER_DAY ,N_PAST_DAYS + N_FUTURE_DAYS ))
    print("数据已Reshape为 (样本数, 天数, 每天点数) 结构。")

    # 分割训练集和测试集

    Z_train= Z[:]
    print("数据已分割为训练集和测试集。")

    # 保存
    np.save('Z_train.npy', Z_train)

    print("已保存为 Z_train.npy")

    print("\n--- 数据准备完成 ---")
    print(f"训练集形状: {Z_train.shape}")

    print("每个样本包含过去与未来完整时间段，可直接用于 TimeGAN。")

except FileNotFoundError:
    print(f"错误：找不到文件 {INPUT_CSV}")

except Exception as e:
    print(f"未知错误: {e}")
