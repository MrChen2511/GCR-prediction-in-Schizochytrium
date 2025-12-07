import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import shap
import warnings

# --- 准备工作 ---
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# --- 步骤 1: 复用您的模型和数据处理代码 ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_preprocess_data(file_path):
    """从单个CSV文件加载并预处理"""
    try:
        combined_data = pd.read_csv(file_path, skiprows=1, header=None, 
                                 names=['Time', 'Stirrer', 'pH', 'DO', 'CER', 'OUR', 'RQ', 'Kla', 'Glu'])
        print(f"Processing {file_path}: {len(combined_data)} rows")
        for col in combined_data.columns:
            combined_data[col] = combined_data[col].astype(str).str.replace(r'[\(\)]', '', regex=True)
            combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
        for col in ['Time', 'Stirrer', 'pH', 'DO', 'CER', 'OUR', 'RQ', 'Kla', 'Glu']:
            combined_data = combined_data[combined_data[col] >= 0]
        combined_data = combined_data.dropna()
        if len(combined_data) == 0:
            raise ValueError(f"Warning: {file_path} has no valid data after cleaning")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        raise
    return combined_data

def create_sequences(data, features, target, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(features[i:(i + time_steps)])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

class SugarConsumptionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_size, output_size))
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output

# --- 步骤 2: 主程序 ---

def analyze_shap():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(code_dir)
    
    print("Step 1: Loading and preprocessing data...")
    data_file = "/srv/nfs/home/njnu_ljq/CZL_LSTM/Datacsv/combined_data.csv"
    data = load_and_preprocess_data(data_file)
    
    feature_names_for_model = ['Time', 'Stirrer', 'pH', 'DO', 'CER', 'OUR', 'RQ', 'Kla']
    target_name = 'Glu'
    
    print(f"Loading model with {len(feature_names_for_model)} features to match 'best_model.pth': {feature_names_for_model}")
    
    features_raw = data[feature_names_for_model].values
    target = data[target_name].values.reshape(-1, 1)
    
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features_raw)
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(target)
    
    TIME_STEPS = 5
    n_features = len(feature_names_for_model)
    X, y = create_sequences(data, scaled_features, scaled_target, TIME_STEPS)
    
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Data loaded and prepared successfully.")

    print("\nStep 2: Loading the pre-trained model...")
    input_size = X.shape[2]
    model = SugarConsumptionLSTM(input_size).to(device)
    
    model_path = 'best_model.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model weights from '{model_path}' loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    print("\nStep 3: Calculating SHAP values using KernelExplainer...")
    
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    background_summary = shap.kmeans(X_train_2d, 20)

    def model_predict_wrapper(x_2d):
        x_3d = x_2d.reshape(-1, TIME_STEPS, n_features)
        x_tensor = torch.FloatTensor(x_3d).to(device)
        with torch.no_grad():
            output = model(x_tensor)
        return output.cpu().numpy()

    explainer = shap.KernelExplainer(model_predict_wrapper, background_summary)
    
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    # 为整个测试集计算SHAP值
    print(f"\nCalculating SHAP values for the ENTIRE test set ({len(X_test_2d)} samples)...")
    print("!!! WARNING: This will be significantly slower. Please be patient. !!!")
    shap_values = explainer.shap_values(X_test_2d)
    print("SHAP values calculated successfully.")
    
    if isinstance(shap_values, list):
        shap_values_2d = shap_values[0]
    else:
        shap_values_2d = shap_values

    shap_values_3d = shap_values_2d.reshape(-1, TIME_STEPS, n_features)
    shap_values_avg = np.mean(shap_values_3d, axis=1)

    # 使用完整的X_test进行颜色映射
    X_test_df_full = pd.DataFrame(X_test.mean(axis=1), columns=feature_names_for_model)

    print("\nStep 4: Filtering out 'Time' feature from results for plotting...")
    
    time_index = feature_names_for_model.index('Time')
    shap_values_filtered = np.delete(shap_values_avg, time_index, axis=1)
    X_test_df_filtered = X_test_df_full.drop(columns=['Time'])

    print("\nStep 5: Generating SHAP plots with final custom styles...")
    
    # 绘制蜂巢图与条形图的组合图
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200)
    
    shap.summary_plot(shap_values_filtered, X_test_df_filtered, plot_type="dot", show=False, color_bar=True)
    
    ax1 = plt.gca()
    ax1.set_xlim(-0.04, 0.04)
    
    cb = fig.axes[-1]
    cb.set_ylabel("Feature Value", fontsize=12)
    cb.tick_params(labelsize=10)
    
    ax2 = ax1.twiny()
    
    plt.sca(ax2)
    shap.summary_plot(shap_values_filtered, X_test_df_filtered, plot_type="bar", show=False)
    
    for bar in ax2.patches:
        bar.set_alpha(0.2)
        
    # 最终的坐标轴样式控制

    ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12)
    ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)
    ax1.set_ylabel('Features', fontsize=12)
    ax2.set_ylabel('')

    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()

    ax2.set_xlim(0, 0.004)
    
    ax1.tick_params(axis='x', labelsize=10, direction='in', top=False)
    ax1.tick_params(axis='y', labelsize=10, direction='in', right=False, length=0)
    
    ax2.tick_params(axis='x', labelsize=10, direction='in', bottom=False)
    ax2.tick_params(axis='y', labelsize=10, direction='in', left=False, length=0)
    
    linewidth = 1.0
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
    
    plt.tight_layout()
    plt.savefig("SHAP_combined_plot_full_test_set.pdf", format='pdf', bbox_inches='tight')
    print("Saved: SHAP_combined_plot_full_test_set.pdf")
    
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    analyze_shap()