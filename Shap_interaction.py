import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import shap
import warnings
import networkx as nx

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

# --- 步骤 2: 定制的网络图绘制函数 ---
def plot_interaction_network(main_effects_df, interactions_df, output_path="LSTM_Interaction_Network.pdf"):
    """根据特征重要性和交互作用绘制网络图"""
    
    G = nx.Graph()

    # 添加节点 (特征)，其属性'importance'为主效应值
    for _, row in main_effects_df.iterrows():
        G.add_node(row['feature'], importance=row['main_effect_value'])

    # 添加边 (交互作用)，其属性'interaction'为交互作用值
    for _, row in interactions_df.iterrows():
        G.add_edge(row['feature1'], row['feature2'], interaction=row['interaction_value'])

    # 使用圆形布局
    pos = nx.circular_layout(G, scale=1.0)
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.9, 0.05], height_ratios=[0.5, 0.5])
    ax_main = fig.add_subplot(gs[:, 0])

    # --- 绘制节点 ---
    node_importance = [G.nodes[node]['importance'] for node in G.nodes()]
    max_imp_actual = max(node_importance) if node_importance else 1
    
    node_size = [(imp / max_imp_actual) * 4000 + 500 for imp in node_importance]
    norm_vimp = plt.Normalize(vmin=0, vmax=0.004)
    node_colors = [cm.coolwarm(norm_vimp(imp)) for imp in node_importance]

    nx.draw_networkx_nodes(G, pos, ax=ax_main, node_color=node_colors, node_size=node_size, alpha=0.9)

    # --- 绘制边 ---
    if G.edges():
        edge_interactions = [G.edges[u, v]['interaction'] for u, v in G.edges()]
        max_int_actual = max(edge_interactions) if edge_interactions else 1

        edge_width = [(inter / max_int_actual) * 10 + 1 for inter in edge_interactions]
        norm_vint = plt.Normalize(vmin=0, vmax=1)
        edge_colors = [cm.Greys(norm_vint(inter)) for inter in edge_interactions]

        nx.draw_networkx_edges(G, pos, ax=ax_main, width=edge_width, edge_color=edge_colors, alpha=0.8)

    # --- 绘制标签 ---
    nx.draw_networkx_labels(G, pos, ax=ax_main, font_size=16, font_weight='bold')

    # --- 格式化主图 ---
    ax_main.axis('off')
    ax_main.set_aspect('equal')
    ax_main.set_title("Feature Interaction and Main Effects", size=24)

    # --- 添加颜色条 ---
    # 交互作用 Vint
    ax_vint_cbar = fig.add_subplot(gs[0, 1])
    sm_vint = cm.ScalarMappable(cmap=cm.Greys, norm=plt.Normalize(vmin=0, vmax=1))
    cbar_vint = plt.colorbar(sm_vint, cax=ax_vint_cbar, orientation='vertical')
    ax_vint_cbar.set_title('Vint', loc='left', pad=10, fontsize=20)
    cbar_vint.ax.tick_params(labelsize=13)

    # 特征重要性 Vimp
    ax_vimp_cbar = fig.add_subplot(gs[1, 1])
    sm_vimp = cm.ScalarMappable(cmap=cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=0.004))
    cbar_vimp = plt.colorbar(sm_vimp, cax=ax_vimp_cbar, orientation='vertical')
    ax_vimp_cbar.set_title('Vimp', loc='left', pad=10, fontsize=20)
    cbar_vimp.ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=1200)
    print(f"Interaction network plot saved to {output_path}")
    plt.show()

# --- 步骤 3: 主程序 ---
def analyze_shap_network():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(code_dir)
    
    print("Step 1: Loading and preprocessing data...")
    data_file = "/srv/nfs/home/njnu_ljq/CZL_LSTM/Datacsv/combined_data.csv"
    data = load_and_preprocess_data(data_file)
    
    feature_names_for_model = ['Time', 'Stirrer', 'pH', 'DO', 'CER', 'OUR', 'RQ', 'Kla']
    target_name = 'Glu'
    
    features_raw = data[feature_names_for_model].values
    target = data[target_name].values.reshape(-1, 1)
    
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features_raw)
    
    TIME_STEPS = 5
    n_features = len(feature_names_for_model)
    X, y = create_sequences(data, scaled_features, data[target_name].values, TIME_STEPS)
    
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Data loaded and prepared successfully.")

    print("\nStep 2: Loading the pre-trained model...")
    input_size = X.shape[2]
    model = SugarConsumptionLSTM(input_size).to(device)
    model_path = 'best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model weights from '{model_path}' loaded successfully.")

    print("\nStep 3: Calculating standard SHAP values using KernelExplainer...")
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
    X_test_avg = np.mean(X_test, axis=1)
    X_test_avg_df = pd.DataFrame(X_test_avg, columns=feature_names_for_model)

    print("\nStep 4: Calculating Main Effects (Vimp) and Approximating Interactions (Vint)...")
    
    features_to_plot = [f for f in feature_names_for_model if f != 'Time']
    time_index = feature_names_for_model.index('Time')
    
    shap_values_filtered = np.delete(shap_values_avg, time_index, axis=1)
    X_test_df_filtered = X_test_avg_df.drop(columns=['Time'])
    
    main_effects = np.mean(np.abs(shap_values_filtered), axis=0)
    main_effects_df = pd.DataFrame({
        'feature': features_to_plot,
        'main_effect_value': main_effects
    })
    
    interaction_data = []
    for i, feature1 in enumerate(features_to_plot):
        for j, feature2 in enumerate(features_to_plot):
            if i < j:
                shap_vec1 = shap_values_filtered[:, i]
                feature_vec2 = X_test_df_filtered[feature2].values
                interaction_strength = np.abs(np.corrcoef(shap_vec1, feature_vec2)[0, 1])
                interaction_data.append([feature1, feature2, interaction_strength])

    interactions_df = pd.DataFrame(interaction_data, columns=['feature1', 'feature2', 'interaction_value'])

    print("Main effects and interactions calculated.")
    print("\nFeature Importances (Vimp):")
    print(main_effects_df.sort_values(by='main_effect_value', ascending=False))
    print("\nTop Feature Interactions (Vint):")
    print(interactions_df.sort_values(by='interaction_value', ascending=False).head(10))

    print("\nStep 5: Generating and saving the interaction network plot...")
    plot_interaction_network(main_effects_df, interactions_df)

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    analyze_shap_network()