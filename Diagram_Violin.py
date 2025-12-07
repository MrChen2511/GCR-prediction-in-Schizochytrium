import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==============================================================================
# --- 1. 用户自定义设置 ---
# ==============================================================================

# 文件和数据列设置
file_path = '/srv/nfs/home/njnu_ljq/CZL_LSTM/Datacsv/combined_data.csv'
column_index_to_plot = 1 

# 自定义坐标轴标签
x_label1 = 'Stirrer'
y_label1 = 'Range (rpm)'

# 自定义Y轴的显示范围
y_axis_min = 0
y_axis_max = 1000

# 输出文件夹和文件名
output_directory = '/srv/nfs/home/njnu_ljq/CZL_LSTM'
output_filename = 'violin_diagram1.pdf'
full_output_path = os.path.join(output_directory, output_filename)


# ==============================================================================
# --- 2. 加载和准备数据 ---
# ==============================================================================

try:
    df = pd.read_csv(file_path)
    data_series = df.iloc[:, column_index_to_plot].dropna()
except FileNotFoundError:
    print(f"错误：无法在路径 '{file_path}' 找到文件。")
    data_series = pd.Series([])
except IndexError:
    print(f"错误：列索引 '{column_index_to_plot}' 超出范围。")
    data_series = pd.Series([])

if data_series.empty:
    print("数据为空，无法生成图表。")
else:
    # ==============================================================================
    # --- 3. 绘制图表 ---
    # ==============================================================================
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # --- a. 绘制箱形图 ---
    ax.boxplot(data_series, positions=[0.4], widths=0.12, patch_artist=True,
               boxprops=dict(facecolor="#7a538d", alpha=0.7, linewidth=2),
               medianprops=dict(color="black", linewidth=2),
               whiskerprops=dict(linewidth=2),
               capprops=dict(linewidth=2))
    
    # --- b. 绘制小提琴图 ---
    violin_parts = ax.violinplot(data_series, positions=[0.6], widths=0.22,
                                 showmeans=False, showmedians=False, showextrema=False)
    for pc in violin_parts['bodies']:
        pc.set_facecolor("#7a538d")
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(2)
    
    # --- c. 计算并添加统计注释 ---
    stats = {
        'Max': data_series.max(),
        '75%': data_series.quantile(0.75), 
        'MEAN': data_series.mean(),
        '25%': data_series.quantile(0.25), 
        'Min': data_series.min()
    }
    
    text_x_position = 0.33
    for label, value in stats.items():
        ax.axhline(y=value, color='gray', linestyle='--', linewidth=2, alpha=0.8)
        
        ax.text(text_x_position, value, f'{value:.2f} ({label}) ', 
                verticalalignment='center', 
                horizontalalignment='right',
                fontsize=30) 
    
    ax.set_xlabel(x_label1, fontsize=36) 
    ax.set_ylabel(y_label1, fontsize=36) 
    
    ax.tick_params(axis='both', direction='in', labelsize=30, width=2)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xticks([])
    
    ax.set_ylim(y_axis_min, y_axis_max)
    ax.set_xlim(0, 1) 
    
    # ==============================================================================
    # --- 4. 调整布局并保存文件 ---
    # ==============================================================================
    plt.tight_layout()
    plt.savefig(full_output_path, dpi=1200)
    
    print(f"图表已成功保存为 {full_output_path}")