import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge

# 文件和数据列设置
file_path = '/srv/nfs/home/njnu_ljq/CZL_LSTM/Datacsv/combined_data.csv'
# 读取CSV文件
df = pd.read_csv(file_path)

# 计算相关系数矩阵
corr = df.corr()

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8), dpi=1200)
cmap = plt.cm.RdBu_r
norm = plt.Normalize(vmin=-1, vmax=1)

# 循环绘制饼图（下三角部分）和数值（上三角部分）
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        if i > j:  # 下三角部分，绘制饼图
            coeff = corr.iloc[i, j]  # 相关系数
            abs_coeff = np.abs(coeff)  # 相关系数的绝对值

            # 设置饼图的起始角度和结束角度（有色部分表示相关性大小）
            start_angle = 90  # 起始角度
            end_angle = 90 + abs_coeff * 360  # 结束角度

            # 计算中心点位置
            x, y = i, j

            # 添加扇形（饼图效果）
            wedge = Wedge(center=(x, y), r=0.4, theta1=start_angle, theta2=end_angle,
                          facecolor=cmap(norm(coeff)), edgecolor='black', alpha=0.75)
            ax.add_patch(wedge)

            # 填充背景（无色部分）
            bg_wedge = Wedge(center=(x, y), r=0.4, theta1=end_angle, theta2=start_angle + 360,
                             facecolor='white', edgecolor='black', alpha=0.5)
            ax.add_patch(bg_wedge)
        elif i < j:  # 上三角部分，显示数值
            coeff = corr.iloc[i, j]
            color = cmap(norm(coeff))  # 数值的颜色基于相关系数
            ax.text(i, j, f'{coeff:.2f}', ha='center', va='center', color=color, fontsize=14)
        elif i == j:  # --- 对角线部分 ---
            ax.scatter(i, j, s=1, color='white')
            color_for_1 = cmap(norm(1.0)) 
            ax.text(i, j, '1', ha='center', va='center', color=color_for_1, fontsize=14)

# 设置坐标轴标签
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=14)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns, fontsize=14)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 仅用于显示颜色条
cbar = fig.colorbar(sm, ax=ax, label='Correlation Coefficient')
cbar.set_label('Correlation Coefficient', fontsize=14)
cbar.ax.tick_params(labelsize=14)

# 添加标题和布局调整
plt.tight_layout()
plt.savefig("Spearman.pdf", format='pdf', bbox_inches='tight')
plt.show()
