import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import scipy.stats as stats
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df_train = pd.read_excel('/srv/nfs/home/njnu_ljq/CZL_LSTM/Figure/1prediction_train.xlsx')
df_test = pd.read_excel('/srv/nfs/home/njnu_ljq/CZL_LSTM/Figure/1prediction_test.xlsx')

from sklearn import metrics
y_train = df_train['Experimental value']
y_test = df_test['Experimental value']
y_pred_train = df_train['Predicted value']
y_pred_test = df_test['Predicted value']

y_pred_train_list = y_pred_train.tolist()
y_pred_test_list = y_pred_test.tolist()

# 计算训练集的指标
mse_train = metrics.mean_squared_error(y_train, y_pred_train_list)
rmse_train = np.sqrt(mse_train)
mae_train = metrics.mean_absolute_error(y_train, y_pred_train_list)
r2_train = metrics.r2_score(y_train, y_pred_train_list)

# 计算测试集的指标
mse_test = metrics.mean_squared_error(y_test, y_pred_test_list)
rmse_test = np.sqrt(mse_test)
mae_test = metrics.mean_absolute_error(y_test, y_pred_test_list)
r2_test = metrics.r2_score(y_test, y_pred_test_list)

print("训练集评价指标:")
print("均方误差 (MSE):", mse_train)
print("均方根误差 (RMSE):", rmse_train)
print("平均绝对误差 (MAE):", mae_train)
print("拟合优度 (R-squared):", r2_train)

print("\n测试集评价指标:")
print("均方误差 (MSE):", mse_test)
print("均方根误差 (RMSE):", rmse_test)
print("平均绝对误差 (MAE):", mae_test)
print("拟合优度 (R-squared):", r2_test)

# 创建一个包含训练集和测试集真实值与预测值的数据框
data_train = pd.DataFrame({
    'True': y_train,
    'Predicted': y_pred_train,
    'Data Set': 'Train'
})

data_test = pd.DataFrame({
    'True': y_test,
    'Predicted': y_pred_test,
    'Data Set': 'Test'
})

data = pd.concat([data_train, data_test])

# 自定义调色板
palette = {'Train': "#92c5da", 'Test': '#f4ba8a'}

# 创建 JointGrid 对象
plt.figure(figsize=(12, 12), dpi=1200)
g = sns.JointGrid(data=data, x="True", y="Predicted", hue="Data Set", height=10, palette=palette, space=0)

# 绘制中心的散点图
g.plot_joint(sns.scatterplot, alpha=0.5)
# 添加训练集的回归线
sns.regplot(data=data_train, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#b4d4e1', label='Train Regression Line', truncate=False)
# 添加测试集的回归线
sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#f4ba8a', label='Test Regression Line', truncate=False)
# 添加边缘的柱状图
g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)


ax = g.ax_joint

# 1. 创建五行、左对齐的文本内容
info_text = (f"Model: RNN\n"
             f"Train $R^2$ = {r2_train:.3f}\n"
             f"Train MSE = {mse_train:.3f}\n"
             f"Test $R^2$ = {r2_test:.3f}\n"
             f"Test MSE = {mse_test:.3f}")

ax.text(0.7, 0.03, info_text,
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment='bottom',
        horizontalalignment='left',
        linespacing=1.5)

# 2. 添加中心线
ax.plot([0, 6], [0, 6], c="black", alpha=0.5, linestyle='--', label='y=x')

# 3. 修改坐标轴标签和字号
g.set_axis_labels('Actual Values (g/L·h)', 'Predicted Values (g/L·h)', fontsize=20)

# 4. 设置坐标轴范围
ax.set_xlim([0, 6])
ax.set_ylim([0, 6])

# 5. 设置坐标轴刻度线向内，并显示所有边框
ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, labelsize=16)

# 6. 设置刻度，并使用FuncFormatter处理原点'0'的重叠问题
ax.set_xticks(np.arange(0, 7, 1))
ax.set_yticks(np.arange(0, 7, 1))

def y_axis_formatter(x, pos):
    """当刻度值为0时，返回空字符串"""
    if x == 0:
        return ''
    return f'{int(x)}'

ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
# 确保X轴的刻度也是整数
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))

# 7. 去除图例边框并设置字号
ax.legend(frameon=False, fontsize=16, loc='upper left')

plt.savefig("Prediction_error.pdf", format='pdf', bbox_inches='tight')
plt.show()