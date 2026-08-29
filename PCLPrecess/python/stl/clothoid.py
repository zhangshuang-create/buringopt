import numpy as np
import matplotlib.pyplot as plt
#各种螺旋线

# -------------------------------------------------------------------
# 1. 通用数值积分函数 (数值计算 theta, x, y)
# -------------------------------------------------------------------
def generate_intrinsic_curve(kappa_func, s_span, num_points=4000):
    s = np.linspace(s_span[0], s_span[1], num_points)
    kappa = kappa_func(s)

    # 梯形数值积分: theta(s) = ∫ kappa(s) ds
    ds = np.diff(s)
    d_theta = 0.5 * (kappa[:-1] + kappa[1:]) * ds
    theta = np.zeros_like(s)
    theta[1:] = np.cumsum(d_theta)

    # 积分得到坐标: x(s) = ∫ cos(theta) ds, y(s) = ∫ sin(theta) ds
    dx = 0.5 * (np.cos(theta[:-1]) + np.cos(theta[1:])) * ds
    dy = 0.5 * (np.sin(theta[:-1]) + np.sin(theta[1:])) * ds

    x = np.zeros_like(s)
    y = np.zeros_like(s)
    x[1:] = np.cumsum(dx)
    y[1:] = np.cumsum(dy)

    return s, kappa, x, y


# -------------------------------------------------------------------
# 2. 定义 8 大经典代表曲线的曲率方程 kappa(s)
# -------------------------------------------------------------------

# 1. 标准回旋线 (Clothoid): kappa = c * s
kappa_clothoid = lambda s: 0.8 * s

# 2. 对数螺线 (Logarithmic Spiral): kappa = c / s (加微小偏移量防除以0)
kappa_log_spiral = lambda s: 1.2 / (s + 0.08)

# 3. 圆的渐开线 (Involute of Circle): kappa = c / sqrt(s)
kappa_involute = lambda s: 0.8 / np.sqrt(s + 0.04)

# 4. 正圆 (Circle): kappa = const
kappa_circle = lambda s: np.full_like(s, 1.0)

# 5. 二次高阶螺线 (Quadratic Clothoid / G3): kappa = c * s^2
kappa_g3 = lambda s: 0.35 * (s ** 2)

# 6. 渐平指数螺线 (Exponential Spiral): kappa = s * exp(-k*s)
kappa_exp = lambda s: s * np.exp(-0.55 * s)

# 7. 密卷对数积螺线 (Log-Product Spiral): kappa = s / ln(s + 1.1)
kappa_log_prod = lambda s: s / (0.45 * np.log(s + 1.1))

# 8. 周期花瓣螺线 (Sinusoidal Spiral): kappa = s / (1 + A*sin(k*s))
kappa_sinusoidal = lambda s: s / (1.0 + 0.65 * np.sin(3.5 * s))

# 曲线配置列表：(名称, 曲率函数, 弧长范围, 颜色)
all_curves = [
    ("1. 标准回旋线 (Clothoid)", kappa_clothoid, (0, 4.2), '#1f77b4'),  # 蓝色
    ("2. 对数螺线 (Log-Spiral)", kappa_log_spiral, (0, 6.0), '#e377c2'),  # 粉色
    ("3. 圆的渐开线 (Involute)", kappa_involute, (0, 7.0), '#17becf'),  # 青色
    ("4. 正圆 (Circle)", kappa_circle, (0, 6.3), '#ff7f0e'),  # 橙色
    ("5. G3二次螺线 (Quadratic)", kappa_g3, (0, 3.0), '#bcbd22'),  # 黄绿
    ("6. 渐平指数螺线 (Exponential)", kappa_exp, (0, 8.0), '#2ca02c'),  # 绿色
    ("7. 密卷对数积螺线 (Log-Product)", kappa_log_prod, (0, 3.6), '#9467bd'),  # 紫色
    ("8. 周期花瓣螺线 (Sinusoidal)", kappa_sinusoidal, (0, 5.0), '#d62728')  # 红色
]

# -------------------------------------------------------------------
# 3. 绘图 (3x3 网格：1 个大型汇总图 + 8 个独立子图)
# -------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(18, 12))

# --- 左上角 2x2 区域：总览对比主图 ---
ax_main = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=2)
ax_main.set_title("8 大经典内在几何曲线总览对比", fontsize=15, fontweight='bold')
ax_main.set_xlabel("X 坐标", fontsize=11)
ax_main.set_ylabel("Y 坐标", fontsize=11)
ax_main.grid(True, linestyle='--', alpha=0.5)
ax_main.set_aspect('equal')

# --- 8 个子图的位置配置 ---
sub_grid_positions = [
    (0, 2), (0, 3),  # 第一行右侧 2 个
    (1, 2), (1, 3),  # 第二行右侧 2 个
    (2, 0), (2, 1), (2, 2), (2, 3)  # 第三行全部 4 个
]

for i, (name, k_func, s_span, color) in enumerate(all_curves):
    s, kappa, x, y = generate_intrinsic_curve(k_func, s_span)

    # 绘制到主总览图
    ax_main.plot(x, y, label=name, color=color, linewidth=2)
    ax_main.plot(x[0], y[0], 'ko', markersize=3)

    # 绘制到各自子图
    r, c = sub_grid_positions[i]
    ax_sub = plt.subplot2grid((3, 4), (r, c))
    ax_sub.plot(x, y, color=color, linewidth=2)
    ax_sub.plot(x[0], y[0], 'ko', markersize=3)
    ax_sub.set_title(name, fontsize=10, fontweight='bold', color=color)
    ax_sub.grid(True, linestyle=':', alpha=0.6)
    ax_sub.set_aspect('equal')

ax_main.legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.show()