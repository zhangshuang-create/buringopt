import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# 设置绘图字体（支持中文显示）
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 1. 基础参数与坐标轴定义
# =========================================================
H = 20.0  # 顶点高度固定为 20mm
init_angle_deg = 53.13  # 初始夹角 θ ≈ 53.13° (对应初始 a = 10mm)
x = np.linspace(-35, 35, 1000)  # X轴采样范围 (稍微加宽以便观察大夹角)


# 根据高度 H 和顶角 θ (度) 计算底边半宽 a
def get_a_from_angle(angle_deg):
    angle_rad = np.radians(angle_deg)
    return H * np.tan(angle_rad / 2.0)


# =========================================================
# 2. 五种函数及其导数计算函数
# =========================================================
def compute_all_curves(angle_deg):
    a = get_a_from_angle(angle_deg)

    # 0. 理想三角形
    y_tri = np.maximum(0, H * (1 - np.abs(x) / a))

    # 1. 超高斯函数 (p=4)
    y1 = H * np.exp(-((x / a) ** 4))
    dy1 = -4 * H * (x**3 / a**4) * np.exp(-((x / a) ** 4))
    ddy1 = H * np.exp(-((x / a) ** 4)) * (
        16 * (x**6 / a**8) - 12 * (x**2 / a**4)
    )

    # 2. 超柯西分式 (p=4)
    u = x / a
    denom = 1 + u**4
    y2 = H / denom
    dy2 = -4 * H * (u**3 / a) / (denom**2)
    ddy2 = (-12 * H * (u**2 / a**2) * denom + 32 * H * (u**6 / a**2)) / (
        denom**3
    )

    # 3. 高阶平滑多项式
    y3 = np.zeros_like(x)
    dy3 = np.zeros_like(x)
    ddy3 = np.zeros_like(x)
    mask3 = np.abs(x) <= a
    u3 = x[mask3] / a
    y3[mask3] = H * (1 - u3**4) ** 2
    dy3[mask3] = -8 * (H / a) * (u3**3) * (1 - u3**4)
    ddy3[mask3] = (8 * H / a**2) * (u3**2) * (7 * u3**4 - 3)

    # 4. 三角函数 / 升余弦峰
    y4 = np.zeros_like(x)
    dy4 = np.zeros_like(x)
    ddy4 = np.zeros_like(x)
    mask4 = np.abs(x) <= a
    y4[mask4] = H * 0.5 * (1 + np.cos(np.pi * x[mask4] / a))
    dy4[mask4] = -0.5 * np.pi * (H / a) * np.sin(np.pi * x[mask4] / a)
    ddy4[mask4] = -0.5 * (np.pi**2) * (H / a**2) * np.cos(np.pi * x[mask4] / a)

    # 5. 普通抛物线
    y5 = np.zeros_like(x)
    dy5 = np.zeros_like(x)
    ddy5 = np.zeros_like(x)
    mask5 = np.abs(x) <= a
    y5[mask5] = H * (1 - (x[mask5] / a) ** 2)
    dy5[mask5] = -2 * H * x[mask5] / (a**2)
    ddy5[mask5] = -2 * H / (a**2)

    return (
        a,
        y_tri,
        (y1, dy1, ddy1),
        (y2, dy2, ddy2),
        (y3, dy3, ddy3),
        (y4, dy4, ddy4),
        (y5, dy5, ddy5),
    )


# =========================================================
# 3. 初始图像绘制
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
# 留出底部控件区域
plt.subplots_adjust(bottom=0.13, hspace=0.25)

(
    a_init,
    y_tri_init,
    c1_init,
    c2_init,
    c3_init,
    c4_init,
    c5_init,
) = compute_all_curves(init_angle_deg)

# --- 图 1: 轨迹对比 y(x) ---
(line_tri,) = axes[0].plot(
    x,
    y_tri_init,
    "k--",
    linewidth=1.5,
    alpha=0.6,
    label="0. 理想三角形 (C0 阶角)",
)
(line_y1,) = axes[0].plot(
    x,
    c1_init[0],
    "r-",
    linewidth=2,
    label="1. 超高斯函数 (Super-Gaussian) [y''(0)=0]",
)
(line_y2,) = axes[0].plot(
    x,
    c2_init[0],
    "g-",
    linewidth=2,
    label="2. 超柯西分式 (Super-Cauchy) [y''(0)=0]",
)
(line_y3,) = axes[0].plot(
    x, c3_init[0], "b-", linewidth=2, label="3. 高阶平滑多项式 [y''(0)=0]"
)
(line_y4,) = axes[0].plot(
    x,
    c4_init[0],
    "m-.",
    linewidth=2,
    label="4. 三角余弦 (Raised Cosine) [y''(0)≠0]",
)
(line_y5,) = axes[0].plot(
    x, c5_init[0], "c:", linewidth=2.5, label="5. 普通抛物线 (Parabola) [y''(0)≠0]"
)

axes[0].set_ylabel("轨迹位置 y (mm)", fontsize=11)
axes[0].set_title(
    f"五种平滑函数与三角形拟合对比 (当前夹角 θ = {init_angle_deg:.1f}°, 底边半宽 a = {a_init:.2f}mm)",
    fontsize=12,
    fontweight="bold",
)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(fontsize=9, loc="upper right")

# --- 图 2: 斜率 y'(x) ---
(line_dy1,) = axes[1].plot(x, c1_init[1], "r-", linewidth=2, label="超高斯 y'")
(line_dy2,) = axes[1].plot(x, c2_init[1], "g-", linewidth=2, label="超柯西 y'")
(line_dy3,) = axes[1].plot(
    x, c3_init[1], "b-", linewidth=2, label="高阶多项式 y'"
)
(line_dy4,) = axes[1].plot(
    x, c4_init[1], "m-.", linewidth=2, label="三角余弦 y'"
)
(line_dy5,) = axes[1].plot(x, c5_init[1], "c:", linewidth=2.5, label="抛物线 y'")
axes[1].axhline(0, color="black", linestyle="--", alpha=0.4)
axes[1].set_ylabel("斜率 / 一阶导数 y'", fontsize=11)
axes[1].set_title("斜率变化 (接近顶点 x=0 时斜率均单调减小至 0)", fontsize=11)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(fontsize=9, loc="upper right")

# --- 图 3: 加速度 y''(x) ---
(line_ddy1,) = axes[2].plot(
    x, c1_init[2], "r-", linewidth=2, label="1. 超高斯 y'' (顶点 y''=0)"
)
(line_ddy2,) = axes[2].plot(
    x, c2_init[2], "g-", linewidth=2, label="2. 超柯西 y'' (顶点 y''=0)"
)
(line_ddy3,) = axes[2].plot(
    x, c3_init[2], "b-", linewidth=2, label="3. 高阶多项式 y'' (顶点 y''=0)"
)
(line_ddy4,) = axes[2].plot(
    x, c4_init[2], "m-.", linewidth=2, label="4. 三角余弦 y''"
)
(line_ddy5,) = axes[2].plot(
    x, c5_init[2], "c:", linewidth=2.5, label="5. 普通抛物线 y''"
)
axes[2].axhline(0, color="black", linestyle="--", alpha=0.4)
scatter_zero = axes[2].scatter([0], [0], color="red", s=70, zorder=5)

axes[2].set_xlabel("位置 x (mm)", fontsize=11)
axes[2].set_ylabel("加速度 / 二阶导数 y''", fontsize=11)
axes[2].set_title(
    "向心加速度对比 (前三种顶点加速度严格为 0，后两种顶点加速度不为 0)",
    fontsize=11,
)
axes[2].grid(True, linestyle=":", alpha=0.6)
axes[2].legend(fontsize=9, loc="lower right")

# =========================================================
# 4. 创建交互滑块 (Slider)
# =========================================================
ax_angle = plt.axes([0.20, 0.03, 0.60, 0.035], facecolor="lightgoldenrodyellow")
slider_angle = Slider(
    ax=ax_angle,
    label="三角形顶角 θ (°)",
    valmin=15.0,  # 最小夹角 15° (更尖锐)
    valmax=110.0,  # 最大夹角 110° (更平缓)
    valinit=init_angle_deg,
    valfmt="%.1f°",
)


# =========================================================
# 5. 滑块拖动更新回调函数
# =========================================================
def update(val):
    angle_deg = slider_angle.val
    (
        a,
        y_tri,
        (y1, dy1, ddy1),
        (y2, dy2, ddy2),
        (y3, dy3, ddy3),
        (y4, dy4, ddy4),
        (y5, dy5, ddy5),
    ) = compute_all_curves(angle_deg)

    # 1. 更新图 1 y(x) 数据
    line_tri.set_ydata(y_tri)
    line_y1.set_ydata(y1)
    line_y2.set_ydata(y2)
    line_y3.set_ydata(y3)
    line_y4.set_ydata(y4)
    line_y5.set_ydata(y5)
    axes[0].set_title(
        f"五种平滑函数与三角形拟合对比 (当前夹角 θ = {angle_deg:.1f}°, 底边半宽 a = {a:.2f}mm)",
        fontsize=12,
        fontweight="bold",
    )

    # 2. 更新图 2 y'(x) 数据
    line_dy1.set_ydata(dy1)
    line_dy2.set_ydata(dy2)
    line_dy3.set_ydata(dy3)
    line_dy4.set_ydata(dy4)
    line_dy5.set_ydata(dy5)

    # 3. 更新图 3 y''(x) 数据
    line_ddy1.set_ydata(ddy1)
    line_ddy2.set_ydata(ddy2)
    line_ddy3.set_ydata(ddy3)
    line_ddy4.set_ydata(ddy4)
    line_ddy5.set_ydata(ddy5)

    # 更新坐标轴数据范围自适应
    idx_zero = len(x) // 2
    axes[2].set_title(
        f"向心加速度对比 [顶点 y''(0): 前三种=0, 余弦={ddy4[idx_zero]:.2f}, 抛物线={ddy5[idx_zero]:.2f}]",
        fontsize=11,
    )

    fig.canvas.draw_idle()  # 重新绘制图像


# 绑定滑块更新事件
slider_angle.on_changed(update)

plt.show()