import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from scipy.optimize import root_scalar

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

#超椭圆过渡到圆弧
class SuperellipseFilletApp:
    def __init__(self):
        self.a = 200.0          # 半宽 (对称轴 X=200)
        self.b = 300.0          # 半高 (Y=300)
        self.n_val = 0.60       # 超椭圆指数
        self.R_target = 50.0    # 倒圆半径下限 R (mm)
        self.v_robot = 500.0    # 机器人末端线速度 (mm/s)


state = SuperellipseFilletApp()


# 超椭圆显式几何计算（包含端点奇异性截断与平滑钳位）
def superellipse_geom_exact(x, a, b, n):
    # 距离原点/终点的边缘死区距离 (mm)
    eps_boundary = 1.5
    u = np.clip(np.abs(x - a) / a, 1e-6, 1.0 - (eps_boundary / a))
    inside = np.maximum(0.0, 1.0 - u ** n)
    y = b * (inside ** (1.0 / n))

    sgn = np.sign(x - a)
    dy = - (b / a) * sgn * (u ** (n - 1.0)) * (inside ** (1.0 / n - 1.0))

    term1 = (n - 1.0) * (u ** (n - 2.0)) * (inside ** (1.0 / n - 1.0))
    term2 = - (n - 1.0) * (u ** (2.0 * n - 2.0)) * (inside ** (1.0 / n - 2.0))
    ddy = - (b / (a ** 2)) * (term1 + term2)

    kappa = np.abs(ddy) / np.maximum((1.0 + dy ** 2) ** 1.5, 1e-9)

    # 针对 n < 1 边缘端点断崖的平滑截断衰减（保留真实形态，消除数值奇异飞线）
    dist_to_ends = np.minimum(x, 2.0 * a - x)
    clamp_zone = 6.0  # 边缘 6mm 区域平滑截断
    decay = np.clip(dist_to_ends / clamp_zone, 0.0, 1.0)
    kappa = kappa * (decay ** 1.2)

    return y, dy, kappa


# 智能双侧内切判定
def solve_tangent_circle(a, b, n, R):
    need_fillet = False

    def objective(x_t):
        y_t, dy_t, _ = superellipse_geom_exact(x_t, a, b, n)
        if abs(dy_t) < 1e-5:
            return -R
        Yc = y_t - (a - x_t) / dy_t
        dist = np.hypot(x_t - a, y_t - Yc)
        return dist - R

    try:
        sol = root_scalar(objective, bracket=[2.0, a - 0.5], method='brentq')
        x_t_left = sol.root
        y_t_left, dy_t_left, _ = superellipse_geom_exact(x_t_left, a, b, n)
        Yc = y_t_left - (a - x_t_left) / dy_t_left
        need_fillet = True
    except Exception:
        x_t_left = a
        y_t_left = b
        Yc = b - R
        need_fillet = False

    x_t_right = 2.0 * a - x_t_left
    return x_t_left, y_t_left, x_t_right, Yc, need_fillet


# 生成自适应轨迹、曲率、曲率半径及动力学参数
def generate_fillet_trajectory(a, b, n, R, v_robot, n_pts=900):
    x_tl, y_tl, x_tr, Yc, need_fillet = solve_tangent_circle(a, b, n, R)

    if need_fillet and (a - x_tl) > 0.5:
        # 工况 A：尖峰超标，插入恒定圆弧
        x_left = np.linspace(0.0, x_tl, n_pts // 3)
        y_left, _, k_left = superellipse_geom_exact(x_left, a, b, n)

        phi_l = np.arctan2(y_tl - Yc, x_tl - a)
        phi_r = np.arctan2(y_tl - Yc, x_tr - a)
        phi_arr = np.linspace(phi_l, phi_r, n_pts // 3)
        x_arc = a + R * np.cos(phi_arr)
        y_arc = Yc + R * np.sin(phi_arr)
        k_arc = np.full_like(x_arc, 1.0 / R)

        x_right = np.linspace(x_tr, 2.0 * a, n_pts // 3)
        y_right, _, k_right = superellipse_geom_exact(x_right, a, b, n)

        curve_x = np.concatenate([x_left[:-1], x_arc[:-1], x_right])
        curve_y = np.concatenate([y_left[:-1], y_arc[:-1], y_right])
        curve_k = np.concatenate([k_left[:-1], k_arc[:-1], k_right])
    else:
        # 工况 B：顶点曲率本身已达标，保留原始超椭圆
        curve_x = np.linspace(0.0, 2.0 * a, n_pts)
        curve_y, _, curve_k = superellipse_geom_exact(curve_x, a, b, n)

    # 沿弧长积分求解物理时间与动力学参数
    full_curve = np.column_stack([curve_x, curve_y])
    seg_lens = np.linalg.norm(np.diff(full_curve, axis=0), axis=1)
    s_full = np.concatenate([[0], np.cumsum(seg_lens)])

    a_n = (v_robot ** 2) * curve_k              # 法向加速度 a_n = v^2 * kappa
    dt = s_full / v_robot                       # 真实时间基准
    jerk_n = np.gradient(a_n, dt)               # 法向加加速度 Jerk = da_n / dt
    radius = np.where(curve_k > 1e-6, 1.0 / curve_k, 1e6)

    return (curve_x, curve_y), (x_tl, y_tl, x_tr, Yc), curve_k, a_n, jerk_n, radius, need_fillet


# =========================================================================
# 可视化布局与交互控件 (2 x 3 全参数监控)
# =========================================================================
fig, axs = plt.subplots(2, 3, figsize=(18, 9))
plt.subplots_adjust(bottom=0.14, top=0.94, left=0.06, right=0.96, wspace=0.25, hspace=0.32)

ax_slider_n = fig.add_axes([0.15, 0.04, 0.28, 0.03])
slider_n = Slider(ax_slider_n, '超椭圆指数 n: ', 0.35, 2.5, valinit=state.n_val, valstep=0.05)

ax_textbox_r = fig.add_axes([0.58, 0.04, 0.15, 0.035])
text_box_r = TextBox(ax_textbox_r, '内切圆半径 R (mm): ', initial=str(state.R_target))


def update_view():
    for row in axs:
        for ax in row:
            ax.cla()

    (curve_x, curve_y), (x_tl, y_tl, x_tr, Yc), kappa, a_n, jerk_n, radius, need_fillet = generate_fillet_trajectory(
        state.a, state.b, state.n_val, state.R_target, state.v_robot
    )

    # 原始完整超椭圆背景
    t_full = np.linspace(np.pi, 0, 400)
    se_x = state.a + state.a * np.sign(np.cos(t_full)) * (np.abs(np.cos(t_full)) ** (2.0 / state.n_val))
    se_y = state.b * np.sign(np.sin(t_full)) * (np.abs(np.sin(t_full)) ** (2.0 / state.n_val))

    target_k = 1.0 / state.R_target
    target_an = (state.v_robot ** 2) * target_k

    # ---------------- 1. 几何轨迹图 ----------------
    axs[0, 0].plot(se_x, se_y, 'm:', lw=1.5, alpha=0.5)
    axs[0, 0].plot(curve_x, curve_y, 'b-', lw=2.5)

    theta_circle = np.linspace(0, 2 * np.pi, 200)
    full_circle_x = state.a + state.R_target * np.cos(theta_circle)
    full_circle_y = Yc + state.R_target * np.sin(theta_circle)
    axs[0, 0].plot(full_circle_x, full_circle_y, 'gray', linestyle='--', alpha=0.4)

    if need_fillet and (state.a - x_tl) > 0.5:
        axs[0, 0].plot(state.a, Yc, 'ro', markersize=6)
        axs[0, 0].plot([x_tl, x_tr], [y_tl, y_tl], 'g^', markersize=9)
    else:
        axs[0, 0].plot(state.a, state.b, 'm*', markersize=11)

    axs[0, 0].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)
    axs[0, 0].set_xlim(-15, 415)
    geom_max_y = max(np.max(curve_y), state.b, Yc + state.R_target)
    axs[0, 0].set_ylim(-15, geom_max_y * 1.1)
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].set_title(f'超椭圆加工轨迹 (n={state.n_val:.2f}, R={state.R_target:.1f}mm)', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('X 坐标 (mm)')
    axs[0, 0].set_ylabel('Y 坐标 (mm)')
    axs[0, 0].grid(True, linestyle=':')

    # ---------------- 2. 曲率分布图 (已消除两端断崖飞线) ----------------
    axs[0, 1].plot(curve_x, kappa, 'navy', lw=2.5)
    axs[0, 1].axhline(target_k, color='r', linestyle='--')
    axs[0, 1].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)

    if need_fillet and (state.a - x_tl) > 0.5:
        axs[0, 1].plot([x_tl, x_tr], [target_k, target_k], 'g^', markersize=8)

    k_upper = max(np.max(kappa), target_k) * 1.25
    axs[0, 1].set_xlim(-15, 415)
    axs[0, 1].set_ylim(-k_upper * 0.02, k_upper)
    axs[0, 1].set_title(r'曲率分布 $\kappa(X)$ (1/mm)', fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel('X 坐标 (mm)')
    axs[0, 1].set_ylabel(r'曲率 $\kappa$ (1/mm)')
    axs[0, 1].grid(True, linestyle=':')

    # ---------------- 3. 曲率半径分布图 ----------------
    valid_radius = radius[radius < 1e5]
    if len(valid_radius) > 0:
        r_scale_max = max(np.percentile(valid_radius, 95) * 1.25, state.R_target * 3.0, 50.0)
    else:
        r_scale_max = state.R_target * 5.0

    axs[0, 2].plot(curve_x, radius, 'purple', lw=2.5)
    axs[0, 2].axhline(state.R_target, color='r', linestyle='--')
    axs[0, 2].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)

    if need_fillet and (state.a - x_tl) > 0.5:
        axs[0, 2].plot([x_tl, x_tr], [state.R_target, state.R_target], 'g^', markersize=8)

    axs[0, 2].set_xlim(-15, 415)
    axs[0, 2].set_ylim(0, r_scale_max)
    axs[0, 2].set_title(f'曲率半径分布 $R(X)$ (mm)', fontsize=11, fontweight='bold')
    axs[0, 2].set_xlabel('X 坐标 (mm)')
    axs[0, 2].set_ylabel('曲率半径 $R$ (mm)')
    axs[0, 2].grid(True, linestyle=':')

    # ---------------- 4. 法向加速度分布图 a_n(X) (已消除两端尖峰) ----------------
    axs[1, 0].plot(curve_x, a_n, 'darkblue', lw=2.5)
    axs[1, 0].axhline(target_an, color='r', linestyle='--')
    axs[1, 0].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)

    if need_fillet and (state.a - x_tl) > 0.5:
        axs[1, 0].plot([x_tl, x_tr], [target_an, target_an], 'g^', markersize=8)

    an_upper = max(np.max(a_n), target_an) * 1.25
    axs[1, 0].set_xlim(-15, 415)
    axs[1, 0].set_ylim(-an_upper * 0.02, an_upper)
    axs[1, 0].set_title(f'法向加速度 $a_n(X)$ (v={state.v_robot:.0f} mm/s)', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel('X 坐标 (mm)')
    axs[1, 0].set_ylabel(r'加速度 $a_n$ ($\mathrm{mm/s^2}$)')
    axs[1, 0].grid(True, linestyle=':')

    # ---------------- 5. 法向加加速度分布图 Jerk(X) ----------------
    axs[1, 1].plot(curve_x, jerk_n, 'darkgreen', lw=2.2)
    axs[1, 1].axhline(0, color='gray', linestyle=':', alpha=0.6)
    axs[1, 1].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)

    # 忽略端点极窄区后自适应 Jerk 坐标上限
    valid_j = np.abs(jerk_n[int(len(jerk_n)*0.02) : int(len(jerk_n)*0.98)])
    max_j = max(np.max(valid_j), 10.0) if len(valid_j) > 0 else 100.0

    axs[1, 1].set_xlim(-15, 415)
    axs[1, 1].set_ylim(-max_j * 1.35, max_j * 1.35)
    axs[1, 1].set_title('法向加加速度 Jerk ($j_n(X)$)', fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel('X 坐标 (mm)')
    axs[1, 1].set_ylabel(r'Jerk ($\mathrm{mm/s^3}$)')
    axs[1, 1].grid(True, linestyle=':')

    # ---------------- 6. 状态栏与物理指标监控 ----------------
    axs[1, 2].axis('off')
    status_text = (
        f"【实时动力学与几何参数】\n\n"
        f"• 超椭圆指数 n:  {state.n_val:.2f}\n"
        f"• 约束半径 R:    {state.R_target:.1f} mm\n"
        f"• 执行线速度 v:   {state.v_robot:.0f} mm/s\n\n"
        f"• 顶点曲率 κ_max: {np.max(kappa):.4f} 1/mm\n"
        f"• 最大法向加速度:  {np.max(a_n):.1f} mm/s²\n"
        f"• 圆心坐标 (X, Y): ({state.a:.1f}, {Yc:.1f})\n"
        f"• 双侧切点间距:   {abs(x_tr - x_tl):.1f} mm"
    )
    axs[1, 2].text(0.1, 0.5, status_text, fontsize=11, va='center', family='sans-serif',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='whitesmoke', edgecolor='lightgray'))

    fig.canvas.draw_idle()


def on_slider_change(val):
    state.n_val = val
    update_view()


def on_text_submit(text):
    try:
        val = float(text)
        if val > 0:
            state.R_target = val
            update_view()
    except ValueError:
        pass


slider_n.on_changed(on_slider_change)
text_box_r.on_submit(on_text_submit)

update_view()
plt.show()