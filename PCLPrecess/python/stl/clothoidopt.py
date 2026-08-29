import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root_scalar

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
#四段过度螺旋线+圆弧+超椭圆

class SuperellipseConvexFilletApp:
    def __init__(self):
        self.a = 200.0          # 半宽 (对称轴 X=200)
        self.b = 300.0          # 半高 (Y=300)
        self.n_val = 0.60       # 超椭圆指数
        self.R_target = 50.0    # 倒圆半径下限 R (mm)
        self.v_robot = 500.0    # 机器人末端线速度 (mm/s)


state = SuperellipseConvexFilletApp()


# 超椭圆显式几何计算（端点奇异性保护）
def superellipse_geom_exact(x, a, b, n):
    eps_boundary = 1.2
    u = np.clip(np.abs(x - a) / a, 1e-6, 1.0 - (eps_boundary / a))
    inside = np.maximum(0.0, 1.0 - u ** n)
    y = b * (inside ** (1.0 / n))

    sgn = np.sign(x - a)
    dy = - (b / a) * sgn * (u ** (n - 1.0)) * (inside ** (1.0 / n - 1.0))

    term1 = (n - 1.0) * (u ** (n - 2.0)) * (inside ** (1.0 / n - 1.0))
    term2 = - (n - 1.0) * (u ** (2.0 * n - 2.0)) * (inside ** (1.0 / n - 2.0))
    ddy = - (b / (a ** 2)) * (term1 + term2)

    kappa = np.abs(ddy) / np.maximum((1.0 + dy ** 2) ** 1.5, 1e-9)

    # 消除极端边界飞线
    dist_to_ends = np.minimum(x, 2.0 * a - x)
    decay = np.clip(dist_to_ends / 6.0, 0.0, 1.0)
    kappa = kappa * (decay ** 1.2)

    return y, dy, kappa


# 求解严格全凸外延缓和螺线与顶部圆弧的 G2 闭合
def solve_strictly_convex_fillet(a, b, n, R, n_pts=1000):
    k_target = 1.0 / R

    # 1. 提前截断点 x1 选取（提前退出超椭圆，留出充分的外凸螺线过渡区间）
    # 当 n 越小或 R 越大时，提前截取的幅度越大
    x1 = np.clip(a - max(1.6 * R, 40.0) * (1.1 - 0.3 * min(n, 1.5)), 10.0, a - 10.0)
    y1, dy1, k1 = superellipse_geom_exact(x1, a, b, n)
    theta1 = np.arctan(dy1)  # 起步切向角 (theta1 > 0)

    # 2. 构造严格单调递增曲率模型 (全凸外弯)
    # kappa(u) = k1 + (k_target - k1) * u^2, u in [0, 1]
    # 通过单参数寻根求解螺线长度 Ls，使得终端法线圆心 Xc 严格等于 a
    def center_x_error(Ls):
        u_sample = np.linspace(0, 1, 80)
        s_sample = u_sample * Ls
        k_sample = k1 + (k_target - k1) * (u_sample ** 2)
        # 航向角单调减小（顺时针向右转弯，严格外凸）
        th_sample = theta1 - cumulative_trapezoid(k_sample, s_sample, initial=0.0)
        dx_end = np.trapz(np.cos(th_sample), s_sample)
        th_end = th_sample[-1]
        x_end = x1 + dx_end
        # 终端圆心 X 坐标: Xc = x_end + R * sin(th_end)
        xc = x_end + R * np.sin(th_end)
        return xc - a

    try:
        sol = root_scalar(center_x_error, bracket=[1.0, 3.5 * a], method='brentq')
        Ls_opt = sol.root
    except Exception:
        Ls_opt = max(5.0, (a - x1) * 1.1)

    # 3. 生成高精度左侧过渡螺线 (纯外凸)
    n_sp = n_pts // 5
    u_dense = np.linspace(0, 1, n_sp)
    s_dense = u_dense * Ls_opt
    k_sp_l = k1 + (k_target - k1) * (u_dense ** 2)
    th_sp_l = theta1 - cumulative_trapezoid(k_sp_l, s_dense, initial=0.0)

    dx_dense = cumulative_trapezoid(np.cos(th_sp_l), s_dense, initial=0.0)
    dy_dense = cumulative_trapezoid(np.sin(th_sp_l), s_dense, initial=0.0)

    x_sp_l = x1 + dx_dense
    y_sp_l = y1 + dy_dense

    x2 = x_sp_l[-1]
    y2 = y_sp_l[-1]
    theta2 = th_sp_l[-1]

    # 4. 生成顶部圆弧段（严格定曲率外凸圆弧）
    Yc = y2 - R * np.cos(theta2)
    phi_l = np.pi / 2.0 + theta2
    phi_r = np.pi / 2.0 - theta2
    phi_arr = np.linspace(phi_l, phi_r, n_pts // 5)
    x_arc = a + R * np.cos(phi_arr)
    y_arc = Yc + R * np.sin(phi_arr)
    k_arc = np.full_like(x_arc, k_target)

    # 5. 生成左侧超椭圆保留段 [0, x1]
    x_se_l = np.linspace(0.0, x1, n_pts // 5)
    y_se_l, _, k_se_l = superellipse_geom_exact(x_se_l, a, b, n)

    # 6. 对称生成右侧螺线与右侧超椭圆
    x_sp_r = 2.0 * a - x_sp_l[::-1]
    y_sp_r = y_sp_l[::-1]
    k_sp_r = k_sp_l[::-1]

    x_se_r = 2.0 * a - x_se_l[::-1]
    y_se_r = y_se_l[::-1]
    k_se_r = k_se_l[::-1]

    # 7. 全局五段 G2 连续拼接
    full_x = np.concatenate([x_se_l[:-1], x_sp_l[:-1], x_arc[:-1], x_sp_r[:-1], x_se_r])
    full_y = np.concatenate([y_se_l[:-1], y_sp_l[:-1], y_arc[:-1], y_sp_r[:-1], y_se_r])
    full_k = np.concatenate([k_se_l[:-1], k_sp_l[:-1], k_arc[:-1], k_sp_r[:-1], k_se_r])

    segments = {
        'se_l': (x_se_l, y_se_l),
        'sp_l': (x_sp_l, y_sp_l),
        'arc': (x_arc, y_arc),
        'sp_r': (x_sp_r, y_sp_r),
        'se_r': (x_se_r, y_se_r),
    }

    key_points = {
        'P1_l': (x1, y1),
        'P2_l': (x2, y2),
        'P2_r': (2.0 * a - x2, y2),
        'P1_r': (2.0 * a - x1, y1),
        'Yc': Yc
    }

    return (full_x, full_y), segments, key_points, full_k


# 动力学计算
def compute_dynamics(full_x, full_y, full_k, v_robot):
    full_curve = np.column_stack([full_x, full_y])
    seg_lens = np.linalg.norm(np.diff(full_curve, axis=0), axis=1)
    s_full = np.concatenate([[0], np.cumsum(seg_lens)])

    a_n = (v_robot ** 2) * full_k
    dt = s_full / v_robot
    jerk_n = np.gradient(a_n, dt)
    radius = np.where(full_k > 1e-6, 1.0 / full_k, 1e6)

    return a_n, jerk_n, radius


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

    (full_x, full_y), segs, pts, kappa = solve_strictly_convex_fillet(
        state.a, state.b, state.n_val, state.R_target
    )
    a_n, jerk_n, radius = compute_dynamics(full_x, full_y, kappa, state.v_robot)

    # 原始完整超椭圆基准
    t_full = np.linspace(np.pi, 0, 400)
    se_x = state.a + state.a * np.sign(np.cos(t_full)) * (np.abs(np.cos(t_full)) ** (2.0 / state.n_val))
    se_y = state.b * np.sign(np.sin(t_full)) * (np.abs(np.sin(t_full)) ** (2.0 / state.n_val))

    target_k = 1.0 / state.R_target
    target_an = (state.v_robot ** 2) * target_k
    Yc = pts['Yc']

    # ---------------- 1. 几何轨迹图（清晰分段高亮） ----------------
    axs[0, 0].plot(se_x, se_y, 'm:', lw=1.5, alpha=0.45)

    # 蓝色：超椭圆保留段；亮绿/橙色：纯外凸过渡螺线；红色：顶部外凸圆弧
    axs[0, 0].plot(segs['se_l'][0], segs['se_l'][1], 'blue', lw=2.5)
    axs[0, 0].plot(segs['se_r'][0], segs['se_r'][1], 'blue', lw=2.5)
    axs[0, 0].plot(segs['sp_l'][0], segs['sp_l'][1], 'lime', lw=3.5)
    axs[0, 0].plot(segs['sp_r'][0], segs['sp_r'][1], 'orange', lw=3.5)
    axs[0, 0].plot(segs['arc'][0], segs['arc'][1], 'red', lw=3.0)

    # 绘制参考圆与特征交接点
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    full_circle_x = state.a + state.R_target * np.cos(theta_circle)
    full_circle_y = Yc + state.R_target * np.sin(theta_circle)
    axs[0, 0].plot(full_circle_x, full_circle_y, 'gray', linestyle='--', alpha=0.35)

    axs[0, 0].plot(state.a, Yc, 'ro', markersize=6)
    axs[0, 0].plot([pts['P1_l'][0], pts['P1_r'][0]], [pts['P1_l'][1], pts['P1_r'][1]], 'bs', markersize=6)
    axs[0, 0].plot([pts['P2_l'][0], pts['P2_r'][0]], [pts['P2_l'][1], pts['P2_r'][1]], 'g^', markersize=8)

    axs[0, 0].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)
    axs[0, 0].set_xlim(-15, 415)
    geom_max_y = max(np.max(full_y), state.b, Yc + state.R_target)
    axs[0, 0].set_ylim(-15, geom_max_y * 1.1)
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].set_title(f'四段纯外凸平滑轨迹 (蓝:超椭圆, 绿/橙:凸螺线, 红:圆弧)', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('X 坐标 (mm)')
    axs[0, 0].set_ylabel('Y 坐标 (mm)')
    axs[0, 0].grid(True, linestyle=':')

    # ---------------- 2. 曲率分布图 (严格单调外凸连续) ----------------
    axs[0, 1].plot(full_x, kappa, 'navy', lw=2.5)
    axs[0, 1].axhline(target_k, color='r', linestyle='--')
    axs[0, 1].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)
    axs[0, 1].plot([pts['P2_l'][0], pts['P2_r'][0]], [target_k, target_k], 'g^', markersize=8)

    k_upper = max(np.max(kappa), target_k) * 1.25
    axs[0, 1].set_xlim(-15, 415)
    axs[0, 1].set_ylim(-k_upper * 0.02, k_upper)
    axs[0, 1].set_title(r'曲率分布 $\kappa(X)$ (G2 严格平滑无突变)', fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel('X 坐标 (mm)')
    axs[0, 1].set_ylabel(r'曲率 $\kappa$ (1/mm)')
    axs[0, 1].grid(True, linestyle=':')

    # ---------------- 3. 曲率半径分布图 ----------------
    valid_radius = radius[radius < 1e5]
    r_scale_max = max(np.percentile(valid_radius, 95) * 1.25, state.R_target * 3.0, 50.0) if len(valid_radius) > 0 else state.R_target * 5.0

    axs[0, 2].plot(full_x, radius, 'purple', lw=2.5)
    axs[0, 2].axhline(state.R_target, color='r', linestyle='--')
    axs[0, 2].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)
    axs[0, 2].plot([pts['P2_l'][0], pts['P2_r'][0]], [state.R_target, state.R_target], 'g^', markersize=8)

    axs[0, 2].set_xlim(-15, 415)
    axs[0, 2].set_ylim(0, r_scale_max)
    axs[0, 2].set_title(f'曲率半径分布 $R(X)$ (mm)', fontsize=11, fontweight='bold')
    axs[0, 2].set_xlabel('X 坐标 (mm)')
    axs[0, 2].set_ylabel('曲率半径 $R$ (mm)')
    axs[0, 2].grid(True, linestyle=':')

    # ---------------- 4. 法向加速度分布图 a_n(X) ----------------
    axs[1, 0].plot(full_x, a_n, 'darkblue', lw=2.5)
    axs[1, 0].axhline(target_an, color='r', linestyle='--')
    axs[1, 0].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)
    axs[1, 0].plot([pts['P2_l'][0], pts['P2_r'][0]], [target_an, target_an], 'g^', markersize=8)

    an_upper = max(np.max(a_n), target_an) * 1.25
    axs[1, 0].set_xlim(-15, 415)
    axs[1, 0].set_ylim(-an_upper * 0.02, an_upper)
    axs[1, 0].set_title(f'法向加速度 $a_n(X)$ (v={state.v_robot:.0f} mm/s)', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel('X 坐标 (mm)')
    axs[1, 0].set_ylabel(r'加速度 $a_n$ ($\mathrm{mm/s^2}$)')
    axs[1, 0].grid(True, linestyle=':')

    # ---------------- 5. 法向加加速度 Jerk(X) ----------------
    axs[1, 1].plot(full_x, jerk_n, 'darkgreen', lw=2.2)
    axs[1, 1].axhline(0, color='gray', linestyle=':', alpha=0.6)
    axs[1, 1].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)

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
        f"【纯外凸缓和过渡动力学监控】\n\n"
        f"• 超椭圆指数 n:    {state.n_val:.2f}\n"
        f"• 约束半径 R:      {state.R_target:.1f} mm\n"
        f"• 提前截断点 X1:   {pts['P1_l'][0]:.1f} mm\n"
        f"• 圆弧接入点 X2:   {pts['P2_l'][0]:.1f} mm\n\n"
        f"• 顶点曲率 κ_max:   {np.max(kappa):.4f} 1/mm\n"
        f"• 最大法向加速度:    {np.max(a_n):.1f} mm/s²\n"
        f"• 绿色/橙色过渡段:  纯外凸单调增弯螺线 (无反折)\n"
        f"• 红色顶部段:       严格定曲率圆弧"
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