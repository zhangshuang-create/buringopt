import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from scipy.integrate import cumulative_trapezoid
from scipy.special import comb
from scipy.optimize import root_scalar
#螺旋线和贝塞尔的过渡比较
# 字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# =========================================================================
# 1. 算法一：7次贝塞尔 C3 连续平滑过渡 (数值寻根锁定曲率上限)
# =========================================================================
def eval_bezier_c3_properties(L_trans, P_corner, e1, e2, v_linear, n_pts=600):
    w = np.array([0.0, 0.22, 0.48, 0.78])
    P0 = P_corner - e1 * L_trans
    P1 = P0 + e1 * (L_trans * w[1])
    P2 = P0 + e1 * (L_trans * w[2])
    P3 = P0 + e1 * (L_trans * w[3])

    P7 = P_corner + e2 * L_trans
    P6 = P7 - e2 * (L_trans * w[1])
    P5 = P7 - e2 * (L_trans * w[2])
    P4 = P7 - e2 * (L_trans * w[3])
    ctrl_pts = np.array([P0, P1, P2, P3, P4, P5, P6, P7])

    u = np.linspace(0, 1, n_pts)
    n = 7
    B = np.zeros((len(u), n + 1))
    for i in range(n + 1):
        B[:, i] = comb(n, i) * (u ** i) * ((1 - u) ** (n - i))
    curve = B @ ctrl_pts

    seg_lens = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s_arc = np.concatenate([[0], np.cumsum(seg_lens)])

    dx_ds = np.gradient(curve[:, 0], s_arc)
    dy_ds = np.gradient(curve[:, 1], s_arc)
    d2x_ds2 = np.gradient(dx_ds, s_arc)
    d2y_ds2 = np.gradient(dy_ds, s_arc)
    kappa = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / ((dx_ds ** 2 + dy_ds ** 2) ** 1.5)

    return curve, ctrl_pts, s_arc, kappa


def generate_c3_bezier_exact_mm(P_in, P_corner, P_out, kappa_max_target, v_linear):
    v1 = P_corner - P_in
    v2 = P_out - P_corner
    d1 = np.linalg.norm(v1)
    d2 = np.linalg.norm(v2)
    e1 = v1 / d1
    e2 = v2 / d2

    def objective(L_val):
        _, _, _, k_arr = eval_bezier_c3_properties(L_val, P_corner, e1, e2, v_linear, n_pts=300)
        return np.max(k_arr) - kappa_max_target

    try:
        sol = root_scalar(objective, bracket=[1.0, min(d1 * 0.95, d2 * 0.95)], method='brentq')
        L_trans_opt = sol.root
    except Exception:
        L_trans_opt = min(d1 * 0.45, d2 * 0.45)

    curve, ctrl_pts, s_arc, kappa = eval_bezier_c3_properties(
        L_trans_opt, P_corner, e1, e2, v_linear, n_pts=600
    )

    a_n = (v_linear ** 2) * kappa
    dt = s_arc / v_linear
    jerk_n = np.gradient(a_n, dt)
    radius = np.where(kappa > 1e-6, 1.0 / kappa, np.inf)

    return curve, ctrl_pts, kappa, a_n, jerk_n, radius


# =========================================================================
# 2. 算法二：G3 三次螺线分段合成
# =========================================================================
def solve_g3_half_spiral_mm(theta_in, target_k_max):
    delta_theta = theta_in
    L = 2.0 * delta_theta / target_k_max

    n_pts = 300
    s = np.linspace(0, L, n_pts)
    u = s / L

    kappa = target_k_max * (3.0 * (u ** 2) - 2.0 * (u ** 3))
    theta = theta_in - cumulative_trapezoid(kappa, s, initial=0.0)

    dx = cumulative_trapezoid(np.cos(theta), s, initial=0.0)
    dy = cumulative_trapezoid(np.sin(theta), s, initial=0.0)
    return s, kappa, dx, dy, L


def generate_g3_composite_transition_mm(P_in, P_corner, P_out, target_k_max, v_robot):
    v_in = P_corner - P_in
    theta_in = np.arctan2(v_in[1], v_in[0])

    s_l, k_l, dx_l, dy_l, L_l = solve_g3_half_spiral_mm(theta_in, target_k_max)

    t_tangent = dx_l[-1] / np.cos(theta_in)
    e1 = v_in / np.linalg.norm(v_in)
    P_start_l = P_corner - t_tangent * e1

    x_left = P_start_l[0] + dx_l
    y_left = P_start_l[1] + dy_l
    x_right = 2.0 * P_corner[0] - x_left[::-1]
    y_right = y_left[::-1]
    k_right = k_l[::-1]

    curve_x = np.concatenate([x_left[:-1], x_right])
    curve_y = np.concatenate([y_left[:-1], y_right])
    curve_kappa = np.concatenate([k_l[:-1], k_right])

    s_full = np.linspace(0, 2 * L_l, len(curve_x))
    a_n = (v_robot ** 2) * curve_kappa
    dt = s_full / v_robot
    jerk_n = np.gradient(a_n, dt)
    radius = np.where(curve_kappa > 1e-6, 1.0 / curve_kappa, np.inf)

    return (x_left, y_left), (x_right, y_right), (curve_x, curve_y), curve_kappa, a_n, jerk_n, radius


# =========================================================================
# 3. 初始参数与画布创建
# =========================================================================
P_in = np.array([0.0, 0.0])
P_corner = np.array([200.0, 300.0])
P_out = np.array([400.0, 0.0])

R_min_init = 20.0
v_robot = 500.0  # 恒定线速度 500 mm/s
a_ellip, b_ellip = 200.0, 300.0
theta_param = np.linspace(0, np.pi, 400)

fig = plt.figure(figsize=(21, 10))
gs = fig.add_gridspec(2, 4, left=0.04, right=0.98, top=0.94, bottom=0.10, wspace=0.25, hspace=0.35)

axs = np.empty((2, 4), dtype=object)
for i in range(2):
    for j in range(4):
        axs[i, j] = fig.add_subplot(gs[i, j])

# 文本输入框
ax_textbox = fig.add_axes([0.38, 0.02, 0.24, 0.035])
text_box = TextBox(ax_textbox, '设置最小曲率半径 R_min (mm): ', initial=str(R_min_init))


def draw_all(R_min_target):
    for i in range(2):
        for j in range(4):
            axs[i, j].cla()

    target_k_max = 1.0 / max(R_min_target, 0.1)

    # 重新计算数据
    curve_bez, ctrl_pts_bez, kappa_bez, a_n_bez, jerk_bez, r_bez = generate_c3_bezier_exact_mm(
        P_in, P_corner, P_out, target_k_max, v_robot
    )
    left_sp, right_sp, composite_sp, kappa_sp, a_n_sp, jerk_sp, r_sp = generate_g3_composite_transition_mm(
        P_in, P_corner, P_out, target_k_max, v_robot
    )

    # 统一横坐标（以全局实际 X 坐标为准）
    x_bez = curve_bez[:, 0]
    x_sp = composite_sp[0]

    # 统一 X / Y 轴显示区间
    xlim_global = (-10.0, 410.0)
    ylim_geom = (-10.0, 330.0)
    ylim_kappa = (-0.001, target_k_max * 1.25)
    max_jerk_val = max(np.max(np.abs(jerk_bez)), np.max(np.abs(jerk_sp)), 1.0)
    ylim_jerk = (-max_jerk_val * 1.2, max_jerk_val * 1.2)
    ylim_radius = (0.0, R_min_target * 5.0)

    # ----------------- 第一行：7次贝塞尔 C3 -----------------
    axs[0, 0].plot([P_in[0], P_corner[0], P_out[0]], [P_in[1], P_corner[1], P_out[1]], 'r--', lw=1.5, label='原始尖角')
    axs[0, 0].plot(ctrl_pts_bez[:, 0], ctrl_pts_bez[:, 1], 'yo:', alpha=0.6, label='贝塞尔控制点')
    axs[0, 0].plot(x_bez, curve_bez[:, 1], 'b-', lw=2.5, label='C3 轨迹')
    axs[0, 0].axvline(200.0, color='gray', linestyle='-.', alpha=0.5)
    axs[0, 0].set_xlim(xlim_global)
    axs[0, 0].set_ylim(ylim_geom)
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].set_title('【方案一】7次贝塞尔 C3 几何平滑过渡', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('X 坐标 (mm)')
    axs[0, 0].set_ylabel('Y 坐标 (mm)')
    axs[0, 0].legend(loc='lower center', fontsize=8)
    axs[0, 0].grid(True, linestyle=':')

    axs[0, 1].plot(x_bez, kappa_bez, 'navy', lw=2, label=r'$\kappa(X)$')
    axs[0, 1].axhline(target_k_max, color='r', linestyle='--', label=f'上限 ({target_k_max:.4f})')
    axs[0, 1].axvline(200.0, color='gray', linestyle='-.', alpha=0.5, label='顶点 (X=200)')
    axs[0, 1].set_xlim(xlim_global)
    axs[0, 1].set_ylim(ylim_kappa)
    axs[0, 1].set_title(r'曲率变化 $\kappa(X)$ (1/mm)', fontsize=11)
    axs[0, 1].set_xlabel('X 坐标 (mm)')
    axs[0, 1].set_ylabel(r'曲率 $\kappa$ (1/mm)')
    axs[0, 1].legend(loc='upper right', fontsize=8)
    axs[0, 1].grid(True, linestyle=':')

    axs[0, 2].plot(x_bez, jerk_bez, 'darkgreen', lw=2, label='法向 Jerk')
    axs[0, 2].axhline(0, color='gray', linestyle=':', alpha=0.6)
    axs[0, 2].axvline(200.0, color='gray', linestyle='-.', alpha=0.5)
    axs[0, 2].set_xlim(xlim_global)
    axs[0, 2].set_ylim(ylim_jerk)
    axs[0, 2].set_title(r'法向加加速度 Jerk ($\mathrm{mm/s^3}$)', fontsize=11)
    axs[0, 2].set_xlabel('X 坐标 (mm)')
    axs[0, 2].set_ylabel(r'Jerk ($\mathrm{mm/s^3}$)')
    axs[0, 2].legend(loc='upper right', fontsize=8)
    axs[0, 2].grid(True, linestyle=':')

    axs[0, 3].plot(x_bez, r_bez, 'purple', lw=2, label=r'$R(X)$')
    axs[0, 3].axhline(R_min_target, color='r', linestyle='--', label=f'R_min ({R_min_target:.0f} mm)')
    axs[0, 3].axvline(200.0, color='gray', linestyle='-.', alpha=0.5)
    axs[0, 3].set_xlim(xlim_global)
    axs[0, 3].set_ylim(ylim_radius)
    axs[0, 3].set_title('曲率半径分布 $R(X)$ (mm)', fontsize=11)
    axs[0, 3].set_xlabel('X 坐标 (mm)')
    axs[0, 3].set_ylabel('曲率半径 $R$ (mm)')
    axs[0, 3].legend(loc='upper right', fontsize=8)
    axs[0, 3].grid(True, linestyle=':')

    # ----------------- 第二行：G3 三次螺线 -----------------
    for n_val, color, ls, alpha_val in [(1.0, 'gray', '--', 0.6), (0.7, 'm', ':', 0.4), (0.5, 'purple', ':', 0.3)]:
        cos_t, sin_t = np.cos(theta_param), np.sin(theta_param)
        x_se = 200.0 + a_ellip * np.sign(cos_t) * (np.abs(cos_t) ** (2.0 / n_val))
        y_se = b_ellip * np.sign(sin_t) * (np.abs(sin_t) ** (2.0 / n_val))
        mask = y_se >= 0
        axs[1, 0].plot(x_se[mask], y_se[mask], color=color, linestyle=ls, alpha=alpha_val,
                       label=f'超椭圆 (n={n_val})' if n_val == 1.0 else None)

    axs[1, 0].plot([P_in[0], P_corner[0], P_out[0]], [P_in[1], P_corner[1], P_out[1]], 'r--', lw=1.5, label='原始尖角')
    axs[1, 0].plot(left_sp[0], left_sp[1], 'c-', lw=2.8, alpha=0.9, label=r'左支 ($\theta \to 0^{\circ}$)')
    axs[1, 0].plot(right_sp[0], right_sp[1], 'orange', lw=2.8, alpha=0.9, label=r'右支')
    axs[1, 0].plot(x_sp, composite_sp[1], 'b:', lw=1.8, label='合成倒圆')

    top_x = x_sp[len(x_sp) // 2]
    top_y = composite_sp[1][len(composite_sp[1]) // 2]
    axs[1, 0].axvline(200.0, color='gray', linestyle='-.', alpha=0.6)
    axs[1, 0].plot(top_x, top_y, 'm*', markersize=11, label=r'顶点 (水平相切)')
    axs[1, 0].annotate('', xy=(top_x + 25.0, top_y), xytext=(top_x - 25.0, top_y),
                       arrowprops=dict(arrowstyle="->", color='m', lw=2))

    axs[1, 0].set_xlim(xlim_global)
    axs[1, 0].set_ylim(ylim_geom)
    axs[1, 0].set_aspect('equal', adjustable='box')
    axs[1, 0].set_title('【方案二】G3 三次螺线分段合成 (顶点水平平滑倒圆)', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel('X 坐标 (mm)')
    axs[1, 0].set_ylabel('Y 坐标 (mm)')
    axs[1, 0].legend(loc='lower center', fontsize=8)
    axs[1, 0].grid(True, linestyle=':')

    axs[1, 1].plot(x_sp, kappa_sp, 'navy', lw=2.5, label=r'$\kappa(X)$')
    axs[1, 1].axhline(target_k_max, color='r', linestyle='--', label=f'上限 ({target_k_max:.4f})')
    axs[1, 1].axvline(200.0, color='gray', linestyle='-.', alpha=0.5, label='顶点 (X=200)')
    axs[1, 1].set_xlim(xlim_global)
    axs[1, 1].set_ylim(ylim_kappa)
    axs[1, 1].set_title(r'曲率变化 $\kappa(X)$ (1/mm)', fontsize=11)
    axs[1, 1].set_xlabel('X 坐标 (mm)')
    axs[1, 1].set_ylabel(r'曲率 $\kappa$ (1/mm)')
    axs[1, 1].legend(loc='upper right', fontsize=8)
    axs[1, 1].grid(True, linestyle=':')

    axs[1, 2].plot(x_sp, jerk_sp, 'darkgreen', lw=2.0, label='法向 Jerk')
    axs[1, 2].axhline(0, color='gray', linestyle=':', alpha=0.6)
    axs[1, 2].axvline(200.0, color='gray', linestyle='-.', alpha=0.5)
    axs[1, 2].set_xlim(xlim_global)
    axs[1, 2].set_ylim(ylim_jerk)
    axs[1, 2].set_title(r'法向加加速度 Jerk ($\mathrm{mm/s^3}$)', fontsize=11)
    axs[1, 2].set_xlabel('X 坐标 (mm)')
    axs[1, 2].set_ylabel(r'Jerk ($\mathrm{mm/s^3}$)')
    axs[1, 2].legend(loc='upper right', fontsize=8)
    axs[1, 2].grid(True, linestyle=':')

    axs[1, 3].plot(x_sp, r_sp, 'purple', lw=2.5, label=r'$R(X)$')
    axs[1, 3].axhline(R_min_target, color='r', linestyle='--', label=f'R_min ({R_min_target:.0f} mm)')
    axs[1, 3].axvline(200.0, color='gray', linestyle='-.', alpha=0.5)
    axs[1, 3].set_xlim(xlim_global)
    axs[1, 3].set_ylim(ylim_radius)
    axs[1, 3].set_title('曲率半径分布 $R(X)$ (mm)', fontsize=11)
    axs[1, 3].set_xlabel('X 坐标 (mm)')
    axs[1, 3].set_ylabel('曲率半径 $R$ (mm)')
    axs[1, 3].legend(loc='upper right', fontsize=8)
    axs[1, 3].grid(True, linestyle=':')

    fig.canvas.draw_idle()


def submit(text):
    try:
        val = float(text)
        if val > 0:
            draw_all(val)
    except ValueError:
        pass


text_box.on_submit(submit)

draw_all(R_min_init)
plt.show()