import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
#螺旋线控制方向，曲率变化等

class InteractiveSpiral:
    def __init__(self):
        self.p0 = np.array([0.0, 0.0])
        self.p1 = np.array([30.0, 20.0])
        self.target_heading = np.radians(45.0)
        self.arrow_len = 8.0

        self.s_star_r = 0.45
        self.alpha = 0.5
        self.bend_factor = 1.3  # 弯曲绕行系数 (1.0 为紧绷，>1.0 为深弯)

        self.is_locked = False
        self.selected_target = None


state = InteractiveSpiral()


def solve_spiral_with_heading_and_bend(p0, p1, theta_target, s_star_r, alpha, bend_factor, n_pts=300):
    delta = p1 - p0
    d_chord = max(np.linalg.norm(delta), 1e-4)
    chord_angle = np.arctan2(delta[1], delta[0])

    # 目标弧长由弦长和弯曲度滑块共同决定
    L_target = d_chord * bend_factor

    # 优化起点航向 theta_0 与曲率缩放 k_scale，使轨迹精确到达终点且切线对齐
    def objective(vars_opt):
        k_scale, theta_0 = vars_opt
        s = np.linspace(0, L_target, n_pts)
        s_star = s_star_r * L_target
        kappa = k_scale * ((s - s_star) / L_target) * (1.0 + alpha * (s / L_target))

        theta = theta_0 + cumulative_trapezoid(kappa, s, initial=0.0)
        dx = cumulative_trapezoid(np.cos(theta), s, initial=0.0)[-1]
        dy = cumulative_trapezoid(np.sin(theta), s, initial=0.0)[-1]

        pos_err = (dx - delta[0]) ** 2 + (dy - delta[1]) ** 2
        ang_err = (np.arctan2(np.sin(theta[-1] - theta_target), np.cos(theta[-1] - theta_target))) ** 2
        return pos_err + (d_chord ** 2) * ang_err

    x0 = [4.0 / d_chord, chord_angle]
    bounds = [(-100.0 / d_chord, 100.0 / d_chord), (-2 * np.pi, 2 * np.pi)]

    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 80, 'ftol': 1e-6})
    k_scale_opt, theta_0_opt = res.x

    s = np.linspace(0, L_target, n_pts)
    s_star = s_star_r * L_target
    kappa = k_scale_opt * ((s - s_star) / L_target) * (1.0 + alpha * (s / L_target))
    theta = theta_0_opt + cumulative_trapezoid(kappa, s, initial=0.0)

    x = p0[0] + cumulative_trapezoid(np.cos(theta), s, initial=0.0)
    y = p0[1] + cumulative_trapezoid(np.sin(theta), s, initial=0.0)

    return s, kappa, x, y, s_star


# 渲染窗口
fig = plt.figure(figsize=(12, 6.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=0.08, right=0.92, bottom=0.32, top=0.92)

ax_curve = fig.add_subplot(gs[0])
ax_kappa = fig.add_subplot(gs[1])

s, kappa, x, y, s_star = solve_spiral_with_heading_and_bend(
    state.p0, state.p1, state.target_heading, state.s_star_r, state.alpha, state.bend_factor
)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(vmin=-0.1, vmax=0.1)
lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidth=3.5)
lc.set_array(kappa[:-1])
line_obj = ax_curve.add_collection(lc)

idx_inflection = np.argmin(np.abs(s - s_star))
inflect_pt, = ax_curve.plot(x[idx_inflection], y[idx_inflection], 'm*', markersize=14, label=r'拐点 ($\kappa=0$)',
                            zorder=4)
start_pt, = ax_curve.plot(state.p0[0], state.p0[1], 'ko', markersize=8, label='起点 P0', zorder=5)
end_pt, = ax_curve.plot(state.p1[0], state.p1[1], 'bo', markersize=8, label='终点 P1', zorder=5)

h_tip = state.p1 + state.arrow_len * np.array([np.cos(state.target_heading), np.sin(state.target_heading)])
arrow_line, = ax_curve.plot([state.p1[0], h_tip[0]], [state.p1[1], h_tip[1]], color='green', lw=2.5, zorder=6)
handle_pt, = ax_curve.plot(h_tip[0], h_tip[1], marker='o', color='limegreen', markeredgecolor='darkgreen',
                           markersize=10, label='方向手柄 (可拖拽)', zorder=7)

cbar = fig.colorbar(lc, ax=ax_curve, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label(r'曲率 $\kappa$')

ax_curve.set_title('G3 二次螺线 (支持终点方向 + 弯曲程度独立控制)', fontsize=12)
ax_curve.set_xlabel('X 坐标')
ax_curve.set_ylabel('Y 坐标')
ax_curve.grid(True, linestyle=':')
ax_curve.axis('equal')
ax_curve.legend(loc='upper left')

kappa_line, = ax_kappa.plot(s, kappa, 'navy', lw=2)
inflect_line = ax_kappa.axvline(s_star, color='m', linestyle='--', alpha=0.7, label=r'拐点弧长 $s^*$')
zero_line = ax_kappa.axhline(0, color='gray', linestyle=':', alpha=0.6)
ax_kappa.set_title(r'曲率分布 $\kappa(s)$', fontsize=12)
ax_kappa.set_xlabel(r'弧长 $s$')
ax_kappa.set_ylabel(r'曲率 $\kappa$')
ax_kappa.grid(True, linestyle=':')
ax_kappa.legend(loc='lower right')

# 控件
ax_s_star = fig.add_axes([0.15, 0.20, 0.52, 0.025])
ax_bend = fig.add_axes([0.15, 0.15, 0.52, 0.025])
ax_shape = fig.add_axes([0.15, 0.10, 0.52, 0.025])
ax_btn = fig.add_axes([0.74, 0.095, 0.16, 0.13])

s_slider = Slider(ax_s_star, r'拐点位置 ($s^*/L$)', -0.5, 1.5, valinit=state.s_star_r)
bend_slider = Slider(ax_bend, r'弯曲度/弧长倍数', 1.02, 2.5, valinit=state.bend_factor)
shape_slider = Slider(ax_shape, r'非对称因子 ($\alpha$)', -1.0, 2.0, valinit=state.alpha)
btn_lock = Button(ax_btn, '锁定拖拽', color='lightgreen', hovercolor='0.9')


def update(val=None):
    state.s_star_r = s_slider.val
    state.bend_factor = bend_slider.val
    state.alpha = shape_slider.val

    s_new, k_new, x_new, y_new, s_star_new = solve_spiral_with_heading_and_bend(
        state.p0, state.p1, state.target_heading, state.s_star_r, state.alpha, state.bend_factor
    )

    pts = np.array([x_new, y_new]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc.set_segments(segs)
    lc.set_array(k_new[:-1])

    max_k = max(abs(k_new.min()), abs(k_new.max()), 1e-4)
    lc.set_clim(-max_k, max_k)
    cbar.update_normal(lc)

    idx_inf = np.argmin(np.abs(s_new - s_star_new))
    start_pt.set_data([state.p0[0]], [state.p0[1]])
    end_pt.set_data([state.p1[0]], [state.p1[1]])

    if 0 <= state.s_star_r <= 1:
        inflect_pt.set_visible(True)
        inflect_pt.set_data([x_new[idx_inf]], [y_new[idx_inf]])
    else:
        inflect_pt.set_visible(False)

    h_tip = state.p1 + state.arrow_len * np.array([np.cos(state.target_heading), np.sin(state.target_heading)])
    arrow_line.set_data([state.p1[0], h_tip[0]], [state.p1[1], h_tip[1]])
    handle_pt.set_data([h_tip[0]], [h_tip[1]])

    margin = 8.0
    all_x = [x_new.min(), state.p0[0], state.p1[0], h_tip[0]]
    all_y = [y_new.min(), state.p0[1], state.p1[1], h_tip[1]]
    ax_curve.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax_curve.set_ylim(min(all_y) - margin, max(all_y) + margin)

    kappa_line.set_data(s_new, k_new)
    inflect_line.set_xdata([s_star_new, s_star_new])
    ax_kappa.set_xlim(0, s_new[-1])
    ax_kappa.set_ylim(k_new.min() - 0.01, k_new.max() + 0.01)

    fig.canvas.draw_idle()


def on_press(event):
    if state.is_locked or event.inaxes != ax_curve or event.button != 1:
        return

    h_tip = state.p1 + state.arrow_len * np.array([np.cos(state.target_heading), np.sin(state.target_heading)])
    d0 = np.hypot(event.xdata - state.p0[0], event.ydata - state.p0[1])
    d1 = np.hypot(event.xdata - state.p1[0], event.ydata - state.p1[1])
    dh = np.hypot(event.xdata - h_tip[0], event.ydata - h_tip[1])

    pick_r = 4.0
    if dh < pick_r:
        state.selected_target = 'heading'
    elif d0 < pick_r and d0 <= d1:
        state.selected_target = 'p0'
    elif d1 < pick_r:
        state.selected_target = 'p1'


def on_motion(event):
    if state.is_locked or state.selected_target is None or event.inaxes != ax_curve:
        return

    if state.selected_target == 'p0':
        state.p0 = np.array([event.xdata, event.ydata])
    elif state.selected_target == 'p1':
        state.p1 = np.array([event.xdata, event.ydata])
    elif state.selected_target == 'heading':
        v = np.array([event.xdata - state.p1[0], event.ydata - state.p1[1]])
        if np.linalg.norm(v) > 1e-3:
            state.target_heading = np.arctan2(v[1], v[0])

    update()


def on_release(event):
    state.selected_target = None


def toggle_lock(event):
    state.is_locked = not state.is_locked
    btn_lock.label.set_text('解锁拖拽' if state.is_locked else '锁定拖拽')
    btn_lock.color = 'salmon' if state.is_locked else 'lightgreen'
    fig.canvas.draw_idle()


s_slider.on_changed(update)
bend_slider.on_changed(update)
shape_slider.on_changed(update)
btn_lock.on_clicked(toggle_lock)

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

update()
plt.show()