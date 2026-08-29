import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, CheckButtons
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import root_scalar, minimize, curve_fit
from scipy.interpolate import splprep, splev, CubicSpline, BSpline
from scipy.signal import savgol_filter  # 引入高保真 Savitzky-Golay 滤波器
import math

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class TrajectoryBenchmarkApp:
    def __init__(self):
        self.a = 200.0          # 半宽 (对称轴 X=200)
        self.b = 300.0          # 半高 (Y=300)
        self.n_val = 1.05       # 超椭圆指数
        self.R_target = 50.0    # 倒圆半径下限 R (mm)
        self.v_robot = 500.0    # 机器人末端线速度 (mm/s)
        self.x_cutoff = 20.0    # 左右两端物理截断死区 (mm)
        self.tol_pct = 10.0     # 允许超出半径的百分比容差上限 (默认 10%)
        # 13 种算法的列表
        self.raw_method_names = [
            '原始复合基准 (超椭圆+圆弧)',
            '内切参考圆 (Tangent Circle)',
            '广义外凸螺线 (Spiral - G2平滑)',
            '五次多项式 (Quintic)',
            '七次多项式 (7th Poly)',
            '余弦 S 型 (Cosine S-Curve)',
            '五阶贝塞尔 (5th Bézier)',
            '七阶贝塞尔 (7th Bézier)',
            '三次 B 样条 (3rd B-Spline)',
            '四次 B 样条 (4th B-Spline)',
            '五次 B 样条 (5th B-Spline)',
            '七次 B 样条 (7th B-Spline)',
            '三次自然样条 (Cubic Spline)'
        ]
        self.visible_flags = {
            '原始复合基准 (超椭圆+圆弧)': True,
            '内切参考圆 (Tangent Circle)': True,
            '广义外凸螺线 (Spiral - G2平滑)': True,
            '五次多项式 (Quintic)': False,
            '七次多项式 (7th Poly)': False,
            '余弦 S 型 (Cosine S-Curve)': False,
            '五阶贝塞尔 (5th Bézier)': False,
            '七阶贝塞尔 (7th Bézier)': False,
            '三次 B 样条 (3rd B-Spline)': False,
            '四次 B 样条 (4th B-Spline)': True,
            '五次 B 样条 (5th B-Spline)': True,
            '七次 B 样条 (7th B-Spline)': False,
            '三次自然样条 (Cubic Spline)': False
        }


state = TrajectoryBenchmarkApp()


# 1. 超椭圆几何解析
def superellipse_geom_exact(x, a, b, n):
    u = np.clip(np.abs(x - a) / a, 1e-6, 1.0 - 1e-6)
    inside = np.maximum(0.0, 1.0 - u ** n)
    y = b * (inside ** (1.0 / n))

    sgn = np.sign(x - a)
    dy = - (b / a) * sgn * (u ** (n - 1.0)) * (inside ** (1.0 / n - 1.0))

    term1 = (n - 1.0) * (u ** (n - 2.0)) * (inside ** (1.0 / n - 1.0))
    term2 = - (n - 1.0) * (u ** (2.0 * n - 2.0)) * (inside ** (1.0 / n - 2.0))
    ddy = - (b / (a ** 2)) * (term1 + term2)

    kappa = np.abs(ddy) / np.maximum((1.0 + dy ** 2) ** 1.5, 1e-9)
    return y, dy, kappa


# 2. 严格几何相切圆求解
def solve_strict_tangent_circle(a, b, n, R):
    def objective(x_t):
        y_t, dy_t, _ = superellipse_geom_exact(x_t, a, b, n)
        if abs(dy_t) < 1e-5:
            return -R
        Yc_tmp = y_t - (a - x_t) / dy_t
        dist = np.hypot(x_t - a, y_t - Yc_tmp)
        return dist - R

    try:
        sol = root_scalar(objective, bracket=[state.x_cutoff, a - 0.2], method='brentq')
        x_tl = sol.root
        y_tl, dy_tl, _ = superellipse_geom_exact(x_tl, a, b, n)
        Yc = y_tl - (a - x_tl) / dy_tl
        theta_tangent = np.arctan(dy_tl)
    except Exception:
        x_tl = a - max(R * 0.8, 5.0)
        y_tl, dy_tl, _ = superellipse_geom_exact(x_tl, a, b, n)
        Yc = b - R
        theta_tangent = np.arctan(dy_tl)

    return x_tl, y_tl, Yc, theta_tangent


# 3. 严格真实空间曲率与高精度动力学计算（加入高保真平滑滤波）
def compute_dynamics_and_curvature(x_arr, y_arr, v_robot):
    dx_raw = np.diff(x_arr)
    dy_raw = np.diff(y_arr)
    ds_raw = np.hypot(dx_raw, dy_raw)
    s_raw = np.concatenate([[0], np.cumsum(ds_raw)])

    s_uniform = np.linspace(0, s_raw[-1], len(x_arr))
    x_u = np.interp(s_uniform, s_raw, x_arr)
    y_u = np.interp(s_uniform, s_raw, y_arr)

    # 动态确定 Savitzky-Golay 滤波器窗口（奇数，保持轻微滤波不改变几何形貌）
    n_pts = len(x_u)
    win_len = min(25, n_pts if n_pts % 2 != 0 else n_pts - 1)
    if win_len >= 7:
        x_u = savgol_filter(x_u, window_length=win_len, polyorder=3)
        y_u = savgol_filter(y_u, window_length=win_len, polyorder=3)

    dx = np.gradient(x_u, s_uniform)
    dy = np.gradient(y_u, s_uniform)
    ddx = np.gradient(dx, s_uniform)
    ddy = np.gradient(dy, s_uniform)

    speed = np.hypot(dx, dy)
    kappa_true = np.abs(dx * ddy - dy * ddx) / np.maximum(speed ** 3, 1e-12)

    # 对曲率进行轻度滤波，消除高阶差分带来的数值抖动
    if win_len >= 7:
        kappa_true = savgol_filter(kappa_true, window_length=win_len, polyorder=3)
        kappa_true = np.maximum(0.0, kappa_true)

    a_n = (v_robot ** 2) * kappa_true
    dkappa_ds = np.gradient(kappa_true, s_uniform)

    # 对 Jerk 微分结果二次平滑
    if win_len >= 7:
        dkappa_ds = savgol_filter(dkappa_ds, window_length=win_len, polyorder=3)

    jerk_n = (v_robot ** 3) * dkappa_ds
    radius = np.where(kappa_true > 1e-6, 1.0 / kappa_true, 1e6)

    return x_u, y_u, kappa_true, a_n, jerk_n, radius


# 4. 全区间最大曲率约束的 B 样条通用优化算子
def fit_high_order_bspline_constrained(x_pts, y_pts, k=3, n_internal_knots=8, R_target=50.0, tol_pct=10.0):
    u = np.linspace(0, 1, len(x_pts))
    internal_knots = np.linspace(0, 1, n_internal_knots + 2)[1:-1]
    knots = np.concatenate([[0.0] * (k + 1), internal_knots, [1.0] * (k + 1)])
    n_bases = len(knots) - k - 1

    A = np.zeros((len(u), n_bases))
    for i in range(n_bases):
        c = np.zeros(n_bases)
        c[i] = 1.0
        spl = BSpline(knots, c, k)
        A[:, i] = spl(u)

    P_start = np.array([x_pts[0], y_pts[0]])
    P_end = np.array([x_pts[-1], y_pts[-1]])

    A_inner = A[:, 1:-1]
    rhs_x = x_pts - A[:, 0] * P_start[0] - A[:, -1] * P_end[0]
    rhs_y = y_pts - A[:, 0] * P_start[1] - A[:, -1] * P_end[1]

    px_inner_0, _, _, _ = np.linalg.lstsq(A_inner, rhs_x, rcond=None)
    py_inner_0, _, _, _ = np.linalg.lstsq(A_inner, rhs_y, rcond=None)

    kappa_cap = 1.0 / (R_target * max(0.2, (1.0 - tol_pct / 100.0)))
    px_all = np.concatenate([[P_start[0]], px_inner_0, [P_end[0]]])
    x_fit = A @ px_all
    dx = np.gradient(x_fit)

    def bspline_loss(py_vars):
        py_all = np.concatenate([[P_start[1]], py_vars, [P_end[1]]])
        y_fit = A @ py_all

        err_fit = np.mean((y_fit - y_pts) ** 2)

        dy = np.gradient(y_fit)
        ddy = np.gradient(dy)
        ddx = np.gradient(dx)
        denom = np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
        kappa_all = np.abs(dx * ddy - dy * ddx) / denom

        excess = np.maximum(0.0, kappa_all - kappa_cap)
        penalty = 1e6 * np.sum(excess ** 2)
        smooth_pen = 1e-4 * np.sum(np.diff(py_vars, n=2) ** 2)

        return err_fit + penalty + smooth_pen

    res = minimize(bspline_loss, py_inner_0, method='L-BFGS-B', options={'maxiter': 300})
    py_opt = np.concatenate([[P_start[1]], res.x, [P_end[1]]])

    return x_fit, A @ py_opt


# 5. 通用 N 阶贝塞尔拟合优化算子
def fit_bezier_constrained(x_pts, y_pts, deg=5, R_target=50.0, tol_pct=10.0, s_samples=None):
    if s_samples is None:
        ds = np.hypot(np.diff(x_pts), np.diff(y_pts))
        s_acc = np.concatenate([[0], np.cumsum(ds)])
        u_chord = s_acc / s_acc[-1]
    else:
        u_chord = s_samples / s_samples[-1]

    P_start = np.array([x_pts[0], y_pts[0]])
    P_end = np.array([x_pts[-1], y_pts[-1]])

    B_matrix = np.zeros((len(u_chord), deg + 1))
    for i in range(deg + 1):
        B_matrix[:, i] = [math.comb(deg, i) * (val ** i) * ((1.0 - val) ** (deg - i)) for val in u_chord]

    A_inner = B_matrix[:, 1:deg]
    y_target_x = x_pts - B_matrix[:, 0] * P_start[0] - B_matrix[:, deg] * P_end[0]
    y_target_y = y_pts - B_matrix[:, 0] * P_start[1] - B_matrix[:, deg] * P_end[1]

    px_inner_0, _, _, _ = np.linalg.lstsq(A_inner, y_target_x, rcond=None)
    py_inner_0, _, _, _ = np.linalg.lstsq(A_inner, y_target_y, rcond=None)

    px_all_bz = np.concatenate([[P_start[0]], px_inner_0, [P_end[0]]])
    x_fit_bz = B_matrix @ px_all_bz
    dx_bz = np.gradient(x_fit_bz)
    ddx_bz = np.gradient(dx_bz)

    kappa_cap = 1.0 / (R_target * max(0.2, (1.0 - tol_pct / 100.0)))

    def bezier_loss(py_vars):
        py_all = np.concatenate([[P_start[1]], py_vars, [P_end[1]]])
        y_fit = B_matrix @ py_all

        err_fit = np.mean((y_fit - y_pts) ** 2)

        dy = np.gradient(y_fit)
        ddy = np.gradient(dy)
        denom = np.maximum((dx_bz ** 2 + dy ** 2) ** 1.5, 1e-9)
        kappa_all = np.abs(dx_bz * ddy - dy * ddx_bz) / denom

        excess = np.maximum(0.0, kappa_all - kappa_cap)
        penalty = 1e6 * np.sum(excess ** 2)

        return err_fit + penalty

    res_bz = minimize(bezier_loss, py_inner_0, method='L-BFGS-B', options={'maxiter': 300})
    py_all_bz = np.concatenate([[P_start[1]], res_bz.x, [P_end[1]]])
    return x_fit_bz, B_matrix @ py_all_bz


# 6. 生成基准与所有拟合算法
def generate_all_methods(a, b, n, R, v_robot, x_cut_off, tol_pct, n_pts=1000):
    x_tl, y_tl, Yc, theta_tl = solve_strict_tangent_circle(a, b, n, R)
    k_target = 1.0 / R

    x_min_valid = x_cut_off
    x_max_valid = 2.0 * a - x_cut_off

    # ---------------- 原始基准：超椭圆 + 顶部圆弧 ----------------
    x_base_raw = np.linspace(x_min_valid, x_max_valid, 1200)
    y_base_raw = []
    x_tr = 2.0 * a - x_tl

    for x_val in x_base_raw:
        if x_val < x_tl or x_val > x_tr:
            y_v, _, _ = superellipse_geom_exact(x_val, a, b, n)
            y_base_raw.append(y_v)
        else:
            dx_c = x_val - a
            dy_c = np.sqrt(max(0.0, R ** 2 - dx_c ** 2))
            y_base_raw.append(Yc + dy_c)

    x_base_raw = np.array(x_base_raw)
    y_base_raw = np.array(y_base_raw)

    ds_base = np.hypot(np.diff(x_base_raw), np.diff(y_base_raw))
    s_base_acc = np.concatenate([[0], np.cumsum(ds_base)])
    s_samples = np.linspace(0, s_base_acc[-1], n_pts)
    x_base_full = np.interp(s_samples, s_base_acc, x_base_raw)
    y_base_full = np.interp(s_samples, s_base_acc, y_base_raw)

    P_start = np.array([x_base_full[0], y_base_full[0]])
    P_end = np.array([x_base_full[-1], y_base_full[-1]])

    kappa_cap = 1.0 / (R * max(0.2, (1.0 - tol_pct / 100.0)))
    R_min_allowed = R * (1.0 - tol_pct / 100.0)

    # ---------------- 广义外凸螺线过渡 ----------------
    delta_x = max((a - x_tl) * 0.55, 12.0)
    x1 = max(x_min_valid, x_tl - delta_x)
    y1, dy1, k1 = superellipse_geom_exact(x1, a, b, n)
    theta1 = np.arctan(dy1)
    P1 = np.array([x1, y1])
    P2 = np.array([x_tl, y_tl])
    theta2 = theta_tl

    dx_target = P2[0] - P1[0]
    dy_target = P2[1] - P1[1]
    chord_len = np.hypot(dx_target, dy_target)
    n_seg = n_pts // 5
    u_dense = np.linspace(0, 1, n_seg)

    def spiral_loss(params):
        L_val, p_pow = params
        s_arr = u_dense * L_val
        k_arr = k1 + (k_target - k1) * (u_dense ** p_pow)
        th_arr = theta1 - cumulative_trapezoid(k_arr, s_arr, initial=0.0)
        dx_val = trapezoid(np.cos(th_arr), s_arr)
        dy_val = trapezoid(np.sin(th_arr), s_arr)
        return (dx_val - dx_target) ** 2 + (dy_val - dy_target) ** 2 + (th_arr[-1] - theta2) ** 2 * 1e3

    res_sp = minimize(spiral_loss, [chord_len * 1.1, 2.0],
                      bounds=[(chord_len * 0.95, chord_len * 2.5), (0.5, 5.0)], method='L-BFGS-B')
    L_sp, p_sp = res_sp.x
    s_sp = u_dense * L_sp
    k_sp = k1 + (k_target - k1) * (u_dense ** p_sp)
    th_sp = theta1 - cumulative_trapezoid(k_sp, s_sp, initial=0.0)
    dx_sp = cumulative_trapezoid(np.cos(th_sp), s_sp, initial=0.0)
    dy_sp = cumulative_trapezoid(np.sin(th_sp), s_sp, initial=0.0)

    x_sp_l = P1[0] + dx_sp * (dx_target / max(dx_sp[-1], 1e-5))
    y_sp_l = P1[1] + dy_sp * (dy_target / max(dy_sp[-1], 1e-5))
    x_sp_r = 2.0 * a - x_sp_l[::-1]
    y_sp_r = y_sp_l[::-1]
    x_se_l = np.linspace(x_min_valid, x1, n_seg)
    y_se_l, _, _ = superellipse_geom_exact(x_se_l, a, b, n)
    x_se_r = 2.0 * a - x_se_l[::-1]
    y_se_r = y_se_l[::-1]

    phi_l = np.arctan2(y_tl - Yc, x_tl - a)
    phi_r = np.arctan2(y_tl - Yc, x_tr - a)
    phi_arr = np.linspace(phi_l, phi_r, n_seg)
    x_arc = a + R * np.cos(phi_arr)
    y_arc = Yc + R * np.sin(phi_arr)

    methods_xy = {}

    # 1. 原始复合基准
    methods_xy['原始复合基准 (超椭圆+圆弧)'] = (x_base_full, y_base_full)

    # 2. 广义外凸螺线
    methods_xy['广义外凸螺线 (Spiral - G2平滑)'] = (
        np.concatenate([x_se_l[:-1], x_sp_l[:-1], x_arc[:-1], x_sp_r[:-1], x_se_r]),
        np.concatenate([y_se_l[:-1], y_sp_l[:-1], y_arc[:-1], y_sp_r[:-1], y_se_r])
    )

    # 3. 五阶贝塞尔
    methods_xy['五阶贝塞尔 (5th Bézier)'] = fit_bezier_constrained(
        x_base_full, y_base_full, deg=5, R_target=R, tol_pct=tol_pct, s_samples=s_samples
    )

    # 4. 七阶贝塞尔
    methods_xy['七阶贝塞尔 (7th Bézier)'] = fit_bezier_constrained(
        x_base_full, y_base_full, deg=7, R_target=R, tol_pct=tol_pct, s_samples=s_samples
    )

    # 5 & 6. 多项式
    x0, y0 = P_start
    x1_b, y1_b = P_end
    y_line = y0 + (y1_b - y0) * (x_base_full - x0) / (x1_b - x0)
    phi_x = (x_base_full - x0) * (x_base_full - x1_b)
    residual = (y_base_full - y_line) / np.where(np.abs(phi_x) < 1e-6, 1.0, phi_x)

    def poly_constrained_fit(deg_res):
        p_init = np.polyfit(x_base_full, residual, deg_res)

        def poly_loss(p_v):
            y_pred = y_line + phi_x * np.polyval(p_v, x_base_full)
            err = np.mean((y_pred - y_base_full) ** 2)

            dx = np.gradient(x_base_full)
            dy = np.gradient(y_pred)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            denom = np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
            k_all = np.abs(dx * ddy - dy * ddx) / denom

            excess = np.maximum(0.0, k_all - kappa_cap)
            pen = 1e6 * np.sum(excess ** 2)
            return err + pen

        res_p = minimize(poly_loss, p_init, method='L-BFGS-B', options={'maxiter': 300})
        return x_base_full, y_line + phi_x * np.polyval(res_p.x, x_base_full)

    methods_xy['五次多项式 (Quintic)'] = poly_constrained_fit(3)
    methods_xy['七次多项式 (7th Poly)'] = poly_constrained_fit(5)

    # 7. 余弦 S 型拟合
    def cos_model(x_in, a_p, b_p, c_p):
        return a_p * np.cos(b_p * (x_in - a)) + c_p
    try:
        popt, _ = curve_fit(cos_model, x_base_full, y_base_full,
                            p0=[np.max(y_base_full) - np.min(y_base_full), np.pi / (2 * a), np.min(y_base_full)],
                            maxfev=3000)
        y_cos_raw = cos_model(x_base_full, *popt)
        y_cos_adj = y_cos_raw - (y_cos_raw[0] - y0) * (x_base_full - x1_b) / (x0 - x1_b) - (y_cos_raw[-1] - y1_b) * (x_base_full - x0) / (x1_b - x0)
        methods_xy['余弦 S 型 (Cosine S-Curve)'] = (x_base_full, y_cos_adj)
    except Exception:
        methods_xy['余弦 S 型 (Cosine S-Curve)'] = (x_base_full, y_base_full)

    # 8. 三次 B 样条
    methods_xy['三次 B 样条 (3rd B-Spline)'] = fit_high_order_bspline_constrained(
        x_base_full, y_base_full, k=3, n_internal_knots=8, R_target=R, tol_pct=tol_pct
    )

    # 9. 四次 B 样条
    methods_xy['四次 B 样条 (4th B-Spline)'] = fit_high_order_bspline_constrained(
        x_base_full, y_base_full, k=4, n_internal_knots=8, R_target=R, tol_pct=tol_pct
    )

    # 10. 五次 B 样条
    methods_xy['五次 B 样条 (5th B-Spline)'] = fit_high_order_bspline_constrained(
        x_base_full, y_base_full, k=5, n_internal_knots=8, R_target=R, tol_pct=tol_pct
    )

    # 11. 七次 B 样条
    methods_xy['七次 B 样条 (7th B-Spline)'] = fit_high_order_bspline_constrained(
        x_base_full, y_base_full, k=7, n_internal_knots=8, R_target=R, tol_pct=tol_pct
    )

    # 12. 三次自然样条
    u_chord = s_samples / s_samples[-1]
    idx_samp = np.linspace(0, len(x_base_full) - 1, 40, dtype=int)
    idx_samp[0] = 0
    idx_samp[-1] = len(x_base_full) - 1
    cs_x = CubicSpline(u_chord[idx_samp], x_base_full[idx_samp], bc_type='natural')
    cs_y = CubicSpline(u_chord[idx_samp], y_base_full[idx_samp], bc_type='natural')
    methods_xy['三次自然样条 (Cubic Spline)'] = (cs_x(u_chord), cs_y(u_chord))

    # 计算各拟合算法真实动力学与 RMSE 误差
    results = {}
    for name, (raw_x, raw_y) in methods_xy.items():
        x_u, y_u, kappa_t, a_n_t, jerk_n_t, radius_t = compute_dynamics_and_curvature(raw_x, raw_y, v_robot)

        y_base_interp = np.interp(x_u, x_base_full, y_base_full)
        rmse_val = np.sqrt(np.mean((y_u - y_base_interp) ** 2))

        results[name] = {
            'x': x_u, 'y': y_u,
            'kappa': kappa_t, 'a_n': a_n_t,
            'jerk_n': jerk_n_t, 'radius': radius_t,
            'rmse': rmse_val
        }

    # 0. 内切参考圆
    theta_ref_full = np.linspace(0.0, 2.0 * np.pi, 600)
    x_circle_full = a + R * np.cos(theta_ref_full)
    y_circle_full = Yc + R * np.sin(theta_ref_full)

    results['内切参考圆 (Tangent Circle)'] = {
        'x': x_circle_full, 'y': y_circle_full,
        'kappa': np.full_like(x_circle_full, k_target),
        'a_n': np.full_like(x_circle_full, (v_robot ** 2) * k_target),
        'jerk_n': np.zeros_like(x_circle_full),
        'radius': np.full_like(x_circle_full, R),
        'rmse': 0.0
    }

    key_pts = {
        'P1': P1, 'P2': P2, 'Yc': Yc,
        'P_start': P_start, 'P_end': P_end,
        'kappa_cap': kappa_cap, 'R_min_allowed': R_min_allowed
    }
    return results, key_pts


# =========================================================================
# 可视化布局与交互控件
# =========================================================================
fig, axs = plt.subplots(2, 3, figsize=(19, 9.5))
plt.subplots_adjust(bottom=0.14, top=0.94, left=0.06, right=0.96, wspace=0.25, hspace=0.32)

# 指数滑块
ax_slider_n = fig.add_axes([0.08, 0.04, 0.18, 0.03])
slider_n = Slider(ax_slider_n, '超椭圆指数 n: ', 0.35, 2.5, valinit=state.n_val, valstep=0.05)

# 倒圆半径 R 滑动条
ax_slider_r = fig.add_axes([0.34, 0.04, 0.22, 0.03])
slider_r = Slider(ax_slider_r, '倒圆半径 R: ', 5.0, 120.0, valinit=state.R_target, valstep=1.0)

# 容差上限输入框
ax_textbox_tol = fig.add_axes([0.62, 0.04, 0.10, 0.035])
text_box_tol = TextBox(ax_textbox_tol, '半径容差 (%): ', initial=str(state.tol_pct))

# 右侧复选框容器
ax_check = fig.add_axes([0.75, 0.05, 0.23, 0.40])
check_box = None

method_colors = {
    '原始复合基准 (超椭圆+圆弧)': '#000000',
    '内切参考圆 (Tangent Circle)': '#FF00FF',
    '广义外凸螺线 (Spiral - G2平滑)': '#1f77b4',
    '五次多项式 (Quintic)': '#2ca02c',
    '七次多项式 (7th Poly)': '#ff7f0e',
    '余弦 S 型 (Cosine S-Curve)': '#17becf',
    '五阶贝塞尔 (5th Bézier)': '#6b8e23',
    '七阶贝塞尔 (7th Bézier)': '#9467bd',
    '三次 B 样条 (3rd B-Spline)': '#d62728',
    '四次 B 样条 (4th B-Spline)': '#20b2aa',
    '五次 B 样条 (5th B-Spline)': '#8c564b',
    '七次 B 样条 (7th B-Spline)': '#e377c2',
    '三次自然样条 (Cubic Spline)': '#008080'
}

method_styles = {
    '原始复合基准 (超椭圆+圆弧)': '--',
    '内切参考圆 (Tangent Circle)': ':',
    '广义外凸螺线 (Spiral - G2平滑)': '-',
    '五次多项式 (Quintic)': '--',
    '七次多项式 (7th Poly)': '-.',
    '余弦 S 型 (Cosine S-Curve)': ':',
    '五阶贝塞尔 (5th Bézier)': '-.',
    '七阶贝塞尔 (7th Bézier)': '-',
    '三次 B 样条 (3rd B-Spline)': '-.',
    '四次 B 样条 (4th B-Spline)': '-',
    '五次 B 样条 (5th B-Spline)': '--',
    '七次 B 样条 (7th B-Spline)': '-.',
    '三次自然样条 (Cubic Spline)': '--'
}


def update_view(rebuild_check=True):
    global check_box

    for i in range(2):
        for j in range(3):
            if (i, j) == (1, 2):
                continue
            axs[i, j].cla()

    results, pts = generate_all_methods(
        state.a, state.b, state.n_val, state.R_target, state.v_robot, state.x_cutoff, state.tol_pct
    )

    t_full = np.linspace(np.pi, 0, 400)
    se_x = state.a + state.a * np.sign(np.cos(t_full)) * (np.abs(np.cos(t_full)) ** (2.0 / state.n_val))
    se_y = state.b * np.sign(np.sin(t_full)) * (np.abs(np.sin(t_full)) ** (2.0 / state.n_val))

    target_k = 1.0 / state.R_target
    target_an = (state.v_robot ** 2) * target_k
    Yc = pts['Yc']
    P1, P2 = pts['P1'], pts['P2']
    P_start, P_end = pts['P_start'], pts['P_end']
    kappa_cap = pts['kappa_cap']
    R_min_allowed = pts['R_min_allowed']

    dynamic_labels = []
    for raw_name in state.raw_method_names:
        if raw_name in results:
            rmse_val = results[raw_name]['rmse']
            if raw_name in ['原始复合基准 (超椭圆+圆弧)', '内切参考圆 (Tangent Circle)']:
                lbl = raw_name
            else:
                lbl = f"{raw_name}\n[RMSE={rmse_val:.3f}mm]"
        else:
            lbl = raw_name
        dynamic_labels.append(lbl)

    if rebuild_check:
        ax_check.cla()
        active_statuses = [state.visible_flags[name] for name in state.raw_method_names]
        check_box = CheckButtons(ax_check, dynamic_labels, active_statuses)
        ax_check.set_title("【算法选择与 RMSE 评估】", fontsize=10.5, fontweight='bold')

        def on_checkbox_clicked(label_text):
            for i, d_lbl in enumerate(dynamic_labels):
                if d_lbl == label_text:
                    raw_name = state.raw_method_names[i]
                    state.visible_flags[raw_name] = not state.visible_flags[raw_name]
                    break
            update_view(rebuild_check=False)

        check_box.on_clicked(on_checkbox_clicked)

    active_results = {k: v for k, v in results.items() if state.visible_flags.get(k, True)}

    # ---------------- 1. 几何轨迹图 ----------------
    axs[0, 0].plot(se_x, se_y, 'gray', linestyle=':', lw=1.5, alpha=0.45, label='完整超椭圆轮廓')

    for name, data in active_results.items():
        lw_val = 2.5 if name == '内切参考圆 (Tangent Circle)' else 2.2
        if name in ['原始复合基准 (超椭圆+圆弧)', '内切参考圆 (Tangent Circle)']:
            lbl = name
        else:
            lbl = f"{name} (RMSE={data['rmse']:.3f}mm)"
        axs[0, 0].plot(data['x'], data['y'], color=method_colors[name],
                       linestyle=method_styles[name], lw=lw_val, label=lbl)

    theta_circle = np.linspace(0, 2 * np.pi, 200)
    full_circle_x = state.a + state.R_target * np.cos(theta_circle)
    full_circle_y = Yc + state.R_target * np.sin(theta_circle)
    axs[0, 0].plot(full_circle_x, full_circle_y, 'gray', linestyle='--', alpha=0.35)

    axs[0, 0].plot(state.a, Yc, 'ro', markersize=6)
    axs[0, 0].plot([P1[0], 2 * state.a - P1[0]], [P1[1], P1[1]], 'bs', markersize=5)
    axs[0, 0].plot([P2[0], 2 * state.a - P2[0]], [P2[1], P2[1]], 'g^', markersize=8)
    axs[0, 0].plot([P_start[0], P_end[0]], [P_start[1], P_end[1]], 'kD', markersize=6, label='固定边界端点')

    axs[0, 0].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)
    axs[0, 0].set_xlim(-10, 410)
    geom_max_y = max(state.b, Yc + state.R_target)
    axs[0, 0].set_ylim(-15, geom_max_y * 1.15)
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].set_title(f'几何轨迹对比 (R={state.R_target:.1f}mm, 允许容差±{state.tol_pct:.1f}%)', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('X 坐标 (mm)')
    axs[0, 0].set_ylabel('Y 坐标 (mm)')
    if len(active_results) > 0:
        axs[0, 0].legend(loc='lower center', fontsize=6.8, ncol=2)
    axs[0, 0].grid(True, linestyle=':')

    # ---------------- 2. 真实曲率分布图 κ(X) ----------------
    for name, data in active_results.items():
        k_min = np.min(data['kappa'])
        k_max = np.max(data['kappa'])
        if name in ['原始复合基准 (超椭圆+圆弧)', '内切参考圆 (Tangent Circle)']:
            label_with_stat = f"{name} ({k_min:.4f}, {k_max:.4f})"
        else:
            label_with_stat = f"{name} (RMSE={data['rmse']:.3f}, {k_min:.4f}, {k_max:.4f})"
        lw_val = 2.4 if name == '内切参考圆 (Tangent Circle)' else 2.0
        axs[0, 1].plot(data['x'], data['kappa'], color=method_colors[name],
                       linestyle=method_styles[name], lw=lw_val, label=label_with_stat)

    axs[0, 1].axhline(target_k, color='black', linestyle=':', label=f'1/R 目标 ({target_k:.4f})')
    axs[0, 1].axhline(kappa_cap, color='red', linestyle='--', label=f'κ 约束上限 ({kappa_cap:.4f})')
    axs[0, 1].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)

    max_k_all = max([np.max(d['kappa']) for d in active_results.values()]) if len(active_results) > 0 else target_k
    k_upper = max(max_k_all, kappa_cap) * 1.2
    axs[0, 1].set_xlim(state.x_cutoff - 5, 2 * state.a - state.x_cutoff + 5)
    axs[0, 1].set_ylim(-k_upper * 0.02, k_upper)
    axs[0, 1].set_title(r'真实曲率分布 $\kappa(X)$ (1/mm) (严格限顶)', fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel('X 坐标 (mm)')
    axs[0, 1].set_ylabel(r'曲率 $\kappa$ (1/mm)')
    if len(active_results) > 0:
        axs[0, 1].legend(loc='upper right', fontsize=6.8)
    axs[0, 1].grid(True, linestyle=':')

    # ---------------- 3. 真实曲率半径分布图 R(X) ----------------
    for name, data in active_results.items():
        valid_r = data['radius'][data['radius'] < 1e5]
        r_min = np.min(valid_r) if len(valid_r) > 0 else state.R_target
        r_max = np.max(valid_r) if len(valid_r) > 0 else state.R_target
        if name in ['原始复合基准 (超椭圆+圆弧)', '内切参考圆 (Tangent Circle)']:
            label_with_stat = f"{name} ({r_min:.1f}, {r_max:.1f})"
        else:
            label_with_stat = f"{name} (RMSE={data['rmse']:.3f}, {r_min:.1f}, {r_max:.1f})"
        lw_val = 2.4 if name == '内切参考圆 (Tangent Circle)' else 2.0
        axs[0, 2].plot(data['x'], data['radius'], color=method_colors[name],
                       linestyle=method_styles[name], lw=lw_val, label=label_with_stat)

    axs[0, 2].axhline(state.R_target, color='black', linestyle=':', label=f'R 目标 ({state.R_target:.1f} mm)')
    axs[0, 2].axhline(R_min_allowed, color='red', linestyle='--', label=f'R 允许下限 ({R_min_allowed:.1f} mm)')
    axs[0, 2].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)

    r_scale_max = max(state.R_target * 2.2, 50.0)
    axs[0, 2].set_xlim(state.x_cutoff - 5, 2 * state.a - state.x_cutoff + 5)
    axs[0, 2].set_ylim(0, r_scale_max)
    axs[0, 2].set_title(f'真实曲率半径分布 $R(X)$ (下限保障: {R_min_allowed:.1f}mm)', fontsize=11, fontweight='bold')
    axs[0, 2].set_xlabel('X 坐标 (mm)')
    axs[0, 2].set_ylabel('曲率半径 $R$ (mm)')
    if len(active_results) > 0:
        axs[0, 2].legend(loc='upper right', fontsize=6.8)
    axs[0, 2].grid(True, linestyle=':')

    # ---------------- 4. 真实法向加速度分布图 a_n(X) ----------------
    for name, data in active_results.items():
        an_min = np.min(data['a_n'])
        an_max = np.max(data['a_n'])
        label_with_minmax = f"{name} ({an_min:.0f}, {an_max:.0f})"
        lw_val = 2.4 if name == '内切参考圆 (Tangent Circle)' else 2.0
        axs[1, 0].plot(data['x'], data['a_n'], color=method_colors[name],
                       linestyle=method_styles[name], lw=lw_val, label=label_with_minmax)

    axs[1, 0].axhline(target_an, color='black', linestyle=':', label=f'an 目标 ({target_an:.0f} mm/s²)')
    axs[1, 0].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)

    max_an_all = max([np.max(d['a_n']) for d in active_results.values()]) if len(active_results) > 0 else target_an
    an_upper = max(max_an_all, target_an) * 1.2
    axs[1, 0].set_xlim(state.x_cutoff - 5, 2 * state.a - state.x_cutoff + 5)
    axs[1, 0].set_ylim(-an_upper * 0.02, an_upper)
    axs[1, 0].set_title(f'真实法向加速度 $a_n(X)$ (v={state.v_robot:.0f} mm/s)', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel('X 坐标 (mm)')
    axs[1, 0].set_ylabel(r'加速度 $a_n$ ($\mathrm{mm/s^2}$)')
    if len(active_results) > 0:
        axs[1, 0].legend(loc='upper right', fontsize=6.8)
    axs[1, 0].grid(True, linestyle=':')

    # ---------------- 5. 真实法向加加速度 Jerk(X) ----------------
    for name, data in active_results.items():
        valid_j_seg = data['jerk_n'][int(len(data['jerk_n']) * 0.02): int(len(data['jerk_n']) * 0.98)]
        j_min = np.min(valid_j_seg) if len(valid_j_seg) > 0 else np.min(data['jerk_n'])
        j_max = np.max(valid_j_seg) if len(valid_j_seg) > 0 else np.max(data['jerk_n'])
        label_with_minmax = f"{name} ({j_min:.0f}, {j_max:.0f})"
        lw_val = 2.4 if name == '内切参考圆 (Tangent Circle)' else 1.8
        axs[1, 1].plot(data['x'], data['jerk_n'], color=method_colors[name],
                       linestyle=method_styles[name], lw=lw_val, label=label_with_minmax)

    axs[1, 1].axhline(0, color='gray', linestyle=':', alpha=0.6)
    axs[1, 1].axvline(state.a, color='gray', linestyle='-.', alpha=0.4)

    all_j = [np.abs(d['jerk_n'][int(len(d['jerk_n']) * 0.02): int(len(d['jerk_n']) * 0.98)]) for d in active_results.values()]
    max_j = max([np.max(j) for j in all_j if len(j) > 0] + [50.0])

    axs[1, 1].set_xlim(state.x_cutoff - 5, 2 * state.a - state.x_cutoff + 5)
    axs[1, 1].set_ylim(-max_j * 1.35, max_j * 1.35)
    axs[1, 1].set_title('法向加加速度 Jerk ($j_n(X)$) 真实平稳度对比', fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel('X 坐标 (mm)')
    axs[1, 1].set_ylabel(r'Jerk ($\mathrm{mm/s^3}$)')
    if len(active_results) > 0:
        axs[1, 1].legend(loc='upper right', fontsize=6.8)
    axs[1, 1].grid(True, linestyle=':')

    axs[1, 2].axis('off')
    fig.canvas.draw_idle()


# =========================================================================
# 事件监听回调绑定
# =========================================================================
def on_scroll(event):
    if event.inaxes is None or event.inaxes in [axs[1, 2], ax_check]:
        return

    ax = event.inaxes
    base_scale = 1.25
    scale_factor = 1.0 / base_scale if event.button == 'up' else base_scale

    xdata, ydata = event.xdata, event.ydata
    if xdata is None or ydata is None:
        return

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * (1.0 - relx), xdata + new_width * relx])
    ax.set_ylim([ydata - new_height * (1.0 - rely), ydata + new_height * rely])
    ax.figure.canvas.draw_idle()


slider_n.on_changed(lambda val: (setattr(state, 'n_val', val), update_view(rebuild_check=True)))
slider_r.on_changed(lambda val: (setattr(state, 'R_target', val), update_view(rebuild_check=True)))
text_box_tol.on_submit(lambda text: (setattr(state, 'tol_pct', float(text)) if float(text) >= 0 else None, update_view(rebuild_check=True)))
fig.canvas.mpl_connect('scroll_event', on_scroll)

update_view(rebuild_check=True)
plt.show()