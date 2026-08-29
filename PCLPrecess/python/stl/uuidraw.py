import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math

# 尝试导入科学计算与图片处理库，并提供友好的提示
try:
    import numpy as np
    from scipy.interpolate import make_interp_spline, CubicHermiteSpline
    # 【修改这里】加入 least_squares
    from scipy.optimize import curve_fit, minimize, least_squares
    from PIL import Image, ImageTk
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "缺少依赖库",
        "本程序需要 numpy, scipy 和 Pillow 库支持。\n请在终端运行以下命令安装：\npip install numpy scipy Pillow"
    )
    exit()


# ------------------ 数学拟合与插值核心算法 ------------------

def fit_circle(pts):
    """
    代数圆拟合 (Least Squares Circle Fit)
    拟合方程: (x-xc)^2 + (y-yc)^2 = R^2
    """
    n = len(pts)
    if n < 3:
        return None
    A = np.zeros((n, 3))
    b = np.zeros(n)
    for i, (x, y) in enumerate(pts):
        A[i] = [2 * x, 2 * y, 1]
        b[i] = x ** 2 + y ** 2
    try:
        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, C = w[0], w[1], w[2]
        R2 = C + xc ** 2 + yc ** 2
        if R2 < 0:
            return None
        return xc, yc, np.sqrt(R2)
    except:
        return None


def fit_semicircle(pts):
    """
    半圆拟合: 严格以 P1 和 P3 为直径两端，P2 仅用于确定半圆所在的半平面 (弯曲方向)。
    输入 pts 必须刚好有 3 个点。
    """
    if len(pts) != 3:
        return None
    p1 = np.array(pts[0])
    p2 = np.array(pts[1])
    p3 = np.array(pts[2])

    # 1. 计算圆心 C (直径 P1-P3 的中点)
    C = (p1 + p3) / 2.0

    # 2. 计算半径 R (直径的一半)
    R = np.linalg.norm(p3 - p1) / 2.0
    if R < 1e-5:
        return None

    # 3. 计算从 C 指向 P1 的方向向量 u_dir
    u_dir = (p1 - C) / R

    # 4. 计算垂直于 u_dir 的法向量 n_dir (旋转90度)
    n_dir = np.array([-u_dir[1], u_dir[0]])

    # 5. 用 P2 确定半圆的弯曲朝向 (利用点积判定投影方向)
    v_p2 = p2 - C
    if np.dot(v_p2, n_dir) < 0:
        n_dir = -n_dir

    # 6. 生成 180 度的半圆离散点 (theta 从 0 渐变到 pi)
    theta = np.linspace(0, np.pi, 200)
    semi_pts = []
    for t_val in theta:
        pt = C + R * np.cos(t_val) * u_dir + R * np.sin(t_val) * n_dir
        semi_pts.append((pt[0], pt[1]))

    return semi_pts


def fit_g3_spiral_least_squares(pts, num_samples=300):
    """
    真正的单条 G3 二次回旋螺线最小二乘拟合 (Least-Squares G3 Spiral Fitting)
    基于标准内在几何方程: kappa(s) = c * s^2 (Quadratic Clothoid)
    - 绝不插值！全局最优逼近所有控制点。
    - 【防尾巴机制】: 渲染前自动裁剪，首尾严格收束在首末控制点附近。
    """
    pts_arr = np.array(pts, dtype=float)
    n = len(pts_arr)
    if n < 3:
        return None

    # 1. 计算控制点的累计弦长，用于参数初始匹配
    ds_input = np.linalg.norm(pts_arr[1:] - pts_arr[:-1], axis=1)
    s_input = np.zeros(n)
    s_input[1:] = np.cumsum(ds_input)
    L_est = s_input[-1]
    if L_est < 1e-5:
        return None

    s_norm_input = s_input / L_est

    # 2. 估计优化的初始值 [x0, y0, theta0, c, L]
    x0_init, y0_init = pts_arr[0]
    dir_init = pts_arr[1] - pts_arr[0]
    theta0_init = np.arctan2(dir_init[1], dir_init[0])

    p1, p2, p3 = pts_arr[0], pts_arr[n // 2], pts_arr[-1]
    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    denom = np.linalg.norm(p1 - p2) * np.linalg.norm(p2 - p3) * np.linalg.norm(p1 - p3)
    kappa_char = (4.0 * area / denom) if denom > 1e-5 else 0.01

    cross_prod = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    sign = 1.0 if cross_prod >= 0 else -1.0
    c_init = sign * (kappa_char / (L_est ** 2 + 1e-5)) * 3.0
    p0 = np.array([x0_init, y0_init, theta0_init, c_init, L_est])

    # 3. 单条 G3 螺线坐标生成函数 (内部高密度积分，消除截断误差)
    def generate_single_g3_spiral(params, s_eval):
        x0, y0, theta0, c, L = params
        L_abs = max(abs(L), 1e-3)

        num_int_dense = max(1000, len(s_eval) * 50)
        s_dense = np.linspace(0, L_abs, num_int_dense)

        kappa_dense = c * (s_dense ** 2)
        ds_dense = np.diff(s_dense)

        d_theta_dense = 0.5 * (kappa_dense[:-1] + kappa_dense[1:]) * ds_dense
        theta_dense = np.zeros(num_int_dense)
        theta_dense[1:] = np.cumsum(d_theta_dense) + theta0

        dx_dense = 0.5 * (np.cos(theta_dense[:-1]) + np.cos(theta_dense[1:])) * ds_dense
        dy_dense = 0.5 * (np.sin(theta_dense[:-1]) + np.sin(theta_dense[1:])) * ds_dense

        x_dense = np.zeros(num_int_dense)
        y_dense = np.zeros(num_int_dense)
        x_dense[1:] = np.cumsum(dx_dense) + x0
        y_dense[1:] = np.cumsum(dy_dense) + y0

        if len(s_eval) == num_int_dense and np.allclose(s_eval, s_dense):
            return np.column_stack([x_dense, y_dense])

        s_eval_safe = np.clip(s_eval, 0, L_abs)
        x_interp = np.interp(s_eval_safe, s_dense, x_dense)
        y_interp = np.interp(s_eval_safe, s_dense, y_dense)
        return np.column_stack([x_interp, y_interp])

    # 4. 最小二乘残差函数
    def residuals(params):
        L_abs = max(abs(params[4]), 1e-3)
        s_eval_input = s_norm_input * L_abs
        pred_pts = generate_single_g3_spiral(params, s_eval_input)
        return (pred_pts - pts_arr).flatten()

    # 5. 鲁棒最小二乘优化 (soft_l1 损失函数可抑制离群误点的拉扯)
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 1e-3],
              [np.inf, np.inf, np.inf, np.inf, np.inf])
    try:
        res = least_squares(residuals, p0, method='trf', bounds=bounds,
                            loss='soft_l1', f_scale=L_est * 0.05,
                            xtol=1e-6, ftol=1e-6, max_nfev=2000)
        best_params = res.x
    except Exception:
        try:
            res = least_squares(residuals, p0, method='lm', xtol=1e-6, ftol=1e-6, max_nfev=2000)
            best_params = res.x
            best_params[4] = abs(best_params[4])
        except Exception:
            best_params = p0

    # 6. 生成高密度螺线
    L_best = abs(best_params[4])
    s_dense = np.linspace(0, L_best, num_samples)
    dense = generate_single_g3_spiral(best_params, s_dense)

    # 7. 【关键新增 - 防尾巴裁剪】:
    #    只保留 "距首控制点最近处" 到 "距末控制点最近处" 之间的弧段，
    #    彻底切除 s≈0 处的直线入场段和末端任何无关延伸。
    dist_first = np.sum((dense - pts_arr[0]) ** 2, axis=1)
    dist_last = np.sum((dense - pts_arr[-1]) ** 2, axis=1)
    i0 = int(np.argmin(dist_first))
    i1 = int(np.argmin(dist_last))
    if i0 > i1:
        i0, i1 = i1, i0
    trimmed = dense[i0:i1 + 1] if (i1 - i0) >= 2 else dense

    return list(map(tuple, trimmed))

def fit_trig(pts):
    """
    三角函数拟合: y = A * sin(B * x + C) + D
    """
    n = len(pts)
    if n < 3:
        return None
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    # 启发式初值估计
    D_guess = np.mean(y)
    A_guess = (np.max(y) - np.min(y)) / 2.0
    if A_guess == 0:
        A_guess = 1.0
    x_range = np.max(x) - np.min(x)
    B_guess = 2 * np.pi / (x_range if x_range > 0 else 1.0)
    C_guess = 0.0

    def func(t, A, B, C, D):
        return A * np.sin(B * t + C) + D

    try:
        popt, _ = curve_fit(func, x, y, p0=[A_guess, B_guess, C_guess, D_guess], maxfev=3000)
        return popt
    except:
        return [A_guess, B_guess, C_guess, D_guess]


def fit_spline(pts, degree=3):
    """
    参数化B样条拟合 ( B-Spline )
    """
    n = len(pts)
    if n < degree + 1:
        return None
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    u = np.zeros(n)
    for i in range(1, n):
        dist = np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
        u[i] = u[i - 1] + dist
    if u[-1] == 0:
        return None
    u_norm = u / u[-1]

    try:
        spl_x = make_interp_spline(u_norm, x, k=degree)
        spl_y = make_interp_spline(u_norm, y, k=degree)
        return spl_x, spl_y
    except:
        return None


def get_curvature_normal(p1, p2, p3):
    """
    根据三点计算中心指向，用于估算向心加速度矢量方向
    """
    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    if abs(d) < 1e-5:
        return np.array([0.0, 0.0])
    ux = ((p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p3[1]) + (p2[0] ** 2 + p2[1] ** 2) * (p3[1] - p1[1]) + (
            p3[0] ** 2 + p3[1] ** 2) * (p1[1] - p2[1])) / d
    uy = ((p1[0] ** 2 + p1[1] ** 2) * (p3[0] - p2[0]) + (p2[0] ** 2 + p2[1] ** 2) * (p1[0] - p3[0]) + (
            p3[0] ** 2 + p3[1] ** 2) * (p2[0] - p1[0])) / d
    center = np.array([ux, uy])
    normal = center - np.array(p2)
    norm = np.linalg.norm(normal)
    if norm < 1e-5:
        return np.array([0.0, 0.0])
    return normal / norm


class QuinticHermiteSpline:
    """
    分段五次埃尔米特插值 (解析求解版)
    """

    def __init__(self, t, points, velocities, accelerations):
        self.t = t
        self.points = points
        self.velocities = velocities
        self.accelerations = accelerations
        self.num_segments = len(t) - 1
        self.segments = []

        for i in range(self.num_segments):
            t0, t1 = t[i], t[i + 1]
            dt = t1 - t0
            P0, P1 = points[i], points[i + 1]
            V0, V1 = velocities[i], velocities[i + 1]
            A0, A1 = accelerations[i], accelerations[i + 1]

            c0 = P0
            c1 = V0 * dt
            c2 = 0.5 * A0 * (dt ** 2)

            b1 = P1 - P0 - V0 * dt - 0.5 * A0 * (dt ** 2)
            b2 = V1 * dt - V0 * dt - A0 * (dt ** 2)
            b3 = A1 * (dt ** 2) - A0 * (dt ** 2)

            c3 = 10 * b1 - 4 * b2 + 0.5 * b3
            c4 = -15 * b1 + 7 * b2 - b3
            c5 = 6 * b1 - 3 * b2 + 0.5 * b3

            self.segments.append((t0, t1, dt, c0, c1, c2, c3, c4, c5))

    def __call__(self, eval_t):
        return np.array([self._eval_single(ti) for ti in eval_t])

    def _eval_single(self, ti):
        if ti <= self.t[0]:
            idx = 0
        elif ti >= self.t[-1]:
            idx = self.num_segments - 1
        else:
            idx = np.searchsorted(self.t, ti) - 1
            idx = max(0, min(idx, self.num_segments - 1))

        t0, t1, dt, c0, c1, c2, c3, c4, c5 = self.segments[idx]
        tau = (ti - t0) / dt if dt > 0 else 0.0
        return c0 + c1 * tau + c2 * (tau ** 2) + c3 * (tau ** 3) + c4 * (tau ** 4) + c5 * (tau ** 5)


class SepticHermiteSpline:
    """
    分段七次埃尔米特插值 (同时满足位置、速度、加速度、Jerk连续)
    """

    def __init__(self, t, points, velocities, accelerations, jerks):
        self.t = t
        self.points = points
        self.velocities = velocities
        self.accelerations = accelerations
        self.jerks = jerks
        self.num_segments = len(t) - 1
        self.segments = []

        # 建立七次Hermite高阶解算矩阵 M (4x4)
        M = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [4.0, 5.0, 6.0, 7.0],
            [12.0, 20.0, 30.0, 42.0],
            [24.0, 60.0, 120.0, 210.0]
        ])

        for i in range(self.num_segments):
            t0, t1 = t[i], t[i + 1]
            dt = t1 - t0
            P0, P1 = points[i], points[i + 1]
            V0, V1 = velocities[i], velocities[i + 1]
            A0, A1 = accelerations[i], accelerations[i + 1]
            J0, J1 = jerks[i], jerks[i + 1]

            # tau=0 边界条件确定 c0 - c3
            c0 = P0
            c1 = V0 * dt
            c2 = 0.5 * A0 * (dt ** 2)
            c3 = (1.0 / 6.0) * J0 * (dt ** 3)

            # tau=1 边界条件构建右侧矢量
            b1 = P1 - (c0 + c1 + c2 + c3)
            b2 = V1 * dt - (c1 + 2 * c2 + 3 * c3)
            b3 = A1 * (dt ** 2) - (2 * c2 + 6 * c3)
            b4 = J1 * (dt ** 3) - 6 * c3

            # 2D坐标轴分开求解高阶系数 (c4 - c7)
            B = np.vstack([b1, b2, b3, b4])  # shape (4, 2)
            C_high = np.linalg.solve(M, B)

            c4 = C_high[0]
            c5 = C_high[1]
            c6 = C_high[2]
            c7 = C_high[3]

            self.segments.append((t0, t1, dt, c0, c1, c2, c3, c4, c5, c6, c7))

    def __call__(self, eval_t):
        return np.array([self._eval_single(ti) for ti in eval_t])

    def _eval_single(self, ti):
        if ti <= self.t[0]:
            idx = 0
        elif ti >= self.t[-1]:
            idx = self.num_segments - 1
        else:
            idx = np.searchsorted(self.t, ti) - 1
            idx = max(0, min(idx, self.num_segments - 1))

        t0, t1, dt, c0, c1, c2, c3, c4, c5, c6, c7 = self.segments[idx]
        tau = (ti - t0) / dt if dt > 0 else 0.0
        return (c0 + c1 * tau + c2 * (tau ** 2) + c3 * (tau ** 3) +
                c4 * (tau ** 4) + c5 * (tau ** 5) + c6 * (tau ** 6) + c7 * (tau ** 7))


# ------------------ 主 GUI 程序界面 ------------------

class TrajectoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多工业轨迹拟合与插值分析工具 (运动学物理量分析)")
        self.root.geometry("1200x850")

        # 变量初始化
        self.control_points = []  # 当前激活段的控制点
        self.temp_clicks = []  # 用于几何绘图的临时点击记录
        self.draw_color = "black"  # 默认画笔颜色
        self.active_tool = "ctrl_point"  # 默认工具：添加控制点

        # 运动学物理量存储 (线速度 V, 向心加速度 a_n, 向心加加速度变化率 j_n)
        self.current_kinematics_data = None
        self.canvas_last_mouse = None
        self.slope_hover_info = None
        self.slope_plot_bounds = None
        self.slope_last_mouse = None

        # 背景图片变量初始化
        self.orig_img = None
        self.tk_image = None
        self.img_x = 400
        self.img_y = 300
        self.img_scale = 1.0
        self.img_angle = 0.0
        self.img_locked = True

        self._setup_ui()

    def _setup_ui(self):
        # ------------------ 1. 左侧控制面板 ------------------
        left_panel = tk.Frame(self.root, width=280, bg="#f5f5f7", bd=1, relief="raised")
        left_panel.pack(side="left", fill="y")
        left_panel.pack_propagate(False)

        # 标题
        title = tk.Label(left_panel, text="轨迹控制与算法选择", font=("Helvetica", 11, "bold"), bg="#f5f5f7",
                         fg="#333333")
        title.pack(pady=10)

        # 背景图片控制分组
        group_img = tk.LabelFrame(left_panel, text="背景图片控制", font=("Helvetica", 9, "bold"), bg="#f5f5f7",
                                  fg="#555555", padx=8, pady=8)
        group_img.pack(fill="x", padx=10, pady=4)

        tk.Button(group_img, text="导入背景图片", command=self.import_image, bg="#ffffff",
                  activebackground="#eaeaea").pack(fill="x", pady=2)

        btn_img_frame = tk.Frame(group_img, bg="#f5f5f7")
        btn_img_frame.pack(fill="x", pady=2)
        self.btn_lock_img = tk.Button(btn_img_frame, text="锁定图片", command=self.lock_image, bg="#ffffff",
                                      state="disabled")
        self.btn_lock_img.pack(side="left", fill="x", expand=True, padx=1)
        self.btn_unlock_img = tk.Button(btn_img_frame, text="解锁图片", command=self.unlock_image, bg="#ffffff",
                                        state="disabled")
        self.btn_unlock_img.pack(side="right", fill="x", expand=True, padx=1)

        # 缩放控制（滑块 + 输入框）
        self.lbl_scale = tk.Label(group_img, text="缩放比例:", bg="#f5f5f7", font=("Helvetica", 8))
        self.lbl_scale.pack(anchor="w", pady=(4, 0))

        scale_frame = tk.Frame(group_img, bg="#f5f5f7")
        scale_frame.pack(fill="x", pady=2)
        self.slider_scale = tk.Scale(scale_frame, from_=0.1, to=5.0, resolution=0.1, orient="horizontal",
                                     showvalue=False, command=self.on_scale_slider_change, state="disabled",
                                     bg="#f5f5f7", highlightthickness=0)
        self.slider_scale.set(1.0)
        self.slider_scale.pack(side="left", fill="x", expand=True)
        self.entry_scale = tk.Entry(scale_frame, width=6, font=("Helvetica", 9), state="disabled")
        self.entry_scale.insert(0, "1.0")
        self.entry_scale.pack(side="right", padx=(5, 0))
        self.entry_scale.bind("<Return>", self.on_scale_entry_change)
        self.entry_scale.bind("<FocusOut>", self.on_scale_entry_change)

        # 旋转控制（滑块 + 输入框）
        self.lbl_rotate = tk.Label(group_img, text="旋转角度 (°):", bg="#f5f5f7", font=("Helvetica", 8))
        self.lbl_rotate.pack(anchor="w", pady=(4, 0))

        rotate_frame = tk.Frame(group_img, bg="#f5f5f7")
        rotate_frame.pack(fill="x", pady=2)
        self.slider_rotate = tk.Scale(rotate_frame, from_=0, to=360, resolution=0.1, orient="horizontal",
                                      showvalue=False, command=self.on_rotate_slider_change, state="disabled",
                                      bg="#f5f5f7", highlightthickness=0)
        self.slider_rotate.set(0)
        self.slider_rotate.pack(side="left", fill="x", expand=True)
        self.entry_rotate = tk.Entry(rotate_frame, width=6, font=("Helvetica", 9), state="disabled")
        self.entry_rotate.insert(0, "0.0")
        self.entry_rotate.pack(side="right", padx=(5, 0))
        self.entry_rotate.bind("<Return>", self.on_rotate_entry_change)
        self.entry_rotate.bind("<FocusOut>", self.on_rotate_entry_change)

        self.lbl_img_tip = tk.Label(group_img,
                                    text="提示: 解锁后，可通过[右键拖拽]移动背景图，滑块右侧输入框支持精确数值修改(输入后按回车生效)。",
                                    font=("Helvetica", 8), fg="#777777", bg="#f5f5f7", justify="left", wrap=220)
        self.lbl_img_tip.pack(anchor="w", pady=(4, 0))

        # 几何拟合分组
        group_fit = tk.LabelFrame(left_panel, text="几何拟合算法", font=("Helvetica", 9, "bold"), bg="#f5f5f7",
                                  fg="#555555", padx=8, pady=8)
        group_fit.pack(fill="x", padx=10, pady=4)

        tk.Button(group_fit, text="圆弧拟合", command=self.do_circle_fit, bg="#ffffff",
                  activebackground="#eaeaea").pack(fill="x", pady=2)

        # 半圆拟合按钮
        tk.Button(group_fit, text="半圆拟合 (3点)", command=self.do_semicircle_fit, bg="#ffffff",
                  activebackground="#eaeaea").pack(fill="x", pady=2)

        # 单条 G3 螺线最小二乘拟合按钮
        tk.Button(group_fit, text="单条 G3 螺线拟合 (最小二乘)", command=self.do_g3_spiral_fit, bg="#e8f5e9",
                  activebackground="#c8e6c9", font=("Helvetica", 9, "bold"), fg="#2e7d32").pack(fill="x", pady=2)

        tk.Button(group_fit, text="三角函数拟合", command=self.do_trig_fit, bg="#ffffff",
                  activebackground="#eaeaea").pack(fill="x", pady=2)
        tk.Button(group_fit, text="三次样条拟合", command=lambda: self.do_spline_fit(3), bg="#ffffff",
                  activebackground="#eaeaea").pack(fill="x", pady=2)
        tk.Button(group_fit, text="五次样条拟合", command=lambda: self.do_spline_fit(5), bg="#ffffff",
                  activebackground="#eaeaea").pack(fill="x", pady=2)

        # Hermite插值分组
        group_hermite = tk.LabelFrame(left_panel, text="Hermite分段插值", font=("Helvetica", 9, "bold"), bg="#f5f5f7",
                                      fg="#555555", padx=8, pady=8)
        group_hermite.pack(fill="x", padx=10, pady=4)

        v_frame = tk.Frame(group_hermite, bg="#f5f5f7")
        v_frame.pack(fill="x", pady=2)
        tk.Label(v_frame, text="恒定线速度 V (px/s):", bg="#f5f5f7", font=("Helvetica", 8, "bold")).pack(side="left")
        self.entry_v = tk.Entry(v_frame, width=8)
        self.entry_v.insert(0, "200")
        self.entry_v.pack(side="right")

        a_frame = tk.Frame(group_hermite, bg="#f5f5f7")
        a_frame.pack(fill="x", pady=2)
        tk.Label(a_frame, text="向心加速度 A (px/s²):", bg="#f5f5f7", font=("Helvetica", 8)).pack(side="left")
        self.entry_a = tk.Entry(a_frame, width=8)
        self.entry_a.insert(0, "100")
        self.entry_a.pack(side="right")

        # 加速度输入框
        j_frame = tk.Frame(group_hermite, bg="#f5f5f7")
        j_frame.pack(fill="x", pady=2)
        tk.Label(j_frame, text="加加速度 J (px/s³):", bg="#f5f5f7", font=("Helvetica", 8)).pack(side="left")
        self.entry_j = tk.Entry(j_frame, width=8)
        self.entry_j.insert(0, "50")
        self.entry_j.pack(side="right")

        tk.Button(group_hermite, text="三次 Hermite 插值", command=lambda: self.do_hermite_interpolation(3),
                  bg="#ffffff", activebackground="#eaeaea").pack(fill="x", pady=2)
        tk.Button(group_hermite, text="五次 Hermite 插值", command=lambda: self.do_hermite_interpolation(5),
                  bg="#ffffff", activebackground="#eaeaea").pack(fill="x", pady=2)

        # 七次 Hermite 插值按钮
        tk.Button(group_hermite, text="七次 Hermite 插值", command=lambda: self.do_hermite_interpolation(7),
                  bg="#ffffff", activebackground="#eaeaea").pack(fill="x", pady=2)

        # 系统操作
        group_sys = tk.Frame(left_panel, bg="#f5f5f7")
        group_sys.pack(side="bottom", fill="x", padx=10, pady=10)

        tk.Button(group_sys, text="撤销上一步", command=self.undo, bg="#ffe4e1", activebackground="#fcd0cd").pack(
            fill="x", pady=2)
        tk.Button(group_sys, text="清除所有内容", command=self.clear_all, bg="#ffcccb",
                  activebackground="#fca5a2").pack(fill="x", pady=2)

        # ------------------ 2. 右侧画板与顶部工具栏 ------------------
        right_panel = tk.Frame(self.root, bg="#ffffff")
        right_panel.pack(side="right", fill="both", expand=True)

        # 顶部工具栏
        toolbar = tk.Frame(right_panel, height=45, bg="#eaeaea")
        toolbar.pack(side="top", fill="x")
        toolbar.pack_propagate(False)

        # 功能选择按钮
        self.btn_ctrl = tk.Button(toolbar, text="添加拟合控制点", command=lambda: self.set_tool("ctrl_point"),
                                  relief="sunken", bg="#d0d0d0")
        self.btn_ctrl.pack(side="left", padx=5, pady=5)
        self.btn_point = tk.Button(toolbar, text="普通画点", command=lambda: self.set_tool("point"), bg="#fcfcfc")
        self.btn_point.pack(side="left", padx=5, pady=5)
        self.btn_line = tk.Button(toolbar, text="画直线", command=lambda: self.set_tool("line"), bg="#fcfcfc")
        self.btn_line.pack(side="left", padx=5, pady=5)
        self.btn_arc = tk.Button(toolbar, text="三点画圆弧", command=lambda: self.set_tool("arc"), bg="#fcfcfc")
        self.btn_arc.pack(side="left", padx=5, pady=5)

        # 分割线
        tk.Frame(toolbar, width=2, bg="#cccccc").pack(side="left", fill="y", padx=10, pady=5)

        # 颜色选择按钮
        tk.Label(toolbar, text="画笔颜色:", bg="#eaeaea", font=("Helvetica", 9)).pack(side="left", padx=2)
        colors = [("黑色", "black"), ("红色", "red"), ("绿色", "green"), ("蓝色", "blue")]
        for name, col in colors:
            btn = tk.Button(toolbar, text=name, fg="white", bg=col, activebackground=col,
                            command=lambda c=col: self.set_color(c))
            btn.pack(side="left", padx=3, pady=5)

        # 分割线
        tk.Frame(toolbar, width=2, bg="#cccccc").pack(side="left", fill="y", padx=10, pady=5)

        # 隐藏控制点复选框
        self.hide_ctrl_var = tk.BooleanVar(value=False)
        self.chk_hide_ctrl = tk.Checkbutton(toolbar, text="隐藏控制点", variable=self.hide_ctrl_var,
                                            command=self.toggle_control_points, bg="#eaeaea",
                                            activebackground="#eaeaea", font=("Helvetica", 9))
        self.chk_hide_ctrl.pack(side="left", padx=5, pady=5)

        # 分割线
        tk.Frame(toolbar, width=2, bg="#cccccc").pack(side="left", fill="y", padx=10, pady=5)

        # 开始下一段拟合 按钮
        self.btn_next_seg = tk.Button(toolbar, text="开始下一段拟合", command=self.start_next_segment, bg="#e1f5fe",
                                      activebackground="#b3e5fc", font=("Helvetica", 9, "bold"), fg="#0277bd")
        self.btn_next_seg.pack(side="left", padx=5, pady=5)

        # 状态栏
        self.status = tk.Label(right_panel,
                               text="状态: 准备就绪。当前模式 [添加拟合控制点]，在画布上点击添加定位点进行拟合。", bd=1,
                               relief="sunken", anchor="w", font=("Helvetica", 9), bg="#f5f5f7")
        self.status.pack(side="bottom", fill="x")

        # ------------------ 3. 上下分割绘制区域 (轨迹画板 + 运动学分析图) ------------------
        paned = ttk.PanedWindow(right_panel, orient=tk.VERTICAL)
        paned.pack(side="top", fill="both", expand=True)

        # 上半区域：轨迹 Canvas
        frame_traj = tk.Frame(paned, bg="#ffffff")
        paned.add(frame_traj, weight=3)

        self.canvas = tk.Canvas(frame_traj, bg="#ffffff", bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self.draw_grid())
        self.canvas.bind("<Motion>", self.on_canvas_mouse_move)
        self.canvas.bind("<Leave>", self.on_canvas_mouse_leave)

        # 绑定右键拖拽移动背景图事件 (兼容多平台)
        self.canvas.bind("<Button-3>", self.on_right_click_press)
        self.canvas.bind("<B3-Motion>", self.on_right_click_drag)
        self.canvas.bind("<Button-2>", self.on_right_click_press)
        self.canvas.bind("<B2-Motion>", self.on_right_click_drag)

        # 下半区域：运动学物理量分析画板 (线速度 V, 向心加速度 a_n, 向心加加速度变化率 j_n)
        frame_slope = tk.Frame(paned, bg="#ffffff")
        paned.add(frame_slope, weight=2)

        self.slope_canvas = tk.Canvas(frame_slope, bg="#ffffff", bd=0, highlightthickness=0)
        self.slope_canvas.pack(fill="both", expand=True)
        self.slope_canvas.bind("<Configure>", lambda e: self.redraw_kinematics_chart())
        self.slope_canvas.bind("<Motion>", self.on_slope_mouse_move)
        self.slope_canvas.bind("<Leave>", self.on_slope_mouse_leave)

        self.root.update()
        self.draw_grid()

    def draw_grid(self):
        """
        绘制带有像素坐标数值刻度的标尺与网格线
        """
        self.canvas.delete("grid")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w <= 0 or h <= 0:
            return

        # 1. 绘制辅助细网格线 (每 50 像素)
        for x in range(0, w, 50):
            if x % 100 != 0:
                self.canvas.create_line(x, 0, x, h, fill="#f5f5f8", tags="grid")
        for y in range(0, h, 50):
            if y % 100 != 0:
                self.canvas.create_line(0, y, w, y, fill="#f5f5f8", tags="grid")

        # 2. 绘制主标尺刻度线与坐标数值 (每 100 像素)
        for x in range(0, w, 100):
            # X 轴竖向虚线网格
            self.canvas.create_line(x, 0, x, h, fill="#e1e1e6", dash=(2, 2), tags="grid")
            # 在顶部标注 X 轴像素坐标数值
            if x > 0:
                self.canvas.create_text(x + 12, 12, text=f"X:{x}", fill="#666666",
                                        font=("Helvetica", 8, "bold"), tags="grid")

        for y in range(0, h, 100):
            # Y 轴横向虚线网格
            self.canvas.create_line(0, y, w, y, fill="#e1e1e6", dash=(2, 2), tags="grid")
            # 在左侧标注 Y 轴像素坐标数值
            if y > 0:
                self.canvas.create_text(22, y + 8, text=f"Y:{y}", fill="#666666",
                                        font=("Helvetica", 8, "bold"), tags="grid")

        self._restack_layers()

    def _restack_layers(self):
        """
        严格规范画布图层顺序：背景图 -> 网格 -> 用户绘图 -> 拟合高亮线 -> 控制点
        """
        self.canvas.tag_lower("grid")
        self.canvas.tag_lower("bg_image")

        if self.canvas_last_mouse is not None:
            self._draw_canvas_hover(*self.canvas_last_mouse)

    def on_canvas_mouse_move(self, event):
        self.canvas_last_mouse = (event.x, event.y)
        self._draw_canvas_hover(event.x, event.y)

    def on_canvas_mouse_leave(self, event=None):
        self.canvas_last_mouse = None
        if hasattr(self, 'canvas'):
            self.canvas.delete("canvas_hover")

    def _draw_canvas_hover(self, x, y):
        self.canvas.delete("canvas_hover")
        w = self.canvas.winfo_width()
        if w <= 0:
            return

        text = f"X = {x:.1f}  Y = {y:.1f}"
        box_w = 150
        box_h = 28
        x2 = w - 10
        y1 = 10
        x1 = max(10, x2 - box_w)
        y2 = y1 + box_h

        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ffffff", outline="#c7c7cc",
                                     width=1, tags="canvas_hover")
        self.canvas.create_text(x2 - 8, y1 + 7, text=text, anchor="ne",
                                font=("Helvetica", 9, "bold"), fill="#1c1c1e",
                                tags="canvas_hover")
        self.canvas.tag_raise("canvas_hover")

    # ------------------ 背景图片导入与操作 ------------------

    def import_image(self):
        """
        从文件系统中导入任意背景图片
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            try:
                self.orig_img = Image.open(file_path)
                # 复位空间变量
                self.img_x = 400
                self.img_y = 300
                self.img_scale = 1.0
                self.img_angle = 0.0
                self.img_locked = False  # 导入后默认为“未锁定”

                # 同步更新滑块UI和输入框
                self.slider_scale.config(state="normal")
                self.slider_rotate.config(state="normal")
                self.entry_scale.config(state="normal")
                self.entry_rotate.config(state="normal")

                self.slider_scale.set(1.0)
                self.slider_rotate.set(0.0)

                self.entry_scale.delete(0, tk.END)
                self.entry_scale.insert(0, "1.0")
                self.entry_rotate.delete(0, tk.END)
                self.entry_rotate.insert(0, "0.0")

                # 启用背景图控制按钮
                self.btn_lock_img.config(state="normal")
                self.btn_unlock_img.config(state="normal")

                self.update_background_image()
                self.status.config(
                    text="状态: 成功载入背景图。当前处于[未锁定]状态，可通过滑块、右侧数值框或[右键拖拽]调整其位置、大小和旋转。")
            except Exception as e:
                messagebox.showerror("载入失败", f"无法正常打开该图片文件: {str(e)}")

    def lock_image(self):
        """
        锁定图片：无法再通过滑块、输入框或鼠标右键进行位置及姿态调整
        """
        if self.orig_img:
            self.img_locked = True
            self.slider_scale.config(state="disabled")
            self.slider_rotate.config(state="disabled")
            self.entry_scale.config(state="disabled")
            self.entry_rotate.config(state="disabled")
            self.status.config(text="状态: 背景图已被[锁定]。现在可以放心绘制图形或控制点，无法意外触发移动。")

    def unlock_image(self):
        """
        解锁图片：允许用户继续通过拖拽、滑块和输入框调整背景图
        """
        if self.orig_img:
            self.img_locked = False
            self.slider_scale.config(state="normal")
            self.slider_rotate.config(state="normal")
            self.entry_scale.config(state="normal")
            self.entry_rotate.config(state="normal")
            self.status.config(text="状态: 背景图已被[解锁]。可再次进行调整。")

    # 缩放控制（滑块与输入框联动）
    def on_scale_slider_change(self, val):
        if self.orig_img and not self.img_locked:
            val_f = float(val)
            self.img_scale = val_f
            self.entry_scale.delete(0, tk.END)
            self.entry_scale.insert(0, f"{val_f:.1f}")
            self.update_background_image()

    def on_scale_entry_change(self, event=None):
        if self.orig_img and not self.img_locked:
            try:
                val = float(self.entry_scale.get())
                val = max(0.1, min(val, 5.0))
                val = round(val, 1)
                self.img_scale = val
                self.slider_scale.set(val)
                self.entry_scale.delete(0, tk.END)
                self.entry_scale.insert(0, f"{val:.1f}")
                self.update_background_image()
            except ValueError:
                # 恢复之前有效值
                self.entry_scale.delete(0, tk.END)
                self.entry_scale.insert(0, f"{self.img_scale:.1f}")

    # 旋转控制（滑块与输入框联动）
    def on_rotate_slider_change(self, val):
        if self.orig_img and not self.img_locked:
            val_f = float(val)
            self.img_angle = val_f
            self.entry_rotate.delete(0, tk.END)
            self.entry_rotate.insert(0, f"{val_f:.1f}")
            self.update_background_image()

    def on_rotate_entry_change(self, event=None):
        if self.orig_img and not self.img_locked:
            try:
                val = float(self.entry_rotate.get())
                val = max(0.0, min(val, 360.0))
                val = round(val, 1)
                self.img_angle = val
                self.slider_rotate.set(val)
                self.entry_rotate.delete(0, tk.END)
                self.entry_rotate.insert(0, f"{val:.1f}")
                self.update_background_image()
            except ValueError:
                # 恢复之前有效值
                self.entry_rotate.delete(0, tk.END)
                self.entry_rotate.insert(0, f"{self.img_angle:.1f}")

    def on_right_click_press(self, event):
        """
        记录右键拖拽图片的起始坐标
        """
        if self.orig_img and not self.img_locked:
            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def on_right_click_drag(self, event):
        """
        执行右键拖拽位移计算
        """
        if self.orig_img and not self.img_locked:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.img_x += dx
            self.img_y += dy
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.update_background_image()

    def update_background_image(self):
        """
        计算图片的旋转与缩放，并极速渲染到画布最底层
        """
        if self.orig_img is None:
            return
        try:
            rotated = self.orig_img.rotate(-self.img_angle, expand=True, resample=Image.BICUBIC)

            new_w = int(rotated.width * self.img_scale)
            new_h = int(rotated.height * self.img_scale)
            if new_w <= 0 or new_h <= 0:
                return

            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                try:
                    resample_filter = Image.ANTIALIAS
                except AttributeError:
                    resample_filter = Image.BICUBIC

            resized = rotated.resize((new_w, new_h), resample_filter)

            self.tk_image = ImageTk.PhotoImage(resized)
            self.canvas.delete("bg_image")
            self.canvas.create_image(self.img_x, self.img_y, image=self.tk_image, tags="bg_image")

            # 复位图层叠放关系
            self._restack_layers()
        except Exception as e:
            print(f"背景图渲染出现异常: {e}")

    # ------------------ 工具选择与画板点击 ------------------

    def set_tool(self, tool):
        self.active_tool = tool
        self.temp_clicks.clear()

        # 样式复位
        self.btn_ctrl.config(relief="raised", bg="#fcfcfc")
        self.btn_point.config(relief="raised", bg="#fcfcfc")
        self.btn_line.config(relief="raised", bg="#fcfcfc")
        self.btn_arc.config(relief="raised", bg="#fcfcfc")

        if tool == "ctrl_point":
            self.btn_ctrl.config(relief="sunken", bg="#d0d0d0")
            self.status.config(text="状态: 切换到 [添加拟合控制点] 模式。点击即可在画布上增加轨迹拟合的控制点。")
        elif tool == "point":
            self.btn_point.config(relief="sunken", bg="#d0d0d0")
            self.status.config(text="状态: 切换到 [普通画点] 模式。点击画布可留下当前选中颜色的独立打点。")
        elif tool == "line":
            self.btn_line.config(relief="sunken", bg="#d0d0d0")
            self.status.config(text="状态: 切换到 [画直线] 模式。依次点击起点和终点来画出一条直线。")
        elif tool == "arc":
            self.btn_arc.config(relief="sunken", bg="#d0d0d0")
            self.status.config(text="状态: 切换到 [三点画圆弧] 模式。依次点击三点来画出穿过这三点的圆弧。")

    def set_color(self, color):
        self.draw_color = color
        self.status.config(text=f"状态: 笔刷颜色切换为 {color}。")

    def on_canvas_click(self, event):
        x, y = event.x, event.y

        # 1. 普通画点
        if self.active_tool == "point":
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=self.draw_color, outline=self.draw_color,
                                    tags="user_shapes")

        # 2. 画直线
        elif self.active_tool == "line":
            self.temp_clicks.append((x, y))
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=self.draw_color, outline="", tags="temp")
            if len(self.temp_clicks) == 2:
                p1, p2 = self.temp_clicks
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=self.draw_color, width=2, tags="user_shapes")
                self.canvas.delete("temp")
                self.temp_clicks.clear()

        # 3. 三点画圆弧
        elif self.active_tool == "arc":
            self.temp_clicks.append((x, y))
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=self.draw_color, outline="", tags="temp")
            if len(self.temp_clicks) == 3:
                p1, p2, p3 = self.temp_clicks
                self.canvas.delete("temp")
                self.temp_clicks.clear()

                # 计算几何圆弧
                res = fit_circle([p1, p2, p3])
                if res:
                    xc, yc, R = res
                    theta1 = math.atan2(p1[1] - yc, p1[0] - xc)
                    theta2 = math.atan2(p2[1] - yc, p2[0] - xc)
                    theta3 = math.atan2(p3[1] - yc, p3[0] - xc)

                    diff21 = (theta2 - theta1) % (2 * math.pi)
                    diff31 = (theta3 - theta1) % (2 * math.pi)
                    if diff21 < diff31:
                        angles = np.linspace(theta1, theta1 + diff31, 100)
                    else:
                        angles = np.linspace(theta1, theta1 - (2 * math.pi - diff31), 100)

                    arc_pts = [(xc + R * math.cos(a), yc + R * math.sin(a)) for a in angles]
                    for i in range(len(arc_pts) - 1):
                        self.canvas.create_line(arc_pts[i][0], arc_pts[i][1], arc_pts[i + 1][0], arc_pts[i + 1][1],
                                                fill=self.draw_color, width=2, tags="user_shapes")
                else:
                    messagebox.showwarning("警告", "所选三点共线，无法构成圆弧！")

        # 4. 添加拟合控制点 (默认)
        elif self.active_tool == "ctrl_point":
            self.control_points.append((x, y))
            idx = len(self.control_points)

            # 判断当前是否处于“隐藏控制点”勾选状态
            state = "hidden" if self.hide_ctrl_var.get() else "normal"

            # 添加 "current_ctrl" 标签来区分当前激活段与已存段
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ff3b30", outline="#000000", width=1.5,
                                    tags=(f"ctrl_{idx}", "ctrl_all", "current_ctrl"), state=state)
            self.canvas.create_text(x + 10, y - 10, text=f"P{idx}", fill="#333333", font=("Helvetica", 9, "bold"),
                                    tags=(f"ctrl_{idx}", "ctrl_all", "current_ctrl"), state=state)
            self.status.config(text=f"状态: 已成功添加控制点 P{idx} ({x}, {y})。")

    # ------------------ 系统清除与多段操作 ------------------

    def toggle_control_points(self):
        """
        隐藏/显示所有控制点和标签
        """
        state = "hidden" if self.hide_ctrl_var.get() else "normal"
        self.canvas.itemconfigure("ctrl_all", state=state)
        self.status.config(
            text="状态: 已隐藏所有轨迹控制点。" if self.hide_ctrl_var.get() else "状态: 控制点已重新显示。")

    def start_next_segment(self):
        """
        将当前高亮拟合结果进行静态归档保存，并重置控制点列表以便开启下一段轨迹的重新标定
        """
        if not self.control_points:
            messagebox.showinfo("提示", "当前分段没有控制点，无需开启新分段。")
            return

        # 重置运动学物理量分析图 (开启新分段时不覆盖显示)
        self.current_kinematics_data = None
        self.redraw_kinematics_chart()

        # 1. 保存当前的高亮拟合曲线图层 (重新打标以避开后续 do_fit 时自动删除 highlight_layer 的动作)
        self.canvas.addtag_withtag("saved_fit", "highlight_layer")
        self.canvas.dtag("saved_fit", "highlight_layer")

        # 2. 将当前控制点“静态化”重置为已保存控制点
        self.canvas.addtag_withtag("saved_ctrl", "current_ctrl")

        # 3. 将已存段的控制点变为低调的暗灰色，防止与新控制点的红色冲突，影响视觉辨识
        for item in self.canvas.find_withtag("current_ctrl"):
            if self.canvas.type(item) == "oval":
                self.canvas.itemconfigure(item, fill="#9e9e9e", outline="#616161")
            elif self.canvas.type(item) == "text":
                self.canvas.itemconfigure(item, fill="#757575")

        self.canvas.dtag("saved_ctrl", "current_ctrl")

        # 4. 清空活跃的控制点数据，准备开始下一段拟合
        self.control_points.clear()
        self.status.config(text="状态: 已成功保存当前轨迹段。您可以重新打点并选择算法开始下一段拟合。")

    def undo(self):
        if self.active_tool == "ctrl_point" and self.control_points:
            idx = len(self.control_points)
            self.canvas.delete(f"ctrl_{idx}")
            self.control_points.pop()
            if not self.control_points:
                self.current_kinematics_data = None
                self.redraw_kinematics_chart()
            self.status.config(text=f"状态: 已撤销控制点 P{idx}。")
        else:
            self.canvas.delete("user_shapes")
            self.status.config(text="状态: 已清除用户自绘图层。")

    def clear_all(self):
        """
        全面重置复位画布及所有变量状态
        """
        # 如果当前有背景图片，弹出带“是/否/取消”的确认对话框
        if self.orig_img is not None:
            ans = messagebox.askyesnocancel(
                "清除确认",
                "是否在清除时保留当前的背景图片？\n\n"
                "【是】 保留背景图，仅清除轨迹与控制点。\n"
                "【否】 全部清除（包括背景图）。\n"
                "【取消】 取消本次清除操作。"
            )
            if ans is None:
                self.status.config(text="状态: 已取消清除操作。")
                return
            elif ans is True:
                # 只清除轨迹和点，重新渲染网格和背景图
                self.canvas.delete("all")
                self.control_points.clear()
                self.temp_clicks.clear()

                # 清除运动学分析图
                self.current_kinematics_data = None
                self.redraw_kinematics_chart()

                self.draw_grid()
                self.update_background_image()
                self.status.config(text="状态: 已成功清除所有轨迹与控制点，背景图已保留。")
                return
            else:
                # 用户点击了“否”：执行完全清除
                pass

        # 执行完全清除（没有背景图，或者用户点击了“否”）
        self.canvas.delete("all")
        self.control_points.clear()
        self.temp_clicks.clear()

        # 运动学数据全面复位
        self.current_kinematics_data = None
        self.redraw_kinematics_chart()

        # 背景图归零复位
        self.orig_img = None
        self.tk_image = None
        self.img_x = 400
        self.img_y = 300
        self.img_scale = 1.0
        self.img_angle = 0.0
        self.img_locked = True

        # 控制点隐藏状态复位
        self.hide_ctrl_var.set(False)

        # 复位背景图控制控件
        self.btn_lock_img.config(state="disabled")
        self.btn_unlock_img.config(state="disabled")

        self.slider_scale.config(state="disabled")
        self.slider_scale.set(1.0)
        self.entry_scale.config(state="normal")
        self.entry_scale.delete(0, tk.END)
        self.entry_scale.insert(0, "1.0")
        self.entry_scale.config(state="disabled")

        self.slider_rotate.config(state="disabled")
        self.slider_rotate.set(0.0)
        self.entry_rotate.config(state="normal")
        self.entry_rotate.delete(0, tk.END)
        self.entry_rotate.insert(0, "0.0")
        self.entry_rotate.config(state="disabled")

        # 复位加加速度输入框
        self.entry_j.config(state="normal")
        self.entry_j.delete(0, tk.END)
        self.entry_j.insert(0, "50")

        self.draw_grid()
        self.status.config(text="状态: 全画板已全面复位重置（包含所有历史拟合段和背景图）。")

    # ------------------ 运动学物理量 (线速度、向心加速度、Jerk变化率) 分析与绘图 ------------------

    def update_kinematics_data(self, dense_points):
        """
        根据假设的恒定线速度 V，计算轨迹上各点的向心加速度 a_n 与向心加加速度变化率 j_n (Jerk)
        - 线速度 V: 由 UI 框输入 (恒定值, px/s)
        - 向心加速度 a_n: V^2 * curvature (px/s^2)
        - 向心加加速度变化率 j_n: d(a_n)/dt = V * d(a_n)/ds (px/s^3)
        """
        if dense_points is None or len(dense_points) < 5:
            self.current_kinematics_data = None
            self.redraw_kinematics_chart()
            return

        # 尝试读取左侧面板输入的线速度 V
        try:
            V_line = float(self.entry_v.get())
            if V_line <= 0:
                V_line = 200.0
        except ValueError:
            V_line = 200.0

        pts = np.array(dense_points)
        n = len(pts)

        # 1. 计算累计弧长 s
        ds = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        s = np.zeros(n)
        s[1:] = np.cumsum(ds)

        # 2. 计算几何曲率 kappa (基于 Menger 3点外接圆曲率)
        kappa = np.zeros(n)
        for i in range(1, n - 1):
            p_prev, p_curr, p_next = pts[i - 1], pts[i], pts[i + 1]
            a = np.linalg.norm(p_prev - p_curr)
            b = np.linalg.norm(p_curr - p_next)
            c = np.linalg.norm(p_prev - p_next)

            # 三角形面积
            area = 0.5 * abs(p_prev[0] * (p_curr[1] - p_next[1]) +
                             p_curr[0] * (p_next[1] - p_prev[1]) +
                             p_next[0] * (p_prev[1] - p_curr[1]))

            denom = a * b * c
            if denom > 1e-6:
                kappa[i] = (4.0 * area) / denom
            else:
                kappa[i] = 0.0

        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]

        # 移动平均平滑滤波，消除离散采样噪点
        def smooth(vals, window=5):
            if len(vals) < window:
                return vals
            kernel = np.ones(window) / window
            return np.convolve(vals, kernel, mode='same')

        kappa = smooth(kappa, window=5)

        # 3. 计算物理量
        # (1) 恒定线速度 V
        v_arr = np.full(n, V_line)

        # (2) 向心加速度 a_n = V^2 * kappa (px/s^2)
        a_n = (V_line ** 2) * kappa

        # (3) 向心加加速度变化率 j_n = d(a_n)/dt = V * d(a_n)/ds (px/s^3)
        j_n = np.zeros(n)
        for i in range(1, n - 1):
            delta_s = s[i + 1] - s[i - 1]
            if delta_s > 1e-5:
                j_n[i] = V_line * (a_n[i + 1] - a_n[i - 1]) / delta_s
            else:
                j_n[i] = 0.0

        j_n[0] = j_n[1]
        j_n[-1] = j_n[-2]
        j_n = smooth(j_n, window=5)

        x_vals = pts[:, 0]

        # 打包运动学数据 (针对 X 位置)
        self.current_kinematics_data = {
            'v': (x_vals, v_arr, "#007aff", f"线速度 V ({V_line:.0f} px/s)"),
            'a_n': (x_vals, a_n, "#ff9500", "向心加速度 a_n (px/s²)"),
            'j_n': (x_vals, j_n, "#34c759", "向心加加速度变化率 j_n (px/s³)")
        }
        self.redraw_kinematics_chart()

    def redraw_kinematics_chart(self):
        """
        在下半部分画布上绘制假设匀速通过轨迹时的 [线速度 / 向心加速度 / 向心加加速度变化率]
        """
        if not hasattr(self, 'slope_canvas'):
            return

        self.slope_canvas.delete("all")
        w = self.slope_canvas.winfo_width()
        h = self.slope_canvas.winfo_height()

        if w < 50 or h < 50:
            return

        pad_l, pad_r, pad_t, pad_b = 60, 30, 30, 35
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        self.slope_plot_bounds = {
            "pad_l": pad_l,
            "pad_r": pad_r,
            "pad_t": pad_t,
            "pad_b": pad_b,
            "plot_w": plot_w,
            "plot_h": plot_h,
        }

        # 标题栏
        self.slope_canvas.create_text(
            w / 2, 14,
            text="匀速过弯运动学物理量分析 (恒定线速度 V / 向心加速度 a_n / 向心加加速度变化率 j_n vs X 轴位置)",
            font=("Helvetica", 10, "bold"),
            fill="#333333"
        )

        if self.current_kinematics_data is None or plot_w <= 0 or plot_h <= 0:
            self.slope_canvas.create_rectangle(pad_l, pad_t, w - pad_r, h - pad_b, outline="#dddddd", width=1)
            self.slope_canvas.create_text(
                w / 2, h / 2,
                text="暂无运动学数据，请添加控制点进行轨迹拟合/插值",
                font=("Helvetica", 9),
                fill="#888888"
            )
            return

        # 收集全部 X 位置和全部物理数值求最值（同图映射坐标轴）
        all_x, all_v = [], []
        for key, (xs, ys, _, _) in self.current_kinematics_data.items():
            all_x.extend(xs)
            all_v.extend(ys)

        if not all_x or not all_v:
            return

        min_x, max_x = min(all_x), max(all_x)
        min_v, max_v = min(all_v), max(all_v)

        # 扩展极值边界
        if min_x == max_x:
            min_x -= 10
            max_x += 10
        else:
            dx_margin = (max_x - min_x) * 0.05
            min_x -= dx_margin
            max_x += dx_margin

        if min_v == max_v:
            min_v -= 1.0
            max_v += 1.0
        else:
            dv_margin = (max_v - min_v) * 0.1
            min_v -= dv_margin
            max_v += dv_margin

        self.slope_plot_bounds.update({
            "min_x": min_x,
            "max_x": max_x,
            "min_v": min_v,
            "max_v": max_v,
        })

        # 绘制背景底框与参考网格
        self.slope_canvas.create_rectangle(pad_l, pad_t, w - pad_r, h - pad_b, outline="#cccccc", fill="#fafafa", width=1)

        # 绘制 Zero (数值=0) 物理基准线
        if min_v <= 0 <= max_v:
            y0 = pad_t + plot_h * (1.0 - (0 - min_v) / (max_v - min_v))
            self.slope_canvas.create_line(pad_l, y0, w - pad_r, y0, fill="#aaaaaa", dash=(4, 4), width=1.5)

        # 绘制 Y 轴刻度 (物理数值)
        y_ticks = 5
        for i in range(y_ticks + 1):
            val_v = min_v + i * (max_v - min_v) / y_ticks
            cy = pad_t + plot_h * (1.0 - i / y_ticks)
            self.slope_canvas.create_line(pad_l - 3, cy, pad_l, cy, fill="#666666")
            self.slope_canvas.create_line(pad_l, cy, w - pad_r, cy, fill="#e8e8e8", dash=(2, 2))
            self.slope_canvas.create_text(pad_l - 6, cy, text=f"{val_v:.1f}", anchor="e", font=("Helvetica", 8), fill="#555555")

        # 绘制 X 轴刻度 (X像素位置)
        x_ticks = 6
        for i in range(x_ticks + 1):
            val_x = min_x + i * (max_x - min_x) / x_ticks
            cx = pad_l + i * plot_w / x_ticks
            cy_bot = h - pad_b
            self.slope_canvas.create_line(cx, cy_bot, cx, cy_bot + 3, fill="#666666")
            self.slope_canvas.create_line(cx, pad_t, cx, cy_bot, fill="#e8e8e8", dash=(2, 2))
            self.slope_canvas.create_text(cx, cy_bot + 12, text=f"{val_x:.0f}", anchor="n", font=("Helvetica", 8), fill="#555555")

        # 坐标轴标识
        self.slope_canvas.create_text(pad_l - 40, pad_t - 10, text="物理量数值", anchor="w", font=("Helvetica", 8, "bold"), fill="#333333")
        self.slope_canvas.create_text(w - pad_r, h - pad_b + 22, text="位置 X (px)", anchor="e", font=("Helvetica", 8, "bold"), fill="#333333")

        # 按顺序绘制三组运动学物理曲线 (v: 深蓝, a_n: 橙色, j_n: 绿色)
        order_keys = ['v', 'a_n', 'j_n']
        widths = {'v': 2.5, 'a_n': 2.0, 'j_n': 1.5}

        for key in order_keys:
            if key not in self.current_kinematics_data:
                continue
            xs, ys, color, label = self.current_kinematics_data[key]
            pts_canvas = []
            for xi, vi in zip(xs, ys):
                cx = pad_l + (xi - min_x) / (max_x - min_x) * plot_w
                cy = pad_t + plot_h * (1.0 - (vi - min_v) / (max_v - min_v))
                pts_canvas.append((cx, cy))

            for i in range(len(pts_canvas) - 1):
                p1, p2 = pts_canvas[i], pts_canvas[i + 1]
                self.slope_canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=widths[key])

        # 右上角绘制物理图例 (Legend)
        leg_x = w - pad_r - 10
        leg_y = pad_t + 10
        for idx, key in enumerate(order_keys):
            if key not in self.current_kinematics_data:
                continue
            _, _, color, label = self.current_kinematics_data[key]
            ly = leg_y + idx * 16
            self.slope_canvas.create_line(leg_x - 170, ly, leg_x - 140, ly, fill=color, width=2.5)
            self.slope_canvas.create_text(leg_x - 135, ly, text=label, anchor="w", font=("Helvetica", 8, "bold"), fill="#333333")

        if self.slope_last_mouse is not None:
            self._draw_slope_hover(*self.slope_last_mouse)

    def on_slope_mouse_move(self, event):
        self.slope_last_mouse = (event.x, event.y)
        self._draw_slope_hover(event.x, event.y)

    def on_slope_mouse_leave(self, event=None):
        self.slope_last_mouse = None
        self.slope_hover_info = None
        if hasattr(self, 'slope_canvas'):
            self.slope_canvas.delete("slope_hover")

    def _draw_slope_hover(self, x, y):
        if not self.slope_plot_bounds or self.current_kinematics_data is None:
            self.on_slope_mouse_leave()
            return

        pad_l = self.slope_plot_bounds["pad_l"]
        pad_r = self.slope_plot_bounds["pad_r"]
        pad_t = self.slope_plot_bounds["pad_t"]
        pad_b = self.slope_plot_bounds["pad_b"]
        plot_w = self.slope_plot_bounds["plot_w"]
        plot_h = self.slope_plot_bounds["plot_h"]
        min_x = self.slope_plot_bounds["min_x"]
        max_x = self.slope_plot_bounds["max_x"]
        min_v = self.slope_plot_bounds["min_v"]
        max_v = self.slope_plot_bounds["max_v"]

        plot_left = pad_l
        plot_top = pad_t
        plot_right = self.slope_canvas.winfo_width() - pad_r
        plot_bottom = self.slope_canvas.winfo_height() - pad_b

        if x < plot_left or x > plot_right or y < plot_top or y > plot_bottom:
            self.slope_canvas.delete("slope_hover")
            self.slope_hover_info = None
            return

        if max_x == min_x or max_v == min_v:
            return

        data_x = min_x + (x - pad_l) / plot_w * (max_x - min_x)
        data_y = max_v - (y - pad_t) / plot_h * (max_v - min_v)

        parts = [f"X = {data_x:.1f}    Y = {data_y:.1f}"]

        curve_text = []
        for key in ["v", "a_n", "j_n"]:
            if key not in self.current_kinematics_data:
                continue
            xs, ys, _, _ = self.current_kinematics_data[key]
            if len(xs) < 2:
                continue
            order = np.argsort(xs)
            xs_sorted = np.asarray(xs)[order]
            ys_sorted = np.asarray(ys)[order]
            curve_y = float(np.interp(data_x, xs_sorted, ys_sorted))
            curve_text.append(f"{key}: {curve_y:.1f}")

        if curve_text:
            parts.append("    ".join(curve_text))

        text = "\n".join(parts)
        self.slope_hover_info = text

        self.slope_canvas.delete("slope_hover")
        box_w = 330
        box_h = 44 if curve_text else 28
        x2 = self.slope_canvas.winfo_width() - 10
        y1 = 10
        x1 = max(10, x2 - box_w)
        y2 = y1 + box_h

        self.slope_canvas.create_rectangle(x1, y1, x2, y2, fill="#ffffff", outline="#c7c7cc",
                                           width=1, tags="slope_hover")
        self.slope_canvas.create_text(x2 - 8, y1 + 8, text=text, anchor="ne",
                                      font=("Helvetica", 8, "bold"), fill="#1c1c1e",
                                      tags="slope_hover")
        self.slope_canvas.tag_raise("slope_hover")

    # ------------------ 曲线高亮绘制层 ------------------

    def draw_fit_highlight(self, dense_points):
        """
        高亮渲染当前激活段的计算轨迹，并同步更新运动学分析图
        """
        self.canvas.delete("highlight_layer")
        if dense_points is None or len(dense_points) < 2:
            return

        # 1. 同步计算并更新运动学物理量分析图表
        self.update_kinematics_data(dense_points)

        # 2. 投影微影层，获得立体悬浮质感
        for i in range(len(dense_points) - 1):
            p1, p2 = dense_points[i], dense_points[i + 1]
            self.canvas.create_line(p1[0], p1[1] + 1.5, p2[0], p2[1] + 1.5, fill="#555555", width=5.5,
                                    tags="highlight_layer")

        # 3. 黄色核心高亮层
        for i in range(len(dense_points) - 1):
            p1, p2 = dense_points[i], dense_points[i + 1]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#ffd60a", width=3.5, tags="highlight_layer")

        self.canvas.tag_raise("highlight_layer")
        # 确保红色的当前控制点层永远浮动在最上层 (除非当前被勾选隐藏)
        for idx in range(1, len(self.control_points) + 1):
            self.canvas.tag_raise(f"ctrl_{idx}")

    # ------------------ 算法调用与绘图集成 ------------------

    def do_circle_fit(self):
        if len(self.control_points) < 3:
            messagebox.showwarning("数据不足", "圆弧拟合需要至少 3 个控制点。")
            return

        res = fit_circle(self.control_points)
        if res:
            xc, yc, R = res
            pts_all = [(xc + R * math.cos(a), yc + R * math.sin(a)) for a in np.linspace(0, 2 * np.pi, 200)]
            self.draw_fit_highlight(pts_all)
            self.status.config(text=f"状态: 圆弧拟合成功。圆心: ({xc:.1f}, {yc:.1f}), 半径 R: {R:.1f} 像素。")
        else:
            messagebox.showerror("计算错误", "拟合计算失败，请确保控制点集不要完全共线。")

    def do_semicircle_fit(self):
        """
        执行半圆拟合：起点为 P1，终点为 P3，中间控制点 P2 决定半圆所在的半平面。
        限制输入点数必须刚好为 3 个点。
        """
        if len(self.control_points) != 3:
            messagebox.showwarning(
                "点数不匹配",
                "半圆拟合需要刚好 3 个控制点：\n- P1：起点\n- P3：终点\n- P2：确定弯曲半平面"
            )
            return

        pts = fit_semicircle(self.control_points)
        if pts:
            self.draw_fit_highlight(pts)
            p1, _, p3 = self.control_points
            dist = np.linalg.norm(np.array(p1) - np.array(p3))
            self.status.config(text=f"状态: 半圆拟合完成。直径端点: P1-P3, 直径长度: {dist:.1f} 像素。")
        else:
            messagebox.showerror("计算错误", "半圆拟合解算异常，请重试。")

    def do_g3_spiral_fit(self):
        """
        执行单条 G3 二次回旋螺线最小二乘拟合 (True Least-Squares Fitting)
        注意：这是拟合(Approximation)，不是插值(Interpolation)！
        采用单条 G3 螺线全局逼近所有控制点，最小化残差平方和，保证全曲线 G3 连续。
        """
        if len(self.control_points) < 3:
            messagebox.showwarning("数据不足", "单条 G3 螺线拟合需要至少 3 个控制点。")
            return

        pts = fit_g3_spiral_least_squares(self.control_points)
        if pts:
            self.draw_fit_highlight(pts)
            self.status.config(text="状态: 单条 G3 螺线最小二乘拟合完成。全局平滑逼近控制点，实现零 Jerk 冲击。")
        else:
            messagebox.showerror("计算错误", "G3 螺线求解异常，请微调控制点。")

    def do_trig_fit(self):
        if len(self.control_points) < 3:
            messagebox.showwarning("数据不足", "三角函数逼近需要至少 3 个控制点。")
            return

        popt = fit_trig(self.control_points)
        if popt:
            A, B, C, D = popt
            x_vals = [p[0] for p in self.control_points]
            min_x, max_x = min(x_vals) - 50, max(x_vals) + 50

            x_dense = np.linspace(min_x, max_x, 300)
            y_dense = A * np.sin(B * x_dense + C) + D

            dense_points = list(zip(x_dense, y_dense))
            self.draw_fit_highlight(dense_points)
            self.status.config(text=f"状态: 三角函数拟合完成。公式: y = {A:.1f}*sin({B:.4f}*x + {C:.2f}) + {D:.1f}。")
        else:
            messagebox.showerror("计算错误", "拟合解算产生异常，请微调控制点。")

    def do_spline_fit(self, degree):
        if len(self.control_points) < degree + 1:
            messagebox.showwarning("数据不足", f"{degree}次样条拟合需要至少 {degree + 1} 个控制点。")
            return

        res = fit_spline(self.control_points, degree)
        if res:
            spl_x, spl_y = res
            u_dense = np.linspace(0, 1, 300)
            x_dense = spl_x(u_dense)
            y_dense = spl_y(u_dense)

            dense_points = list(zip(x_dense, y_dense))
            self.draw_fit_highlight(dense_points)
            self.status.config(text=f"状态: 参数化 {degree} 次样条曲线拟合成功。")
        else:
            messagebox.showerror("计算错误", "样条求解异常，请重试。")

    def do_hermite_interpolation(self, degree):
        n = len(self.control_points)
        if n < 2:
            messagebox.showwarning("数据不足", "Hermite 分段插值需要至少 2 个控制点。")
            return

        try:
            V_avg = float(self.entry_v.get())
            A_avg = float(self.entry_a.get())
            J_avg = float(self.entry_j.get())
            if V_avg <= 0 or A_avg < 0 or J_avg < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "请确认速度（正数）、加速度（非负数）和加加速度（非负数）输入合规。")
            return

        pts = np.array(self.control_points)
        t = np.zeros(n)
        for i in range(1, n):
            dist = np.linalg.norm(pts[i] - pts[i - 1])
            dt = dist / V_avg if V_avg > 0 else 1.0
            t[i] = t[i - 1] + dt

        velocities = np.zeros_like(pts)
        for i in range(n):
            if i == 0:
                direction = pts[1] - pts[0]
            elif i == n - 1:
                direction = pts[-1] - pts[-2]
            else:
                dir1 = (pts[i] - pts[i - 1]) / np.linalg.norm(pts[i] - pts[i - 1])
                dir2 = (pts[i + 1] - pts[i]) / np.linalg.norm(pts[i + 1] - pts[i])
                direction = dir1 + dir2
            norm_val = np.linalg.norm(direction)
            if norm_val > 0:
                direction = direction / norm_val
            velocities[i] = direction * V_avg

        if degree == 3:
            try:
                spl = CubicHermiteSpline(t, pts, velocities)
                t_dense = np.linspace(0, t[-1], 300)
                dense_points = spl(t_dense)
                self.draw_fit_highlight(dense_points)
                self.status.config(text=f"状态: 三次 Hermite 插值完成。")
            except Exception as e:
                messagebox.showerror("计算错误", f"插值发生错误: {str(e)}")

        elif degree == 5:
            accelerations = np.zeros_like(pts)
            for i in range(n):
                if i == 0 or i == n - 1:
                    accelerations[i] = np.array([0.0, 0.0])
                else:
                    normal_dir = get_curvature_normal(pts[i - 1], pts[i], pts[i + 1])
                    accelerations[i] = normal_dir * A_avg
            try:
                spl = QuinticHermiteSpline(t, pts, velocities, accelerations)
                t_dense = np.linspace(0, t[-1], 300)
                dense_points = spl(t_dense)
                self.draw_fit_highlight(dense_points)
                self.status.config(text=f"状态: 五次 Hermite 插值完成。")
            except Exception as e:
                messagebox.showerror("计算错误", f"分段五次 Hermite 求解异常: {str(e)}")

        elif degree == 7:
            accelerations = np.zeros_like(pts)
            jerks = np.zeros_like(pts)
            for i in range(n):
                v_norm = np.linalg.norm(velocities[i])
                unit_T = velocities[i] / v_norm if v_norm > 0 else np.array([0.0, 0.0])
                # 切向加加速度 (代表减速冲击控制)
                jerks[i] = -unit_T * J_avg

                if i == 0 or i == n - 1:
                    accelerations[i] = np.array([0.0, 0.0])
                    jerks[i] = np.array([0.0, 0.0])
                else:
                    normal_dir = get_curvature_normal(pts[i - 1], pts[i], pts[i + 1])
                    accelerations[i] = normal_dir * A_avg
            try:
                spl = SepticHermiteSpline(t, pts, velocities, accelerations, jerks)
                t_dense = np.linspace(0, t[-1], 300)
                dense_points = spl(t_dense)
                self.draw_fit_highlight(dense_points)
                self.status.config(text=f"状态: 七次 Hermite 插值完成。满足了位置、速度、加速度及加加速度的四重平滑过渡。")
            except Exception as e:
                messagebox.showerror("计算错误", f"分段七次 Hermite 求解异常: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryApp(root)
    root.mainloop()
