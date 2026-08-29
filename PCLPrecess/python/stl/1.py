import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.interpolate import splprep, splev
from scipy.optimize import curve_fit
#几个简单的拟合看斜率变化

# ----------------- 几何与曲率计算 -----------------
def generate_superellipse_upper(n, a, b, num_pts=600):
    t = np.linspace(0, np.pi, num_pts)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    x = a * np.sign(cos_t) * (np.abs(cos_t) ** (2.0 / n))
    y = b * np.sign(sin_t) * (np.abs(sin_t) ** (2.0 / n))
    return x, y


def compute_curvature_and_derivative(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    ds = np.sqrt(dx ** 2 + dy ** 2)
    s = np.cumsum(ds)
    s = (s - s[0]) / (s[-1] - s[0] + 1e-12)

    curvature = np.abs(dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5 + 1e-12)
    dkappa_ds = np.gradient(curvature, s + 1e-12)

    return s, curvature, dkappa_ds


# ----------------- 修复后的拟合算法 -----------------
def fit_spiral(x, y):
    """
    广义外凸螺线：以顶点(0, b)为中心，采用关于 Y 轴对称的极坐标偶次展开
    确保在顶点处导数为0，避免出现中心折角和曲率突变
    """
    theta = np.arctan2(y, x)
    phi = theta - np.pi / 2.0  # 顶点处 phi = 0
    r = np.sqrt(x ** 2 + y ** 2)

    def symmetric_spiral(p, r0, c2, c4):
        return r0 + c2 * (p ** 2) + c4 * (p ** 4)

    try:
        popt, _ = curve_fit(symmetric_spiral, phi, r, p0=[np.max(y), 0.1, 0.01], maxfev=3000)
        r_fit = symmetric_spiral(phi, *popt)
        return r_fit * np.cos(theta), r_fit * np.sin(theta)
    except:
        return x, y


def fit_quintic(x, y):
    p = np.polyfit(x, y, 5)
    return x, np.polyval(p, x)


def fit_7th_poly(x, y):
    p = np.polyfit(x, y, 7)
    return x, np.polyval(p, x)


def fit_bspline(x, y):
    tck, u = splprep([x, y], s=0.0005, k=3)
    u_fine = np.linspace(0, 1, len(x))
    new_pts = splev(u_fine, tck)
    return new_pts[0], new_pts[1]


def fit_bezier_7th(x, y):
    def bezier_basis(u, n, i):
        import math
        return math.comb(n, i) * (u ** i) * ((1 - u) ** (n - i))

    u = np.linspace(0, 1, len(x))
    A = np.zeros((len(x), 8))
    for i in range(8):
        A[:, i] = bezier_basis(u, 7, i)

    px, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    py, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return A @ px, A @ py


def fit_cosine_s_curve(x, y):
    def cos_func(x_in, a_p, b_p, c_p):
        return a_p * np.cos(b_p * x_in) + c_p

    try:
        popt, _ = curve_fit(cos_func, x, y, p0=[np.max(y), np.pi / (2 * np.max(np.abs(x))), 0], maxfev=3000)
        return x, cos_func(x, *popt)
    except:
        return x, y


FIT_METHODS = {
    '广义外凸螺线 (Spiral)': {'func': fit_spiral, 'color': '#ff7f0e'},
    '五次多项式 (Quintic)': {'func': fit_quintic, 'color': '#2ca02c'},
    '七次多项式 (7th Poly)': {'func': fit_7th_poly, 'color': '#d62728'},
    '余弦 S 型 (Cosine S-Curve)': {'func': fit_cosine_s_curve, 'color': '#9467bd'},
    '七阶贝塞尔 (Bézier)': {'func': fit_bezier_7th, 'color': '#8c564b'},
    'B 样条曲线 (B-Spline)': {'func': fit_bspline, 'color': '#e377c2'}
}

# ----------------- GUI 布局与绘图 -----------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 8.5))
fig.subplots_adjust(left=0.12, right=0.74, bottom=0.15, top=0.92, hspace=0.35, wspace=0.25)

ax_shape = fig.add_subplot(2, 2, (1, 3))
ax_curv = fig.add_subplot(2, 2, 2)
ax_dcurv = fig.add_subplot(2, 2, 4)

# 控件
ax_slider_angle = plt.axes([0.03, 0.25, 0.025, 0.55])
slider_angle = Slider(ax_slider_angle, '开角\n(度)', 20.0, 160.0, valinit=62.0, valstep=1.0, orientation='vertical')

ax_slider_n = plt.axes([0.16, 0.05, 0.52, 0.03])
slider_n = Slider(ax_slider_n, '超椭圆指数 n', 0.5, 6.0, valinit=1.35, valstep=0.05)

ax_check = plt.axes([0.77, 0.20, 0.21, 0.65])
ORIG_LABEL = '原始超椭圆 (Superellipse)'
all_labels = [ORIG_LABEL] + list(FIT_METHODS.keys())

# 默认勾选螺线
init_status = [False, True, False, False, False, False, False]
check_buttons = CheckButtons(ax_check, all_labels, init_status)


def update(val=None):
    n_val = slider_n.val
    angle_deg = slider_angle.val

    b_val = 1.0
    a_val = b_val * np.tan(np.radians(angle_deg / 2.0))

    x_orig, y_orig = generate_superellipse_upper(n_val, a=a_val, b=b_val)
    s_orig, curv_orig, dcurv_orig = compute_curvature_and_derivative(x_orig, y_orig)

    ax_shape.cla()
    ax_curv.cla()
    ax_dcurv.cla()

    status_dict = dict(zip(all_labels, check_buttons.get_status()))

    if status_dict[ORIG_LABEL]:
        ax_shape.plot(x_orig, y_orig, 'k--', lw=2.2, label='原始超椭圆')
        ax_curv.plot(s_orig, curv_orig, 'k--', lw=2.0, label='原始超椭圆')
        ax_dcurv.plot(s_orig, dcurv_orig, 'k--', lw=2.0, label='原始超椭圆')

    for label, method_info in FIT_METHODS.items():
        if status_dict[label]:
            try:
                x_fit, y_fit = method_info['func'](x_orig, y_orig)
                s_fit, curv_fit, dcurv_fit = compute_curvature_and_derivative(x_fit, y_fit)

                c = method_info['color']
                ax_shape.plot(x_fit, y_fit, color=c, lw=1.8, label=label)
                ax_curv.plot(s_fit, curv_fit, color=c, lw=1.8, label=label)
                ax_dcurv.plot(s_fit, dcurv_fit, color=c, lw=1.8, label=label)
            except Exception:
                pass

    ax_shape.set_title(f'上半超椭圆及拟合曲线 (n={n_val:.2f}, 开角={angle_deg:.1f}°)')
    ax_shape.set_xlabel('X')
    ax_shape.set_ylabel('Y')

    margin_x = a_val * 0.2
    ax_shape.set_xlim(-a_val - margin_x, a_val + margin_x)
    ax_shape.set_ylim(-0.2, b_val * 1.3)
    ax_shape.set_aspect('equal', adjustable='box')
    ax_shape.grid(True, linestyle=':')
    ax_shape.legend(loc='upper right', fontsize=8)

    ax_curv.set_title('曲率分布 $\kappa(s)$')
    ax_curv.set_xlabel('归一化弧长 $s$')
    ax_curv.set_ylabel('曲率 $\kappa$')
    ax_curv.grid(True, linestyle=':')

    ax_dcurv.set_title('曲率变化率 $d\kappa / ds$')
    ax_dcurv.set_xlabel('归一化弧长 $s$')
    ax_dcurv.set_ylabel('$d\kappa / ds$')
    ax_dcurv.grid(True, linestyle=':')

    fig.canvas.draw_idle()


slider_n.on_changed(update)
slider_angle.on_changed(update)
check_buttons.on_clicked(update)

update()
plt.show()