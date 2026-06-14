import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import sys
import time

# ============================================================
# 参数
# ============================================================
g = 1.62           # 月球重力加速度 (m/s²)
Fmax = 2000        # 最大推力 (N)
k = 0.001          # 燃料消耗率
m0 = 500           # 初始质量 (kg)
h0 = 100           # 初始高度 (m)
v0 = -10           # 初始速度 (m/s, 向下)

# ============================================================
# ODE 系统 (状态 + 协态)
#   y = [h, v, m, λ1, λ2, λ3]
#   H = u + λ1·v + λ2·(-g + u/m) + λ3·(-k·u)
#     = λ1·v - λ2·g + u · (1 + λ2/m - k·λ3)
#   切换函数 S = 1 + λ2/m - k·λ3
#   Pontryagin: S>0 → u=0, S<0 → u=Fmax (极小化 H)
# ============================================================
def dynamics(t, y, eps):
    h, v, m, l1, l2, l3 = y
    S = 1.0 + l2 / m - k * l3
    u = Fmax * 0.5 * (1.0 - np.tanh(S / eps))
    dh = v
    dv = -g + u / m
    dm = -k * u
    dl1 = 0.0
    dl2 = -l1
    dl3 = l2 * u / (m * m)
    return [dh, dv, dm, dl1, dl2, dl3]

# ============================================================
# 打靶残差: 给定 [λ10, λ20, λ30, tf], 返回 [h(tf), v(tf), λ3(tf), H(tf)]
# ============================================================
def shooting_residuals(x, eps):
    l10, l20, l30, tf = x
    if tf <= 0.1:
        return [1e6, 1e6, 1e6, 1e6]

    y0 = [h0, v0, m0, l10, l20, l30]

    try:
        sol = solve_ivp(
            lambda t, y: dynamics(t, y, eps),
            [0.0, tf], y0,
            method='RK45',
            max_step=tf / 200.0,
            rtol=1e-9, atol=1e-9
        )
    except Exception:
        return [1e6, 1e6, 1e6, 1e6]

    if not sol.success or sol.t[-1] < tf * 0.99:
        return [1e6, 1e6, 1e6, 1e6]

    hf, vf, mf, l1f, l2f, l3f = sol.y[:, -1]
    Sf = 1.0 + l2f / mf - k * l3f
    uf = Fmax * 0.5 * (1.0 - np.tanh(Sf / eps))
    Hf = uf * Sf + l1f * vf - l2f * g

    return [hf, vf, l3f, Hf]

# ============================================================
# 延拓法主循环
# ============================================================
print("=" * 60)
print("打靶法求解 (Pontryagin 极小值原理 + 延拓法)")
print("=" * 60)

# 初始猜测: λ10, λ20, λ30, tf
# 粗略分析: λ1≈-45, λ2(0)≈-335, λ3(0)≈200, tf≈10-12
x0 = np.array([-45.0, -335.0, 200.0, 10.0])

eps_seq = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

for eps in eps_seq:
    t_start = time.time()
    res = least_squares(
        lambda x: shooting_residuals(x, eps),
        x0,
        bounds=([-1e4, -1e4, -1e4, 0.5],
                [1e4, 1e4, 1e4, 200.0]),
        method='trf',
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=100,
        verbose=0
    )
    x0 = res.x.copy()
    elapsed = time.time() - t_start
    cost = 0.5 * np.sum(np.array(res.fun)**2)
    print(f"eps={eps:6.4f} | λ10={x0[0]:+9.4f} λ20={x0[1]:+9.4f} "
          f"λ30={x0[2]:+9.4f} tf={x0[3]:8.4f}s "
          f"| cost={cost:.2e} nfev={res.nfev:3d} t={elapsed:.1f}s")
    sys.stdout.flush()

print()
l10_opt, l20_opt, l30_opt, tf_opt = x0
tf_opt = max(tf_opt, 0.5)

# ============================================================
# 使用精确 bang-bang 控制重新积分
# ============================================================
def dynamics_exact(t, y):
    h, v, m, l1, l2, l3 = y
    S = 1.0 + l2 / m - k * l3
    u = 0.0 if S > 0 else Fmax
    dh = v
    dv = -g + u / m
    dm = -k * u
    dl1 = 0.0
    dl2 = -l1
    dl3 = l2 * u / (m * m)
    return [dh, dv, dm, dl1, dl2, dl3]

def switch_event(t, y):
    _, _, m, _, l2, l3 = y
    return 1.0 + l2 / m - k * l3
switch_event.terminal = False
switch_event.direction = 0

sol = solve_ivp(
    dynamics_exact,
    [0.0, tf_opt],
    [h0, v0, m0, l10_opt, l20_opt, l30_opt],
    method='RK45', max_step=tf_opt / 1000.0,
    rtol=1e-12, atol=1e-12,
    events=switch_event, dense_output=True
)

t = sol.t
h, v, m, l1, l2, l3 = sol.y

# 终端验证
hf, vf, mf = h[-1], v[-1], m[-1]
l1f, l2f, l3f = l1[-1], l2[-1], l3[-1]
Sf = 1.0 + l2f / mf - k * l3f
uf = 0.0 if Sf > 0 else Fmax
Hf = uf * Sf + l1f * vf - l2f * g

print("终端条件验证:")
print(f"  h(tf)    = {hf:.6e} m     (目标: 0)")
print(f"  v(tf)    = {vf:.6e} m/s   (目标: 0)")
print(f"  λ3(tf)   = {l3f:.6e}       (目标: 0)")
print(f"  H(tf)    = {Hf:.6e}       (目标: 0)")
print()

# 切换点
if len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
    print(f"切换函数过零点: {sol.t_events[0]}")

# ============================================================
# 计算控制量
# ============================================================
u_opt = np.zeros_like(t)
S_arr = np.zeros_like(t)
for i in range(len(t)):
    S = 1.0 + l2[i] / m[i] - k * l3[i]
    S_arr[i] = S
    u_opt[i] = 0.0 if S > 0 else Fmax

# ============================================================
# 绘图
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(t, h, 'b-', linewidth=1.5)
plt.ylabel('Height h (m)')
plt.grid(True, alpha=0.3)
plt.title('Height')

plt.subplot(3, 2, 2)
plt.plot(t, v, 'b-', linewidth=1.5)
plt.ylabel('Velocity v (m/s)')
plt.grid(True, alpha=0.3)
plt.title('Velocity')

plt.subplot(3, 2, 3)
plt.plot(t, m, 'b-', linewidth=1.5)
plt.ylabel('Mass m (kg)')
plt.grid(True, alpha=0.3)
plt.title('Mass')

plt.subplot(3, 2, 4)
plt.plot(t, u_opt / 1000, 'r-', linewidth=1.5)
plt.ylabel('Thrust u (kN)')
plt.ylim([-0.1, 2.2])
plt.grid(True, alpha=0.3)
plt.title('Control (Thrust)')

plt.subplot(3, 2, 5)
plt.plot(t, S_arr, 'b-', linewidth=1.5)
plt.axhline(0, color='r', linestyle='--', alpha=0.7)
plt.ylabel('Switching Function S')
plt.xlabel('Time t (s)')
plt.grid(True, alpha=0.3)
plt.title('Switching Function S = 1 + λ2/m - k·λ3')

plt.subplot(3, 2, 6)
plt.plot(t, l1, label='λ1')
plt.plot(t, l2, label='λ2')
plt.plot(t, l3, label='λ3')
plt.xlabel('Time t (s)')
plt.ylabel('Costates')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Costates')

plt.tight_layout()
plt.show()

# ============================================================
# 结果总结
# ============================================================
print("=" * 60)
print(f"最优终端时间:   {tf_opt:.4f} s")
print(f"初始质量:       {m0:.2f} kg")
print(f"最终质量:       {mf:.2f} kg")
print(f"燃料消耗:       {m0 - mf:.2f} kg")
print(f"初始协态:       λ1={l10_opt:.4f}, λ2={l20_opt:.4f}, λ3={l30_opt:.4f}")
print("=" * 60)
