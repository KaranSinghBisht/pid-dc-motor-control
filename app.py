import numpy as np
import plotly.graph_objects as go
import streamlit as st
from simple_pid import PID
from motor import DCMotorParams, DCMotorSim

st.set_page_config(page_title="DC Motor PID", layout="wide")

# ---- session defaults ----
def _ss_set_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

_ss_set_default("page", "Simulation")
_ss_set_default("setpoint_rpm", 1200)
_ss_set_default("Kp", 0.5)
_ss_set_default("Ki", 20.0)
_ss_set_default("Kd", 0.0)
_ss_set_default("sim_time", 5.0)
_ss_set_default("dt", 0.002)
_ss_set_default("volts_limit", 12)
_ss_set_default("J", 0.01)
_ss_set_default("b", 0.1)
_ss_set_default("Kt", 0.01)
_ss_set_default("Ke", 0.01)
_ss_set_default("R", 1.0)
_ss_set_default("L", 0.5)

# ---- helpers ----
def performance_metrics(t, y, target, tol=0.02):
    y = np.asarray(y); target = float(target)
    eps = max(tol * abs(target), 1e-9)
    y10, y90 = 0.1 * target, 0.9 * target

    def first_cross(level):
        idx = np.where(y >= level)[0]
        return t[idx[0]] if (len(idx) > 0 and target > 0) else np.nan

    t10, t90 = first_cross(y10), first_cross(y90)
    rise = (t90 - t10) if (not np.isnan(t10) and not np.isnan(t90)) else np.nan
    overshoot = max((np.max(y) - target) / abs(target), 0.0) * 100.0 if target != 0 else 0.0

    settle = np.nan
    idx = np.where(np.abs(y - target) > eps)[0]
    if len(idx) == 0:
        settle = 0.0
    else:
        last_out = idx[-1]
        if last_out < len(t) - 1:
            settle = t[last_out + 1]
    return rise, overshoot, settle

def simulate(params, pid_gains, setpoint_rad, dt, sim_time, volts_limit):
    plant = DCMotorSim(params, dt=dt)
    plant.reset(0.0, 0.0)
    Kp, Ki, Kd = pid_gains
    pid = PID(Kp, Ki, Kd, setpoint=setpoint_rad)
    pid.output_limits = (-volts_limit, volts_limit)

    steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, steps, endpoint=False)
    w_hist = np.zeros(steps); u_hist = np.zeros(steps)
    sp_hist = np.full(steps, setpoint_rad)

    for k in range(steps):
        u = pid(plant.w)
        w_next, _ = plant.step(u)
        w_hist[k] = w_next; u_hist[k] = u

    return t, w_hist, u_hist, sp_hist

def radps_to_rpm(x): return (x * 60.0) / (2 * np.pi)

# ---- presets ----
def apply_preset(name: str):
    if name == "Small Hobby Motor (balanced)":
        st.session_state.update(dict(
            setpoint_rpm=1200, volts_limit=12,
            J=0.0001, b=0.001, Kt=0.05, Ke=0.05, R=1.0, L=0.01,
            Kp=0.30, Ki=3.0, Kd=0.001, sim_time=6.0, dt=0.002
        ))
    elif name == "Fast Response (more overshoot)":
        st.session_state.update(dict(
            setpoint_rpm=1500, volts_limit=24,
            J=0.0001, b=0.001, Kt=0.05, Ke=0.05, R=1.0, L=0.01,
            Kp=0.80, Ki=8.0, Kd=0.0, sim_time=6.0, dt=0.002
        ))
    elif name == "Conservative/Robust (low overshoot)":
        st.session_state.update(dict(
            setpoint_rpm=1200, volts_limit=12,
            J=0.0002, b=0.002, Kt=0.04, Ke=0.04, R=1.2, L=0.02,
            Kp=0.20, Ki=1.5, Kd=0.003, sim_time=8.0, dt=0.002
        ))

# ---- sidebar page selector ----
with st.sidebar:
    st.selectbox("Page", ["Simulation", "Report"], key="page")

st.title("DC Motor Speed Control using PID (Python)")

# =======================
# SIMULATION PAGE
# =======================
if st.session_state.page == "Simulation":
    with st.sidebar:
        st.header("Quick Presets")
        c1, c2, c3 = st.columns(3)
        if c1.button("Hobby Motor"): apply_preset("Small Hobby Motor (balanced)")
        if c2.button("Fast"): apply_preset("Fast Response (more overshoot)")
        if c3.button("Conservative"): apply_preset("Conservative/Robust (low overshoot)")

        st.header("PID & Plant Settings")
        setpoint_rpm = st.slider("Setpoint (RPM)", 0, 3000, st.session_state.setpoint_rpm, 50, key="setpoint_rpm")
        setpoint_rad = (setpoint_rpm * 2 * np.pi) / 60.0

        st.subheader("PID gains")
        Kp = st.number_input("Kp", value=float(st.session_state.Kp), step=0.05, format="%.3f", key="Kp")
        Ki = st.number_input("Ki", value=float(st.session_state.Ki), step=0.5, format="%.3f", key="Ki")
        Kd = st.number_input("Kd", value=float(st.session_state.Kd), step=0.001, format="%.3f", key="Kd")

        st.subheader("Simulation options")
        sim_time = st.slider("Simulation time (s)", 1.0, 12.0, float(st.session_state.sim_time), 0.5, key="sim_time")
        dt = st.select_slider("Time step dt (s)", options=[0.001, 0.002, 0.005, 0.01], value=float(st.session_state.dt), key="dt")
        volts_limit = st.slider("Voltage limit (±V)", 1, 48, int(st.session_state.volts_limit), 1, key="volts_limit")

        st.subheader("Motor Params")
        J = st.number_input("J (kg·m²)", value=float(st.session_state.J), step=0.0001, format="%.5f", key="J")
        b = st.number_input("b (N·m·s)", value=float(st.session_state.b), step=0.0005, format="%.5f", key="b")
        Kt = st.number_input("Kt (N·m/A)", value=float(st.session_state.Kt), step=0.001, format="%.5f", key="Kt")
        Ke = st.number_input("Ke (V·s/rad)", value=float(st.session_state.Ke), step=0.001, format="%.5f", key="Ke")
        R = st.number_input("R (Ω)", value=float(st.session_state.R), step=0.1, format="%.3f", key="R")
        L = st.number_input("L (H)", value=float(st.session_state.L), step=0.005, format="%.3f", key="L")

        run = st.button("Run Simulation", type="primary", use_container_width=True)

    if run:
        params = DCMotorParams(J=J, b=b, Kt=Kt, Ke=Ke, R=R, L=L)
        t, w_hist, u_hist, sp_hist = simulate(
            params=params, pid_gains=(Kp, Ki, Kd),
            setpoint_rad=setpoint_rad, dt=dt, sim_time=sim_time, volts_limit=volts_limit
        )
        w_rpm = radps_to_rpm(w_hist); sp_rpm = radps_to_rpm(sp_hist)
        rise, overshoot, settle = performance_metrics(t, w_hist, setpoint_rad, tol=0.02)

        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=w_rpm, mode="lines", name="Speed (RPM)"))
            fig.add_trace(go.Scatter(x=t, y=sp_rpm, mode="lines", name="Setpoint (RPM)", line=dict(dash="dash")))
            fig.update_layout(title="Motor Speed Response", xaxis_title="Time (s)", yaxis_title="RPM", height=450)
            st.plotly_chart(fig, use_container_width=True)

            fig_u = go.Figure()
            fig_u.add_trace(go.Scatter(x=t, y=u_hist, mode="lines", name="Control Voltage (V)"))
            fig_u.update_layout(title="Control Signal", xaxis_title="Time (s)", yaxis_title="Voltage (V)", height=300)
            st.plotly_chart(fig_u, use_container_width=True)

        with col2:
            st.subheader("Performance")
            fmt = lambda x: "—" if (x is None or np.isnan(x)) else f"{x:.3f} s"
            st.metric("Rise time (10→90%)", fmt(rise))
            st.metric("Overshoot", f"{overshoot:.2f} %")
            st.metric("Settling time (±2%)", fmt(settle))
            st.subheader("Notes")
            st.write("- Increase **Kp** → faster rise (watch overshoot)\n- **Ki** removes steady-state error\n- **Kd** damps overshoot/noise")
    else:
        st.info("Set parameters in the sidebar, choose a preset if you like, then click **Run Simulation**.")

# =======================
# REPORT PAGE
# =======================
else:
    st.header("EEPC20 – CONTROL SYSTEMS — Mini Project")
    st.subheader("National Institute of Technology, Tiruchirappalli")

    st.markdown("### Team Members")
    st.write(
        "- **Anjesh Singh** — 107123014\n"
        "- **Kamble Aniket** — 107123052\n"
        "- **Karan Singh Bisht** — 107123054"
    )
    st.markdown("### Title")
    st.write("**DC Motor Speed Control using PID (Python Simulation + Dashboard)**")

    st.markdown("### Aim")
    st.write(
        "Design and evaluate a PID controller for an armature-controlled DC motor to track a desired "
        "speed (RPM) with minimal steady-state error, acceptable rise time, and bounded overshoot, "
        "demonstrated via a Python/Streamlit dashboard."
    )

    st.markdown("### Theory")
    st.latex(r"""\dot{\omega} = \frac{K_t\,i - b\,\omega}{J}, \qquad \dot{i} = \frac{V - R\,i - K_e\,\omega}{L}""")
    st.write("where $\\omega$ is angular speed, $i$ is armature current, and $V$ is applied voltage.")
    st.latex(r"""\left.\frac{\omega}{V}\right|_{ss} \approx \frac{K_t}{R\,b + K_t K_e}""")
    st.write("A PID controller computes the control voltage")
    st.latex(r"""V(t) = K_p\,e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d\,\frac{de(t)}{dt}, \quad e(t)=\omega_{\mathrm{ref}}-\omega(t)""")
    st.write("Tuning trades off rise time, overshoot, and noise sensitivity.")

    st.markdown("### Method")
    st.write(
        "1. Implemented a discrete-time simulator of the DC motor in Python.\n"
        "2. Closed the loop with a PID controller (`simple-pid`).\n"
        "3. Built an interactive Streamlit app to vary motor parameters and PID gains.\n"
        "4. Reported key time-domain metrics: rise time (10→90%), percent overshoot, and 2% settling time."
    )

    st.markdown("### Results (Representative)")
    st.write(
        "Using a light ‘small hobby motor’ configuration (preset), the controller tracks a 1200 RPM step "
        "with moderate rise time and low overshoot at gains around Kp≈0.30, Ki≈3.0, Kd≈0.001 under a 12 V limit. "
        "A faster preset reduces rise time but increases overshoot; a conservative preset improves damping at the cost of speed."
    )

    st.markdown("### Conclusion")
    st.write(
        "The PID controller meets the speed-tracking objective over a range of motor parameters. "
        "Interactive tuning highlights the trade-offs among rise time, overshoot, and steady-state error."
    )