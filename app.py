import numpy as np
import plotly.graph_objects as go
import streamlit as st
from motor import DCMotorParams, DCMotorSim   # uses your motor.py

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(page_title="DC Motor PID", layout="wide")


# =======================
# SESSION DEFAULTS
# =======================
def _ss_set_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


_ss_set_default("page", "Simulation")

# default PID (balanced-ish)
_ss_set_default("Kp", 0.8)
_ss_set_default("Ki", 1.2)
_ss_set_default("Kd", 0.02)

_ss_set_default("setpoint_rpm", 1200)
_ss_set_default("sim_time", 6.0)
_ss_set_default("dt", 0.002)
_ss_set_default("volts_limit", 12)

# =======================
# FIXED MOTOR PARAMETERS
# (chosen so 12V ≈ 1200 RPM at steady state)
# =======================
BASE_PARAMS = DCMotorParams(
    J=0.01,
    b=0.00228,
    Kt=0.05,
    Ke=0.05,
    R=1.0,
    L=0.01,
)


# =======================
# HELPERS
# =======================
def performance_metrics(t, y, target, tol=0.02, window_sec=0.1):
    """
    - Rise time: 10→90% of step magnitude
    - Overshoot: % above target (for positive step)
    - Settling time: earliest time after which the response
      stays within ±tol*|target| for the rest of the simulation.
    """
    y = np.asarray(y, dtype=float)
    target = float(target)
    n = len(t)
    if n < 2:
        return np.nan, 0.0, np.nan

    dt = float(t[1] - t[0])

    # ----- Rise time -----
    y0 = float(y[0])
    step = target - y0
    rise = np.nan
    if abs(step) > 1e-9:
        y10 = y0 + 0.10 * step
        y90 = y0 + 0.90 * step

        if step > 0:
            idx10 = np.where(y >= y10)[0]
            idx90 = np.where(y >= y90)[0]
        else:
            idx10 = np.where(y <= y10)[0]
            idx90 = np.where(y <= y90)[0]

        if len(idx10) and len(idx90):
            t10 = t[idx10[0]]
            t90 = t[idx90[0]]
            if t90 >= t10:
                rise = t90 - t10

    # ----- Overshoot (positive step only) -----
    overshoot = 0.0
    if step > 0 and abs(target) > 1e-9:
        overshoot = max((np.max(y) - target) / abs(target) * 100.0, 0.0)

    # ----- Settling time: stay in band for rest of sim -----
    eps = max(tol * abs(target), 1e-3)
    outside = np.where(np.abs(y - target) > eps)[0]

    if len(outside) == 0:
        settle = t[0]
    else:
        last_out = outside[-1]
        if last_out < n - 1:
            # from next sample onward it is within band
            if np.all(np.abs(y[last_out + 1:] - target) <= eps):
                settle = t[last_out + 1]
            else:
                settle = np.nan
        else:
            settle = np.nan

    return rise, overshoot, settle


def simulate(pid_gains, setpoint_rad, dt, sim_time, volts_limit):
    """
    Closed-loop simulation with our own discrete PID.
    This avoids real-time issues from simple_pid.
    """
    plant = DCMotorSim(BASE_PARAMS, dt=dt)
    plant.reset(0.0, 0.0)

    Kp, Ki, Kd = pid_gains

    steps = int(sim_time / dt)
    t = np.linspace(0.0, sim_time, steps, endpoint=False)

    w_hist = np.zeros(steps)
    u_hist = np.zeros(steps)
    sp_hist = np.full(steps, setpoint_rad)
    e_hist = np.zeros(steps)

    integral = 0.0
    e_prev = setpoint_rad - plant.w

    for k in range(steps):
        e = setpoint_rad - plant.w
        integral += e * dt
        derivative = (e - e_prev) / dt if k > 0 else 0.0

        # raw PID
        u = Kp * e + Ki * integral + Kd * derivative

        # saturation + simple anti-windup:
        if u > volts_limit:
            u = volts_limit
            # prevent windup when pushing against upper limit
            if e > 0:
                integral -= e * dt
        elif u < -volts_limit:
            u = -volts_limit
            if e < 0:
                integral -= e * dt

        w_next, _ = plant.step(u)

        w_hist[k] = w_next
        u_hist[k] = u
        sp_hist[k] = setpoint_rad
        e_hist[k] = e

        plant.w = w_next
        e_prev = e

    return t, w_hist, u_hist, sp_hist, e_hist


def radps_to_rpm(x):
    return (x * 60.0) / (2.0 * np.pi)


def fmt_rise(x):
    if x is None or np.isnan(x):
        return "—"
    return f"{x:.3f} s"


def fmt_settle(x):
    if x is None or np.isnan(x):
        return "Not settled"
    return f"{x:.3f} s"


# =======================
# PID PRESETS (ONLY GAINS)
# =======================
def apply_preset(name: str):
    """
    Only adjusts Kp, Ki, Kd.
    Plant is fixed so differences in response are purely due to tuning.
    """
    if name == "Balanced":
        st.session_state.update(dict(Kp=0.8, Ki=1.2, Kd=0.02))
    elif name == "Fast (Overshoot)":
        st.session_state.update(dict(Kp=1.6, Ki=2.5, Kd=0.01))
    elif name == "Slow (No Overshoot)":
        st.session_state.update(dict(Kp=0.3, Ki=0.3, Kd=0.08))


# =======================
# SIDEBAR PAGE SELECTOR
# =======================
with st.sidebar:
    st.selectbox("Page", ["Simulation", "Report"], key="page")

st.title("DC Motor Speed Control using PID (Python)")


# =======================
# SIMULATION PAGE
# =======================
if st.session_state.page == "Simulation":
    with st.sidebar:
        # --- Presets ---
        st.header("PID Presets")
        c1, c2, c3 = st.columns(3)
        if c1.button("Balanced"):
            apply_preset("Balanced")
        if c2.button("Fast"):
            apply_preset("Fast (Overshoot)")
        if c3.button("Slow"):
            apply_preset("Slow (No Overshoot)")

        # --- Setpoint & gains ---
        st.header("Setpoint & PID")

        setpoint_rpm = st.slider(
            "Setpoint (RPM)",
            min_value=0,
            max_value=3000,
            value=int(st.session_state.setpoint_rpm),
            step=50,
            key="setpoint_rpm",
        )
        setpoint_rad = (setpoint_rpm * 2.0 * np.pi) / 60.0

        st.subheader("PID Gains")
        Kp = st.number_input(
            "Kp", value=float(st.session_state.Kp),
            step=0.1, format="%.3f", key="Kp"
        )
        Ki = st.number_input(
            "Ki", value=float(st.session_state.Ki),
            step=0.1, format="%.3f", key="Ki"
        )
        Kd = st.number_input(
            "Kd", value=float(st.session_state.Kd),
            step=0.005, format="%.3f", key="Kd"
        )

        # --- Simulation options ---
        st.header("Simulation")
        sim_time = st.slider(
            "Simulation time (s)",
            min_value=1.0,
            max_value=20.0,
            value=float(st.session_state.sim_time),
            step=0.5,
            key="sim_time",
        )
        dt = st.select_slider(
            "Time step dt (s)",
            options=[0.001, 0.002, 0.005, 0.01],
            value=float(st.session_state.dt),
            key="dt",
        )
        volts_limit = st.slider(
            "Voltage limit (±V)",
            min_value=2,
            max_value=24,
            value=int(st.session_state.volts_limit),
            step=1,
            key="volts_limit",
        )

        run = st.button(
            "Run Simulation",
            type="primary",
            use_container_width=True,
        )

    if run:
        t, w_hist, u_hist, sp_hist, e_hist = simulate(
            pid_gains=(Kp, Ki, Kd),
            setpoint_rad=setpoint_rad,
            dt=dt,
            sim_time=sim_time,
            volts_limit=volts_limit,
        )

        w_rpm = radps_to_rpm(w_hist)
        sp_rpm = radps_to_rpm(sp_hist)
        e_rpm = radps_to_rpm(e_hist)

        rise, overshoot, settle = performance_metrics(
            t, w_hist, setpoint_rad, tol=0.02
        )

        col1, col2 = st.columns([2, 1], gap="large")

        # ---- plots ----
        with col1:
            # Speed vs setpoint
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t, y=w_rpm,
                mode="lines",
                name="Speed (RPM)",
            ))
            fig.add_trace(go.Scatter(
                x=t, y=sp_rpm,
                mode="lines",
                name="Setpoint (RPM)",
                line=dict(dash="dash"),
            ))
            fig.update_layout(
                title="Motor Speed Response",
                xaxis_title="Time (s)",
                yaxis_title="RPM",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Error plot (in RPM)
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(
                x=t, y=e_rpm,
                mode="lines",
                name="Error (RPM)",
            ))
            fig_e.update_layout(
                title="Speed Error (Setpoint - Actual)",
                xaxis_title="Time (s)",
                yaxis_title="Error (RPM)",
                height=220,
            )
            st.plotly_chart(fig_e, use_container_width=True)

            # Control signal
            fig_u = go.Figure()
            fig_u.add_trace(go.Scatter(
                x=t, y=u_hist,
                mode="lines",
                name="Control Voltage (V)",
            ))
            fig_u.update_layout(
                title="Control Signal (Armature Voltage)",
                xaxis_title="Time (s)",
                yaxis_title="Voltage (V)",
                height=220,
            )
            st.plotly_chart(fig_u, use_container_width=True)

        # ---- metrics ----
        with col2:
            st.subheader("Performance")
            st.metric("Rise time (10→90%)", fmt_rise(rise))
            st.metric("Overshoot", f"{overshoot:.2f} %")
            st.metric("Settling time (±2%)", fmt_settle(settle))

            st.subheader("How to interpret")
            st.write(
                "- **Kp ↑** → faster rise, usually more overshoot.\n"
                "- **Ki ↑** → removes steady-state error; too large → overshoot & slow settling (integral windup).\n"
                "- **Kd ↑** → adds damping; reduces overshoot but too high → sluggish.\n"
                "- **Voltage limit** models actuator saturation; if too low, you may get slow response or steady-state error."
            )
    else:
        st.info(
            "Pick PID gains (or a preset) and click **Run Simulation** to see how "
            "speed, error and performance indices change."
        )

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
        "To design and study a PID controller for an armature-controlled DC motor so that "
        "the motor speed follows a desired setpoint with acceptable rise time, low overshoot, "
        "and negligible steady-state error."
    )

    st.markdown("### Theory")
    st.latex(
        r"""\dot{\omega} = \frac{K_t i - b\omega}{J},
            \qquad
            \dot{i} = \frac{V - Ri - K_e\omega}{L}"""
    )
    st.write(
        "where $\\omega$ is angular speed, $i$ is armature current, and $V$ is the applied armature voltage."
    )
    st.latex(
        r"""\left.\frac{\omega}{V}\right|_{ss}
            \approx
            \frac{K_t}{Rb + K_t K_e}"""
    )
    st.write(
        "A PID controller uses the error $e(t) = \\omega_{ref}(t) - \\omega(t)$ to generate the control voltage:"
    )
    st.latex(
        r"""V(t) = K_p e(t)
        + K_i \int_0^t e(\tau)\,d\tau
        + K_d \frac{de(t)}{dt}."""
    )

    st.markdown("### Method")
    st.write(
        "1. Modelled the DC motor using its differential equations.\n"
        "2. Implemented a discrete-time simulator (`DCMotorSim`).\n"
        "3. Implemented a digital PID controller in Python using the error.\n"
        "4. Built a Streamlit UI to vary $K_p, K_i, K_d$, setpoint, and voltage limit.\n"
        "5. Computed rise time, percent overshoot, and 2% settling time from the simulated response."
    )

    st.markdown("### Observations")
    st.write(
        "- Higher $K_p$ reduces rise time but increases overshoot (underdamped).\n"
        "- Higher $K_i$ eliminates steady-state error but can cause overshoot and longer settling.\n"
        "- Higher $K_d$ adds damping, reducing overshoot and improving settling.\n"
        "These trends match standard control systems theory."
    )

    st.markdown("### Conclusion")
    st.write(
        "The project demonstrates a complete closed-loop speed control of a DC motor using PID. "
        "The interactive tool clearly links theoretical concepts (error, rise time, overshoot, "
        "settling time, steady-state error) with the simulated response."
    )
