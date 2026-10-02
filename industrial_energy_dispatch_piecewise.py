"""
Industrial Energy Dispatch Demo -- piecewise-linear CHP model.

A compact interview/portfolio model for industrial energy optimization.

Model features
--------------
- 24-hour joint dispatch optimization
- CHP with on/off state, startup cost and ramping
- load-dependent CHP electrical and thermal efficiency represented by
  a piecewise-linear performance curve
- gas boiler
- photovoltaic generation
- battery with SOC dynamics and round-trip losses
- grid import/export limits and asymmetric buy/sell prices
- baseline comparison and balance checks

Why piecewise-linear?
---------------------
The underlying CHP efficiency curves are nonlinear functions of load.  A MILP
solver cannot use those nonlinear formulas directly.  We sample the nonlinear
curves at four operating points (25, 50, 75, 100 % load) and linearly
interpolate inside exactly one adjacent segment.  Thus the physical model is
more realistic than a constant-efficiency CHP, while the full optimization
remains a MILP.

All profiles and parameters are illustrative demonstration data, not plant
measurements.
"""

from dataclasses import dataclass
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Params:
    gas_price: float = 38.0

    # CHP
    chp_max_fuel: float = 9.0
    chp_min_fraction: float = 0.25
    chp_ramp: float = 4.5
    startup_cost: float = 14.0

    # Boiler
    boiler_max_fuel: float = 8.0
    eta_boiler: float = 0.90

    # Battery
    battery_capacity: float = 6.0
    battery_power: float = 2.0
    battery_roundtrip_efficiency: float = 0.90
    battery_cycle_cost: float = 1.2

    # Grid
    grid_import_max: float = 9.0
    grid_export_max: float = 5.0
    export_ratio: float = 0.70


def chp_eta_el(load: float) -> float:
    """Illustrative nonlinear electrical-efficiency curve."""
    return 0.28 + 0.30 * load - 0.10 * load**2


def chp_eta_th(load: float) -> float:
    """Illustrative nonlinear thermal-efficiency curve."""
    return 0.30 + 0.20 * load - 0.20 * load**2


def chp_curve(p: Params) -> pd.DataFrame:
    """Operating points used for the piecewise-linear CHP approximation."""
    loads = np.array([p.chp_min_fraction, 0.50, 0.75, 1.00], dtype=float)
    fuel = loads * p.chp_max_fuel
    eta_el = np.array([chp_eta_el(x) for x in loads])
    eta_th = np.array([chp_eta_th(x) for x in loads])

    return pd.DataFrame({
        "load": loads,
        "fuel": fuel,
        "eta_el": eta_el,
        "eta_th": eta_th,
        "p_el": eta_el * fuel,
        "q_heat": eta_th * fuel,
    })


def demo_data() -> pd.DataFrame:
    """Illustrative hourly demand, PV and electricity-price profiles."""
    hour = np.arange(24)

    electric = np.array([
        4.8, 4.5, 4.3, 4.2, 4.3, 4.8,
        5.6, 6.2, 6.6, 6.8, 7.0, 7.2,
        7.4, 7.2, 7.0, 6.9, 6.8, 6.7,
        6.8, 7.1, 7.4, 6.8, 5.8, 5.2,
    ])

    heat = np.array([
        5.8, 5.6, 5.5, 5.4, 5.5, 5.7,
        6.0, 6.2, 6.4, 6.5, 6.4, 6.3,
        6.2, 6.1, 6.0, 5.9, 5.8, 5.7,
        5.8, 6.0, 6.1, 6.0, 5.9, 5.8,
    ])

    pv = np.array([
        0, 0, 0, 0, 0, 0.1,
        0.4, 0.9, 1.5, 2.2, 2.8, 3.2,
        3.4, 3.1, 2.6, 1.9, 1.1, 0.4,
        0.1, 0, 0, 0, 0, 0,
    ])

    buy = np.array([
        75, 70, 68, 66, 68, 75,
        90, 110, 125, 118, 105, 95,
        88, 82, 90, 110, 145, 175,
        220, 245, 210, 160, 120, 95,
    ], dtype=float)

    return pd.DataFrame({
        "hour": hour,
        "electric_demand": electric,
        "heat_demand": heat,
        "pv": pv,
        "buy_price": buy,
    })


def solve_dispatch(data: pd.DataFrame, p: Params) -> tuple[pd.DataFrame, float]:
    """Solve one joint 24-hour MILP with a piecewise-linear CHP curve."""
    h_count = len(data)
    if h_count != 24:
        raise ValueError("This demo expects exactly 24 hourly rows.")

    curve = chp_curve(p)
    # Four operating points define three adjacent interpolation segments.
    n_segments = len(curve) - 1

    # Allocate one flat MILP vector x.  Each named block gets a slice.
    cursor = 0

    def alloc(length: int) -> slice:
        nonlocal cursor
        s = slice(cursor, cursor + length)
        cursor += length
        return s

    F_CHP = alloc(h_count)
    P_CHP = alloc(h_count)
    Q_CHP = alloc(h_count)
    F_B = alloc(h_count)
    P_IMP = alloc(h_count)
    P_EXP = alloc(h_count)
    P_CHG = alloc(h_count)
    P_DIS = alloc(h_count)
    SOC = alloc(h_count)
    ON = alloc(h_count)
    START = alloc(h_count)

    # SEG selects one of the 3 adjacent CHP curve segments in each hour.
    # ALPHA interpolates between that segment's two endpoints.
    SEG = alloc(h_count * n_segments)
    ALPHA = alloc(h_count * n_segments)

    n = cursor

    def seg_idx(block: slice, t: int, s: int) -> int:
        return block.start + t * n_segments + s

    buy = data["buy_price"].to_numpy(float)
    sell = p.export_ratio * buy
    el = data["electric_demand"].to_numpy(float)
    heat = data["heat_demand"].to_numpy(float)
    pv = data["pv"].to_numpy(float)

    # -----------------------------------------------------------------
    # Objective: min c^T x
    # -----------------------------------------------------------------
    c = np.zeros(n)
    c[F_CHP] = p.gas_price
    c[F_B] = p.gas_price
    c[P_IMP] = buy
    c[P_EXP] = -sell
    c[P_CHG] = p.battery_cycle_cost
    c[P_DIS] = p.battery_cycle_cost
    c[START] = p.startup_cost

    # -----------------------------------------------------------------
    # Bounds and binary variables
    # -----------------------------------------------------------------
    lower = np.zeros(n)
    upper = np.full(n, np.inf)

    upper[F_CHP] = p.chp_max_fuel
    upper[P_CHP] = curve["p_el"].max()
    upper[Q_CHP] = curve["q_heat"].max()
    upper[F_B] = p.boiler_max_fuel
    upper[P_IMP] = p.grid_import_max
    upper[P_EXP] = p.grid_export_max
    upper[P_CHG] = p.battery_power
    upper[P_DIS] = p.battery_power
    upper[SOC] = p.battery_capacity
    upper[ON] = 1.0
    upper[START] = 1.0
    upper[SEG] = 1.0
    upper[ALPHA] = 1.0

    bounds = Bounds(lower, upper)

    integrality = np.zeros(n)
    integrality[ON] = 1
    integrality[START] = 1
    integrality[SEG] = 1

    # -----------------------------------------------------------------
    # Linear constraints: lb <= A x <= ub
    # -----------------------------------------------------------------
    rows: list[np.ndarray] = []
    lows: list[float] = []
    highs: list[float] = []

    def add_constraint(coeffs: np.ndarray, low: float, high: float) -> None:
        rows.append(coeffs)
        lows.append(low)
        highs.append(high)

    # CHP piecewise-linear performance curve.
    fuel_pts = curve["fuel"].to_numpy()
    pel_pts = curve["p_el"].to_numpy()
    qheat_pts = curve["q_heat"].to_numpy()

    for t in range(h_count):
        # Exactly one segment when CHP is ON; no segment when OFF.
        a = np.zeros(n)
        for s in range(n_segments):
            a[seg_idx(SEG, t, s)] = 1.0
        a[ON.start + t] = -1.0
        add_constraint(a, 0.0, 0.0)

        # 0 <= alpha_ts <= segment_selected_ts.
        for s in range(n_segments):
            a = np.zeros(n)
            a[seg_idx(ALPHA, t, s)] = 1.0
            a[seg_idx(SEG, t, s)] = -1.0
            add_constraint(a, -np.inf, 0.0)

        # F_CHP = sum_s [fuel_left_s * y_s + delta_fuel_s * alpha_s]
        a = np.zeros(n)
        a[F_CHP.start + t] = 1.0
        for s in range(n_segments):
            a[seg_idx(SEG, t, s)] -= fuel_pts[s]
            a[seg_idx(ALPHA, t, s)] -= fuel_pts[s + 1] - fuel_pts[s]
        add_constraint(a, 0.0, 0.0)

        # P_CHP = same interpolation on electrical output curve.
        a = np.zeros(n)
        a[P_CHP.start + t] = 1.0
        for s in range(n_segments):
            a[seg_idx(SEG, t, s)] -= pel_pts[s]
            a[seg_idx(ALPHA, t, s)] -= pel_pts[s + 1] - pel_pts[s]
        add_constraint(a, 0.0, 0.0)

        # Q_CHP = same interpolation on thermal output curve.
        a = np.zeros(n)
        a[Q_CHP.start + t] = 1.0
        for s in range(n_segments):
            a[seg_idx(SEG, t, s)] -= qheat_pts[s]
            a[seg_idx(ALPHA, t, s)] -= qheat_pts[s + 1] - qheat_pts[s]
        add_constraint(a, 0.0, 0.0)

    # Energy balances.
    for t in range(h_count):
        # Heat: CHP + boiler = heat demand.
        a = np.zeros(n)
        a[Q_CHP.start + t] = 1.0
        a[F_B.start + t] = p.eta_boiler
        add_constraint(a, heat[t], heat[t])

        # Electricity: CHP + PV + import - export + battery = demand.
        a = np.zeros(n)
        a[P_CHP.start + t] = 1.0
        a[P_IMP.start + t] = 1.0
        a[P_EXP.start + t] = -1.0
        a[P_DIS.start + t] = 1.0
        a[P_CHG.start + t] = -1.0
        rhs = el[t] - pv[t]
        add_constraint(a, rhs, rhs)

    # Battery SOC dynamics.
    eta_c = sqrt(p.battery_roundtrip_efficiency)
    eta_d = sqrt(p.battery_roundtrip_efficiency)
    initial_soc = p.battery_capacity / 2

    for t in range(h_count):
        a = np.zeros(n)
        a[SOC.start + t] = 1.0
        a[P_CHG.start + t] = -eta_c
        a[P_DIS.start + t] = 1.0 / eta_d

        if t == 0:
            add_constraint(a, initial_soc, initial_soc)
        else:
            a[SOC.start + t - 1] = -1.0
            add_constraint(a, 0.0, 0.0)

    # End with the same stored energy as at the beginning.
    a = np.zeros(n)
    a[SOC.stop - 1] = 1.0
    add_constraint(a, initial_soc, initial_soc)

    # Startup_t >= ON_t - ON_(t-1).
    for t in range(h_count):
        a = np.zeros(n)
        a[ON.start + t] = 1.0
        a[START.start + t] = -1.0
        if t > 0:
            a[ON.start + t - 1] = -1.0
        add_constraint(a, -np.inf, 0.0)

    # CHP fuel ramping.
    for t in range(1, h_count):
        a = np.zeros(n)
        a[F_CHP.start + t] = 1.0
        a[F_CHP.start + t - 1] = -1.0
        add_constraint(a, -p.chp_ramp, p.chp_ramp)

    constraints = LinearConstraint(
        np.vstack(rows),
        np.array(lows),
        np.array(highs),
    )

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": 30},
    )

    if not result.success:
        raise RuntimeError(result.message)

    x = result.x

    out = data.copy()
    out["sell_price"] = sell
    out["chp_on"] = np.rint(x[ON]).astype(int)
    out["chp_start"] = np.rint(x[START]).astype(int)
    out["chp_fuel"] = x[F_CHP]
    out["chp_el"] = x[P_CHP]
    out["chp_heat"] = x[Q_CHP]
    out["boiler_fuel"] = x[F_B]
    out["boiler_heat"] = p.eta_boiler * x[F_B]
    out["grid_import"] = x[P_IMP]
    out["grid_export"] = x[P_EXP]
    out["battery_charge"] = x[P_CHG]
    out["battery_discharge"] = x[P_DIS]
    out["battery_soc"] = x[SOC]

    # Effective CHP efficiencies reconstructed from optimized outputs.
    running = out["chp_fuel"] > 1e-8
    out["chp_eta_el"] = np.where(running, out["chp_el"] / out["chp_fuel"], np.nan)
    out["chp_eta_th"] = np.where(running, out["chp_heat"] / out["chp_fuel"], np.nan)

    out["hourly_cost"] = (
        p.gas_price * (out["chp_fuel"] + out["boiler_fuel"])
        + out["buy_price"] * out["grid_import"]
        - out["sell_price"] * out["grid_export"]
        + p.battery_cycle_cost * (out["battery_charge"] + out["battery_discharge"])
        + p.startup_cost * out["chp_start"]
    )

    return out, float(result.fun)


def baseline_cost(data: pd.DataFrame, p: Params) -> float:
    """Simple benchmark: boiler supplies heat and grid supplies electricity."""
    boiler_fuel = data["heat_demand"].to_numpy(float) / p.eta_boiler
    if np.any(boiler_fuel > p.boiler_max_fuel + 1e-9):
        raise ValueError("Baseline is infeasible: boiler capacity is too small.")

    grid = data["electric_demand"].to_numpy(float) - data["pv"].to_numpy(float)
    if np.any(grid > p.grid_import_max + 1e-9):
        raise ValueError("Baseline is infeasible: grid import limit is too small.")

    buy = data["buy_price"].to_numpy(float)
    sell = p.export_ratio * buy

    return float(
        p.gas_price * boiler_fuel.sum()
        + np.sum(np.maximum(grid, 0) * buy)
        - np.sum(np.maximum(-grid, 0) * sell)
    )


def validate(result: pd.DataFrame, p: Params) -> None:
    """Check balances and the main technical limits."""
    heat_residual = result["chp_heat"] + result["boiler_heat"] - result["heat_demand"]
    electric_residual = (
        result["chp_el"]
        + result["pv"]
        + result["grid_import"]
        - result["grid_export"]
        + result["battery_discharge"]
        - result["battery_charge"]
        - result["electric_demand"]
    )
    ramp = result["chp_fuel"].diff().abs().fillna(0)

    assert heat_residual.abs().max() < 1e-6
    assert electric_residual.abs().max() < 1e-6
    assert result["grid_import"].max() <= p.grid_import_max + 1e-6
    assert result["grid_export"].max() <= p.grid_export_max + 1e-6
    assert result["battery_soc"].between(-1e-6, p.battery_capacity + 1e-6).all()
    assert ramp.max() <= p.chp_ramp + 1e-6


def plot_results(result: pd.DataFrame, p: Params) -> None:
    """Create four figures with explicit units for the portfolio/demo output."""
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        constrained_layout=True
    )

    # -------------------------------------------------
    # 1. Electricity dispatch. Grid export is plotted below zero only to make
    #    the direction of power flow visually explicit; grid_export itself is
    #    stored as a non-negative export magnitude in the result table.
    # -------------------------------------------------
    ax = axes[0, 0]

    ax.plot(
        result["hour"],
        result["electric_demand"],
        label="Electric demand",
        linewidth=2
    )

    ax.plot(
        result["hour"],
        result["chp_el"],
        label="CHP electricity"
    )

    ax.plot(
        result["hour"],
        result["pv"],
        label="PV"
    )

    ax.plot(
        result["hour"],
        result["grid_import"],
        label="Grid import"
    )

    ax.plot(
        result["hour"],
        -result["grid_export"],
        label="Grid export"
    )

    ax.set_title("Electricity dispatch")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)


    # -------------------------------------------------
    # 2. Heat dispatch
    # -------------------------------------------------
    ax = axes[0, 1]

    ax.plot(
        result["hour"],
        result["heat_demand"],
        label="Heat demand",
        linewidth=2
    )

    ax.plot(
        result["hour"],
        result["chp_heat"],
        label="CHP heat"
    )

    ax.plot(
        result["hour"],
        result["boiler_heat"],
        label="Boiler heat"
    )

    ax.set_title("Heat dispatch")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)


    # -------------------------------------------------
    # 3. Electricity price and battery state. These quantities have different
    #    units and scales, so they use separate y-axes.
    # -------------------------------------------------
    ax1 = axes[1, 0]

    price_line = ax1.plot(
        result["hour"],
        result["buy_price"],
        label="Electricity price",
        linewidth=2,
        color = "tab:blue"
    )

    ax1.set_title("Electricity price and battery state")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Electricity price [EUR/MWh]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(alpha=0.4)

    # second y-axis
    ax2 = ax1.twinx()

    soc_line = ax2.plot(
        result["hour"],
        result["battery_soc"],
        label="Battery SOC",
        linewidth=2,
        color="tab:orange"
    )
    ax2.set_ylabel("Battery state of charge [MWh]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, p.battery_capacity * 1.05)

    # common legend
    lines = price_line + soc_line
    labels = [line.get_label() for line in lines]

    ax1.legend(
        lines,
        labels,
        fontsize=8,
        loc="upper left"
    )

    # -------------------------------------------------
    # 4. CHP nonlinear reference curves and the piecewise-linear curves used
    #    by the MILP. The breakpoint markers are the operating points used by
    #    the optimization model.
    # -------------------------------------------------
    ax = axes[1, 1]

    curve = chp_curve(p)

    fine_load = np.linspace(
        p.chp_min_fraction,
        1.0,
        200
    )

    fine_fuel = fine_load * p.chp_max_fuel

    true_el = (
        np.array([chp_eta_el(x) for x in fine_load])
        * fine_fuel
    )

    true_heat = (
        np.array([chp_eta_th(x) for x in fine_load])
        * fine_fuel
    )

    ax.plot(
        fine_fuel,
        true_el,
        label="Electrical – nonlinear",
        linewidth=2
    )

    ax.plot(
        curve["fuel"],
        curve["p_el"],
        "o--",
        label="Electrical – piecewise"
    )

    ax.plot(
        fine_fuel,
        true_heat,
        label="Thermal – nonlinear",
        linewidth=2
    )

    ax.plot(
        curve["fuel"],
        curve["q_heat"],
        "o--",
        label="Thermal – piecewise"
    )

    ax.set_title("CHP performance curve")
    ax.set_xlabel("CHP fuel input [MW]")
    ax.set_ylabel("Output [MW]")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)


    # -------------------------------------------------
    # Whole figure
    # -------------------------------------------------
    fig.suptitle(
        "Industrial Energy Dispatch – Optimized 24-hour Operation",
        fontsize=16
    )

    plt.savefig(
        "energy_dispatch_dashboard.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()






def main() -> None:
    p = Params()
    data = demo_data()

    result, opt_cost = solve_dispatch(data, p)
    base_cost = baseline_cost(data, p)
    validate(result, p)

    saving = base_cost - opt_cost
    saving_pct = 100 * saving / base_cost

    print("\n--- Industrial Energy Dispatch Demo: piecewise-linear CHP ---")
    print(f"Baseline daily cost:  {base_cost:,.2f} EUR")
    print(f"Optimized daily cost: {opt_cost:,.2f} EUR")
    print(f"Daily saving:          {saving:,.2f} EUR")
    print(f"Saving:                {saving_pct:.2f} %")
    print(f"CHP startups:          {int(result['chp_start'].sum())}")
    print(f"Peak grid import:      {result['grid_import'].max():.2f} MW")
    print(f"Peak grid export:      {result['grid_export'].max():.2f} MW")

    print("\nCHP piecewise operating points:")
    print(chp_curve(p).round(4).to_string(index=False))

    result.to_csv("dispatch_results_piecewise.csv", index=False)
    print("\nSaved: dispatch_results_piecewise.csv")

    plot_results(result, p)


if __name__ == "__main__":
    main()
