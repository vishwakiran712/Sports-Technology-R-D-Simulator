import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QTabWidget,
    QGroupBox, QHeaderView, QComboBox, QFileDialog, QSplitter, QTextEdit,
    QFormLayout, QMessageBox, QDoubleSpinBox, QProgressBar
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ReportLab PDF Dependencies
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ==========================================
# 1. Physics & Biomechanics Simulation Engine
# ==========================================
class SportsTechSimEngine:
    """Consolidated physics, equipment, environmental, and biomechanical simulation model."""

    TECH_PRESETS = {
        "New Sports Shoe": {"mass": 0.22, "stiffness": 140.0, "damping": 12.0, "inertia": 0.008, "energy_return": 0.72},
        "New Racket": {"mass": 0.30, "stiffness": 210.0, "damping": 8.0, "inertia": 0.032, "energy_return": 0.85},
        "New Bat": {"mass": 0.85, "stiffness": 320.0, "damping": 15.0, "inertia": 0.120, "energy_return": 0.60},
        "New Wearable Sensor": {"mass": 0.03, "stiffness": 50.0, "damping": 2.0, "inertia": 0.001, "energy_return": 0.10},
        "New Training Tech": {"mass": 1.20, "stiffness": 180.0, "damping": 25.0, "inertia": 0.050, "energy_return": 0.40},
        "New Sports Surface": {"mass": 5.00, "stiffness": 90.0, "damping": 30.0, "inertia": 0.500, "energy_return": 0.65}
    }

    BASELINE_EQUIPMENT = {
        "mass": 0.28,           # kg
        "stiffness": 100.0,     # kN/m
        "damping": 18.0,        # Ns/m
        "inertia": 0.012,       # kg*m^2
        "energy_return": 0.52   # coefficient 0-1
    }

    @classmethod
    def simulate(cls, athlete, equip, env):
        """Calculates performance, energy efficiency, mechanical loading, and fatigue."""
        # 1. Environmental Air Density Correction
        # Temperature (C), Altitude (m), Pressure approximation
        temp_k = env["temperature"] + 273.15
        p_atm = 101325 * (1 - 2.25577e-5 * env["altitude"])**5.25588
        air_density = p_atm / (287.058 * temp_k)
        
        # 2. Aero Drag Force Impact
        v_rel = athlete["speed"] + env["wind"]
        c_d = 0.9  # General body drag coefficient
        frontal_area = 0.55 * (athlete["mass"] / 75.0)**0.66
        aero_drag = 0.5 * air_density * c_d * frontal_area * max(0.0, v_rel)**2

        # 3. Biomechanical & Surface Interaction
        # Effective mass penalty on limb acceleration
        effective_mass = athlete["mass"] + (equip["mass"] * 4.2)
        
        # Energy restoration & Mechanical Efficiency
        stiffness_ratio = equip["stiffness"] / env["surface_stiffness"]
        optimal_stiffness_tuning = max(0.4, 1.0 - abs(stiffness_ratio - 1.1) * 0.5)
        
        energy_restitution = equip["energy_return"] * optimal_stiffness_tuning
        ground_time_s = athlete["ground_contact"] * (1.0 - (equip["stiffness"] - 100.0) * 0.0015)
        ground_time_s = max(0.07, ground_time_s)

        # 4. Performance Calculations
        # Power & Force Vectors
        raw_power = athlete["power"] * (athlete["strength"] / 200.0)
        net_power = raw_power * (1.0 + (energy_restitution * 0.18)) - (aero_drag * athlete["speed"])
        
        # Sprint Velocity & Jump Displacement
        sprint_speed = (net_power / (effective_mass * 9.81)) * (athlete["timing_efficiency"] * 10.0)
        sprint_speed = min(12.5, max(2.0, sprint_speed))
        
        jump_height_cm = ((athlete["power"] / athlete["mass"]) * 1.8) * (1.0 + energy_restitution * 0.25)
        
        # Mechanical Joint Loading (kN)
        joint_impact_kn = (effective_mass * (sprint_speed / ground_time_s)) / 1000.0
        joint_impact_kn *= (1.0 - (equip["damping"] / 100.0))

        # Efficiency & Fatigue Indicators
        mechanical_efficiency = (sprint_speed * athlete["mass"]) / max(1.0, net_power) * 100.0
        fatigue_index = (joint_impact_kn * 1.8) + (equip["mass"] * 12.0) - (athlete["endurance"] * 0.15)
        stability_index = (1.0 / (equip["inertia"] + 0.01)) * athlete["symmetry"] * optimal_stiffness_tuning

        return {
            "sprint_speed_mps": sprint_speed,
            "jump_height_cm": jump_height_cm,
            "net_power_watts": net_power,
            "mechanical_efficiency_pct": mechanical_efficiency,
            "joint_loading_kn": joint_impact_kn,
            "fatigue_index": fatigue_index,
            "stability_index": stability_index,
            "ground_contact_s": ground_time_s
        }

    @classmethod
    def optimize(cls, athlete, env, target_objective="Maximize Performance"):
        """Uses SciPy Differential Evolution to search for optimal equipment parameter combinations."""
        bounds = [
            (0.10, 1.50),    # mass (kg)
            (50.0, 400.0),   # stiffness (kN/m)
            (2.0, 40.0),     # damping (Ns/m)
            (0.001, 0.200),  # inertia (kg*m^2)
            (0.20, 0.95)     # energy_return (0-1)
        ]

        def cost_fn(x):
            eq = {
                "mass": x[0],
                "stiffness": x[1],
                "damping": x[2],
                "inertia": x[3],
                "energy_return": x[4]
            }
            res = cls.simulate(athlete, eq, env)
            
            if target_objective == "Maximize Performance":
                return -res["sprint_speed_mps"]
            elif target_objective == "Maximize Jump Height":
                return -res["jump_height_cm"]
            elif target_objective == "Minimize Joint Loading":
                return res["joint_loading_kn"]
            elif target_objective == "Maximize Mechanical Efficiency":
                return -res["mechanical_efficiency_pct"]
            else:
                return -res["net_power_watts"]

        opt_res = differential_evolution(cost_fn, bounds, maxiter=80, popsize=12, seed=42)
        
        return {
            "mass": float(opt_res.x[0]),
            "stiffness": float(opt_res.x[1]),
            "damping": float(opt_res.x[2]),
            "inertia": float(opt_res.x[3]),
            "energy_return": float(opt_res.x[4])
        }


# ==========================================
# 2. GUI Desktop Application
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sports Technology R&D Simulator — Decision Support System")
        self.setGeometry(30, 30, 1600, 960)

        # Base State
        self.athlete_params = {
            "height": 1.85, "mass": 78.0, "strength": 240.0, "power": 35.0,
            "speed": 8.5, "endurance": 75.0, "joint_angle": 45.0, "angular_vel": 12.0,
            "symmetry": 0.92, "timing_efficiency": 0.88, "ground_contact": 0.125
        }
        self.env_params = {
            "surface_stiffness": 120.0, "temperature": 22.0, "wind": 0.5, "altitude": 150.0
        }
        self.proposed_equip = SportsTechSimEngine.TECH_PRESETS["New Sports Shoe"].copy()
        self.baseline_equip = SportsTechSimEngine.BASELINE_EQUIPMENT.copy()
        
        self.init_ui()
        self.run_simulation_pipeline()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top Control & Workflow Ribbon
        ribbon_group = QGroupBox("R&D Workflow & Technology Concept Configuration")
        ribbon_layout = QHBoxLayout(ribbon_group)

        self.cmb_concept = QComboBox()
        self.cmb_concept.addItems(list(SportsTechSimEngine.TECH_PRESETS.keys()))
        self.cmb_concept.currentTextChanged.connect(self.on_concept_changed)

        self.cmb_objective = QComboBox()
        self.cmb_objective.addItems([
            "Maximize Performance", "Maximize Jump Height",
            "Minimize Joint Loading", "Maximize Mechanical Efficiency"
        ])

        btn_run = QPushButton("Run Simulation & Optimization")
        btn_run.setStyleSheet("background-color: #0D47A1; color: white; font-weight: bold; padding: 6px 14px;")
        btn_run.clicked.connect(self.run_simulation_pipeline)

        btn_pdf = QPushButton("Export PDF Technical Report")
        btn_pdf.setStyleSheet("background-color: #1B5E20; color: white; font-weight: bold; padding: 6px 14px;")
        btn_pdf.clicked.connect(self.export_pdf_report)

        btn_csv = QPushButton("Export Data & Results (CSV)")
        btn_csv.setStyleSheet("background-color: #E65100; color: white; font-weight: bold; padding: 6px 14px;")
        btn_csv.clicked.connect(self.export_csv_data)

        ribbon_layout.addWidget(QLabel("Technology Concept:"))
        ribbon_layout.addWidget(self.cmb_concept)
        ribbon_layout.addWidget(QLabel("Optimization Target:"))
        ribbon_layout.addWidget(self.cmb_objective)
        ribbon_layout.addWidget(btn_run)
        ribbon_layout.addStretch()
        ribbon_layout.addWidget(btn_csv)
        ribbon_layout.addWidget(btn_pdf)

        main_layout.addWidget(ribbon_group)

        # Main Workspace Splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left Column: Inputs & Parameters Panel
        inputs_widget = QWidget()
        inputs_layout = QVBoxLayout(inputs_widget)

        self.tab_inputs = QTabWidget()
        
        # Tab 1: Equipment Parameters
        tab_eq = QWidget()
        lay_eq = QFormLayout(tab_eq)
        self.spn_eq_mass = self.create_spinbox(0.01, 10.0, 0.01, self.proposed_equip["mass"])
        self.spn_eq_stiff = self.create_spinbox(10.0, 1000.0, 5.0, self.proposed_equip["stiffness"])
        self.spn_eq_damp = self.create_spinbox(0.0, 100.0, 1.0, self.proposed_equip["damping"])
        self.spn_eq_iner = self.create_spinbox(0.001, 1.0, 0.001, self.proposed_equip["inertia"], decimals=3)
        self.spn_eq_ret = self.create_spinbox(0.05, 0.98, 0.01, self.proposed_equip["energy_return"])

        lay_eq.addRow("Mass (kg):", self.spn_eq_mass)
        lay_eq.addRow("Stiffness (kN/m):", self.spn_eq_stiff)
        lay_eq.addRow("Damping (Ns/m):", self.spn_eq_damp)
        lay_eq.addRow("Inertia (kg·m²):", self.spn_eq_iner)
        lay_eq.addRow("Energy Return (0-1):", self.spn_eq_ret)
        self.tab_inputs.addTab(tab_eq, "Equipment")

        # Tab 2: Athlete & Biomechanics
        tab_ath = QWidget()
        lay_ath = QFormLayout(tab_ath)
        self.spn_ath_mass = self.create_spinbox(40.0, 150.0, 0.5, self.athlete_params["mass"])
        self.spn_ath_str = self.create_spinbox(50.0, 500.0, 5.0, self.athlete_params["strength"])
        self.spn_ath_pow = self.create_spinbox(10.0, 100.0, 1.0, self.athlete_params["power"])
        self.spn_ath_gc = self.create_spinbox(0.05, 0.30, 0.005, self.athlete_params["ground_contact"], decimals=3)
        self.spn_ath_sym = self.create_spinbox(0.5, 1.0, 0.01, self.athlete_params["symmetry"])

        lay_ath.addRow("Body Mass (kg):", self.spn_ath_mass)
        lay_ath.addRow("Strength Index:", self.spn_ath_str)
        lay_ath.addRow("Power Density (W/kg):", self.spn_ath_pow)
        lay_ath.addRow("Ground Contact (s):", self.spn_ath_gc)
        lay_ath.addRow("Symmetry Index:", self.spn_ath_sym)
        self.tab_inputs.addTab(tab_ath, "Athlete")

        # Tab 3: Environment
        tab_env = QWidget()
        lay_env = QFormLayout(tab_env)
        self.spn_env_surf = self.create_spinbox(20.0, 500.0, 5.0, self.env_params["surface_stiffness"])
        self.spn_env_temp = self.create_spinbox(-10.0, 45.0, 1.0, self.env_params["temperature"])
        self.spn_env_wind = self.create_spinbox(-10.0, 10.0, 0.5, self.env_params["wind"])
        self.spn_env_alt = self.create_spinbox(0.0, 4000.0, 50.0, self.env_params["altitude"])

        lay_env.addRow("Surface Stiffness (kN/m):", self.spn_env_surf)
        lay_env.addRow("Temperature (°C):", self.spn_env_temp)
        lay_env.addRow("Wind Velocity (m/s):", self.spn_env_wind)
        lay_env.addRow("Altitude (m):", self.spn_env_alt)
        self.tab_inputs.addTab(tab_env, "Environment")

        inputs_layout.addWidget(self.tab_inputs)

        # Simulation Disclaimer
        disclaimer = QLabel(
            "<b>NOTICE:</b> Simulations are derived from numerical physics & biomechanical mathematical models. "
            "Outputs serve for conceptual R&D decision support and do NOT represent experimentally validated real-world results."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("background-color: #FFF3E0; border: 1px solid #FFE0B2; padding: 8px; font-size: 11px;")
        inputs_layout.addWidget(disclaimer)

        splitter.addWidget(inputs_widget)

        # Center Column: Canvas & Charts
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        self.fig = Figure(figsize=(8, 9))
        self.canvas = FigureCanvas(self.fig)
        chart_layout.addWidget(self.canvas)
        splitter.addWidget(chart_widget)

        # Right Column: Executive Dashboard & Engineering Interpretation
        dash_widget = QWidget()
        dash_layout = QVBoxLayout(dash_widget)
        self.tabs_dash = QTabWidget()

        # Tab 1: Delta Comparison Table
        tab_comp = QWidget()
        lay_comp = QVBoxLayout(tab_comp)
        self.table_comp = QTableWidget()
        lay_comp.addWidget(self.table_comp)
        self.tabs_dash.addTab(tab_comp, "Performance Delta")

        # Tab 2: Optimization Results Table
        tab_opt = QWidget()
        lay_opt = QVBoxLayout(tab_opt)
        self.table_opt = QTableWidget()
        lay_opt.addWidget(self.table_opt)
        self.tabs_dash.addTab(tab_opt, "Optimization Target")

        # Tab 3: Executive Summary & Technical Drivers
        tab_summary = QWidget()
        lay_summary = QVBoxLayout(tab_summary)
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        lay_summary.addWidget(self.txt_summary)
        self.tabs_dash.addTab(tab_summary, "Executive Summary")

        dash_layout.addWidget(self.tabs_dash)
        splitter.addWidget(dash_widget)

        splitter.setSizes([380, 750, 470])
        main_layout.addWidget(splitter)

    def create_spinbox(self, min_v, max_v, step, default, decimals=2):
        spn = QDoubleSpinBox()
        spn.setRange(min_v, max_v)
        spn.setSingleStep(step)
        spn.setValue(default)
        spn.setDecimals(decimals)
        spn.valueChanged.connect(self.read_inputs_and_update)
        return spn

    def on_concept_changed(self, concept_name):
        preset = SportsTechSimEngine.TECH_PRESETS[concept_name]
        self.spn_eq_mass.setValue(preset["mass"])
        self.spn_eq_stiff.setValue(preset["stiffness"])
        self.spn_eq_damp.setValue(preset["damping"])
        self.spn_eq_iner.setValue(preset["inertia"])
        self.spn_eq_ret.setValue(preset["energy_return"])
        self.run_simulation_pipeline()

    def read_inputs_and_update(self):
        self.proposed_equip = {
            "mass": self.spn_eq_mass.value(),
            "stiffness": self.spn_eq_stiff.value(),
            "damping": self.spn_eq_damp.value(),
            "inertia": self.spn_eq_iner.value(),
            "energy_return": self.spn_eq_ret.value()
        }
        self.athlete_params["mass"] = self.spn_ath_mass.value()
        self.athlete_params["strength"] = self.spn_ath_str.value()
        self.athlete_params["power"] = self.spn_ath_pow.value()
        self.athlete_params["ground_contact"] = self.spn_ath_gc.value()
        self.athlete_params["symmetry"] = self.spn_ath_sym.value()

        self.env_params["surface_stiffness"] = self.spn_env_surf.value()
        self.env_params["temperature"] = self.spn_env_temp.value()
        self.env_params["wind"] = self.spn_env_wind.value()
        self.env_params["altitude"] = self.spn_env_alt.value()

    def run_simulation_pipeline(self):
        self.read_inputs_and_update()

        # 1. Run Baseline vs Proposed Simulation
        self.base_res = SportsTechSimEngine.simulate(self.athlete_params, self.baseline_equip, self.env_params)
        self.prop_res = SportsTechSimEngine.simulate(self.athlete_params, self.proposed_equip, self.env_params)

        # 2. Run Parameter Sensitivity Analysis (+10% Stiffness, -5% Mass, +8% Energy Return)
        sens_equip = self.proposed_equip.copy()
        sens_equip["stiffness"] *= 1.10
        sens_equip["mass"] *= 0.95
        sens_equip["energy_return"] = min(0.98, sens_equip["energy_return"] * 1.08)
        self.sens_res = SportsTechSimEngine.simulate(self.athlete_params, sens_equip, self.env_params)

        # 3. Run Optimization Search
        obj_target = self.cmb_objective.currentText()
        self.opt_equip = SportsTechSimEngine.optimize(self.athlete_params, self.env_params, obj_target)
        self.opt_res = SportsTechSimEngine.simulate(self.athlete_params, self.opt_equip, self.env_params)

        # 4. Render Interface Components
        self.render_charts()
        self.update_tables()
        self.update_executive_summary()

    def render_charts(self):
        self.fig.clear()

        # Subplot 1: Baseline vs Proposed Performance Outcome Comparison
        ax1 = self.fig.add_subplot(211)
        categories = ["Sprint Speed\n(m/s)", "Jump Height\n(cm)", "Power Output\n(W/100)", "Joint Load\n(kN)", "Efficiency\n(%)"]
        base_vals = [
            self.base_res["sprint_speed_mps"],
            self.base_res["jump_height_cm"],
            self.base_res["net_power_watts"] / 100.0,
            self.base_res["joint_loading_kn"],
            self.base_res["mechanical_efficiency_pct"]
        ]
        prop_vals = [
            self.prop_res["sprint_speed_mps"],
            self.prop_res["jump_height_cm"],
            self.prop_res["net_power_watts"] / 100.0,
            self.prop_res["joint_loading_kn"],
            self.prop_res["mechanical_efficiency_pct"]
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax1.bar(x - width/2, base_vals, width, label="Baseline Concept", color="#78909C")
        ax1.bar(x + width/2, prop_vals, width, label="Proposed Prototype", color="#0D47A1")
        ax1.set_ylabel("Simulated Value")
        ax1.set_title("Performance & Biomechanical Profile Delta", fontsize=10, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=8)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Parameter Sensitivity Impact
        ax2 = self.fig.add_subplot(212)
        params = ["Stiffness (+10%)", "Mass (-5%)", "Energy Return (+8%)"]
        
        # Partial sensitivity impacts on Sprint Speed
        base_speed = self.prop_res["sprint_speed_mps"]
        
        eq_stiff = self.proposed_equip.copy(); eq_stiff["stiffness"] *= 1.10
        p_stiff = ((SportsTechSimEngine.simulate(self.athlete_params, eq_stiff, self.env_params)["sprint_speed_mps"] - base_speed) / base_speed) * 100.0

        eq_mass = self.proposed_equip.copy(); eq_mass["mass"] *= 0.95
        p_mass = ((SportsTechSimEngine.simulate(self.athlete_params, eq_mass, self.env_params)["sprint_speed_mps"] - base_speed) / base_speed) * 100.0

        eq_ret = self.proposed_equip.copy(); eq_ret["energy_return"] = min(0.98, eq_ret["energy_return"] * 1.08)
        p_ret = ((SportsTechSimEngine.simulate(self.athlete_params, eq_ret, self.env_params)["sprint_speed_mps"] - base_speed) / base_speed) * 100.0

        impacts = [p_stiff, p_mass, p_ret]
        colors = ["#2E7D32" if i >= 0 else "#C62828" for i in impacts]

        ax2.barh(params, impacts, color=colors, alpha=0.85)
        ax2.axvline(0, color="black", lw=1)
        ax2.set_xlabel("Sprint Performance Impact (%)")
        ax2.set_title("Parameter Sensitivity Contribution Analysis", fontsize=10, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_tables(self):
        # 1. Comparison Table
        metrics = [
            ("Sprint Speed (m/s)", "sprint_speed_mps"),
            ("Jump Height (cm)", "jump_height_cm"),
            ("Net Power (W)", "net_power_watts"),
            ("Mechanical Efficiency (%)", "mechanical_efficiency_pct"),
            ("Joint Impact Load (kN)", "joint_loading_kn"),
            ("Fatigue Index", "fatigue_index")
        ]

        self.table_comp.setRowCount(len(metrics))
        self.table_comp.setColumnCount(4)
        self.table_comp.setHorizontalHeaderLabels(["Metric", "Baseline", "Proposed", "Delta (%)"])

        for idx, (label, key) in enumerate(metrics):
            b_v = self.base_res[key]
            p_v = self.prop_res[key]
            delta = ((p_v - b_v) / abs(b_v)) * 100.0 if b_v != 0 else 0.0

            self.table_comp.setItem(idx, 0, QTableWidgetItem(label))
            self.table_comp.setItem(idx, 1, QTableWidgetItem(f"{b_v:.2f}"))
            self.table_comp.setItem(idx, 2, QTableWidgetItem(f"{p_v:.2f}"))
            self.table_comp.setItem(idx, 3, QTableWidgetItem(f"{delta:+.2f} %"))

        self.table_comp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 2. Optimization Table
        opt_params = [
            ("Mass (kg)", "mass"),
            ("Stiffness (kN/m)", "stiffness"),
            ("Damping (Ns/m)", "damping"),
            ("Inertia (kg·m²)", "inertia"),
            ("Energy Return", "energy_return")
        ]

        self.table_opt.setRowCount(len(opt_params))
        self.table_opt.setColumnCount(3)
        self.table_opt.setHorizontalHeaderLabels(["Parameter", "Proposed Concept", "Optimized Design"])

        for idx, (label, key) in enumerate(opt_params):
            p_v = self.proposed_equip[key]
            o_v = self.opt_equip[key]

            self.table_opt.setItem(idx, 0, QTableWidgetItem(label))
            self.table_opt.setItem(idx, 1, QTableWidgetItem(f"{p_v:.3f}"))
            self.table_opt.setItem(idx, 2, QTableWidgetItem(f"{o_v:.3f}"))

        self.table_opt.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def update_executive_summary(self):
        concept = self.cmb_concept.currentText()
        speed_delta = ((self.prop_res["sprint_speed_mps"] - self.base_res["sprint_speed_mps"]) / self.base_res["sprint_speed_mps"]) * 100.0
        load_delta = ((self.prop_res["joint_loading_kn"] - self.base_res["joint_loading_kn"]) / self.base_res["joint_loading_kn"]) * 100.0

        summary_md = f"""
### Executive R&D Summary: {concept}

**Opportunity**
Enhance athletic output by optimizing equipment biomechanics and energy restitution profiles.

**Current Limitation**
Baseline equipment dissipates mechanical energy through non-optimal damping ({self.baseline_equip['damping']} Ns/m) and suboptimal mass placement ({self.baseline_equip['mass']} kg).

**Proposed Intervention**
Implementation of high-restitution material matrices offering **{self.proposed_equip['energy_return']*100:.0f}% energy return** with tuned structural stiffness (**{self.proposed_equip['stiffness']:.1f} kN/m**).

**Simulated Impact**
* **Performance Delta**: **{speed_delta:+.2f}%** shift in simulated velocity.
* **Mechanical Load Shift**: **{load_delta:+.2f}%** change in peak joint loading.

**Key Technical Drivers**
1. **Energy Restitution**: Substantially enhances force propagation efficiency during ground contact.
2. **Mass Reduction**: Reduces rotational limb inertia, accelerating stride turnover rate.

**Engineering Limitations**
Simulated outputs do not capture material durability degradation or real-world athlete motor adaptation over extended usage.
"""
        self.txt_summary.setMarkdown(summary_md)

    def export_csv_data(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Simulation Results", "", "CSV Files (*.csv)")
        if not path:
            return

        # Build Export Dataframe
        df_base = pd.DataFrame([self.base_res]).add_prefix("baseline_")
        df_prop = pd.DataFrame([self.prop_res]).add_prefix("proposed_")
        df_opt = pd.DataFrame([self.opt_res]).add_prefix("optimized_")
        
        export_df = pd.concat([df_base, df_prop, df_opt], axis=1)
        export_df.to_csv(path, index=False)
        QMessageBox.information(self, "Export Successful", f"Simulation parameters and outputs exported to:\n{path}")

    def export_pdf_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF Technical Report", "", "PDF Files (*.pdf)")
        if not path:
            return

        try:
            chart_img_path = os.path.join(os.getcwd(), "temp_chart.png")
            self.fig.savefig(chart_img_path, dpi=200)

            doc = SimpleDocTemplate(path, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
            story = []
            styles = getSampleStyleSheet()

            # Header Title
            title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0D47A1'), spaceAfter=10)
            story.append(Paragraph("Sports Technology R&D Technical Evaluation Report", title_style))
            story.append(Paragraph(f"<b>Concept evaluated:</b> {self.cmb_concept.currentText()} | <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
            story.append(Spacer(1, 12))

            # Executive Summary Section
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            summary_plain = self.txt_summary.toPlainText()
            for line in summary_plain.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 3))

            story.append(Spacer(1, 10))

            # Performance Charts
            story.append(Paragraph("Simulated Performance & Sensitivity Analysis", styles['Heading2']))
            story.append(RLImage(chart_img_path, width=500, height=380))
            story.append(Spacer(1, 10))

            # Results Table
            story.append(Paragraph("Performance & Biomechanical Metrics Comparison", styles['Heading2']))
            table_data = [["Metric", "Baseline Concept", "Proposed Concept", "Delta (%)"]]
            
            for row_idx in range(self.table_comp.rowCount()):
                table_data.append([
                    self.table_comp.item(row_idx, 0).text(),
                    self.table_comp.item(row_idx, 1).text(),
                    self.table_comp.item(row_idx, 2).text(),
                    self.table_comp.item(row_idx, 3).text()
                ])

            t = Table(table_data, colWidths=[180, 100, 100, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0D47A1')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(t)

            doc.build(story)
            
            # Clean up temp image
            if os.path.exists(chart_img_path):
                os.remove(chart_img_path)

            QMessageBox.information(self, "PDF Report Generated", f"Technical report successfully generated:\n{path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to generate PDF report:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())