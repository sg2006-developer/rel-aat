"""
Smart Irrigation Controller — Desktop GUI
Run with: python app.py
Requires: pip install numpy matplotlib
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from environment import IrrigationEnv, ACTIONS, MOISTURE_LEVELS, WEATHER_CONDITIONS, TIME_SLOTS
from agent import QLearningAgent


# ------------------------------------------------------------------ #
#  Colour palette                                                      #
# ------------------------------------------------------------------ #
BG        = "#1e2a38"
BG_DARK   = "#0d1b2a"
ACCENT    = "#16a085"
TEXT      = "#ecf0f1"
MUTED     = "#95a5a6"
BORDER    = "#2c3e50"

MOISTURE_COLORS = {
    "Very Dry":  "#e74c3c",
    "Dry":       "#e67e22",
    "Optimal":   "#27ae60",
    "Moist":     "#2980b9",
    "Saturated": "#8e44ad",
}
ACTION_COLORS = {
    "No Water":   "#e74c3c",
    "Water Low":  "#f39c12",
    "Water High": "#27ae60",
}


# ------------------------------------------------------------------ #
#  Main Application                                                    #
# ------------------------------------------------------------------ #

class IrrigationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Irrigation Controller — RL Simulation")
        self.root.geometry("1150x730")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._build_header()
        self._build_body()
        self._check_policies()

    # ---------------------------------------------------------------- #
    #  Layout builders                                                  #
    # ---------------------------------------------------------------- #

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=ACCENT, pady=9)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Smart Irrigation Controller",
                 font=("Courier", 17, "bold"), bg=ACCENT, fg="white").pack()
        tk.Label(hdr,
                 text="Reinforcement Learning — Q-Learning  |  SDG 2: Zero Hunger",
                 font=("Courier", 9), bg=ACCENT, fg="#d5f5e3").pack()

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Left sidebar (fixed width)
        left = tk.Frame(body, bg=BG, width=310)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        # Right area (charts + tabs)
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_state_panel(left)
        self._build_controls(left)
        self._build_log(left)
        self._build_charts(right)

    def _build_state_panel(self, parent):
        frm = self._lframe(parent, " Current State ")
        frm.pack(fill="x", pady=(0, 7))

        self._sv = {}   # key -> (StringVar, Label)
        fields = [
            ("Day",          "day"),
            ("Soil Moisture","moisture"),
            ("Weather",      "weather"),
            ("Time of Day",  "time"),
            ("Last Action",  "action"),
            ("Step Reward",  "reward"),
            ("Total Water",  "water"),
            ("Crop Health",  "health"),
        ]
        for label, key in fields:
            row = tk.Frame(frm, bg=BG)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=f"{label:<14}", font=("Courier", 9),
                     bg=BG, fg=MUTED, width=14, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            lbl = tk.Label(row, textvariable=var, font=("Courier", 9, "bold"),
                           bg=BG, fg=TEXT, anchor="w")
            lbl.pack(side="left", padx=3)
            self._sv[key] = (var, lbl)

        # Episode progress bar
        tk.Label(frm, text="Episode progress:", font=("Courier", 8),
                 bg=BG, fg=MUTED).pack(anchor="w", padx=8, pady=(4, 0))
        self.progress = ttk.Progressbar(frm, length=270, mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=(2, 7))

    def _build_controls(self, parent):
        frm = self._lframe(parent, " Controls ")
        frm.pack(fill="x", pady=(0, 7))

        self.btn_train = self._btn(frm, "Train Both Policies",   ACCENT,   self._start_training)
        self.btn_v1    = self._btn(frm, "Simulate — Policy V1",  "#2980b9", lambda: self._start_sim("policy_v1.pkl", "V1"))
        self.btn_v2    = self._btn(frm, "Simulate — Policy V2",  "#8e44ad", lambda: self._start_sim("policy_v2_explored.pkl", "V2"))
        self.btn_cmp   = self._btn(frm, "Compare Policies",      "#d35400", self._start_compare)

        tk.Label(frm, text="Simulation Speed (delay/step)",
                 font=("Courier", 8), bg=BG, fg=MUTED).pack(anchor="w", padx=8, pady=(4, 0))
        self.speed = tk.DoubleVar(value=0.04)
        tk.Scale(frm, from_=0.005, to=0.3, resolution=0.005, orient="horizontal",
                 variable=self.speed, bg=BG, fg=TEXT, troughcolor=BORDER,
                 highlightbackground=BG, font=("Courier", 8), showvalue=True
                 ).pack(fill="x", padx=8, pady=(0, 7))

    def _build_log(self, parent):
        frm = self._lframe(parent, " Output Log ")
        frm.pack(fill="both", expand=True)
        self.log_box = scrolledtext.ScrolledText(
            frm, height=8, font=("Courier", 8),
            bg=BG_DARK, fg="#2ecc71", insertbackground="white",
            relief="flat", state="disabled"
        )
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_charts(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        self.notebook = nb

        # Tab 1 — Live Simulation
        t1 = tk.Frame(nb, bg=BG); nb.add(t1, text="  Live Simulation  ")
        self.sim_fig = Figure(figsize=(6.5, 5), facecolor=BG)
        self.sim_fig.subplots_adjust(hspace=0.5)
        self.ax_r = self._dark_ax(self.sim_fig.add_subplot(311))
        self.ax_m = self._dark_ax(self.sim_fig.add_subplot(312))
        self.ax_w = self._dark_ax(self.sim_fig.add_subplot(313))
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, t1)
        self.sim_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 2 — Training Progress
        t2 = tk.Frame(nb, bg=BG); nb.add(t2, text="  Training Progress  ")
        self.trn_fig = Figure(figsize=(6.5, 5), facecolor=BG)
        self.trn_fig.subplots_adjust(hspace=0.5)
        self.ax_tr = self._dark_ax(self.trn_fig.add_subplot(311))
        self.ax_tw = self._dark_ax(self.trn_fig.add_subplot(312))
        self.ax_te = self._dark_ax(self.trn_fig.add_subplot(313))
        self.trn_canvas = FigureCanvasTkAgg(self.trn_fig, t2)
        self.trn_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 3 — Policy Comparison
        t3 = tk.Frame(nb, bg=BG); nb.add(t3, text="  Policy Comparison  ")
        self.cmp_fig = Figure(figsize=(6.5, 5), facecolor=BG)
        self.cmp_canvas = FigureCanvasTkAgg(self.cmp_fig, t3)
        self.cmp_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        self._reset_sim_chart()

    # ---------------------------------------------------------------- #
    #  Widget helpers                                                   #
    # ---------------------------------------------------------------- #

    def _lframe(self, parent, title):
        return tk.LabelFrame(parent, text=title, bg=BG, fg=ACCENT,
                             font=("Courier", 9, "bold"), relief="ridge", bd=2)

    def _btn(self, parent, text, color, cmd):
        b = tk.Button(parent, text=text, font=("Courier", 9, "bold"),
                      bg=color, fg="white", relief="flat", cursor="hand2",
                      pady=6, bd=0, command=cmd)
        b.pack(fill="x", padx=8, pady=3)
        return b

    def _dark_ax(self, ax):
        ax.set_facecolor(BG_DARK)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        return ax

    # ---------------------------------------------------------------- #
    #  State panel updates (always call via root.after)                #
    # ---------------------------------------------------------------- #

    def _sv_set(self, key, val, color=None):
        var, lbl = self._sv[key]
        var.set(val)
        lbl.configure(fg=color if color else TEXT)

    def _update_state_panel(self, env, action, reward):
        m_name = MOISTURE_LEVELS[env.moisture_idx]
        a_name = ACTIONS[action]
        self._sv_set("day",      str(env.step_count))
        self._sv_set("moisture", m_name, MOISTURE_COLORS.get(m_name))
        self._sv_set("weather",  WEATHER_CONDITIONS[env.weather_idx])
        self._sv_set("time",     TIME_SLOTS[env.time_idx])
        self._sv_set("action",   a_name, ACTION_COLORS.get(a_name))
        self._sv_set("reward",   f"{reward:+.2f}",
                     "#27ae60" if reward >= 0 else "#e74c3c")
        self._sv_set("water",    str(env.total_water))
        self._sv_set("health",   f"{env.crop_health:.1f} / 100")
        self.progress["value"] = (env.step_count / 365) * 100

    # ---------------------------------------------------------------- #
    #  Log                                                              #
    # ---------------------------------------------------------------- #

    def _log(self, msg):
        def _do():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(0, _do)

    # ---------------------------------------------------------------- #
    #  Policy file check                                                #
    # ---------------------------------------------------------------- #

    def _check_policies(self):
        v1 = os.path.exists("policy_v1.pkl")
        v2 = os.path.exists("policy_v2_explored.pkl")
        self._log("=== Smart Irrigation Controller ===")
        self._log(f"  policy_v1.pkl       : {'FOUND' if v1 else 'NOT FOUND'}")
        self._log(f"  policy_v2_explored  : {'FOUND' if v2 else 'NOT FOUND'}")
        if not v1 and not v2:
            self._log("  >> Click 'Train Both Policies' to begin.")
        self.btn_v1.configure(state="normal"  if v1        else "disabled")
        self.btn_v2.configure(state="normal"  if v2        else "disabled")
        self.btn_cmp.configure(state="normal" if v1 and v2 else "disabled")

    # ================================================================ #
    #  TRAINING                                                         #
    # ================================================================ #

    def _start_training(self):
        self.btn_train.configure(state="disabled", text="Training...")
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        from train import POLICY_CONFIGS

        for cfg_key, cfg in POLICY_CONFIGS.items():
            self._log(f"\n--- {cfg['label']} ---")
            env   = IrrigationEnv()
            agent = QLearningAgent(
                n_states=env.n_states, n_actions=env.n_actions,
                alpha=cfg["alpha"], gamma=cfg["gamma"],
                epsilon=cfg["epsilon"], epsilon_min=cfg["epsilon_min"],
                epsilon_decay=cfg["epsilon_decay"],
            )
            n_ep  = cfg["n_episodes"]
            r_log, w_log, e_log = [], [], []

            for ep in range(1, n_ep + 1):
                state   = env.reset()
                total_r = 0.0
                done    = False
                while not done:
                    a = agent.choose_action(state)
                    ns, r, done = env.step(a)
                    agent.update(state, a, r, ns, done)
                    state = ns; total_r += r
                agent.decay_epsilon()
                r_log.append(total_r)
                w_log.append(env.total_water)
                e_log.append(agent.epsilon)

                pct = (ep / n_ep) * 100
                self.root.after(0, lambda p=pct: self.progress.__setitem__("value", p))

                if ep % 100 == 0:
                    avg_r = np.mean(r_log[-100:])
                    avg_w = np.mean(w_log[-100:])
                    self._log(f"  Ep {ep:4d} | R: {avg_r:8.1f} | W: {avg_w:5.1f} | ε: {agent.epsilon:.3f}")
                    self._redraw_train(r_log, w_log, e_log, cfg["label"])

            agent.save(cfg["save_path"])
            agent.save("q_table.pkl")
            self._log(f"  Saved: {cfg['save_path']}")

        self.root.after(0, lambda: self.progress.__setitem__("value", 0))
        self._log("\n  All policies trained.")
        self.root.after(0, self._post_train)

    def _post_train(self):
        self.btn_train.configure(state="normal", text="Train Both Policies")
        self._check_policies()

    def _redraw_train(self, r_log, w_log, e_log, label):
        def _do():
            W = 50
            xs = list(range(1, len(r_log) + 1))

            for ax in [self.ax_tr, self.ax_tw, self.ax_te]:
                ax.clear(); self._dark_ax(ax)

            self.ax_tr.plot(xs, r_log, alpha=0.25, color=ACCENT, lw=0.7)
            if len(r_log) >= W:
                ma = np.convolve(r_log, np.ones(W)/W, mode="valid")
                self.ax_tr.plot(range(W, len(r_log)+1), ma, color=ACCENT, lw=1.6)
            self.ax_tr.set_title(f"Reward — {label}", color=TEXT, fontsize=8, pad=3)

            self.ax_tw.plot(xs, w_log, alpha=0.25, color="#e67e22", lw=0.7)
            if len(w_log) >= W:
                ma = np.convolve(w_log, np.ones(W)/W, mode="valid")
                self.ax_tw.plot(range(W, len(w_log)+1), ma, color="#e67e22", lw=1.6)
            self.ax_tw.set_title("Water Used", color=TEXT, fontsize=8, pad=3)

            self.ax_te.plot(xs, e_log, color="#9b59b6", lw=1.6)
            self.ax_te.set_title("Exploration Rate (ε)", color=TEXT, fontsize=8, pad=3)

            self.trn_fig.tight_layout(pad=1.5)
            self.trn_canvas.draw()
            self.notebook.select(1)
        self.root.after(0, _do)

    # ================================================================ #
    #  SIMULATION                                                       #
    # ================================================================ #

    def _reset_sim_chart(self):
        for ax, title, color in [
            (self.ax_r, "Cumulative Reward",   ACCENT),
            (self.ax_m, "Soil Moisture Level", "#2980b9"),
            (self.ax_w, "Water Used",          "#e67e22"),
        ]:
            ax.clear(); self._dark_ax(ax)
            ax.set_title(title, color=TEXT, fontsize=8, pad=3)
        self.ax_m.set_yticks(range(5))
        self.ax_m.set_yticklabels(MOISTURE_LEVELS, fontsize=6, color=MUTED)
        self.sim_fig.tight_layout(pad=1.5)
        self.sim_canvas.draw()

    def _start_sim(self, path, label):
        if not os.path.exists(path):
            self._log(f"  {path} not found. Train first.")
            return
        self._reset_sim_chart()
        self._log(f"\n--- Simulation: {label} ---")
        self.notebook.select(0)
        self.btn_v1.configure(state="disabled")
        self.btn_v2.configure(state="disabled")
        threading.Thread(target=self._run_sim, args=(path, label), daemon=True).start()

    def _run_sim(self, path, label):
        env   = IrrigationEnv()
        agent = QLearningAgent(env.n_states, env.n_actions)
        agent.load(path)

        state         = env.reset()
        done          = False
        cum_r         = 0.0
        r_hist, m_hist, w_hist = [], [], []

        while not done:
            action           = agent.choose_action_greedy(state)
            next_state, r, done = env.step(action)
            cum_r           += r

            r_hist.append(cum_r)
            m_hist.append(env.moisture_idx)
            w_hist.append(env.total_water)

            snap = (env.moisture_idx, env.weather_idx, env.time_idx,
                    action, r, env.total_water, env.crop_health, env.step_count)

            def _ui(s=snap, rh=list(r_hist), mh=list(m_hist), wh=list(w_hist)):
                mi, wi, ti, ac, rew, tw, ch, sc = s
                # State panel
                m_name = MOISTURE_LEVELS[mi]
                a_name = ACTIONS[ac]
                self._sv_set("day",      str(sc))
                self._sv_set("moisture", m_name, MOISTURE_COLORS.get(m_name))
                self._sv_set("weather",  WEATHER_CONDITIONS[wi])
                self._sv_set("time",     TIME_SLOTS[ti])
                self._sv_set("action",   a_name, ACTION_COLORS.get(a_name))
                self._sv_set("reward",   f"{rew:+.2f}",
                             "#27ae60" if rew >= 0 else "#e74c3c")
                self._sv_set("water",    str(tw))
                self._sv_set("health",   f"{ch:.1f} / 100")
                self.progress["value"] = (sc / 365) * 100
                # Charts
                days = list(range(1, len(rh)+1))
                for ax in [self.ax_r, self.ax_m, self.ax_w]:
                    ax.clear(); self._dark_ax(ax)
                self.ax_r.plot(days, rh, color=ACCENT,    lw=1.3)
                self.ax_r.set_title("Cumulative Reward",  color=TEXT, fontsize=8, pad=3)
                self.ax_m.plot(days, mh, color="#2980b9", lw=1.3)
                self.ax_m.set_title("Soil Moisture Level",color=TEXT, fontsize=8, pad=3)
                self.ax_m.set_yticks(range(5))
                self.ax_m.set_yticklabels(MOISTURE_LEVELS, fontsize=6, color=MUTED)
                self.ax_m.set_ylim(-0.5, 4.5)
                self.ax_w.plot(days, wh, color="#e67e22", lw=1.3)
                self.ax_w.set_title("Water Used",         color=TEXT, fontsize=8, pad=3)
                self.sim_fig.tight_layout(pad=1.5)
                self.sim_canvas.draw()

            self.root.after(0, _ui)
            time.sleep(self.speed.get())
            state = next_state

        self._log(f"  Done | Water: {env.total_water} | Health: {env.crop_health:.1f} | Reward: {cum_r:.1f}")
        self.root.after(0, lambda: [
            self.progress.__setitem__("value", 0),
            self.btn_v1.configure(state="normal" if os.path.exists("policy_v1.pkl") else "disabled"),
            self.btn_v2.configure(state="normal" if os.path.exists("policy_v2_explored.pkl") else "disabled"),
        ])

    # ================================================================ #
    #  COMPARISON                                                       #
    # ================================================================ #

    def _start_compare(self):
        self.btn_cmp.configure(state="disabled", text="Comparing...")
        threading.Thread(target=self._run_compare, daemon=True).start()

    def _run_compare(self):
        from evaluate import run_episode, FixedScheduleAgent

        n = 30
        env      = IrrigationEnv()
        baseline = FixedScheduleAgent(env)
        results  = {}

        self._log("\n--- Policy Comparison (30 episodes each) ---")

        b_r, b_w, b_h = [], [], []
        for _ in range(n):
            res = run_episode(env, baseline.choose_action)
            b_r.append(res["total_reward"]); b_w.append(res["total_water"]); b_h.append(res["crop_health"])
        results["Fixed\nBaseline"] = (b_r, b_w, b_h)

        for name, path in [("V1\nConservative", "policy_v1.pkl"),
                            ("V2\nExploratory",  "policy_v2_explored.pkl")]:
            if not os.path.exists(path):
                continue
            agent = QLearningAgent(env.n_states, env.n_actions)
            agent.load(path)
            er, ew, eh = [], [], []
            for _ in range(n):
                res = run_episode(env, agent.choose_action_greedy)
                er.append(res["total_reward"]); ew.append(res["total_water"]); eh.append(res["crop_health"])
            results[name] = (er, ew, eh)
            flat = name.replace("\n", " ")
            self._log(f"  {flat:<20} R={np.mean(er):7.1f}  W={np.mean(ew):5.1f}  H={np.mean(eh):.1f}")

        self.root.after(0, lambda: self._draw_compare(results))

    def _draw_compare(self, results):
        self.cmp_fig.clear()
        self.cmp_fig.patch.set_facecolor(BG)
        self.cmp_fig.suptitle("Policy Comparison — RL vs Fixed Baseline",
                               color=TEXT, fontsize=10, fontweight="bold")

        names  = list(results.keys())
        colors = ["#e74c3c", ACCENT, "#8e44ad"]
        for i, (title, idx) in enumerate(
            [("Avg Total Reward", 0), ("Avg Water Used\n(lower = better)", 1), ("Avg Crop Health", 2)], 1
        ):
            ax = self._dark_ax(self.cmp_fig.add_subplot(1, 3, i))
            vals = [np.mean(results[n][idx]) for n in names]
            bars = ax.bar(range(len(names)), vals,
                          color=colors[:len(names)], edgecolor=BORDER, linewidth=0.7)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + abs(bar.get_height())*0.02,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=8, color=TEXT, fontweight="bold")
            ax.set_title(title, color=TEXT, fontsize=8)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=7, color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=7)

        self.cmp_fig.tight_layout(rect=[0, 0, 1, 0.92])
        self.cmp_canvas.draw()
        self.notebook.select(2)
        self._log("  Comparison complete.")
        self.btn_cmp.configure(state="normal", text="Compare Policies")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    root = tk.Tk()
    IrrigationApp(root)
    root.mainloop()
