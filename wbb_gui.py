#!/usr/bin/env python3
"""
wbb_gui.py - WELAB Wii Balance Board posture app (all-in-one GUI).

Buttons for everything: connect (Demo/Live), tare, collect NEUTRAL/SLOUCHED
trials (adjustable duration), train & compare models (with a feature-importance
ranking), and APPLY the trained model to live monitoring only when you choose.
"""

import argparse
import csv
import math
import os
import queue
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk, messagebox, simpledialog
from typing import Optional

from wbb_core import (
    Sample, SwayWindow, SyntheticSource, BoardCalibration,
    BOARD_X_CM, BOARD_Y_CM, board_cm_to_canvas, confidence_ellipse_points,
)
from wbb_bridge import BridgeSource, Tare, resample_uniform
from wbb_dataset import Dataset, DEFAULT_WINDOW_S, DEFAULT_HOP_S, BASELINE_LABEL
from wbb_monitor import PostureClassifier, AlarmController
import wbb_train

TRACE_SECONDS = 4.0
ROLLING_SECONDS = DEFAULT_WINDOW_S
POLL_MS = 25
DEFAULT_COLLECT_S = 30.0


def _demo_source(fs):
    return SyntheticSource(
        cal=BoardCalibration.identity_counts(), fs=fs, duration_s=36000.0,
        body_kg=70.0,
        cop_x_fn=lambda t: 1.5 * math.cos(2 * math.pi * 0.3 * t),
        cop_y_fn=lambda t: 1.0 * math.sin(2 * math.pi * 0.25 * t))


def _demo_trial(label, dur, fs):
    slouch = (label == "slouched")
    bias = -4.0 if slouch else 1.0
    amp = 2.5 if slouch else 1.2
    src = SyntheticSource(cal=BoardCalibration.identity_counts(), fs=fs,
                          duration_s=dur, body_kg=70.0,
                          cop_x_fn=lambda t: 0.5 * math.cos(2 * math.pi * 0.3 * t),
                          cop_y_fn=lambda t: bias + amp * math.sin(2 * math.pi * 0.35 * t))
    return resample_uniform(list(src.stream()), fs)


class _Worker(threading.Thread):
    def __init__(self, source, q, stop_event, paced):
        super().__init__(daemon=True)
        self.source, self.q, self.stop_event, self.paced = source, q, stop_event, paced

    def run(self):
        last_t = None
        try:
            for s in self.source.stream():
                if self.stop_event.is_set():
                    break
                if self.paced and last_t is not None and s.t - last_t > 0:
                    time.sleep(min(s.t - last_t, 0.05))
                last_t = s.t
                self.q.put(s)
        except Exception as exc:
            self.q.put(("__error__", str(exc)))


class WBBApp(ttk.Frame):
    def __init__(self, master, port=8674, fs=100.0, db="wbb_db",
                 model_path="posture_model.joblib", demo_default=False):
        super().__init__(master, padding=8)
        self.port = port
        self.fs = fs
        self.db_var = tk.StringVar(value=db)
        self._db_after = None
        self.model_path = model_path
        self.model_loaded_from = None

        self.q = queue.Queue()
        self.train_q = queue.Queue()
        self.fig_q = queue.Queue()
        self.val_q = queue.Queue()
        self.valfig_q = queue.Queue()
        self.cmp_q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        self.tare = None
        self.tare_capture = None
        self.rolling = deque()
        self.last_sample = None

        self.collecting = None
        self.baselining = None               # capturing a live neutral baseline
        self.live_baseline = None            # subtracted before prediction if normalized
        self.classifier = None
        self.alarm = None
        self.monitor_active = False          # model applied to live ONLY when True
        self._apply_after_baseline = False
        self.clf_window_s = ROLLING_SECONDS
        self.buf_seconds = ROLLING_SECONDS

        self.val_db = tk.StringVar(value=(db.rstrip("/\\") + "_val"))
        self.mode = tk.StringVar(value="demo" if demo_default else "live")
        self.subject = tk.StringVar(value="S01")
        self.collect_seconds = tk.StringVar(value="30")
        self.base_seconds = tk.StringVar(value="5")
        self.base_src = tk.StringVar(value="auto")
        self.use_norm = tk.BooleanVar(value=True)
        self.win_seconds = tk.StringVar(value=str(int(DEFAULT_WINDOW_S)))
        self.group_by = tk.StringVar(value="trial")
        self.slouch_thresh = tk.DoubleVar(value=0.5)

        self._build()
        self._refresh_counts()
        self._poll()

    def _dur(self):
        try:
            d = float(self.collect_seconds.get())
            return d if d > 0 else DEFAULT_COLLECT_S
        except ValueError:
            return DEFAULT_COLLECT_S

    # ---------------- layout ----------------
    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, width=470, height=360, bg="#0e1116",
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.canvas.bind("<Configure>", lambda e: self._draw_board())

        right_outer = ttk.Frame(self)
        right_outer.grid(row=0, column=1, sticky="ns")
        pcanvas = tk.Canvas(right_outer, width=290, highlightthickness=0,
                            borderwidth=0)
        vsb = ttk.Scrollbar(right_outer, orient="vertical", command=pcanvas.yview)
        pcanvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        pcanvas.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(pcanvas)
        pcanvas.create_window((0, 0), window=right, anchor="nw")

        def _fit(_e=None):
            # Track the panel's natural width, otherwise anything wider than the
            # canvas is silently cut off (there is no horizontal scrollbar).
            pcanvas.configure(scrollregion=pcanvas.bbox("all"),
                              width=min(right.winfo_reqwidth(), 420))
        right.bind("<Configure>", _fit)

        def _wheel(e):
            pcanvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        pcanvas.bind_all("<MouseWheel>", _wheel)          # Windows / macOS
        pcanvas.bind_all("<Button-4>", lambda e: pcanvas.yview_scroll(-1, "units"))
        pcanvas.bind_all("<Button-5>", lambda e: pcanvas.yview_scroll(1, "units"))

        # 1. Connect
        f1 = ttk.LabelFrame(right, text="1. Connect", padding=6)
        f1.pack(fill="x", pady=(0, 6))
        mrow = ttk.Frame(f1); mrow.pack(fill="x")
        ttk.Radiobutton(mrow, text="Demo", value="demo",
                        variable=self.mode).pack(side="left")
        ttk.Radiobutton(mrow, text="Live board", value="live",
                        variable=self.mode).pack(side="left", padx=(8, 0))
        self.btn_conn = ttk.Button(f1, text="Connect", command=self.toggle_stream)
        self.btn_conn.pack(fill="x", pady=(4, 2))
        self.btn_tare = ttk.Button(f1, text="Zero (tare) - stand OFF board",
                                   command=self.do_tare, state="disabled")
        self.btn_tare.pack(fill="x")
        self.lbl_status = ttk.Label(f1, text="disconnected", foreground="#888")
        self.lbl_status.pack(anchor="w", pady=(3, 0))

        # Live
        f2 = ttk.LabelFrame(right, text="Live", padding=6)
        f2.pack(fill="x", pady=6)
        self.lbl_mass = ttk.Label(f2, text="-- kg", font=("TkDefaultFont", 18, "bold"))
        self.lbl_mass.pack(anchor="w")
        self.lbl_cop = ttk.Label(f2, text="CoP  x=--  y=--")
        self.lbl_cop.pack(anchor="w")
        ttk.Label(f2, text="Left / Right").pack(anchor="w", pady=(4, 0))
        self.bar_lr = tk.Canvas(f2, width=210, height=16, bg="#1b2027",
                                highlightthickness=0); self.bar_lr.pack(anchor="w")
        ttk.Label(f2, text="Anterior / Posterior").pack(anchor="w", pady=(3, 0))
        self.bar_ap = tk.Canvas(f2, width=210, height=16, bg="#1b2027",
                                highlightthickness=0); self.bar_ap.pack(anchor="w")
        self.lbl_metrics = ttk.Label(f2, text="path -- · vel -- · ellipse --")
        self.lbl_metrics.pack(anchor="w", pady=(4, 0))
        self.lbl_posture = ttk.Label(f2, text="posture: (model not applied)",
                                     font=("TkDefaultFont", 12, "bold"))
        self.lbl_posture.pack(anchor="w", pady=(2, 0))
        self.lbl_why = ttk.Label(f2, text="", foreground="#9aa7b4",
                                 wraplength=250, justify="left")
        self.lbl_why.pack(anchor="w")

        # 2. Collect
        f3 = ttk.LabelFrame(right, text="2. Collect data", padding=6)
        f3.pack(fill="x", pady=6)
        srow = ttk.Frame(f3); srow.pack(fill="x")
        ttk.Label(srow, text="Subject:").pack(side="left")
        ttk.Entry(srow, textvariable=self.subject, width=7).pack(side="left", padx=4)
        ttk.Label(srow, text="Duration(s):").pack(side="left")
        ttk.Entry(srow, textvariable=self.collect_seconds, width=5).pack(side="left",
                                                                         padx=4)
        drow = ttk.Frame(f3); drow.pack(fill="x", pady=(3, 0))
        ttk.Label(drow, text="Dataset folder:").pack(side="left")
        e_db = ttk.Entry(drow, textvariable=self.db_var, width=13)
        e_db.pack(side="left", padx=4)
        self.db_var.trace_add("write", self._on_db_typed)
        ttk.Button(drow, text="Refresh", width=8,
                   command=self._refresh_counts).pack(side="left")
        brow = ttk.Frame(f3); brow.pack(fill="x", pady=(4, 0))
        ttk.Label(brow, text="Baseline(s):").pack(side="left")
        ttk.Entry(brow, textvariable=self.base_seconds, width=5).pack(side="left",
                                                                      padx=4)
        ttk.Label(brow, text="(one per subject, first)",
                  foreground="#888").pack(side="left")
        self.btn_base = ttk.Button(
            f3, text="Record BASELINE (neutral)",
            command=lambda: self.start_collect(BASELINE_LABEL))
        self.btn_base.pack(fill="x", pady=(2, 0))
        self.btn_neutral = ttk.Button(
            f3, text="Record NEUTRAL", command=lambda: self.start_collect("neutral"))
        self.btn_neutral.pack(fill="x", pady=(4, 2))
        self.btn_slouched = ttk.Button(
            f3, text="Record SLOUCHED", command=lambda: self.start_collect("slouched"))
        self.btn_slouched.pack(fill="x")
        self.lbl_collect = ttk.Label(f3, text="", font=("TkDefaultFont", 11, "bold"))
        self.lbl_collect.pack(anchor="w", pady=(4, 0))
        self.lbl_counts = ttk.Label(f3, text="dataset: --")
        self.lbl_counts.pack(anchor="w")
        ttk.Button(f3, text="Clear dataset (keeps raw in archive)",
                   command=self.clear_dataset).pack(fill="x", pady=(4, 0))

        # 3. Train
        f4 = ttk.LabelFrame(right, text="3. Train & compare", padding=6)
        f4.pack(fill="x", pady=6)
        trow = ttk.Frame(f4); trow.pack(fill="x")
        ttk.Label(trow, text="Window(s):").pack(side="left")
        ttk.Entry(trow, textvariable=self.win_seconds, width=4).pack(side="left",
                                                                     padx=(2, 8))
        ttk.Label(trow, text="Group:").pack(side="left")
        ttk.Combobox(trow, textvariable=self.group_by, values=["trial", "subject", "LOSO"],
                     width=8, state="readonly").pack(side="left", padx=2)
        ttk.Checkbutton(f4, text="per-subject baseline normalization",
                        variable=self.use_norm).pack(anchor="w")
        brow2 = ttk.Frame(f4); brow2.pack(fill="x")
        ttk.Label(brow2, text="Baseline from:").pack(side="left")
        ttk.Combobox(brow2, textvariable=self.base_src,
                     values=["auto", "neutral"], width=8,
                     state="readonly").pack(side="left", padx=2)
        ttk.Label(brow2, text="(auto = dedicated)",
                  foreground="#888").pack(side="left")
        self.btn_train = ttk.Button(f4, text="Train & Compare models",
                                    command=self.start_train)
        self.btn_train.pack(fill="x")
        self.btn_cmp = ttk.Button(f4, text="Compare: trials-only vs LOSO",
                                  command=self.start_compare)
        self.btn_cmp.pack(fill="x", pady=(2, 0))
        self.txt_train = tk.Text(f4, width=42, height=10, font=("TkFixedFont", 8),
                                 bg="#0e1116", fg="#d6dbe2", wrap="none")
        self.txt_train.pack(fill="x", pady=(4, 0))
        self.txt_train.insert("1.0", "Collect data, then train.\n")
        self.txt_train.config(state="disabled")
        self.btn_figs = ttk.Button(f4, text="Make poster figures (PNG)",
                                   command=self.start_figures)
        self.btn_figs.pack(fill="x", pady=(4, 0))

        # 4. Apply
        f5 = ttk.LabelFrame(right, text="4. Live monitor", padding=6)
        f5.pack(fill="x", pady=(0, 4))
        mrow = ttk.Frame(f5); mrow.pack(fill="x")
        ttk.Button(mrow, text="Load model file...",
                   command=self.choose_model).pack(side="left")
        ttk.Button(mrow, text="Use trained model", width=16,
                   command=self.use_default_model).pack(side="left", padx=(4, 0))
        self.lbl_model = ttk.Label(f5, text="model: (none loaded)",
                                   foreground="#888", wraplength=260,
                                   justify="left")
        self.lbl_model.pack(anchor="w", pady=(2, 0))
        self.btn_baseline = ttk.Button(
            f5, text="Set neutral baseline (stand neutral)",
            command=self.capture_baseline, state="disabled")
        self.btn_baseline.pack(fill="x", pady=(3, 0))
        self.btn_apply = ttk.Button(f5, text="Apply model to live",
                                    command=self.toggle_monitor)
        self.btn_apply.pack(fill="x", pady=(2, 0))
        self.lbl_monitor = ttk.Label(f5, text="not applied", foreground="#888")
        self.lbl_monitor.pack(anchor="w", pady=(2, 0))
        srow = ttk.Frame(f5); srow.pack(fill="x", pady=(4, 0))
        ttk.Label(srow, text="Slouch sensitivity:").pack(side="left")
        self.lbl_thresh = ttk.Label(srow, text="0.50")
        self.lbl_thresh.pack(side="right")
        ttk.Scale(f5, from_=0.20, to=0.80, variable=self.slouch_thresh,
                  command=lambda v: self.lbl_thresh.config(
                      text=f"{float(v):.2f}")).pack(fill="x")
        ttk.Label(f5, text="(lower = flags slouch more easily)",
                  foreground="#888").pack(anchor="w")

        # --- 5. Phase 2 validation ---
        f6 = ttk.LabelFrame(right, text="5. Validate on new subjects (Phase 2)",
                            padding=6)
        f6.pack(fill="x", pady=(6, 4))
        vrow = ttk.Frame(f6); vrow.pack(fill="x")
        ttk.Label(vrow, text="Held-out folder:").pack(side="left")
        ttk.Entry(vrow, textvariable=self.val_db, width=14).pack(side="left", padx=4)
        self.btn_val = ttk.Button(f6, text="Score frozen model on this folder",
                                  command=self.start_validate)
        self.btn_val.pack(fill="x", pady=(3, 0))
        self.btn_valfig = ttk.Button(f6, text="Make Phase 2 figures (PNG)",
                                     command=self.start_val_figures)
        self.btn_valfig.pack(fill="x", pady=(2, 0))
        ttk.Label(f6, text="no re-training - the saved model is used as-is",
                  foreground="#888").pack(anchor="w")
        self.txt_val = tk.Text(f6, width=42, height=9, font=("TkFixedFont", 8),
                               bg="#0e1116", fg="#d6dbe2", wrap="none")
        self.txt_val.pack(fill="x", pady=(4, 0))
        self.txt_val.insert("1.0",
                            "Phase 1: collect + train -> model is saved.\n"
                            "Phase 2: switch the dataset folder above,\n"
                            "collect NEW subjects (baseline first),\n"
                            "then score the frozen model here.\n")
        self.txt_val.config(state="disabled")

        self._draw_board()

    # ---------------- streaming ----------------
    def _build_source(self):
        if self.mode.get() == "demo":
            return _demo_source(self.fs)
        return BridgeSource(host="127.0.0.1", port=self.port,
                            stop_event=self.stop_event)

    def toggle_stream(self):
        if self.worker and self.worker.is_alive():
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        self.stop_event.clear()
        self.worker = _Worker(self._build_source(), self.q, self.stop_event,
                              paced=(self.mode.get() == "demo"))
        self.worker.start()
        self.lbl_status.config(text="streaming...", foreground="#4ec9b0")
        self.btn_conn.config(text="Disconnect")
        self.btn_tare.config(state="normal")
        self.btn_baseline.config(state="normal")

    def stop_stream(self):
        self.stop_event.set()
        self.worker = None
        self.lbl_status.config(text="disconnected", foreground="#888")
        self.btn_conn.config(text="Connect")
        self.btn_tare.config(state="disabled")
        self.btn_baseline.config(state="disabled")

    def do_tare(self):
        self.tare_capture = []
        self.lbl_status.config(text="taring - stay OFF the board",
                               foreground="#d7ba7d")

    # ---------------- collection ----------------
    def _db(self):
        return self.db_var.get().strip() or "wbb_db"

    def _base_dur(self):
        try:
            d = float(self.base_seconds.get())
            return d if d > 0 else 5.0
        except ValueError:
            return 5.0

    def start_collect(self, label):
        if self.collecting is not None:
            return
        is_base = (label == BASELINE_LABEL)
        dur = self._base_dur() if is_base else self._dur()
        if is_base and self.subject.get() in Dataset(self._db()).baseline_subjects():
            if not messagebox.askyesno(
                    "Baseline",
                    f"Subject {self.subject.get()} already has a baseline.\n"
                    "Record another one? (the average of all of them is used)"):
                return
        if self.mode.get() == "demo":
            self._save_trial(label, _demo_trial(
                "neutral" if is_base else label, dur, self.fs))
            return
        if not (self.worker and self.worker.is_alive()):
            messagebox.showinfo("Collect", "Connect to the board first.")
            return
        self.collecting = {"label": label, "samples": [], "t0": None, "target": dur}
        self.lbl_collect.config(
            text=f"COLLECTING {label.upper()} ..."
                 + (" (stand quietly, neutral)" if is_base else ""),
            foreground=("#f48771" if label == "slouched" else "#4ec9b0"))
        self._set_collect_buttons("disabled")

    def _set_collect_buttons(self, state):
        for b in (self.btn_neutral, self.btn_slouched, self.btn_base):
            b.config(state=state)

    def _finish_collect(self):
        c = self.collecting
        self.collecting = None
        self._set_collect_buttons("normal")
        self.lbl_collect.config(text="")
        samples = resample_uniform(c["samples"], self.fs)
        if len(samples) < 2:
            messagebox.showinfo("Collect", "No data captured.")
            return
        self._save_trial(c["label"], samples)

    def _save_trial(self, label, samples):
        w = SwayWindow()
        for s in samples:
            w.add(s)
        feats = w.features()
        tid = Dataset(self._db()).append(label, feats, subject=self.subject.get(),
                                      cop_samples=samples, fs=self.fs)
        self._refresh_counts()
        self.lbl_collect.config(text=f"saved {label} ({tid})", foreground="#888")

    def _refresh_counts(self):
        # Read the folder WITHOUT creating it: Dataset() makes directories, and
        # this runs while the folder name is still being typed.
        db = self._db()
        path = os.path.join(db, "dataset.csv")
        if not os.path.isfile(path):
            self.lbl_counts.config(
                text=f"dataset: empty - nothing collected yet  (folder: {db})",
                foreground="#d7ba7d")
            return
        c, subs = {}, set()
        try:
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    lab = r.get("label", "")
                    c[lab] = c.get(lab, 0) + 1
                    if lab == BASELINE_LABEL:
                        subs.add(r.get("subject", "?"))
        except Exception:
            self.lbl_counts.config(text=f"dataset: unreadable  (folder: {db})",
                                   foreground="#f48771")
            return
        self.lbl_counts.config(
            text=f"dataset: neutral={c.get('neutral', 0)}  "
                 f"slouched={c.get('slouched', 0)}  "
                 f"baseline={c.get(BASELINE_LABEL, 0)}"
                 + (f" [{', '.join(sorted(subs))}]" if subs else "")
                 + f"  (folder: {db})",
            foreground="#9aa7b4")

    def _on_db_typed(self, *_):
        """Folder name edited -> refresh the counts, debounced so it does not
        hit the disk on every keystroke."""
        if self._db_after is not None:
            try:
                self.after_cancel(self._db_after)
            except Exception:
                pass
        self._db_after = self.after(400, self._refresh_counts)

    # ---------------- training ----------------
    def clear_dataset(self):
        from tkinter import messagebox as mb
        if not mb.askyesno(
                "Clear dataset",
                "Archive the current dataset and start fresh?\n\n"
                "The summary index and ALL raw samples are MOVED to an archive "
                "subfolder (nothing is deleted), and collection starts from zero."):
            return
        adir, n = Dataset(self._db()).clear()
        self._refresh_counts()
        messagebox.showinfo("Clear dataset",
                            f"Archived {n} trials to:\n{adir}\n\nDataset is now empty.")

    def start_train(self):
        try:
            win = float(self.win_seconds.get())
            if win <= 0:
                win = DEFAULT_WINDOW_S
        except ValueError:
            win = DEFAULT_WINDOW_S
        self.btn_train.config(state="disabled")
        self._set_train_text("Training... (cross-validating 3 models)\n")

        def work():
            try:
                sel = self.group_by.get()
                grp = "subject" if sel == "LOSO" else sel
                cv = None if sel == "LOSO" else 5
                res = wbb_train.train_and_compare(
                    self._db(), out=self.model_path,
                    window=win, hop=win / 2.0, cv=cv,
                    normalize=self.use_norm.get(), group_by=grp,
                    baseline_source=self.base_src.get())
                self.train_q.put(("ok", res))
            except Exception as e:
                self.train_q.put(("err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def start_validate(self):
        val = self.val_db.get().strip()
        if not val:
            messagebox.showinfo("Validate", "Enter the held-out dataset folder.")
            return
        if os.path.abspath(val) == os.path.abspath(self._db()):
            messagebox.showinfo(
                "Validate",
                "The held-out folder is the same as the training folder.\n\n"
                "Phase 2 only means something if the model has never seen this "
                "data. Collect the new subjects into a different folder.")
            return
        self.btn_val.config(state="disabled", text="Scoring...")
        import wbb_validate

        def work():
            try:
                r = wbb_validate.evaluate_saved_model(val, self.model_path)
                self.val_q.put(("ok", r))
            except Exception as e:
                self.val_q.put(("err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_validate_done(self, status, payload):
        self.btn_val.config(state="normal",
                            text="Score frozen model on this folder")
        self.txt_val.config(state="normal")
        self.txt_val.delete("1.0", "end")
        self.txt_val.insert("1.0", payload if status == "err"
                            else payload["report"])
        self.txt_val.config(state="disabled")

    def start_val_figures(self):
        val = self.val_db.get().strip()
        if not val:
            messagebox.showinfo("Figures", "Enter the held-out dataset folder.")
            return
        if os.path.abspath(val) == os.path.abspath(self._db()):
            messagebox.showinfo(
                "Figures",
                "The held-out folder is the same as the training folder.")
            return
        self.btn_valfig.config(state="disabled", text="Rendering...")
        import make_figures

        def work():
            try:
                r = make_figures.generate_validation_figures(
                    val, self.model_path, out="figures")
                self.valfig_q.put(("ok", r))
            except Exception as e:
                self.valfig_q.put(("err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_val_figures_done(self, status, payload):
        self.btn_valfig.config(state="normal", text="Make Phase 2 figures (PNG)")
        if status == "err":
            messagebox.showerror("Figures", payload)
            return
        res = payload["result"]
        msg = (f"Saved Phase 2 figures to:\n{payload['out']}\n\n"
               + "\n".join("  " + m for m in payload["made"])
               + f"\n\nheld-out subjects: {res['n_subjects']}   "
                 f"trial accuracy: {res['trial_acc']:.2f}")
        if res.get("cv_accuracy") is not None:
            msg += (f"\nPhase 1 CV ({res.get('cv_group_by')}): "
                    f"{res['cv_accuracy']:.2f}")
        messagebox.showinfo("Figures", msg)

    def start_compare(self):
        try:
            win = float(self.win_seconds.get())
            if win <= 0:
                win = DEFAULT_WINDOW_S
        except ValueError:
            win = DEFAULT_WINDOW_S
        self.btn_cmp.config(state="disabled", text="Comparing...")
        self._set_train_text("Comparing designs...\n"
                             "trials-only, subject k-fold, then LOSO.\n"
                             "LOSO fits one model per subject, so this takes "
                             "a while.\n")
        import wbb_train

        def work():
            try:
                r = wbb_train.compare_designs(
                    self._db(), window=win, hop=win / 2.0,
                    normalize=self.use_norm.get(),
                    baseline_source=self.base_src.get())
                try:
                    import make_figures, os as _os
                    _os.makedirs("figures", exist_ok=True)
                    make_figures.fig_design_comparison(
                        r, _os.path.join("figures", "design_comparison.png"))
                    make_figures.fig_loso_per_subject(
                        r, _os.path.join("figures", "loso_per_subject.png"))
                    r["figures"] = _os.path.abspath("figures")
                except Exception as e:
                    r["figures"] = None
                    print("(comparison figures skipped:", e, ")")
                self.cmp_q.put(("ok", r))
            except Exception as e:
                self.cmp_q.put(("err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_compare_done(self, status, payload):
        self.btn_cmp.config(state="normal", text="Compare: trials-only vs LOSO")
        self._set_train_text(payload if status == "err" else payload["report"])
        if status == "ok":
            msg = ("This comparison did NOT save a model.\n\n"
                   "Use Train & Compare with the grouping you want to report, so "
                   "the saved model matches the number in your paper.")
            if payload.get("figures"):
                msg += (f"\n\nFigures saved to:\n{payload['figures']}\n"
                        "  design_comparison.png\n  loso_per_subject.png")
            messagebox.showinfo("Compare", msg)

    def start_figures(self):
        try:
            win = float(self.win_seconds.get())
            if win <= 0:
                win = DEFAULT_WINDOW_S
        except ValueError:
            win = DEFAULT_WINDOW_S
        self.btn_figs.config(state="disabled", text="Making figures...")
        import make_figures

        def work():
            try:
                r = make_figures.generate_figures(
                    self._db(), window=win, hop=win / 2.0,
                    normalize=self.use_norm.get(), out="figures",
                    group_by=self.group_by.get())
                self.fig_q.put(("ok", r))
            except Exception as e:
                self.fig_q.put(("err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_figures_done(self, status, payload):
        self.btn_figs.config(state="normal", text="Make poster figures (PNG)")
        if status == "err":
            messagebox.showinfo("Figures", "Could not make figures:\n" + payload +
                                "\n\nCollect data and train first.")
            return
        msg = (f"Saved figures to:\n{payload['out']}\n\n"
               f"Grouping: {payload['group']}   best: {payload['best']}   "
               f"accuracy {payload['acc']:.2f}\n"
               f"(files suffixed _{payload['group']}; generalization.png compares both)")
        if payload["acc_trial"] is not None and payload["acc_subject"] is not None:
            msg += (f"\n\ntrial acc {payload['acc_trial']:.2f}  vs  "
                    f"subject acc {payload['acc_subject']:.2f}")
        messagebox.showinfo("Figures", msg)
        try:
            os.startfile(payload["out"])   # open the folder on Windows
        except Exception:
            pass

    def _on_train_done(self, status, payload):
        self.btn_train.config(state="normal")
        if status == "err":
            self._set_train_text("Training failed:\n" + payload +
                                 "\n\nCollect more trials of each posture.")
            return
        self._set_train_text(payload["report"])
        # NOTE: do NOT auto-apply. The model is saved; user presses Apply to use it.
        self.lbl_monitor.config(
            text="model saved - press 'Apply model to live' to use it",
            foreground="#d7ba7d")

    def _set_train_text(self, text):
        self.txt_train.config(state="normal")
        self.txt_train.delete("1.0", "end")
        self.txt_train.insert("1.0", text)
        self.txt_train.config(state="disabled")

    # ---------------- apply / monitor ----------------
    def capture_baseline(self):
        if not (self.worker and self.worker.is_alive()):
            messagebox.showinfo("Baseline", "Connect to the board first.")
            return
        if self.baselining is not None:
            return
        # match the baseline length to the trained model's window, if available
        if os.path.exists(self.model_path):
            try:
                self.classifier = PostureClassifier.load(self.model_path)
                if self.classifier.window_s:
                    self.clf_window_s = float(self.classifier.window_s)
                    self.buf_seconds = max(ROLLING_SECONDS, self.clf_window_s)
            except Exception:
                pass
        self.baselining = {"samples": [], "t0": None, "target": self.clf_window_s}
        self.lbl_monitor.config(text="capturing neutral baseline - stand neutral...",
                                foreground="#d7ba7d")

    def _finish_baseline(self):
        b = self.baselining
        self.baselining = None
        samples = resample_uniform(b["samples"], self.fs)
        if len(samples) < 2:
            self.lbl_monitor.config(text="baseline failed (no data)", foreground="#f48771")
            self._apply_after_baseline = False
            return
        w = SwayWindow()
        for s in samples:
            w.add(s)
        f = w.features()
        names = self.classifier.feature_names if self.classifier else None
        if names is None:
            from wbb_dataset import FEATURE_NAMES as names
        self.live_baseline = [float(getattr(f, n)) for n in names]
        self.lbl_monitor.config(text="neutral baseline set", foreground="#4ec9b0")
        # if the user pressed Apply on a normalized model, start monitoring now
        if getattr(self, "_apply_after_baseline", False):
            self._apply_after_baseline = False
            self._activate_monitor()

    def _activate_monitor(self):
        self.monitor_active = True
        self.btn_apply.config(text="Stop monitoring")
        self.lbl_monitor.config(text="applied - monitoring live", foreground="#4ec9b0")
        self.lbl_posture.config(text="posture: warming up...", foreground="#d7ba7d")

    def choose_model(self):
        """Pick any .joblib model - one this app trained, or one trained
        elsewhere. Missing metadata is asked for rather than assumed, because
        guessing it wrong produces confident nonsense in the live view."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select a trained model",
            filetypes=[("Model files", "*.joblib *.pkl"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(self.model_path)) or ".")
        if not path:
            return
        self._load_model_file(path, ask_missing=True)

    def use_default_model(self):
        """Go back to the model this app trains and saves."""
        if not os.path.exists(self.model_path):
            messagebox.showinfo("Model", "No model has been trained yet. "
                                         "Train one in section 3 first.")
            return
        self._load_model_file(self.model_path, ask_missing=False)

    def _load_model_file(self, path, ask_missing):
        try:
            clf = PostureClassifier.load(path)
        except Exception as e:
            messagebox.showerror("Model", f"Could not load:\n{path}\n\n{e}")
            return

        # A model saved outside this app carries no window length and no record
        # of whether it was trained on baseline-normalized features. Both change
        # the numbers fed to it, so ask instead of defaulting.
        if ask_missing and clf.window_s is None:
            win = simpledialog.askfloat(
                "Model settings",
                "This model does not record the window length it was trained "
                "on.\n\nHow many seconds of data should each prediction use?",
                initialvalue=float(self.win_seconds.get() or DEFAULT_WINDOW_S),
                minvalue=1.0, maxvalue=120.0, parent=self)
            if win is None:
                return
            clf.window_s = float(win)
            clf.normalized = messagebox.askyesno(
                "Model settings",
                "Was it trained on per-subject baseline-normalized features?\n\n"
                "Yes  - each person's neutral baseline is subtracted first\n"
                "No   - raw feature values are used\n\n"
                "Getting this wrong makes the live output meaningless.",
                parent=self)

        self.classifier = clf
        self.model_loaded_from = path
        self.alarm = AlarmController(positive=clf.positive)
        if clf.window_s:
            self.clf_window_s = float(clf.window_s)
        self.buf_seconds = max(ROLLING_SECONDS, self.clf_window_s)
        # a different model means a different baseline scale
        self.live_baseline = None
        self.btn_baseline.config(state="normal")
        self.lbl_model.config(
            text=f"model: {os.path.basename(path)}\n{clf.describe()}",
            foreground="#9aa7b4")
        if clf.positive not in ("slouched",):
            messagebox.showinfo(
                "Model",
                f"This model's classes are: {', '.join(clf.labels)}\n\n"
                f"'{clf.positive}' is being treated as the posture to flag. "
                "If that is the wrong one, re-save the model with "
                "positive='<label>'.")
        if self.monitor_active:      # swap under a running monitor
            self.monitor_active = False
            self.btn_apply.config(text="Apply model to live")
            self.lbl_monitor.config(text="model changed - apply again",
                                    foreground="#d7ba7d")

    def toggle_monitor(self):
        if self.monitor_active:
            self.monitor_active = False
            self._set_alarm(False)
            self.btn_apply.config(text="Apply model to live")
            self.lbl_monitor.config(text="not applied", foreground="#888")
            self.lbl_posture.config(text="posture: (model not applied)",
                                    foreground="#888")
            return
        if self.classifier is None:
            # nothing chosen yet: fall back to the model this app trains
            if not os.path.exists(self.model_path):
                messagebox.showinfo(
                    "Apply",
                    "No model loaded.\n\nTrain one in section 3, or use "
                    "'Load model file...' to import one.")
                return
            self._load_model_file(self.model_path, ask_missing=False)
            if self.classifier is None:
                return
        if not (self.worker and self.worker.is_alive()):
            messagebox.showinfo(
                "Apply", "Connect first (section 1), then Apply so live data flows.")
            return
        if self.classifier.normalized and self.live_baseline is None:
            # one-click: capture the neutral baseline now, then start monitoring
            self._apply_after_baseline = True
            self.capture_baseline()
            messagebox.showinfo(
                "Apply",
                "Capturing your neutral baseline now.\nStand neutrally for "
                f"~{int(self.clf_window_s)}s; monitoring starts automatically.")
            return
        self._activate_monitor()

    # ---------------- poll / ingest ----------------
    def _poll(self):
        try:
            self._poll_body()
        except Exception as exc:
            # never let one bad tick kill the update loop
            print("poll error:", repr(exc))
        finally:
            self.after(POLL_MS, self._poll)

    def _poll_body(self):
        got = None
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple) and item and item[0] == "__error__":
                self.lbl_status.config(text="error: " + item[1], foreground="#f48771")
                self.stop_stream()
                continue
            got = item
            self._ingest(got)
        if got is not None:
            self._draw_dynamic()
            self._update_panel()
        if self.collecting and self.collecting["t0"] is not None and self.last_sample:
            remain = self.collecting["target"] - (self.last_sample.t - self.collecting["t0"])
            self.lbl_collect.config(
                text=f"COLLECTING {self.collecting['label'].upper()} - "
                     f"{max(0, remain):4.1f}s left")
        if self.baselining and self.baselining["t0"] is not None and self.last_sample:
            remain = self.baselining["target"] - (self.last_sample.t - self.baselining["t0"])
            self.lbl_monitor.config(
                text=f"baseline capture - {max(0, remain):4.1f}s left",
                foreground="#d7ba7d")
        try:
            status, payload = self.train_q.get_nowait()
            self._on_train_done(status, payload)
        except queue.Empty:
            pass
        try:
            status, payload = self.fig_q.get_nowait()
            self._on_figures_done(status, payload)
        except queue.Empty:
            pass
        try:
            status, payload = self.val_q.get_nowait()
            self._on_validate_done(status, payload)
        except queue.Empty:
            pass
        try:
            status, payload = self.valfig_q.get_nowait()
            self._on_val_figures_done(status, payload)
        except queue.Empty:
            pass
        try:
            status, payload = self.cmp_q.get_nowait()
            self._on_compare_done(status, payload)
        except queue.Empty:
            pass

    def _ingest(self, s):
        if self.tare_capture is not None:
            self.tare_capture.append(s)
            if len(self.tare_capture) >= int(self.fs * 1.5):
                self.tare = Tare.from_samples(self.tare_capture)
                self.tare_capture = None
                self.lbl_status.config(text="streaming (tared)", foreground="#4ec9b0")
        if self.tare is not None:
            s = self.tare.apply(s)
        self.last_sample = s
        self.rolling.append(s)
        while self.rolling and (s.t - self.rolling[0].t) > self.buf_seconds:
            self.rolling.popleft()
        if self.collecting is not None:
            if self.collecting["t0"] is None:
                self.collecting["t0"] = s.t
            self.collecting["samples"].append(s)
            if (s.t - self.collecting["t0"]) >= self.collecting["target"]:
                self._finish_collect()
        if self.baselining is not None:
            if self.baselining["t0"] is None:
                self.baselining["t0"] = s.t
            self.baselining["samples"].append(s)
            if (s.t - self.baselining["t0"]) >= self.baselining["target"]:
                self._finish_baseline()

    def _recent(self, seconds):
        if not self.rolling:
            return []
        t_now = self.rolling[-1].t
        return [p for p in self.rolling if (t_now - p.t) <= seconds]

    # ---------------- drawing ----------------
    def _board_geom(self):
        w = self.canvas.winfo_width() or 470
        h = self.canvas.winfo_height() or 360
        m = 24
        aspect = BOARD_X_CM / BOARD_Y_CM
        bw = w - 2 * m
        bh = bw / aspect
        if bh > h - 2 * m:
            bh = h - 2 * m
            bw = bh * aspect
        return w / 2, h / 2, bw, bh

    def _draw_board(self):
        self.canvas.delete("board")
        cx, cy, bw, bh = self._board_geom()
        x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#3a4250", width=2,
                                     tags="board")
        self.canvas.create_line(cx, y0, cx, y1, fill="#262c36", tags="board")
        self.canvas.create_line(x0, cy, x1, cy, fill="#262c36", tags="board")
        for lbl, ox, oy in (("R", x1 - 10, cy), ("L", x0 + 10, cy),
                            ("Ant", cx, y0 + 10), ("Post", cx, y1 - 10)):
            self.canvas.create_text(ox, oy, text=lbl, fill="#55606e",
                                    font=("TkDefaultFont", 8), tags="board")

    def _draw_dynamic(self):
        self.canvas.delete("dyn")
        cx, cy, bw, bh = self._board_geom()
        if self.last_sample is not None:
            trail = self._recent(TRACE_SECONDS)
            if len(trail) >= 2:
                pts = []
                for p in trail:
                    px, py = board_cm_to_canvas(p.cop_x, p.cop_y, bw, bh, cx, cy)
                    pts += [px, py]
                self.canvas.create_line(*pts, fill="#2d6cdf", width=1, tags="dyn")
        disp = self._recent(ROLLING_SECONDS)
        if len(disp) >= 8:
            xs = [p.cop_x for p in disp]; ys = [p.cop_y for p in disp]
            ell = confidence_ellipse_points(xs, ys, n=48)
            if ell:
                flat = []
                for ex, ey in ell:
                    px, py = board_cm_to_canvas(ex, ey, bw, bh, cx, cy)
                    flat += [px, py]
                self.canvas.create_polygon(*flat, outline="#e0a64e", fill="",
                                           width=1, tags="dyn")
        if self.last_sample is not None:
            px, py = board_cm_to_canvas(self.last_sample.cop_x,
                                        self.last_sample.cop_y, bw, bh, cx, cy)
            self.canvas.create_oval(px - 6, py - 6, px + 6, py + 6,
                                    fill="#4ec9b0", outline="", tags="dyn")

    def _bar(self, canvas, left_pct, lcol, rcol):
        canvas.delete("all")
        w = int(canvas["width"]); h = int(canvas["height"])
        split = int(w * left_pct / 100.0)
        canvas.create_rectangle(0, 0, split, h, fill=lcol, outline="")
        canvas.create_rectangle(split, 0, w, h, fill=rcol, outline="")
        canvas.create_text(w / 2, h / 2, text=f"{left_pct:.0f} / {100 - left_pct:.0f}",
                           fill="#0e1116", font=("TkDefaultFont", 8, "bold"))

    def _update_panel(self):
        s = self.last_sample
        if s is None:
            return
        self.lbl_mass.config(text=f"{s.total:.1f} kg")
        self.lbl_cop.config(text=f"CoP  x={s.cop_x:+.2f}  y={s.cop_y:+.2f} cm")
        l, _ = s.left_right_pct
        a, _ = s.ant_post_pct
        self._bar(self.bar_lr, l, "#5aa0ff", "#ff8c5a")
        self._bar(self.bar_ap, a, "#7fd17f", "#c98fe0")
        disp = self._recent(ROLLING_SECONDS)
        if len(disp) >= 2:
            w = SwayWindow()
            for p in disp:
                w.add(p)
            f = w.features()
            if f:
                self.lbl_metrics.config(
                    text=f"path {f.path_length_cm:.1f}cm · "
                         f"vel {f.mean_velocity_cm_s:.1f}cm/s · "
                         f"ellipse {f.ellipse_area_cm2:.1f}cm2")
        # predict ONLY when the user has applied the model
        if self.monitor_active and self.classifier is not None:
            cw = self._recent(self.clf_window_s)
            # The model was trained on full windows, and features like path length
            # scale with duration, so a half-filled window would be scored against
            # the wrong distribution. Wait until the window is essentially full
            # (same 90% tolerance the training loader uses).
            span = (cw[-1].t - cw[0].t) if len(cw) >= 2 else 0.0
            if len(cw) >= 8 and span >= 0.9 * self.clf_window_s:
                cw = resample_uniform(cw, self.fs)   # match the training pipeline
                w2 = SwayWindow()
                for p in cw:
                    w2.add(p)
                cf = w2.features()
                if cf:
                    self._predict_and_alarm(cf, s.t)
            else:
                self.lbl_posture.config(
                    text=f"posture: filling window... "
                         f"{span:.0f}/{self.clf_window_s:.0f}s",
                    foreground="#d7ba7d")
                self.lbl_why.config(text="")

    def _predict_and_alarm(self, features, t):
        try:
            label, proba = self.classifier.predict(features, baseline=self.live_baseline)
        except Exception as exc:
            self.lbl_posture.config(text="posture: error - " + str(exc)[:40],
                                    foreground="#f48771")
            return
        # sensitivity: for a binary model, flag slouch when p >= threshold
        if proba is not None and len(self.classifier.labels) == 2:
            others = [l for l in self.classifier.labels
                      if l != self.classifier.positive]
            label = self.classifier.positive if proba >= self.slouch_thresh.get() \
                else (others[0] if others else label)
        pos = (label == self.classifier.positive)
        txt = f"posture: {label}" + (f"  (p={proba:.2f})" if proba is not None else "")
        self.lbl_posture.config(text=txt, foreground=("#f48771" if pos else "#4ec9b0"))
        # explainability: which features push this decision right now
        try:
            ex = self.classifier.explain(features, baseline=self.live_baseline)
        except Exception:
            ex = None
        if ex:
            parts = [f"{n} {'->slouch' if c > 0 else '->neutral'}" for n, c in ex]
            self.lbl_why.config(text="why: " + ",  ".join(parts))
        else:
            self.lbl_why.config(text="")
        ev = self.alarm.update(t, label)
        if ev == "ALARM":
            self._set_alarm(True)
        elif ev == "CLEAR":
            self._set_alarm(False)

    def _set_alarm(self, on):
        self.canvas.delete("alarm")
        if on:
            w = self.canvas.winfo_width() or 470
            h = self.canvas.winfo_height() or 360
            self.canvas.create_rectangle(2, 2, w - 2, h - 2, outline="#f44747",
                                         width=4, tags="alarm")
            try:
                import winsound
                winsound.Beep(880, 250)
            except Exception:
                self.bell()


def _check_consistency():
    """Detect a partial file update (mixed old/new files), which is the usual
    cause of live/figure breakage. Returns a list of outdated file names."""
    import inspect
    bad = []
    try:
        from wbb_monitor import PostureClassifier
        if not hasattr(PostureClassifier, "explain"):
            bad.append("wbb_monitor.py")
    except Exception:
        bad.append("wbb_monitor.py")
    try:
        import wbb_train
        if "group_by" not in inspect.signature(wbb_train.train_and_compare).parameters:
            bad.append("wbb_train.py")
    except Exception:
        bad.append("wbb_train.py")
    try:
        import make_figures
        if "group_by" not in inspect.signature(make_figures.generate_figures).parameters:
            bad.append("make_figures.py")
    except Exception:
        bad.append("make_figures.py")
    try:
        from wbb_dataset import Dataset
        if not hasattr(Dataset, "load_windowed_normalized"):
            bad.append("wbb_dataset.py")
    except Exception:
        bad.append("wbb_dataset.py")
    try:
        import wbb_validate
        if not hasattr(wbb_validate, "evaluate_saved_model"):
            bad.append("wbb_validate.py")
    except Exception:
        bad.append("wbb_validate.py")
    return bad


def main():
    ap = argparse.ArgumentParser(description="WELAB Wii Balance Board app")
    ap.add_argument("--demo", action="store_true", help="start in Demo mode")
    ap.add_argument("--port", type=int, default=8674)
    ap.add_argument("--fs", type=float, default=100.0)
    ap.add_argument("--db", default="wbb_db")
    ap.add_argument("--model", default="posture_model.joblib")
    args = ap.parse_args()

    root = tk.Tk()
    root.title("WELAB - Wii Balance Board Posture")
    root.geometry("920x700")
    bad = _check_consistency()
    if bad:
        messagebox.showwarning(
            "Outdated files",
            "Some files are older than the rest, which breaks the live view and "
            "figures.\n\nReplace ALL files in your folder with the ones from "
            "WELAB_WBB.zip (not just one), then reopen.\n\nOutdated: "
            + ", ".join(bad))
    app = WBBApp(root, port=args.port, fs=args.fs, db=args.db,
                 model_path=args.model, demo_default=args.demo)
    app.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
