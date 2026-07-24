"""
test_wbb_gui_static.py - static checks on the GUI.

tkinter is not importable here (and a GUI cannot be clicked in CI), so this
parses wbb_gui.py instead. It exists because a typo like `self.window_s` when the
variable is really `self.win_seconds` raises inside a button callback, which
tkinter swallows: the button simply does nothing and no error is shown. Compiling
does not catch it either. Every attribute a callback touches is verified here.

    python3 tests/test_wbb_gui_static.py
"""
import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
GUI = os.path.join(os.path.dirname(__file__), "..", "wbb_gui.py")

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {extra}")


src = open(GUI).read()
tree = ast.parse(src)
cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WBBApp"][0]
module_fns = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}

methods = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
assigned = {n.attr for n in ast.walk(cls)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "self" and isinstance(n.ctx, ast.Store)}

# names that come from ttk.Frame / Tk, not from this class
TK_INHERITED = {
    "after", "after_cancel", "bell", "columnconfigure", "rowconfigure", "pack",
    "grid", "bind", "bind_all", "winfo_reqwidth", "winfo_width", "winfo_height",
    "config", "configure", "update_idletasks", "master", "tk", "children",
    "destroy", "focus_set",
}

print("every self.X the GUI touches must exist")
undefined_reads = sorted(
    n.attr for n in ast.walk(cls)
    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    and n.value.id == "self" and isinstance(n.ctx, ast.Load)
    and n.attr not in methods and n.attr not in assigned
    and n.attr not in TK_INHERITED)
check("no undefined attribute reads", not undefined_reads, undefined_reads)

called = sorted(
    n.func.attr for n in ast.walk(cls)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"
    and n.func.attr not in methods and n.func.attr not in assigned
    and n.func.attr not in TK_INHERITED)
check("no calls to missing methods", not called, called)

print("every button command resolves to a real method")
# ttk.Button(..., command=self.foo) / command=lambda: self.foo(...)
cmd_targets = []
for n in ast.walk(cls):
    if not isinstance(n, ast.Call):
        continue
    for kw in n.keywords:
        if kw.arg != "command":
            continue
        f = kw.value
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                and f.value.id == "self":
            cmd_targets.append(f.attr)
        elif isinstance(f, ast.Lambda):
            for c in ast.walk(f):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) \
                        and isinstance(c.func.value, ast.Name) \
                        and c.func.value.id == "self":
                    cmd_targets.append(c.func.attr)
check("found the buttons", len(cmd_targets) >= 8, len(cmd_targets))
missing_cmds = sorted(set(c for c in cmd_targets if c not in methods))
check("no button points at a missing method", not missing_cmds, missing_cmds)

print("worker queues are drained by the poll loop")
queues = sorted(a for a in assigned if a.endswith("_q"))
polled = [n.func.value.attr for n in ast.walk(cls)
          if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
          and n.func.attr == "get_nowait"
          and isinstance(n.func.value, ast.Attribute)]
check("every worker queue is polled",
      all(q in polled for q in queues), (queues, polled))

print("the update loop cannot die")
poll = [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "_poll"][0]
has_try = any(isinstance(n, ast.Try) for n in poll.body)
reschedules = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "after" for n in ast.walk(poll))
check("_poll wraps its body in try", has_try)
check("_poll always reschedules itself", reschedules)
finallys = [n for n in ast.walk(poll) if isinstance(n, ast.Try) and n.finalbody]
check("_poll reschedules from a finally block", bool(finallys))

print("startup consistency check covers every module")
check("_check_consistency exists", "_check_consistency" in module_fns)
for mod in ("wbb_monitor.py", "wbb_train.py", "make_figures.py",
            "wbb_dataset.py", "wbb_validate.py"):
    check(f"  checks {mod}", f'"{mod}"' in src)

print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
