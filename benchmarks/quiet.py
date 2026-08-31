"""protocol:gpu-bench v3 quiet gate + the 2026-08-31 addendum.

cputime-diff over >=2s DECIDES; loadavg is advisory and printed for the record.

Summing ALL processes and differencing the totals is WRONG and reads NEGATIVE:
a process that exits between samples takes its whole lifetime cputime out of
the second total, which can mask real load. So the delta is computed over the
pids present in BOTH samples, plus the full cputime of pids that APPEARED
during the window (work that genuinely happened inside it). Vanished pids are
counted as unknown and reported, never silently dropped.
"""
import subprocess, sys, time

def snap():
    out = subprocess.run(["ps", "-eo", "pid,time,comm"], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines()[1:]:
        f = line.split(None, 2)
        if len(f) < 3:
            continue
        p = f[1].split(":")
        try:
            s = int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2]) if len(p) == 3 \
                else int(p[0]) * 60 + float(p[1])
        except ValueError:
            continue
        d[f[0]] = (s, f[2])
    return d

vis = subprocess.run(["ps", "-eo", "pid,user,comm"], capture_output=True, text=True).stdout
if "windowserver" not in vis.lower():
    print("BLINDED: cannot see WindowServer — this check does NOT count as quiet")
    sys.exit(2)
nproc = len(vis.splitlines()) - 1

me = {str(subprocess.os.getpid())}
a = snap(); t0 = time.time()
time.sleep(3.0)
b = snap(); t1 = time.time()
el = t1 - t0

busy, gone, new = 0.0, 0, 0.0
top = []
for pid, (s2, comm) in b.items():
    if pid in me:
        continue
    if pid in a:
        d = s2 - a[pid][0]
        if d > 0:
            busy += d
            top.append((d, comm))
    else:
        new += s2
        top.append((s2, comm + " (new)"))
gone = len(set(a) - set(b))
pct = 100.0 * (busy + new) / el
load = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip()
top.sort(reverse=True)
print(f"visible={nproc} procs (WindowServer seen) | foreign busy-CPU {pct:.1f}% over {el:.1f}s "
      f"| {gone} pids exited | loadavg {load} (advisory)")
print("  top: " + ", ".join(f"{c.split('/')[-1]}={d/el*100:.0f}%" for d, c in top[:4]) if top else "  top: none")
print(f"{'QUIET' if pct <= 150.0 else 'NOT QUIET'}: foreign CPU {pct:.1f}% vs 150% gate")
sys.exit(0 if pct <= 150.0 else 1)
