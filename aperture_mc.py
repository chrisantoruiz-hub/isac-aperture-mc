#!/usr/bin/env python3
"""
ISAC beam phase-space and gas-cell aperture transmission Monte Carlo.

Run:
    python3 aperture_mc.py            # then open http://localhost:8765
    python3 aperture_mc.py --port 9000

Model
-----
1. Beam: species (A, q) and energy [MeV/u] give beta*gamma; the geometric rms
   emittance is eps = eps_n / (beta*gamma). Particles are sampled as a Gaussian
   in (x, x'), (y, y') from Twiss parameters at the waist (z = 0, gas-cell centre),
   and in (dt, dW) from the longitudinal emittance [keV/u*ns].
2. Transport: each ray is back-tracked in free space (no apertures) to a start
   plane upstream, then propagated downstream as a ray (x, y, z) + (dx, dy, dz).
   It is transmitted only if it passes both the entrance aperture (default
   Ø6 mm at z = -105 mm) and the downstream aperture (default Ø8 mm at +105 mm).
   T = N_transmitted / N_total, with binomial error sqrt(T(1-T)/N).

For a single aperture and a round beam the exact answer is shown for comparison:
    centred:  T = 1 - exp(-R^2 / (2 sigma^2))
    offset d: T = P(chi'^2_2(lambda = d^2/sigma^2) <= R^2/sigma^2)   (Rice / Marcum-Q)
"""

import argparse
import json
import math
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

try:
    from scipy.stats import ncx2
except ImportError:  # analytic comparison for offset beams needs scipy
    ncx2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_PARTICLES = 20_000_000
MAX_PLOT_POINTS = 4000


# --------------------------------------------------------------------------- physics
def simulate(n, sigma_x, sigma_y, x0, y0, diameter, seed=None):
    """Return (transmission, std_error, x, y, mask) for one Monte Carlo run."""
    rng = np.random.default_rng(seed)
    r2 = (diameter / 2.0) ** 2
    n_pass = 0
    # chunk to keep memory bounded for very large N
    chunk = 2_000_000
    x_keep = y_keep = m_keep = None
    done = 0
    while done < n:
        m = min(chunk, n - done)
        x = rng.normal(x0, sigma_x, m)
        y = rng.normal(y0, sigma_y, m)
        mask = x * x + y * y <= r2
        n_pass += int(mask.sum())
        if x_keep is None:  # keep a sample from the first chunk for plotting
            k = min(MAX_PLOT_POINTS, m)
            x_keep, y_keep, m_keep = x[:k], y[:k], mask[:k]
        done += m
    t = n_pass / n
    err = math.sqrt(max(t * (1 - t), 0.0) / n)
    return t, err, n_pass, x_keep, y_keep, m_keep


def analytic(sigma_x, sigma_y, x0, y0, diameter):
    """Exact transmission for a round beam (sigma_x == sigma_y); None otherwise."""
    if not math.isclose(sigma_x, sigma_y, rel_tol=1e-12):
        return None
    s, R = sigma_x, diameter / 2.0
    d2 = x0 * x0 + y0 * y0
    if d2 == 0:
        return 1.0 - math.exp(-R * R / (2 * s * s))
    if ncx2 is None:
        return None
    return float(ncx2.cdf(R * R / (s * s), df=2, nc=d2 / (s * s)))


# --------------------------------------------------------------------------- beam / emittance
AMU_MEV = 931.49410242      # atomic mass unit, MeV/c^2
C_MM_PER_NS = 299.792458    # speed of light, mm/ns


def kinematics(w_per_u, A, q):
    """Relativistic factors and rigidity for kinetic energy w_per_u [MeV/u]."""
    gamma = 1.0 + w_per_u / AMU_MEV
    beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
    bg = beta * gamma
    p_total_gev = A * AMU_MEV * bg / 1000.0
    return dict(beta=beta, gamma=gamma, betagamma=bg,
                brho=3.335640952 * p_total_gev / q, a_over_q=A / q)


def twiss_sample(rng, n, eps, alpha, beta):
    """Sample (u, u') from a Gaussian with rms emittance eps and Twiss alpha, beta."""
    u1, u2 = rng.standard_normal(n), rng.standard_normal(n)
    u = math.sqrt(eps * beta) * u1
    up = math.sqrt(eps / beta) * (-alpha * u1 + u2)
    return u, up


def rms_stats(u, up, eps, alpha, beta):
    """MC-measured rms emittance / Twiss, and fraction inside the 2-rms ellipse."""
    uu, pp, up_ = np.var(u), np.var(up), np.mean((u - u.mean()) * (up - up.mean()))
    e = math.sqrt(max(uu * pp - up_ * up_, 0.0))
    g = (1 + alpha * alpha) / beta
    inside = (g * u * u + 2 * alpha * u * up + beta * up * up) <= 4 * eps
    return dict(eps=e, beta=uu / e if e else None, alpha=-up_ / e if e else None,
                sigma=math.sqrt(uu), sigma_p=math.sqrt(pp), frac_2rms=float(inside.mean()))


def beam(p):
    """Build the 6D-ish beam (x-x', y-y', dt-dW) from species, energy, emittances, Twiss."""
    A, q = float(p.get("A", 8)), float(p.get("q", 2))
    w = float(p.get("energy", 1.5))                  # MeV/u
    if A <= 0 or q <= 0 or w <= 0:
        raise ValueError("A, q and energy must be positive")
    # quoted emittances may be rms or 4*rms (2-rms ellipse area / pi)
    div = 4.0 if p.get("emit_def") == "4rms" else 1.0
    eps_n = float(p.get("eps_n", 0.2)) / div        # rms normalised, mm*mrad (pi omitted)
    eps_l = float(p.get("eps_l", 1.5)) / div        # rms longitudinal, keV/u*ns
    k = kinematics(w, A, q)
    eps_x = eps_n / k["betagamma"]                   # rms geometric, mm*mrad
    tw = {key: float(p.get(key, d)) for key, d in
          dict(ax=0.0, bx=1.0, ay=0.0, by=1.0, az=0.0, bz=0.2).items()}
    if min(tw["bx"], tw["by"], tw["bz"]) <= 0:
        raise ValueError("Twiss beta must be positive")
    n = int(min(max(int(p.get("n_beam", 50_000)), 100), 2_000_000))
    seed = None if p.get("seed") in (None, "") else int(p["seed"])
    rng = np.random.default_rng(seed)

    x, xp = twiss_sample(rng, n, eps_x, tw["ax"], tw["bx"])   # mm, mrad (beta in m == mm/mrad)
    y, yp = twiss_sample(rng, n, eps_x, tw["ay"], tw["by"])
    dt, dw = twiss_sample(rng, n, eps_l, tw["az"], tw["bz"])  # ns, keV/u

    sx = rms_stats(x, xp, eps_x, tw["ax"], tw["bx"])
    sy = rms_stats(y, yp, eps_x, tw["ay"], tw["by"])
    sz = rms_stats(dt, dw, eps_l, tw["az"], tw["bz"])

    f_rf = float(p.get("f_rf", 106.08))              # MHz, ISAC DTL
    sig_t, sig_w = sz["sigma"], sz["sigma_p"]
    dw_w = sig_w / (w * 1000.0)
    long = dict(
        sigma_t_ns=sig_t, sigma_w_kev_u=sig_w,
        sigma_z_mm=k["beta"] * C_MM_PER_NS * sig_t,
        sigma_phi_deg=360.0 * f_rf * 1e-3 * sig_t,
        dw_over_w=dw_w, dp_over_p=k["gamma"] / (k["gamma"] + 1.0) * dw_w,
        f_rf=f_rf,
    )
    m = min(MAX_PLOT_POINTS, n)
    r = lambda a: np.round(a[:m], 5).tolist()
    return dict(
        kin=k, n=n,
        eps=dict(n_rms=eps_n, x_rms=eps_x, l_rms=eps_l,
                 x_2rms=4 * eps_x, l_2rms=4 * eps_l, n_2rms=4 * eps_n),
        twiss=tw, stats=dict(x=sx, y=sy, z=sz), long=long,
        points=dict(x=r(x), xp=r(xp), y=r(y), yp=r(yp), dt=r(dt), dw=r(dw)),
    )


# --------------------------------------------------------------------------- ray transport
def beam_setup(p):
    """Kinematics, rms geometric emittance [mm*mrad] and Twiss at the waist plane z = 0."""
    A, q, w = float(p.get("A", 8)), float(p.get("q", 2)), float(p.get("energy", 1.5))
    if A <= 0 or q <= 0 or w <= 0:
        raise ValueError("A, q and energy must be positive")
    div = 4.0 if p.get("emit_def") == "4rms" else 1.0
    k = kinematics(w, A, q)
    eps = float(p.get("eps_n", 0.2)) / div / k["betagamma"]
    tw = {key: float(p.get(key, d)) for key, d in dict(ax=0.0, bx=1.0, ay=0.0, by=1.0).items()}
    if min(tw["bx"], tw["by"]) <= 0:
        raise ValueError("Twiss beta must be positive")
    return k, eps, tw


def sigma_at(eps, alpha, beta, z_mm):
    """rms size [mm] at distance z [mm] from the reference plane after a field-free drift."""
    z = z_mm / 1000.0                       # beta is in m
    g = (1 + alpha * alpha) / beta
    return math.sqrt(eps * (beta - 2 * alpha * z + g * z * z))


def _geom(p):
    g = dict(
        n=int(min(max(int(p.get("n", 200_000)), 1), MAX_PARTICLES)),
        z_start=-abs(float(p.get("L_back", 150.0))),     # mm, start plane upstream of waist
        z_ap=float(p.get("z_ap", -105.0)),               # mm, entrance aperture plane
        d_ap=float(p.get("diameter", 6.0)),              # mm
        exit_on=bool(p.get("exit_on", True)),               # downstream aperture
        z_exit=float(p.get("z_exit", 105.0)),
        d_exit=float(p.get("d_exit", 8.0)),
        x0=float(p.get("x0", 0.0)), y0=float(p.get("y0", 0.0)),        # centroid at waist, mm
        xp0=float(p.get("xp0", 0.0)), yp0=float(p.get("yp0", 0.0)),    # centroid angle, mrad
        seed=None if p.get("seed") in (None, "") else int(p["seed"]),
    )
    if g["d_ap"] <= 0 or (g["exit_on"] and g["d_exit"] <= 0):
        raise ValueError("aperture diameters must be positive")
    if g["z_ap"] < g["z_start"]:
        raise ValueError("entrance aperture is upstream of the start plane; increase the back-track distance")
    return g


def _trace(rng, n, eps, tw, g):
    """
    1) sample the beam at the waist (z = 0) from the Twiss parameters,
    2) back-track each ray in free space (no apertures) to z_start,
    3) build rays (x, y, z_start) + unit direction (dx, dy, dz) and propagate downstream,
       testing each aperture plane.
    """
    x_w, xp = twiss_sample(rng, n, eps, tw["ax"], tw["bx"])
    y_w, yp = twiss_sample(rng, n, eps, tw["ay"], tw["by"])
    x_w += g["x0"]; y_w += g["y0"]
    xp = (xp + g["xp0"]) * 1e-3; yp = (yp + g["yp0"]) * 1e-3          # slopes dx/dz (rad)
    norm = 1.0 / np.sqrt(1.0 + xp * xp + yp * yp)
    dx, dy, dz = xp * norm, yp * norm, norm                            # direction cosines

    def at(z, x0, y0, z0):           # propagate a ray from (x0,y0,z0) to plane z
        t = (z - z0) / dz
        return x0 + dx * t, y0 + dy * t

    xs, ys = at(g["z_start"], x_w, y_w, 0.0)                           # back-track (no aperture)
    xa, ya = at(g["z_ap"], xs, ys, g["z_start"])                       # forward to entrance
    ok = xa * xa + ya * ya <= (g["d_ap"] / 2) ** 2
    ok_entr = ok.copy()
    if g["exit_on"]:
        xe, ye = at(g["z_exit"], xs, ys, g["z_start"])
        ok &= xe * xe + ye * ye <= (g["d_exit"] / 2) ** 2
    return dict(xs=xs, ys=ys, dx=dx, dy=dy, dz=dz, xa=xa, ya=ya, ok=ok, ok_entr=ok_entr, xpm=xp * 1e3, ypm=yp * 1e3)


def _analytic_entrance(eps, tw, g):
    sx = sigma_at(eps, tw["ax"], tw["bx"], g["z_ap"])
    sy = sigma_at(eps, tw["ay"], tw["by"], g["z_ap"])
    cx = g["x0"] + g["xp0"] * 1e-3 * g["z_ap"]
    cy = g["y0"] + g["yp0"] * 1e-3 * g["z_ap"]
    return sx, sy, cx, cy, analytic(sx, sy, cx, cy, g["d_ap"])


def _analytic_exit(eps, tw, g):
    """Exact transmission through the downstream aperture alone (reference only)."""
    sx = sigma_at(eps, tw["ax"], tw["bx"], g["z_exit"])
    sy = sigma_at(eps, tw["ay"], tw["by"], g["z_exit"])
    cx = g["x0"] + g["xp0"] * 1e-3 * g["z_exit"]
    cy = g["y0"] + g["yp0"] * 1e-3 * g["z_exit"]
    return sx, sy, cx, cy, analytic(sx, sy, cx, cy, g["d_exit"])


def transport(p):
    k, eps, tw = beam_setup(p)
    g = _geom(p)
    rng = np.random.default_rng(g["seed"])
    n, n_pass, n_entr, chunk, first = g["n"], 0, 0, 1_000_000, None
    done = 0
    while done < n:
        m = min(chunk, n - done)
        r = _trace(rng, m, eps, tw, g)
        n_pass += int(r["ok"].sum()); n_entr += int(r["ok_entr"].sum())
        if first is None:
            first = r
        done += m
    t = n_pass / n
    err = math.sqrt(max(t * (1 - t), 0.0) / n)
    sx, sy, cx, cy, a = _analytic_entrance(eps, tw, g)

    z_end = max(30.0, g["z_exit"] + 20.0 if g["exit_on"] else 30.0)
    zg = np.linspace(g["z_start"], z_end, 121)
    env = dict(z=zg.tolist(),
               sx=[sigma_at(eps, tw["ax"], tw["bx"], z) for z in zg],
               sy=[sigma_at(eps, tw["ay"], tw["by"], z) for z in zg],
               cx=(g["x0"] + g["xp0"] * 1e-3 * zg).tolist(),
               cy=(g["y0"] + g["yp0"] * 1e-3 * zg).tolist())

    m = min(MAX_PLOT_POINTS, len(first["xs"]))
    nr = min(250, m)
    rd = lambda a, k=m: np.round(a[:k], 5).tolist()
    s_start = sigma_at(eps, tw["ax"], tw["bx"], g["z_start"])
    return dict(
        transmission=t, std_error=err, n=n, n_pass=n_pass, n_entrance=n_entr,
        analytic=None if g["exit_on"] else a,
        analytic_entrance=a,
        analytic_exit=_analytic_exit(eps, tw, g)[4] if g["exit_on"] else None,
        at_entrance=dict(sigma_x=sx, sigma_y=sy, cx=cx, cy=cy),
        at_exit=dict(zip(("sigma_x", "sigma_y", "cx", "cy"), _analytic_exit(eps, tw, g)[:4])),
        start=dict(z=g["z_start"], sigma_x=s_start,
                   # Twiss at the start plane after drifting back by z_start
                   alpha_x=tw["ax"] - (1 + tw["ax"] ** 2) / tw["bx"] * (g["z_start"] / 1000),
                   beta_x=(s_start ** 2) / eps),
        eps=eps, geom=g, env=env, z_end=z_end,
        points=dict(xa=rd(first["xa"]), ya=rd(first["ya"]), ok=first["ok"][:m].astype(int).tolist(),
                    xs=rd(first["xs"]), xps=rd(first["xpm"]), ys=rd(first["ys"]), yps=rd(first["ypm"])),
        rays=dict(xs=rd(first["xs"], nr), slope=rd(first["dx"] / first["dz"], nr),
                  ok=first["ok"][:nr].astype(int).tolist(), ok_entr=first["ok_entr"][:nr].astype(int).tolist()),
    )


def transport_sweep(p):
    """Transmission vs. waist sigma, centroid offset, aperture diameter or aperture position."""
    k, eps, tw = beam_setup(p)
    var = p.get("var", "sigma_w")
    lo, hi = float(p["lo"]), float(p["hi"])
    steps = int(min(max(int(p.get("steps", 25)), 2), 200))
    rows = []
    for v in np.linspace(lo, hi, steps):
        q = dict(p, n=min(int(p.get("n", 200_000)), 1_000_000))
        t2 = dict(tw)
        if var == "sigma_w":
            if v <= 0:
                continue
            t2["bx"] = t2["by"] = v * v / eps      # beta* such that waist sigma = v
        elif var == "x0":
            q["x0"] = v
        elif var == "diameter":
            if v <= 0:
                continue
            q["diameter"] = v
        elif var == "z_ap":
            q["z_ap"] = v
        elif var == "d_exit":
            if v <= 0:
                continue
            q["d_exit"] = v
        elif var == "z_exit":
            q["z_exit"] = v
        else:
            raise ValueError("unknown sweep variable")
        g = _geom(q)
        r = _trace(np.random.default_rng(g["seed"]), g["n"], eps, t2, g)
        t = float(r["ok"].mean())
        a = None if g["exit_on"] else _analytic_entrance(eps, t2, g)[4]
        rows.append(dict(v=float(v), t=t, err=math.sqrt(max(t * (1 - t), 0) / g["n"]), analytic=a))
    return dict(var=var, rows=rows)


# --------------------------------------------------------------------------- HTTP
def _params(p):
    sigma = float(p.get("sigma", 2.0))
    sx = float(p.get("sigma_x", sigma))
    sy = float(p.get("sigma_y", sigma))
    return dict(
        n=int(min(max(int(p.get("n", 100_000)), 1), MAX_PARTICLES)),
        sigma_x=sx,
        sigma_y=sy,
        x0=float(p.get("x0", 0.0)),
        y0=float(p.get("y0", 0.0)),
        diameter=float(p.get("diameter", 6.0)),
        seed=None if p.get("seed") in (None, "") else int(p["seed"]),
    )


def run(p):
    q = _params(p)
    if q["sigma_x"] <= 0 or q["sigma_y"] <= 0 or q["diameter"] <= 0:
        raise ValueError("sigma and diameter must be positive")
    t, err, n_pass, x, y, m = simulate(**q)
    return dict(
        transmission=t,
        std_error=err,
        n=q["n"],
        n_pass=n_pass,
        analytic=analytic(q["sigma_x"], q["sigma_y"], q["x0"], q["y0"], q["diameter"]),
        points=dict(x=np.round(x, 4).tolist(), y=np.round(y, 4).tolist(), ok=m.astype(int).tolist()),
    )


def sweep(p):
    """Transmission vs. one parameter (sigma, x0 or diameter)."""
    q = _params(p)
    var = p.get("var", "sigma")
    lo, hi = float(p["lo"]), float(p["hi"])
    steps = int(min(max(int(p.get("steps", 25)), 2), 200))
    n = min(q["n"], 2_000_000)
    out = []
    for v in np.linspace(lo, hi, steps):
        k = dict(q, n=n)
        if var == "sigma":
            if v <= 0:
                continue
            k["sigma_x"] = k["sigma_y"] = v
        elif var == "x0":
            k["x0"] = v
        elif var == "diameter":
            if v <= 0:
                continue
            k["diameter"] = v
        else:
            raise ValueError("unknown sweep variable")
        t, err, *_ = simulate(**k)
        a = analytic(k["sigma_x"], k["sigma_y"], k["x0"], k["y0"], k["diameter"])
        out.append(dict(v=float(v), t=t, err=err, analytic=a))
    return dict(var=var, rows=out)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/run":
                res = run(p)
            elif self.path == "/api/sweep":
                res = sweep(p)
            elif self.path == "/api/beam":
                res = beam(p)
            elif self.path == "/api/transport":
                res = transport(p)
            elif self.path == "/api/transport_sweep":
                res = transport_sweep(p)
            else:
                return self._send(404, json.dumps({"error": "not found"}))
            self._send(200, json.dumps(res))
        except Exception as e:  # report bad input to the UI
            self._send(400, json.dumps({"error": str(e)}))

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    print(f"Aperture Monte Carlo running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
