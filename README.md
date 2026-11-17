# ISAC Beam Phase Space & Gas-Cell Aperture Transmission

A Monte Carlo tool with a browser interface for estimating how much of an ISAC
beam passes through the entrance and downstream apertures of a gas cell.

## Run

```bash
pip install -r requirements.txt
python3 aperture_mc.py          # opens http://localhost:8765
```

Options: `--port 9000`, `--no-browser`.

## What it does

### 1 · Beam emittance
- Choose a beam species (A, q) and energy in MeV/u. The page shows β, γ, βγ,
  A/q and the magnetic rigidity Bρ, and warns when A/q is above the approximate
  ISAC RFQ, DTL or ISAC-II limits.
- ISAC emittances are the defaults:
  - normalized transverse emittance: 0.2 π mm·mrad
  - longitudinal emittance: 1.5 π keV/u·ns overall, or 0.55 through the DTL
- You choose whether the quoted values are rms or 4 × rms.
- The geometric emittance is ε = εₙ / (βγ). The longitudinal emittance in
  keV/u·ns is treated as invariant under acceleration.
- The waist is set by Twiss α and β at z = 0, or directly by a waist σ.
- Monte Carlo plots show x–x′, y–y′ and Δt–ΔW with the 2-rms ellipses. The
  page also gives bunch length (ns, degrees at the RF frequency, mm), σ_W/W
  and σ_p/p.

### 2 · Ray tracing through the apertures
1. Particles are sampled at the waist, z = 0 (centre of the gas cell).
2. Each ray is traced back upstream in free space, with no apertures, to a
   start plane (150 mm upstream by default).
3. Each ray, (x, y, z) plus direction (dx, dy, dz), is propagated downstream.
4. A ray is transmitted only if it passes **both** apertures:
   - entrance: Ø6 mm at z = −105 mm
   - downstream: Ø8 mm at z = +105 mm
- You can add a beam offset and an angle offset at the waist.
- Plots: a side view of the rays with the ±2σ envelope, x–x′ at the start
  plane, and x–y at the entrance aperture. A parameter sweep can vary waist σ,
  offset, or either aperture's diameter or position.
- For each aperture on its own, the exact drift result is shown alongside the
  Monte Carlo as a check.

## Assumptions
- Gaussian beam with free drift only: no fields, space charge or gas
  scattering.
- Whether the quoted ISAC emittances are rms or 4 × rms is selectable; the
  default is rms.
- The default Twiss values (an upright waist with β = 1 m) are placeholders.
  Enter the real waist β or σ.

## Files
- `aperture_mc.py`: physics and a small HTTP server that uses only the Python
  standard library, plus numpy. scipy is optional and is used for the exact
  result for an offset beam.
- `index.html`: the user interface, with plots drawn on HTML canvas.
