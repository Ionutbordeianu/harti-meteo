#!/usr/bin/env python3
"""
Actualizează hărțile meteo ALARO-4 (ANM) pentru pagina index.html.

Ce face:
  1. citește lista de fișiere de pe https://opendata.geo.meteoromania.ro/nwp/alaro4/
  2. găsește cea mai nouă rulare (ex. 2026111700) și orele de prognoză disponibile
  3. descarcă fiecare fișier GRIB1 (toți parametrii: 18 la sol + 5 pe 12 niveluri
     de presiune), îl decodează și păstrează zona României (19–31°E, 42.5–49.5°N)
  4. calculează izoterma de 0°C și grosimea stratului 500–1000 hPa
  5. scrie în data/:
       hXX.js     parametrii de la sol, 4 km
       hXX_p.js   nivelurile de presiune, 8 km
       mg/*.js    serii de timp pentru meteograme, 8 km, pe zone
       meta.js    descrierea rulării
  6. dacă există github.txt, publică pagina pe GitHub Pages

Dacă datele sunt deja la zi, scriptul se oprește după o singură cerere.
Jurnal: alaro_harti.log (în același folder).
Date: ANM, licență CC BY 4.0.
"""
import base64
import gzip
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

BASE = "https://opendata.geo.meteoromania.ro/nwp/alaro4/"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FORMAT = 6                      # schimbă la modificarea structurii datelor (6 = rafinat pe grila ALADIN 2,3 km)

# grila ALARO-4: 501 x 373, 16.5E..41.5E, 37.99N..51.01N, 0.05 x 0.035 grade
NI, NJ = 501, 373
LON1, LAT1, DLON, DLAT = 16.5, 37.99, 0.05, 0.035
# decupaj: România și vecinătăți (19–31°E, 42.5–49.5°N)
I0, I1, J0, J1 = 50, 290, 128, 329
# grila de ieșire: aceeași ca la ALADIN 2,3 km (relieful comun pentru toate modelele)
FLON0, FLAT0, FDLON, FDLAT, CW, CH = 19.0, 42.5, 0.03, 0.021, 367, 334
G0 = 9.80665
LEVELS = [1000, 975, 950, 925, 900, 850, 700, 600, 500, 400, 300, 200]
UA_STEP = 4                     # nivelurile de presiune la ~9 km
MG_STEP, MG_TILE = 3, 16        # meteograme: ~7 km, zone de 16 x 16 puncte

SURF = {
    (1, 1, 0): "sp", (2, 102, 0): "mslp", (6, 1, 0): "zsfc",
    (61, 1, 0): "prectot", (63, 1, 0): "crain", (79, 1, 0): "lssnow",
    (71, 1, 0): "tcc", (72, 1, 0): "ccc", (73, 1, 0): "lcc", (74, 1, 0): "mcc", (75, 1, 0): "hcc",
    (167, 1, 0): "mocon",
    (33, 105, 10): "u10", (34, 105, 10): "v10", (163, 105, 10): "ug", (164, 105, 10): "vg",
    (11, 105, 2): "t2m", (52, 105, 2): "rh2m",
}
UA_PARAMS = {33: "u", 34: "v", 11: "t", 6: "z", 52: "rh"}
WANT = dict(SURF)
for _p, _n in UA_PARAMS.items():
    for _L in LEVELS:
        WANT[(_p, 100, _L)] = "%s%d" % (_n, _L)

# pas de cuantizare (unitățile GRIB); None = automat
# (aleși după precizia afișată: 0.1°C, 1 km/h, 1 hPa, 50 m, 5%)
STEP = {"t2m": 0.1, "mslp": 10, "sp": 100, "rh2m": 1, "u10": 0.25, "v10": 0.25, "ug": 0.5, "vg": 0.5,
        "prectot": 0.02, "crain": 0.02, "lssnow": 0.02, "tcc": 5, "ccc": 5, "lcc": 5, "mcc": 5, "hcc": 5,
        "mocon": None, "frz": 50, "thick": 0.5}
UA_STEP_BY = {"t": 0.2, "z": 49.0, "u": 0.5, "v": 0.5, "rh": 2}
MG_VARS = ["t2m", "rh2m", "u10", "v10", "ug", "vg", "mslp", "prectot", "lssnow", "crain",
           "tcc", "lcc", "mcc", "hcc", "frz"]

log = logging.getLogger("alaro")


def http(url, rng=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HartiALARO4/1.0"})
            if rng:
                req.add_header("Range", "bytes=%d-%d" % rng)
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
                if rng and r.status != 206:
                    raise IOError("serverul nu a respectat Range (status %s)" % r.status)
                return data
        except Exception as e:
            if k == tries - 1:
                raise
            log.warning("eroare %s (%s), reîncerc", url, e)
            time.sleep(5 * (k + 1))


def latest_run():
    html = http(BASE).decode("utf-8", "replace")
    found = {}
    for run, hh in re.findall(r"alaro4_grib_(\d{10})_(\d{2})\b", html):
        found.setdefault(run, set()).add(int(hh))
    if not found:
        raise RuntimeError("nu am găsit fișiere alaro4_grib_* în lista ANM")
    run = max(found)
    return run, sorted(found[run])


def ibm_float(b):
    s = -1.0 if b[0] & 0x80 else 1.0
    e = (b[0] & 0x7F) - 64
    m = ((b[1] << 16) | (b[2] << 8) | b[3]) / 16777216.0
    return s * m * 16.0 ** e


def s16(hi, lo):
    v = ((hi & 0x7F) << 8) | lo
    return -v if hi & 0x80 else v


def unpack_bits(data, nb, n):
    """valori întregi fără semn de câte nb biți, n bucăți"""
    if nb == 0:
        return np.zeros(n, dtype=np.int64)
    if nb == 8:
        return np.frombuffer(data, dtype=np.uint8, count=n).astype(np.int64)
    if nb == 16:
        return np.frombuffer(data, dtype=">u2", count=n).astype(np.int64)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8, count=(n * nb + 7) // 8))[: n * nb]
    bits = bits.reshape(n, nb).astype(np.int64)
    return bits @ (1 << np.arange(nb - 1, -1, -1, dtype=np.int64))


def parse_messages(buf, out, want=None):
    """parcurge mesajele GRIB1 din buf; adaugă în out câmpurile dorite"""
    want = want or WANT
    o = 0
    while o + 8 <= len(buf):
        if buf[o:o + 4] != b"GRIB":
            raise ValueError("nu găsesc GRIB la poziția %d" % o)
        u24 = lambda i: (buf[i] << 16) | (buf[i + 1] << 8) | buf[i + 2]
        L = u24(o + 4)
        p = o + 8
        pl = u24(p)
        P = buf[p:p + pl]
        par, lt, lev = P[8], P[9], (P[10] << 8) | P[11]
        flag = P[7]
        p1, p2 = P[18], P[19]
        D = s16(P[26], P[27])
        key = want.get((par, lt, lev))
        p += pl
        if flag & 0x80:
            Ni = (buf[p + 6] << 8) | buf[p + 7]
            Nj = (buf[p + 8] << 8) | buf[p + 9]
            scan = buf[p + 27]
            p += u24(p)
        else:
            Ni, Nj, scan = NI, NJ, 64
        if flag & 0x40:
            raise ValueError("mesaj cu bitmap, neprevăzut")
        if key:
            if (Ni, Nj, scan) != (NI, NJ, 64):
                raise ValueError("grilă neașteptată %s" % ((Ni, Nj, scan),))
            B = buf[p:p + u24(p)]
            if (B[3] >> 4) & 4:
                raise ValueError("împachetare complexă, neprevăzută")
            E = s16(B[4], B[5])
            R = ibm_float(B[6:10])
            nb = B[10]
            x = unpack_bits(bytes(B[11:]), nb, Ni * Nj).reshape(Nj, Ni)
            v = ((R + x * 2.0 ** E) * 10.0 ** (-D)).astype(np.float32)
            out[key] = {"v": v, "p1": p1, "p2": p2}
        o += L


def freezing_level(t2m, zs, T, Z):
    """altitudinea (m) a izotermei de 0°C, căutând de la sol în sus.
    T, Z: (nivel, NJ, NI) cu nivelurile de jos în sus; Z în metri. 0 = sub 0°C la sol."""
    K0 = 273.15
    prev_h = zs + 2.0
    prev_t = t2m.astype(np.float64)
    res = np.full(t2m.shape, np.nan)
    res[prev_t <= K0] = 0.0
    for k in range(T.shape[0]):
        h = Z[k].astype(np.float64)
        t = T[k].astype(np.float64)
        above = h > prev_h + 1          # niveluri de deasupra terenului
        t = np.where(above, t, prev_t)
        h = np.where(above, h, prev_h)
        cross = np.isnan(res) & (prev_t > K0) & (t <= K0) & above
        w = np.where(cross, (prev_t - K0) / np.maximum(prev_t - t, 1e-6), 0)
        res = np.where(cross, prev_h + w * (h - prev_h), res)
        prev_t, prev_h = t, h
    res[np.isnan(res)] = prev_h[np.isnan(res)]      # tot stratul peste 0°C: plafonăm la 200 hPa
    return res.astype(np.float32)


def fetch_hour(run, h):
    url = "%salaro4_grib_%s_%02d" % (BASE, run, h)
    fields = {}
    parse_messages(http(url), fields)
    missing = set(WANT.values()) - set(fields)
    if missing:
        raise RuntimeError("+%02dh: lipsesc %s" % (h, ", ".join(sorted(missing))))
    return fields


def crop(fields):
    for f in fields.values():
        f["v"] = f["v"][J0:J1 + 1, I0:I1 + 1]
    return fields


def derive(fields):
    zs = fields["zsfc"]["v"] / G0
    T = np.stack([fields["t%d" % L]["v"] for L in LEVELS])
    Z = np.stack([fields["z%d" % L]["v"] for L in LEVELS]) / G0
    fields["frz"] = {"v": freezing_level(fields["t2m"]["v"], zs, T, Z), "p1": 0, "p2": 0}
    fields["thick"] = {"v": ((fields["z500"]["v"] - fields["z1000"]["v"]) / G0 / 10).astype(np.float32), "p1": 0, "p2": 0}
    return fields


# ---------------- rafinare pe relieful comun (ALADIN 2,3 km) ----------------
_FLON, _FLAT = np.meshgrid(FLON0 + np.arange(CW) * FDLON, FLAT0 + np.arange(CH) * FDLAT)
_fi = (_FLON - (LON1 + I0 * DLON)) / DLON
_fj = (_FLAT - (LAT1 + J0 * DLAT)) / DLAT
_ia = np.clip(np.floor(_fi).astype(int), 0, I1 - I0 - 1); _ja = np.clip(np.floor(_fj).astype(int), 0, J1 - J0 - 1)
_wi = np.clip(_fi - _ia, 0, 1).astype(np.float32); _wj = np.clip(_fj - _ja, 0, 1).astype(np.float32)
_DXM = FDLON * 111320.0 * np.cos(np.radians(_FLAT)); _DYM = FDLAT * 110570.0
SIGMA_M = 3000.0                 # relieful „văzut” de ALARO (4 km)


def to_fine(s):
    s = np.asarray(s, dtype=np.float32)
    a = s[_ja, _ia]; b = s[_ja, _ia + 1]; c = s[_ja + 1, _ia]; d = s[_ja + 1, _ia + 1]
    return ((a * (1 - _wi) + b * _wi) * (1 - _wj) + (c * (1 - _wi) + d * _wi) * _wj).astype(np.float32)


def _blur1(a, sig, axis):
    if sig <= 0.3:
        return a
    r = int(np.ceil(3 * sig))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    pad = [(0, 0), (0, 0)]; pad[axis] = (r, r)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis, np.pad(a, pad, mode="reflect"))


def gblur(a, sigma_m):
    return _blur1(_blur1(a.astype(np.float64), sigma_m / float(np.mean(_DXM)), 1), sigma_m / _DYM, 0).astype(np.float32)


_REL = None


def relief():
    """relieful comun: orografia ALADIN 2,3 km (relief_aladin.npy, copiat din HartiAladin4)"""
    global _REL
    if _REL is None:
        for p in (os.path.join(HERE, "relief_aladin.npy"), os.path.join(HERE, "..", "HartiAladin4", "orog_aladin.npy")):
            if os.path.exists(p):
                a = np.load(p)
                if a.shape == (CH, CW):
                    _REL = a.astype(np.float32)
                    break
        if _REL is None:
            raise RuntimeError("lipsește relieful comun (relief_aladin.npy)")
        gy, gx = np.gradient(_REL.astype(np.float64))
        relief.dzdx = (gx / _DXM).astype(np.float32); relief.dzdy = (gy / _DYM).astype(np.float32)
    return _REL


def _es(Tc):
    return 6.112 * np.exp(17.62 * Tc / (243.12 + Tc))


def _td(Tc, rh):
    g = np.log(np.clip(rh, 1, 100) / 100.0) + 17.62 * Tc / (243.12 + Tc)
    return 243.12 * g / (17.62 - g)


def refine_fields(fields):
    """câmpurile ALARO (4 km) -> grila ALADIN 2,3 km, corectate după relieful comun"""
    zf = relief()
    F = {k: dict(f, v=to_fine(f["v"])) for k, f in fields.items()}
    v = lambda k: F[k]["v"]
    zm = v("zsfc") / G0
    dz = zf - zm
    z850, z700 = v("z850") / G0, v("z700") / G0
    g850 = (v("t850") - v("t2m")) / np.maximum(z850 - zm - 2, 1)
    g700 = (v("t700") - v("t2m")) / np.maximum(z700 - zm - 2, 1)
    gamma = np.clip(np.where(z850 - zm >= 400, g850, g700), -0.0098, 0.010)
    Tc = v("t2m") - 273.15
    Tf = Tc + gamma * dz
    Td = _td(Tc, v("rh2m"))
    Tdf = np.minimum(Td - 0.0017 * dz, Tf)
    rh_c = np.clip(100 * _es(Td) / _es(Tc), 0, 100)
    rh = np.clip(100 * _es(Tdf) / _es(Tf), 0, 100)
    F["t2m"]["v"] = (Tf + 273.15).astype(np.float32)
    F["rh2m"]["v"] = rh.astype(np.float32)
    k = np.clip(1 + 0.35 * dz / 500.0, 0.55, 1.6); kg = np.clip(1 + 0.25 * dz / 500.0, 0.65, 1.5)
    for n in ("u10", "v10"):
        F[n]["v"] = (v(n) * k).astype(np.float32)
    for n in ("ug", "vg"):
        F[n]["v"] = (v(n) * kg).astype(np.float32)
    F["sp"]["v"] = (v("sp") * np.exp(-G0 * dz / (287.05 * (Tf + 273.15)))).astype(np.float32)
    # precipitații: spre versanții expuși vântului de la 850/700 hPa, cantitatea pe zonă păstrată
    low = z850 - zm < 600
    us = np.where(low, v("u700"), v("u850")); vs = np.where(low, v("v700"), v("v850"))
    s = gblur(us * relief.dzdx + vs * relief.dzdy, 3000.0) / 0.08
    w = (1 + 0.45 * np.clip(s, 0, 4)) * np.exp(-0.3 * np.clip(-s, 0, 4)) * (1 + 0.2 * np.clip(dz / 500.0, -1, 2))
    w = np.clip(w, 0.25, 3.5).astype(np.float32)
    ratio = w / np.maximum(gblur(w, SIGMA_M), 1e-3)
    P = np.clip(v("prectot"), 0, None)
    Pf = (P * ratio).astype(np.float32)
    frac = np.where(P > 0.01, np.clip(np.clip(v("lssnow"), 0, None) / np.maximum(P, 1e-6), 0, 1), 0)
    frac = np.clip(frac + (Tc - Tf) / 2.5 * (P > 0.01), 0, 1)
    frac = np.where(Tf > 4, 0, frac)
    F["prectot"]["v"] = Pf
    F["lssnow"]["v"] = (Pf * frac).astype(np.float32)
    F["crain"]["v"] = np.minimum(np.clip(v("crain"), 0, None) * ratio, Pf).astype(np.float32)
    # nori joși / totali după umezeala rafinată
    d = rh - rh_c
    lcc = np.clip(v("lcc"), 0, 100)
    lf = np.clip(np.where(d > 0, lcc + (100 - lcc) * np.clip(d / 20.0, 0, 1) * np.clip((rh - 80) / 15.0, 0, 1),
                          lcc * np.clip(1 + d / 30.0, 0, 1)), 0, 100)
    other = np.maximum(v("mcc"), v("hcc")) / 100.0
    tcc = np.clip(v("tcc") + (lf - lcc) * (1 - other), 0, 100)
    F["lcc"]["v"] = lf.astype(np.float32)
    F["tcc"]["v"] = np.maximum(tcc, np.maximum(lf, np.maximum(v("mcc"), v("hcc")))).astype(np.float32)
    F["zsfc"]["v"] = (zf * G0).astype(np.float32)
    return F


def med_residual(x):
    """predicție din vecini (stânga, sus, stânga-sus) ca în LOCO-I; rămâne doar diferența"""
    a = np.zeros_like(x); b = np.zeros_like(x); c = np.zeros_like(x)
    a[:, 1:] = x[:, :-1]; b[1:, :] = x[:-1, :]; c[1:, 1:] = x[:-1, :-1]
    mx = np.maximum(a, b); mn = np.minimum(a, b)
    p = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    p[0, 1:] = x[0, :-1]; p[1:, 0] = x[:-1, 0]; p[0, 0] = 0
    return x - p


def pack_residual(r):
    """zigzag pe 16 biți, apoi octeții mici și cei mari separat (se arhivează mai bine)"""
    z = (r << 1) ^ (r >> 63)
    if z.max() >= 65536:
        raise ValueError("valori în afara intervalului")
    z = z.astype(np.uint16).ravel()
    return np.concatenate([(z & 255).astype(np.uint8), (z >> 8).astype(np.uint8)]).tobytes()


def enc_field(v, st):
    if st is None:
        rng = float(np.nanmax(v) - np.nanmin(v)) or 1.0
        st = float("%.2g" % (rng / 200))
    k = np.round(v.astype(np.float64) / st).astype(np.int64)
    k0 = int(k.min())
    return k0, st, base64.b64encode(gzip.compress(pack_residual(med_residual(k - k0)), 9)).decode()


def write_js(path, var, key, obj):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("window.%s=window.%s||{};window.%s[%s]=" % (var, var, var, json.dumps(key))
                 + json.dumps(obj, separators=(",", ":")) + ";\n")


def write_hour(tmp, h, fields):
    out = {"h": h, "f": {}}
    for name in list(STEP):
        f = fields[name]
        k0, st, d = enc_field(f["v"], STEP[name])
        out["f"][name] = {"k0": k0, "st": st, "p1": f["p1"], "p2": f["p2"], "d": d}
    write_js(os.path.join(tmp, "h%02d.js" % h), "ALARO_DATA", h, out)
    ua = {"h": h, "f": {}}
    for p, n in UA_PARAMS.items():
        for L in LEVELS:
            v = fields["%s%d" % (n, L)]["v"][::UA_STEP, ::UA_STEP]
            k0, st, d = enc_field(v, UA_STEP_BY[n])
            ua["f"]["%s%d" % (n, L)] = {"k0": k0, "st": st, "d": d}
    write_js(os.path.join(tmp, "h%02d_p.js" % h), "ALARO_UA", h, ua)


def write_orog(tmp, zsfc):
    k0, st, d = enc_field(zsfc / G0, 1)
    rec = {"w": CW, "h": CH, "k0": k0, "st": st, "d": d}
    with open(os.path.join(tmp, "orog.js"), "w", encoding="utf-8") as fh:
        fh.write("window.ALARO_OROG=" + json.dumps(rec, separators=(",", ":")) + ";\n")


def write_meteograms(tmp, mg, nh):
    os.makedirs(os.path.join(tmp, "mg"))
    ny, nx = mg[MG_VARS[0]].shape[1:]
    for ty in range(0, ny, MG_TILE):
        for tx in range(0, nx, MG_TILE):
            th, tw = min(MG_TILE, ny - ty), min(MG_TILE, nx - tx)
            obj = {"tw": tw, "th": th, "v": {}}
            for name in MG_VARS:
                a = mg[name][:, ty:ty + th, tx:tx + tw].astype(np.float64)
                st = STEP[name] or 0.01
                k = np.round(a / st).astype(np.int64)
                k0 = int(k.min())
                r = np.stack([med_residual(k[n] - k0) for n in range(k.shape[0])])
                obj["v"][name] = {"k0": k0, "st": st, "d": base64.b64encode(
                    gzip.compress(pack_residual(r), 9)).decode()}
            key = "%d_%d" % (tx // MG_TILE, ty // MG_TILE)
            write_js(os.path.join(tmp, "mg", "m_%s.js" % key), "ALARO_MG", key, obj)


def update(force):
    run, hours = latest_run()
    old = current_meta()
    if not force and old and old.get("format") == FORMAT and old.get("run") == run and old.get("hours") == hours:
        log.info("deja la zi: rularea %s, %d ore", run, len(hours))
        return
    log.info("rularea %s, ore disponibile: %s", run, hours)
    tmp = DATA + "_nou"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    t0 = time.time()
    mg = {}
    pint = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for n, (h, fields) in enumerate(zip(hours, ex.map(lambda h: (h, refine_fields(crop(derive(fetch_hour(run, h))))), hours))):
            fields = fields[1]
            write_hour(tmp, h, fields)
            pint.append([fields["prectot"]["p1"], fields["prectot"]["p2"]])
            if n == 0:
                write_orog(tmp, fields["zsfc"]["v"])
            for name in MG_VARS:
                if name not in mg:
                    sub = fields[name]["v"][::MG_STEP, ::MG_STEP]
                    mg[name] = np.zeros((len(hours),) + sub.shape, dtype=np.float32)
                mg[name][n] = fields[name]["v"][::MG_STEP, ::MG_STEP]
            log.info("+%02dh gata", h)
            del fields
    write_meteograms(tmp, mg, len(hours))
    meta = {"format": FORMAT, "run": run, "hours": hours, "pint": pint, "levels": LEVELS,
            "grid": {"w": CW, "h": CH, "lon0": FLON0, "lat0": FLAT0, "dlon": FDLON, "dlat": FDLAT},
            "refined": "relief ALADIN 2,3 km",
            "ua_step": UA_STEP, "mg": {"step": MG_STEP, "tile": MG_TILE},
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(os.path.join(tmp, "meta.js"), "w", encoding="utf-8") as fh:
        fh.write("window.ALARO_META=" + json.dumps(meta) + ";\n")
    hist_install(tmp, run)
    log.info("gata: %d ore în %.0f s", len(hours), time.time() - t0)


# ---------------- istoric: ultimele rulări, fiecare în data/<rulare>/ ----------------
KEEP_RUNS = 4
_RUN_RE = re.compile(r"^\d{10}$")


def _read_meta_file(path):
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.read().split("=", 1)[1].rstrip().rstrip(";"))


def hist_runs():
    """rulările păstrate, cea mai nouă prima"""
    if not os.path.isdir(DATA):
        return []
    return sorted([d for d in os.listdir(DATA) if _RUN_RE.match(d) and os.path.exists(os.path.join(DATA, d, "meta.js"))],
                  reverse=True)


def hist_index():
    runs = []
    for r in hist_runs():
        try:
            m = _read_meta_file(os.path.join(DATA, r, "meta.js"))
            runs.append({"run": r, "updated": m.get("updated"), "h0": m["hours"][0], "h1": m["hours"][-1]})
        except Exception:
            log.warning("rularea %s din istoric e incompletă, o ignor", r)
    with open(os.path.join(DATA, "runs.js"), "w", encoding="utf-8") as fh:
        fh.write("window.ALARO_RUNS=" + json.dumps(runs) + ";\n")


def hist_migrate():
    """mută datele din vechiul format (direct în data/) în data/<rulare>/"""
    flat = os.path.join(DATA, "meta.js")
    if not os.path.exists(flat):
        return
    try:
        run = str(_read_meta_file(flat).get("run"))
    except Exception:
        run = ""
    if not _RUN_RE.match(run):
        return
    dst = os.path.join(DATA, run)
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for name in os.listdir(DATA):
        if name == run or _RUN_RE.match(name) or name == "runs.js":
            continue
        os.rename(os.path.join(DATA, name), os.path.join(dst, name))
    log.info("datele existente au fost mutate în istoric (data/%s)", run)
    hist_index()


def hist_install(tmp, run):
    """pune rularea nouă în data/<rulare>/, păstrează doar ultimele KEEP_RUNS"""
    os.makedirs(DATA, exist_ok=True)
    hist_migrate()
    dst = os.path.join(DATA, run)
    old_dir = dst + "_vechi"
    shutil.rmtree(old_dir, ignore_errors=True)
    if os.path.isdir(dst):
        os.rename(dst, old_dir)
    os.rename(tmp, dst)
    shutil.rmtree(old_dir, ignore_errors=True)
    for r in hist_runs()[KEEP_RUNS:]:
        shutil.rmtree(os.path.join(DATA, r), ignore_errors=True)
        log.info("rularea %s a ieșit din istoric", r)
    for name in os.listdir(DATA):          # resturi de la întreruperi
        if name.endswith("_vechi"):
            shutil.rmtree(os.path.join(DATA, name), ignore_errors=True)
    hist_index()


def current_meta():
    try:
        hist_migrate()
    except Exception:
        log.exception("mutarea în istoric a eșuat")
    runs = hist_runs()
    try:
        return _read_meta_file(os.path.join(DATA, runs[0], "meta.js")) if runs else None
    except Exception:
        return None


def _git_sha(data):
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def gh_existing(conf):
    """fișierele deja publicate (cale -> sha), ca să trimitem doar ce s-a schimbat"""
    try:
        ref = gh(conf, "GET", "/git/refs/heads/gh-pages", ok404=True)
        if ref.get("_status") or "object" not in ref:
            return {}
        com = gh(conf, "GET", "/git/commits/" + ref["object"]["sha"])
        t = gh(conf, "GET", "/git/trees/%s?recursive=1" % com["tree"]["sha"])
        if t.get("truncated"):
            return {}
        return {e["path"]: e["sha"] for e in t.get("tree", []) if e.get("type") == "blob"}
    except Exception as e:
        log.warning("nu pot citi ce e deja publicat (%s), trimit tot", e)
        return {}


def gh_blob_sha(conf, existing, local):
    with open(local, "rb") as fh:
        data = fh.read()
    sha = _git_sha(data)
    if existing.get("__any__", {}).get(sha):
        return sha, False
    b = gh(conf, "POST", "/git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})
    return b["sha"], True


# ---------------- publicare pe GitHub Pages (opțional) ----------------
GH_API = os.environ.get("ALARO_GH_API", "https://api.github.com")
GH_CONF = os.path.join(HERE, "github.txt")
GH_MARK = os.path.join(HERE, "github_publicat.txt")


def gh_config():
    if not os.path.exists(GH_CONF):
        return None
    conf = {}
    with open(GH_CONF, encoding="utf-8-sig") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                conf[k.strip().lower()] = v.strip()
    if conf.get("repo") and conf.get("token"):
        return conf
    log.warning("github.txt incomplet (trebuie repo= și token=)")
    return None


def gh(conf, method, path, data=None, ok404=False):
    for i in range(5):
        try:
            return _gh1(conf, method, path, data, ok404)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if i == 4:
                raise
            log.warning("GitHub: conexiune întreruptă (%s), reîncerc...", e)
            time.sleep(5 * (i + 1))


def _gh1(conf, method, path, data=None, ok404=False):
    url = "%s/repos/%s%s" % (GH_API, conf["repo"], path)
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": "Bearer " + conf["token"], "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "HartiALARO4/1.0",
        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            txt = r.read().decode() or "{}"
            return json.loads(txt)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")[:300]
        if ok404 and e.code in (403, 404, 409, 422):
            return {"_status": e.code, "_msg": msg}
        raise RuntimeError("GitHub %s %s -> %s %s" % (method, path, e.code, msg))


def publish(force=False):
    conf = gh_config()
    if not conf:
        return
    meta = current_meta()
    if not meta:
        return
    import hashlib
    pages = [n for n in ("index.html", "comparare.html") if os.path.exists(os.path.join(HERE, n))]
    ph = hashlib.md5(b"".join(open(os.path.join(HERE, n), "rb").read() for n in pages)).hexdigest()[:10]
    stamp = "%s %s %s" % (meta.get("run"), meta.get("updated"), ph)
    if not force and os.path.exists(GH_MARK):
        with open(GH_MARK, encoding="utf-8") as fh:
            if fh.read().strip() == stamp:
                return
    log.info("public pe GitHub (%s)...", conf["repo"])
    files = [(n, os.path.join(HERE, n)) for n in pages]
    for root, _dirs, names in os.walk(DATA):
        for name in sorted(names):
            if name.endswith(".js"):
                full = os.path.join(root, name)
                files.append(("data/" + os.path.relpath(full, DATA).replace(os.sep, "/"), full))
    # un depozit gol nu acceptă încă obiecte git: îl inițializăm cu un README
    first = gh(conf, "POST", "/git/blobs", {"content": "", "encoding": "utf-8"}, ok404=True)
    if first.get("_status") == 409:
        gh(conf, "PUT", "/contents/README.md", {"message": "init", "content": base64.b64encode(
            "Hărți meteo ALARO-4 (date ANM, CC BY 4.0)\n".encode()).decode()})
    tree = [{"path": ".nojekyll", "mode": "100644", "type": "blob",
             "sha": gh(conf, "POST", "/git/blobs", {"content": "", "encoding": "utf-8"})["sha"]}]
    existing = gh_existing(conf)
    existing["__any__"] = {s: True for p, s in existing.items() if p != "__any__"}
    sent = 0
    for path, local in files:
        sha, up = gh_blob_sha(conf, existing, local)
        sent += up
        tree.append({"path": path, "mode": "100644", "type": "blob", "sha": sha})
    log.info("fișiere: %d în total, %d trimise acum", len(files), sent)
    tsha = gh(conf, "POST", "/git/trees", {"tree": tree})["sha"]
    csha = gh(conf, "POST", "/git/commits", {"message": "date " + stamp, "tree": tsha, "parents": []})["sha"]
    r = gh(conf, "PATCH", "/git/refs/heads/gh-pages", {"sha": csha, "force": True}, ok404=True)
    if r.get("_status"):
        gh(conf, "POST", "/git/refs", {"ref": "refs/heads/gh-pages", "sha": csha})
    pages = gh(conf, "GET", "/pages", ok404=True)
    if pages.get("_status") in (403, 404):
        r = gh(conf, "POST", "/pages", {"source": {"branch": "gh-pages", "path": "/"}}, ok404=True)
        if r.get("_status"):
            log.warning("activează GitHub Pages manual: Settings > Pages > Branch: gh-pages / (root)")
    owner, name = conf["repo"].split("/", 1)
    with open(GH_MARK, "w", encoding="utf-8") as fh:
        fh.write(stamp)
    log.info("publicat: https://%s.github.io/%s/", owner.lower(), name)


def main():
    force = "--force" in sys.argv
    try:
        update(force)
    finally:
        try:
            publish("--publica" in sys.argv)
        except Exception:
            log.exception("publicarea pe GitHub a eșuat (se reîncearcă la următoarea rulare)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(HERE, "alaro_harti.log"), encoding="utf-8")]
                        + ([logging.StreamHandler(sys.stdout)] if sys.stdout else []))
    try:
        main()
    except Exception:
        log.exception("actualizarea a eșuat")
        sys.exit(1)
