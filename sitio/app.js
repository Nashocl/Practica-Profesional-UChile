/* Práctica CMM: simulación del Epileptor 2D (Proix et al. 2014, ec. 8) y laboratorios interactivos */
(() => {
'use strict';

const TAU0 = 2857, I1 = 3.1, X0C = 2.91;
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C = {
  ink: css('--ink'), muted: css('--muted'), line: css('--line'), grid: css('--line-soft'),
  accent: css('--accent'), sage: css('--sage'), ochre: css('--ochre'), plate: css('--plate'),
};
const ACC = [180, 80, 44], SAGE = [102, 117, 75], CREAM = [255, 253, 249];
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const mix = (a, b, u) => `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * u)).join(',')})`;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const R = String.raw;
const KATEX_OPTS = { delimiters: [{ left: '$$', right: '$$', display: true }, { left: R`\(`, right: R`\)`, display: false }], throwOnError: false };
const tex = el => { if (window.renderMathInElement && el) renderMathInElement(el, KATEX_OPTS); };
const MATHFONT = 'italic 15px KaTeX_Math, "Times New Roman", serif';

/* ================= modelo ================= */
const sig = u => 1 / (1 + Math.exp(-u));
const hEq8 = (x, x0) => x0 + 3 * sig((x + 0.5) / 0.1);          // Práctica_OFICIAL / ecuación (8)
const hNb = (x, x0) => x0 - 3 + 3 * sig((x - 0.5) / 0.1);       // variante usada en el experimento base del grafo
const cubic = x => -x * x * x - 2 * x * x + 1 + I1;             // nullclina de x

// RK4 para N regiones; nbrs[i] = índices j que entran en la ecuación de i
function simulate({ x0, nbrs, K, T, xi, zi, h = hEq8, dt = 0.1, every = 10 }) {
  const N = x0.length, steps = Math.round(T / dt), ns = Math.floor(steps / every) + 1;
  let s = new Float64Array(2 * N);
  const k1 = new Float64Array(2 * N), k2 = new Float64Array(2 * N), k3 = new Float64Array(2 * N),
    k4 = new Float64Array(2 * N), tmp = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) { s[i] = xi[i]; s[N + i] = zi[i]; }
  const f = (u, o) => {
    for (let i = 0; i < N; i++) {
      const x = u[i], nb = nbrs[i];
      let c = 0;
      for (let q = 0; q < nb.length; q++) c += u[nb[q]] - x;
      o[i] = -x * x * x - 2 * x * x + 1 - u[N + i] + I1;
      o[N + i] = (h(x, x0[i]) - u[N + i] - K * c) / TAU0;
    }
  };
  const t = new Float32Array(ns), xs = [], zs = [];
  for (let i = 0; i < N; i++) { xs.push(new Float32Array(ns)); zs.push(new Float32Array(ns)); }
  const rec = k => { for (let i = 0; i < N; i++) { xs[i][k] = s[i]; zs[i][k] = s[N + i]; } };
  rec(0);
  const M = 2 * N;
  for (let n = 1; n <= steps; n++) {
    f(s, k1);
    for (let m = 0; m < M; m++) tmp[m] = s[m] + 0.5 * dt * k1[m];
    f(tmp, k2);
    for (let m = 0; m < M; m++) tmp[m] = s[m] + 0.5 * dt * k2[m];
    f(tmp, k3);
    for (let m = 0; m < M; m++) tmp[m] = s[m] + dt * k3[m];
    f(tmp, k4);
    for (let m = 0; m < M; m++) s[m] += dt / 6 * (k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m]);
    if (n % every === 0) { const k = n / every; t[k] = n * dt; rec(k); }
  }
  return { t, x: xs, z: zs, n: ns, T };
}

function onsets(t, x, th = -0.5, from = 0) {
  const r = [];
  for (let k = Math.max(1, from); k < x.length; k++) if (x[k - 1] <= th && x[k] > th) r.push(t[k]);
  return r;
}
const meanDiff = a => a.length > 1 ? (a[a.length - 1] - a[0]) / (a.length - 1) : NaN;

// puntos fijos de N = 1 y su estabilidad (barrido de signo + bisección)
function fixedPoints(x0) {
  const g = x => cubic(x) - hEq8(x, x0), out = [];
  const n = 6000;
  let xa = -3, fa = g(xa);
  for (let k = 1; k <= n; k++) {
    const xb = -3 + 6 * k / n, fb = g(xb);
    if (fa * fb < 0) {
      let lo = xa, hi = xb, flo = fa;
      for (let it = 0; it < 60; it++) { const mid = (lo + hi) / 2, fm = g(mid); if (flo * fm <= 0) hi = mid; else { lo = mid; flo = fm; } }
      const x = (lo + hi) / 2, s = sig((x + 0.5) / 0.1), dh = 3 * s * (1 - s) / 0.1;
      const a = -3 * x * x - 4 * x, tr = a - 1 / TAU0, det = (-a + dh) / TAU0, disc = tr * tr - 4 * det;
      const re = disc >= 0 ? (tr + Math.sqrt(disc)) / 2 : tr / 2;
      out.push({ x, z: hEq8(x, x0), re, stable: re < 0 });
    }
    xa = xb; fa = fb;
  }
  return out;
}

/* ================= utilidades de canvas ================= */
function fit(c) {
  const d = Math.min(window.devicePixelRatio || 1, 2);
  if (!c._h) c._h = +c.getAttribute('height') || 200;
  const w = c.clientWidth, h = c._h;
  if (c._w !== w || c._d !== d) {
    c.width = Math.round(w * d); c.height = Math.round(h * d); c.style.height = h + 'px'; c._w = w; c._d = d;
  }
  const g = c.getContext('2d');
  g.setTransform(d, 0, 0, d, 0, 0);
  return { g, w, h };
}
function ticks([a, b], n = 5) {
  const raw = (b - a) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), e = raw / p;
  const st = (e < 1.5 ? 1 : e < 3 ? 2 : e < 7 ? 5 : 10) * p, out = [];
  for (let v = Math.ceil(a / st) * st; v <= b + 1e-9; v += st) out.push(Math.abs(v) < 1e-9 ? 0 : v);
  return out;
}
const fmtT = v => Math.abs(v) >= 1000 ? (v / 1000) + 'k' : String(+v.toFixed(2));
function frame(g, w, h, xr, yr, o = {}) {
  const m = { l: o.l ?? 36, r: 10, t: 8, b: 22 };
  const X = v => m.l + (v - xr[0]) / (xr[1] - xr[0]) * (w - m.l - m.r);
  const Y = v => h - m.b - (v - yr[0]) / (yr[1] - yr[0]) * (h - m.t - m.b);
  g.clearRect(0, 0, w, h);
  g.font = '13.5px KaTeX_Main, "Times New Roman", serif';
  g.lineWidth = 1; g.setLineDash([]);
  g.textAlign = 'right';
  for (const v of ticks(yr, o.ny ?? 4)) {
    g.strokeStyle = C.grid; g.beginPath(); g.moveTo(m.l, Y(v)); g.lineTo(w - m.r, Y(v)); g.stroke();
    g.fillStyle = C.muted; g.fillText(fmtT(v), m.l - 6, Y(v) + 3.5);
  }
  g.textAlign = 'center';
  for (const v of ticks(xr, o.nx ?? 6)) g.fillText(fmtT(v), X(v), h - 6);
  if (o.xl) { g.save(); g.font = MATHFONT; g.textAlign = 'right'; g.fillText(o.xl, w - m.r, h - 6 - 14); g.restore(); }
  return { X, Y, m };
}
function path(g, X, Y, xs, ys, a = 0, b = xs.length) {
  g.beginPath();
  for (let k = a; k < b; k++) { const px = X(xs[k]), py = Y(ys[k]); k === a ? g.moveTo(px, py) : g.lineTo(px, py); }
}
function stroke(g, color, width = 1.5, dash = []) { g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash); g.lineJoin = 'round'; g.stroke(); g.setLineDash([]); }
function dot(g, x, y, r, fill, ring) {
  g.beginPath(); g.arc(x, y, r, 0, 6.2832); g.fillStyle = fill; g.fill();
  if (ring) { g.strokeStyle = ring; g.lineWidth = 2; g.stroke(); }
}
function tween(ms, fn, done) {
  if (reduced) { fn(1); done && done(); return () => {}; }
  let id, t0;
  const step = now => {
    t0 ??= now;
    const u = clamp((now - t0) / ms, 0, 1);
    fn(1 - Math.pow(1 - u, 3));
    if (u < 1) id = requestAnimationFrame(step); else done && done();
  };
  id = requestAnimationFrame(step);
  return () => cancelAnimationFrame(id);
}
function watchVisible(el, cb) {
  new IntersectionObserver(es => es.forEach(e => cb(e.isIntersecting)), { rootMargin: '80px' }).observe(el);
}
function bindRange(input, out, dec, fn) {
  const upd = () => {
    const v = +input.value;
    out.textContent = v.toFixed(dec);
    input.style.setProperty('--fill', ((v - input.min) / (input.max - input.min) * 100) + '%');
  };
  input.addEventListener('input', () => { upd(); fn(); });
  upd();
  return upd;
}
const rafThrottle = fn => { let q = false; return () => { if (q) return; q = true; requestAnimationFrame(() => { q = false; fn(); }); }; };

/* ================= fondo: red de regiones que se reclutan ================= */
function background() {
  const c = $('#bgNet'), g = c.getContext('2d');
  const LINK = 175;
  let w = 0, h = 0, nodes = [], last = 0, nextFire = 1200;
  function size() {
    const d = Math.min(window.devicePixelRatio || 1, 1.5);
    w = innerWidth; h = innerHeight;
    c.width = Math.round(w * d); c.height = Math.round(h * d);
    g.setTransform(d, 0, 0, d, 0, 0);
    const n = clamp(Math.round(w * h / 36000), 16, 46), r = mulberry32(5);
    nodes = Array.from({ length: n }, (_, i) => {
      const a = r() * 6.2832, v = 5 + 9 * r();
      return { x: r() * w, y: r() * h, vx: v * Math.cos(a), vy: v * Math.sin(a), epi: i % 5 === 0, act: 0, wait: -1, gen: 0 };
    });
  }
  // una región epileptogénica dispara y arrastra a sus vecinas con retardo, como en el grafo
  function fire(p, gen) {
    p.act = 1; p.gen = gen;
    if (gen >= 2) return;
    for (const q of nodes) if (q !== p && q.act < 0.2 && q.wait < 0 && Math.hypot(q.x - p.x, q.y - p.y) < LINK) { q.wait = 380 + 500 * Math.random(); q.gen = gen + 1; }
  }
  function draw(dt) {
    g.clearRect(0, 0, w, h);
    for (const p of nodes) {
      p.x += p.vx * dt / 1000; p.y += p.vy * dt / 1000;
      if (p.x < -20) p.x = w + 20; else if (p.x > w + 20) p.x = -20;
      if (p.y < -20) p.y = h + 20; else if (p.y > h + 20) p.y = -20;
      if (p.wait >= 0) { p.wait -= dt; if (p.wait < 0) fire(p, p.gen); }
      p.act = Math.max(0, p.act - dt / 2600);
    }
    g.lineWidth = 1;
    for (let i = 0; i < nodes.length; i++) for (let k = i + 1; k < nodes.length; k++) {
      const a = nodes[i], b = nodes[k], d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d > LINK) continue;
      const hot = Math.min(a.act, b.act) > 0.05 ? Math.max(a.act, b.act) : 0;
      g.strokeStyle = hot ? rgba(ACC, (1 - d / LINK) * (0.12 + 0.3 * hot)) : rgba([118, 104, 90], (1 - d / LINK) * 0.13);
      g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
    }
    for (const p of nodes) {
      if (p.act > 0.02) {
        g.beginPath(); g.arc(p.x, p.y, 4 + 26 * (1 - p.act), 0, 6.2832);
        g.strokeStyle = rgba(ACC, 0.3 * p.act); g.lineWidth = 1.5; g.stroke();
      }
      dot(g, p.x, p.y, p.epi ? 3 : 2.2, p.act > 0.02 ? rgba(ACC, 0.25 + 0.45 * p.act) : rgba(p.epi ? ACC : SAGE, p.epi ? 0.32 : 0.28));
    }
  }
  const loop = now => {
    requestAnimationFrame(loop);
    if (document.hidden || now - last < 33) return;      // ~30 fps bastan para un fondo
    const dt = Math.min(80, now - last || 33); last = now;
    nextFire -= dt;
    if (nextFire < 0) {
      const epi = nodes.filter(p => p.epi && p.x > 0 && p.x < w && p.y > 0 && p.y < h);
      if (epi.length) fire(epi[Math.floor(Math.random() * epi.length)], 0);
      nextFire = 2600 + 2200 * Math.random();
    }
    draw(dt);
  };
  size();
  let rt;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { size(); if (reduced) draw(0); }, 150); });
  if (reduced) draw(0); else requestAnimationFrame(loop);
}

/* ================= Three.js: una sola carga para el cerebro y el grafo 3D ================= */
let threePromise = null;
const loadThree = () => threePromise || (threePromise = import('./vendor/three/three.module.js')
  .catch(() => import('https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js')));

/* ================= portada: cerebro 3D interactivo (Three.js) ================= */
function brain3d() {
  const canvas = $('#brain3d');
  let onResize = () => {};
  loadThree()
    .then(T => { onResize = buildBrain(T, canvas); })
    .catch(() => { canvas.parentElement.style.display = 'none'; });
  return () => onResize();
}

function buildBrain(T, canvas) {
  const renderer = new T.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(32, 1, 0.1, 20);
  camera.position.set(0, 0, 3.45);
  // iluminación de estudio: luz principal cálida, relleno rosado y contraluz para marcar el relieve
  scene.add(new T.HemisphereLight(0xfff3ee, 0x6b2a30, 0.4));
  const key = new T.DirectionalLight(0xfff6ee, 1.4);
  key.position.set(-2.2, 3.2, 3.6);
  const rim = new T.DirectionalLight(0xffd9d0, 0.35);
  rim.position.set(3, 1.2, -3);
  const fill = new T.DirectionalLight(0xffe4e6, 0.12);
  fill.position.set(2.5, -1.5, 2.5);
  scene.add(key, rim, fill);

  const brain = new T.Group();
  scene.add(brain);
  // ---- respaldo: cerebro generado por código, por si el modelo no se puede cargar
  function procedural() {
  const tmp = new T.Color();
  const SULCUS = new T.Color('#4a0f1c'), GYRUS = new T.Color('#e07585'), CREST = new T.Color('#f3a6ae');
  // superficie húmeda: poca rugosidad y una capa de barniz que da los brillos de la referencia
  const cortex = new T.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.42, metalness: 0, clearcoat: 0.55, clearcoatRoughness: 0.3 });

  // ruido de gradiente 3D (Perlin); sus cruces por cero, deformados, dibujan el laberinto de surcos
  const fade = t => t * t * t * (t * (t * 6 - 15) + 10);
  const grad = (i, j, k, x, y, z) => {
    let h = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(k, 1274126177);
    h = Math.imul(h ^ (h >>> 13), 1103515245); h = (h ^ (h >>> 16)) & 15;
    const u = h < 8 ? x : y, v = h < 4 ? y : h === 12 || h === 14 ? x : z;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
  };
  function noise(x, y, z) {
    const i = Math.floor(x), j = Math.floor(y), k = Math.floor(z), fx = x - i, fy = y - j, fz = z - k, u = fade(fx), v = fade(fy), w = fade(fz);
    const L = (a, b, t) => a + (b - a) * t;
    return L(L(L(grad(i, j, k, fx, fy, fz), grad(i + 1, j, k, fx - 1, fy, fz), u), L(grad(i, j + 1, k, fx, fy - 1, fz), grad(i + 1, j + 1, k, fx - 1, fy - 1, fz), u), v),
             L(L(grad(i, j, k + 1, fx, fy, fz - 1), grad(i + 1, j, k + 1, fx - 1, fy, fz - 1), u), L(grad(i, j + 1, k + 1, fx, fy - 1, fz - 1), grad(i + 1, j + 1, k + 1, fx - 1, fy - 1, fz - 1), u), v), w);
  }
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

  function hemisphere(seed) {
    const WS = 288, HS = 208, geo = new T.SphereGeometry(1, WS, HS), pos = geo.attributes.position;
    const col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const ux = pos.getX(i), uy = pos.getY(i), uz = pos.getZ(i);
      let rx = ux < 0 ? 0.13 : 0.45, ry = uy < 0 ? 0.35 : 0.50;
      const rz = uz < 0 ? 0.80 : 0.76;
      if (uz > 0) { rx *= 1 - 0.20 * uz; ry *= 1 - 0.12 * uz; } else { ry *= 1 + 0.06 * uz; }
      // lóbulo temporal: abultamiento lateral e inferior, hacia adelante
      const lobe = ux > 0 ? Math.exp(-((uy + 0.55) ** 2 / 0.07 + (uz - 0.22) ** 2 / 0.22)) : 0;
      const bx = 0.155 + ux * rx + 0.05 * lobe, by = uy * ry - 0.11 * lobe, bz = uz * rz + 0.03 * lobe;
      let nx = ux / rx, ny = uy / ry, nz = uz / rz;
      const nl = Math.hypot(nx, ny, nz); nx /= nl; ny /= nl; nz /= nl;

      // surcos: dos escalas de pliegue con deformación de dominio
      const px = bx + seed, py = by + seed * 0.37, pz = bz - seed * 0.61;
      const wx = 0.13 * noise(px * 2.6, py * 2.6, pz * 2.6), wy = 0.13 * noise(px * 2.6 + 31, py * 2.6 + 17, pz * 2.6 + 5), wz = 0.13 * noise(px * 2.6 + 71, py * 2.6 + 43, pz * 2.6 + 29);
      const f = noise((px + wx) * 7.4, (py + wy) * 7.4, (pz + wz) * 7.4) + 0.25 * noise((px + wx) * 15 + 9, (py + wy) * 15 + 4, (pz + wz) * 15 + 2);
      // perfil del giro: cima redondeada y pliegue en V donde el campo cruza por cero
      const tf = Math.min(1, Math.abs(f) / 0.3);
      let ridge = 1 - (1 - tf) * (1 - tf);
      // cisura de Silvio: surco profundo que separa el lóbulo temporal
      const sy = -0.30 + 0.42 * (0.30 - uz), dSyl = ux > 0.25 && uz < 0.45 && uz > -0.35 ? Math.abs(uy - sy) : 1;
      const sylv = 1 - smooth(0.0, 0.07, dSyl);
      ridge *= 1 - 0.9 * sylv;
      const medial = ux < -0.2 ? 0.35 : 1, depth = 0.08 * medial;
      const d = -depth * (1 - ridge) + 0.012 * ridge - 0.03 * sylv;
      pos.setXYZ(i, bx + nx * d, by + ny * d, bz + nz * d);

      const mottle = 0.5 + 0.5 * noise(px * 3.1 + 50, py * 3.1, pz * 3.1);
      tmp.copy(SULCUS).lerp(GYRUS, Math.pow(tf * (1 - 0.9 * sylv), 0.9)).lerp(CREST, 0.6 * smooth(0.8, 1, ridge) * mottle);
      col[3 * i] = tmp.r; col[3 * i + 1] = tmp.g; col[3 * i + 2] = tmp.b;
    }
    geo.setAttribute('color', new T.BufferAttribute(col, 3));
    geo.computeVertexNormals();
    return { geo, WS, HS };
  }
  const hemis = [1, -1].map((side, n) => {
    const h = hemisphere(n ? 7.3 : 1.9), mesh = new T.Mesh(h.geo, cortex);
    mesh.scale.x = side;                                    // el izquierdo es el espejo, con otro patrón de pliegues
    brain.add(mesh);
    return { ...h, side };
  });

  // cerebelo: dos lóbulos con láminas finas horizontales, más oscuro y pardo que la corteza
  const CB_DARK = new T.Color('#6a2a2c'), CB_LIGHT = new T.Color('#b9666a');
  const cgeo = new T.SphereGeometry(1, 128, 160), cpos = cgeo.attributes.position, ccol = new Float32Array(cpos.count * 3);
  for (let i = 0; i < cpos.count; i++) {
    const ux = cpos.getX(i), uy = cpos.getY(i), uz = cpos.getZ(i);
    const folia = Math.sin(58 * uy + 5 * ux * ux + 2.5 * uz), crease = 1 - Math.pow(Math.abs(folia), 0.55);
    const split = Math.exp(-(ux * ux) / 0.012) * (uz < 0.3 ? 1 : 0);          // hendidura entre los dos lóbulos
    const k = 1 - 0.045 * crease - 0.10 * split;
    cpos.setXYZ(i, ux * 0.40 * k, -0.37 + uy * 0.20 * k, -0.47 + uz * 0.27 * k);
    tmp.copy(CB_DARK).lerp(CB_LIGHT, clamp(1 - crease * 1.2 - 0.5 * split, 0, 1));
    ccol[3 * i] = tmp.r; ccol[3 * i + 1] = tmp.g; ccol[3 * i + 2] = tmp.b;
  }
  cgeo.setAttribute('color', new T.BufferAttribute(ccol, 3));
  cgeo.computeVertexNormals();
  brain.add(new T.Mesh(cgeo, new T.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.5, clearcoat: 0.4, clearcoatRoughness: 0.35 })));

  // tronco encefálico: protuberancia y bulbo que se afina hacia la médula
  const stemMat = new T.MeshPhysicalMaterial({ color: '#c98f84', roughness: 0.5, clearcoat: 0.35, clearcoatRoughness: 0.4 });
  const pons = new T.Mesh(new T.SphereGeometry(1, 40, 28), stemMat);
  pons.scale.set(0.10, 0.10, 0.09); pons.position.set(0, -0.40, -0.21);
  const stem = new T.Mesh(new T.CylinderGeometry(0.085, 0.045, 0.42, 28, 1), stemMat);
  stem.position.set(0, -0.60, -0.235); stem.rotation.x = 0.22;
  brain.add(pons, stem);
  return () => {
    const h = hemis[Math.random() < 0.5 ? 0 : 1], pos = h.geo.attributes.position, nor = h.geo.attributes.normal;
    let ix = Math.floor(Math.random() * h.WS), iy = 24 + Math.floor(Math.random() * (h.HS - 48));
    let dx = (Math.random() < 0.5 ? -1 : 1) * (2 + Math.random() * 4), dy = (Math.random() - 0.5) * 6;
    const pts = [], steps = 9 + Math.floor(Math.random() * 8);
    for (let n = 0; n < steps; n++) {
      const i = iy * (h.WS + 1) + ((ix % h.WS) + h.WS) % h.WS, j = () => (Math.random() - 0.5) * 0.022;
      pts.push(new T.Vector3((pos.getX(i) + nor.getX(i) * 0.02 + j()) * h.side, pos.getY(i) + nor.getY(i) * 0.02 + j(), pos.getZ(i) + nor.getZ(i) * 0.02 + j()));
      ix += Math.round(dx + (Math.random() - 0.5) * 6);
      iy = clamp(iy + Math.round(dy + (Math.random() - 0.5) * 6), 12, h.HS - 12);
    }
    return pts;
  };
  }

  // ---- modelo anatómico: lectura directa del .glb (una malla, texturas PNG embebidas)
  async function loadModel() {
    const res = await fetch('assets/models/brain.glb');
    if (!res.ok) throw new Error('modelo no disponible');
    const buf = await res.arrayBuffer(), dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('no es un GLB');
    const jsonLen = dv.getUint32(12, true), json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLen)));
    const bin = 20 + jsonLen + 8;
    const view = i => { const v = json.bufferViews[i]; return [bin + (v.byteOffset || 0), v.byteLength]; };
    const accessor = (i, Arr, n) => {
      const a = json.accessors[i], start = view(a.bufferView)[0] + (a.byteOffset || 0);
      return new Arr(buf.slice(start, start + a.count * n * Arr.BYTES_PER_ELEMENT));
    };
    const texture = async (ti, srgb) => {
      const im = json.images[json.textures[ti].source], [o, l] = view(im.bufferView);
      const bmp = await createImageBitmap(new Blob([new Uint8Array(buf, o, l)], { type: im.mimeType }));
      const t = new T.Texture(bmp);
      t.flipY = false; t.wrapS = t.wrapT = T.RepeatWrapping; t.anisotropy = 8;
      if (srgb) t.colorSpace = T.SRGBColorSpace;
      t.needsUpdate = true;
      return t;
    };
    const prim = json.meshes[0].primitives[0], mat = json.materials[prim.material];
    const sg = mat.extensions && mat.extensions.KHR_materials_pbrSpecularGlossiness;
    const diffuse = sg ? sg.diffuseTexture : mat.pbrMetallicRoughness.baseColorTexture;
    const [map, normalMap] = await Promise.all([texture(diffuse.index, true), texture(mat.normalTexture.index, false)]);

    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.BufferAttribute(accessor(prim.attributes.POSITION, Float32Array, 3), 3));
    geo.setAttribute('normal', new T.BufferAttribute(accessor(prim.attributes.NORMAL, Float32Array, 3), 3));
    geo.setAttribute('uv', new T.BufferAttribute(accessor(prim.attributes.TEXCOORD_0, Float32Array, 2), 2));
    geo.setIndex(new T.BufferAttribute(accessor(prim.indices, json.accessors[prim.indices].componentType === 5125 ? Uint32Array : Uint16Array, 1), 1));
    // centrar y llevar a la escala de la escena (el más largo de sus ejes mide ~1.7)
    geo.computeBoundingBox();
    const box = geo.boundingBox, c = box.getCenter(new T.Vector3()), s = box.getSize(new T.Vector3());
    geo.translate(-c.x, -c.y, -c.z);
    geo.scale(1.7 / Math.max(s.x, s.y, s.z), 1.7 / Math.max(s.x, s.y, s.z), 1.7 / Math.max(s.x, s.y, s.z));
    const material = new T.MeshPhysicalMaterial({ map, color: new T.Color(1, 0.62, 0.74), normalMap, normalScale: new T.Vector2(1.25, 1.25), roughness: 0.5, clearcoat: 0.3, clearcoatRoughness: 0.35, side: T.DoubleSide });
    brain.add(new T.Mesh(geo, material));

    // vecindad entre vértices (soldando los duplicados de las costuras UV) para que los destellos caminen por la superficie
    const P = geo.attributes.position, N = geo.attributes.normal, idx = geo.index, canon = new Int32Array(P.count), seen = new Map();
    for (let i = 0; i < P.count; i++) {
      const k = P.getX(i).toFixed(4) + ',' + P.getY(i).toFixed(4) + ',' + P.getZ(i).toFixed(4);
      if (!seen.has(k)) seen.set(k, i);
      canon[i] = seen.get(k);
    }
    const nb = new Map();
    const link = (a, b) => { if (a === b) return; (nb.get(a) || nb.set(a, new Set()).get(a)).add(b); };
    for (let i = 0; i < idx.count; i += 3) {
      const a = canon[idx.getX(i)], b = canon[idx.getX(i + 1)], d = canon[idx.getX(i + 2)];
      link(a, b); link(b, a); link(b, d); link(d, b); link(a, d); link(d, a);
    }
    const ids = [...nb.keys()];
    return () => {
      let i = ids[Math.floor(Math.random() * ids.length)];
      const dir = new T.Vector3(Math.random() - .5, Math.random() - .5, Math.random() - .5).normalize();
      const pts = [], used = new Set(), steps = 7 + Math.floor(Math.random() * 6), j = () => (Math.random() - 0.5) * 0.02;
      for (let n = 0; n < steps && i !== undefined; n++) {
        used.add(i);
        pts.push(new T.Vector3(P.getX(i) + N.getX(i) * 0.022 + j(), P.getY(i) + N.getY(i) * 0.022 + j(), P.getZ(i) + N.getZ(i) * 0.022 + j()));
        let best, bs = -1e9;
        for (const k of nb.get(i)) {
          if (used.has(k)) continue;
          const sc = (P.getX(k) - P.getX(i)) * dir.x + (P.getY(k) - P.getY(i)) * dir.y + (P.getZ(k) - P.getZ(i)) * dir.z + 0.03 * Math.random();
          if (sc > bs) { bs = sc; best = k; }
        }
        i = best;
      }
      return pts;
    };
  }


  // ---- destellos eléctricos: tubos quebrados que avanzan sobre la corteza y se apagan
  class Poly extends T.Curve {
    constructor(pts) { super(); this.pts = pts; }
    getPoint(t, target = new T.Vector3()) {
      const n = this.pts.length - 1, f = clamp(t, 0, 1) * n, i = Math.min(n - 1, Math.floor(f));
      return target.copy(this.pts[i]).lerp(this.pts[i + 1], f - i);
    }
  }
  const glowTex = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,240,1)'); gr.addColorStop(0.25, 'rgba(255,200,90,.9)'); gr.addColorStop(1, 'rgba(255,140,30,0)');
    g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
    return new T.CanvasTexture(c);
  })();
  let sparks = [], makePath = () => [];
  function spark() {
    const pts = makePath();
    if (pts.length < 2) return;
    const curve = new Poly(pts), seg = pts.length * 3;
    const core = new T.Mesh(new T.TubeGeometry(curve, seg, 0.0048, 5, false), new T.MeshBasicMaterial({ color: 0xfff6d8, transparent: true }));
    const halo = new T.Mesh(new T.TubeGeometry(curve, seg, 0.014, 6, false), new T.MeshBasicMaterial({ color: 0xffa436, transparent: true, opacity: 0.5, depthWrite: false }));
    const head = new T.Sprite(new T.SpriteMaterial({ map: glowTex, transparent: true, depthWrite: false }));
    head.scale.setScalar(0.13);
    brain.add(halo, core, head);
    sparks.push({ curve, core, halo, head, t: 0, dur: 380 + 260 * Math.random() });
  }
  function stepSparks(dt) {
    sparks = sparks.filter(s => {
      s.t += dt;
      const u = Math.min(1, s.t / s.dur), fade = s.t < s.dur ? 1 : 1 - (s.t - s.dur) / 240;
      if (fade <= 0) {
        brain.remove(s.core, s.halo, s.head);
        s.core.geometry.dispose(); s.halo.geometry.dispose(); s.core.material.dispose(); s.halo.material.dispose(); s.head.material.dispose();
        return false;
      }
      for (const m of [s.core, s.halo]) m.geometry.setDrawRange(0, Math.floor(m.geometry.index.count * u / 6) * 6);
      s.core.material.opacity = fade; s.halo.material.opacity = 0.5 * fade;
      s.curve.getPoint(u, s.head.position);
      s.head.material.opacity = s.t < s.dur ? 1 : fade * 0.6;
      return true;
    });
  }

  // ---- interacción: arrastrar para girar, con inercia; al soltar vuelve a girar sola
  const AUTO = 0.32;
  let yaw = -0.75, pitch = 0.22, vel = AUTO, drag = false, lx = 0, ly = 0, last = 0, nextSpark = 300, vis = true;
  const render = () => { brain.rotation.set(pitch, yaw, 0); renderer.render(scene, camera); };
  canvas.addEventListener('pointerdown', e => { drag = true; lx = e.clientX; ly = e.clientY; vel = 0; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing'; });
  canvas.addEventListener('pointermove', e => {
    if (!drag) return;
    const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
    yaw += dx * 0.011; pitch = clamp(pitch + dy * 0.011, -1.1, 1.1);
    vel = clamp(dx * 0.011 * 60, -6, 6);
    if (reduced) render();
  });
  const release = () => { drag = false; canvas.style.cursor = 'grab'; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  const size = () => { const w = canvas.parentElement.clientWidth; renderer.setSize(w, w, false); render(); };
  const loop = now => {
    requestAnimationFrame(loop);
    if (!vis || document.hidden) { last = now; return; }
    const dt = Math.min(60, now - last || 16); last = now;
    if (!drag) {
      vel += (AUTO - vel) * (1 - Math.exp(-dt / 900));
      yaw += vel * dt / 1000;
    }
    if ((nextSpark -= dt) < 0) { spark(); if (Math.random() < 0.4) spark(); nextSpark = 200 + 480 * Math.random(); }
    stepSparks(dt);
    render();
  };
  size();
  loadModel().then(fn => { makePath = fn; }).catch(() => { makePath = procedural(); }).then(() => render());
  watchVisible(canvas, v => { vis = v; });
  if (!reduced) requestAnimationFrame(loop);
  return size;
}

/* ================= hero: traza en vivo ================= */
function hero() {
  const c = $('#heroTrace');
  const sim = simulate({ x0: [2.5, 3.1], nbrs: [[1], [0]], K: 0.4, T: 70000, xi: [-1.5, -1.5], zi: [2.5, 3.1], every: 40 });
  let p = reduced ? 1 : 0, vis = true, last;
  const draw = () => {
    const { g, w, h } = fit(c);
    g.clearRect(0, 0, w, h);
    const X = v => v / sim.T * w, Y = v => h * 0.5 - v * h * 0.26 - h * 0.08;
    const n = Math.max(2, Math.floor(p * sim.n));
    path(g, X, Y, sim.t, sim.x[1], 0, n); stroke(g, rgba(SAGE, .85), 1.6);
    path(g, X, Y, sim.t, sim.x[0], 0, n); stroke(g, C.accent, 2);
    if (p < 1) {
      dot(g, X(sim.t[n - 1]), Y(sim.x[1][n - 1]), 3.5, C.sage);
      dot(g, X(sim.t[n - 1]), Y(sim.x[0][n - 1]), 4.5, C.accent);
    }
  };
  const loop = now => {
    if (vis && p < 1) { last ??= now; p = Math.min(1, p + (now - last) / 9000); draw(); }
    last = now;
    if (p < 1) requestAnimationFrame(loop);
  };
  watchVisible(c, v => { vis = v; });
  draw();
  if (!reduced) requestAnimationFrame(loop);
  return draw;
}

/* ================= laboratorio N = 1 ================= */
function lab1() {
  const cx = $('#l1x'), cz = $('#l1z'), cp = $('#l1ph'), inp = $('#l1x0');
  const st = { x0: 2.5, sim: null, fps: [], p: 0, vis: false };
  const XR = [-2.4, 1.6], ZR = [1.5, 7];
  const nullX = [], nullZx = [];
  for (let k = 0; k <= 300; k++) { const x = XR[0] + (XR[1] - XR[0]) * k / 300; nullZx.push(x); nullX.push(cubic(x)); }

  function compute() {
    st.x0 = +inp.value;
    st.sim = simulate({ x0: [st.x0], nbrs: [[]], K: 0, T: 60000, xi: [-1.2], zi: [3.0], every: 30 });
    st.fps = fixedPoints(st.x0);
    const on = onsets(st.sim.t, st.sim.x[0]), P = meanDiff(on), osc = on.length >= 2;
    $('#l1read').innerHTML =
      `<div><dt>Estado</dt><dd class="${osc ? 'epi' : 'sano'}">${osc ? 'Crisis autónomas' : 'Reposo estable'}</dd></div>` +
      `<div><dt>Período \\(T\\)</dt><dd>${osc ? Math.round(P).toLocaleString('es-CL') : '—'}</dd></div>` +
      `<div><dt>Crisis en \\(T = 60\\,000\\)</dt><dd>${on.length}</dd></div>`;
    $('#l1fp tbody').innerHTML = st.fps.map(f =>
      `<tr><td>${f.x.toFixed(4)}</td><td>${f.z.toFixed(4)}</td><td>${f.re.toFixed(5)}</td><td><span class="tag ${f.stable ? 'ok' : 'no'}">${f.stable ? 'estable' : 'inestable'}</span></td></tr>`).join('');
    tex($('#l1read'));
    $$('#l1presets .btn').forEach(b => b.setAttribute('aria-pressed', String(Math.abs(+b.dataset.v - st.x0) < 1e-9)));
    draw();
  }
  function series(c, ys, yr, color) {
    const { g, w, h } = fit(c), s = st.sim, { X, Y } = frame(g, w, h, [0, s.T], yr);
    const n = Math.max(2, Math.floor(st.p * s.n));
    path(g, X, Y, s.t, ys); stroke(g, color.replace('1)', '.28)'), 1.3);
    path(g, X, Y, s.t, ys, 0, n); stroke(g, color, 1.8);
    g.strokeStyle = rgba([42, 32, 24], .25); g.lineWidth = 1; g.beginPath(); g.moveTo(X(s.t[n - 1]), 8); g.lineTo(X(s.t[n - 1]), h - 22); g.stroke();
    dot(g, X(s.t[n - 1]), Y(ys[n - 1]), 4, color, C.plate);
  }
  function draw() {
    if (!st.sim) return;
    const s = st.sim;
    series(cx, s.x[0], [-2.2, 1.6], rgba(ACC, 1));
    series(cz, s.z[0], [2.4, 4.6], 'rgba(197,138,42,1)');
    const { g, w, h } = fit(cp), { X, Y } = frame(g, w, h, XR, ZR, { xl: 'x', nx: 5, ny: 6 });
    g.save(); g.beginPath(); g.rect(36, 8, w - 46, h - 30); g.clip();
    path(g, X, Y, nullZx, nullX); stroke(g, C.ink, 1.6);
    path(g, X, Y, nullZx, nullZx.map(x => hEq8(x, st.x0))); stroke(g, C.sage, 1.6, [6, 5]);
    path(g, X, Y, s.x[0], s.z[0]); stroke(g, rgba(ACC, .22), 1.2);
    const n = Math.max(2, Math.floor(st.p * s.n)), a = Math.max(0, n - 900);
    path(g, X, Y, s.x[0], s.z[0], a, n); stroke(g, C.accent, 2.4);
    for (const f of st.fps) dot(g, X(f.x), Y(f.z), 5, f.stable ? C.ink : C.plate, C.ink);
    dot(g, X(s.x[0][n - 1]), Y(s.z[0][n - 1]), 6, C.accent, C.plate);
    g.restore();
    g.fillStyle = C.muted; g.textAlign = 'left'; g.font = MATHFONT; g.fillText('z', 42, 20);
  }
  ['2.0', '2.5', '2.8', '2.91', '3.0', '3.5'].forEach(v => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'btn ghost'; b.dataset.v = v; b.textContent = v;
    b.onclick = () => { inp.value = v; upd(); compute(); };
    $('#l1presets').append(b);
  });
  const upd = bindRange(inp, $('#l1x0o'), 3, rafThrottle(compute));
  let last;
  const loop = now => {
    if (st.vis) { last ??= now; st.p = (st.p + (now - last) / 16000) % 1; draw(); }
    last = now;
    requestAnimationFrame(loop);
  };
  watchVisible($('#lab1'), v => { st.vis = v; });
  compute();
  if (reduced) { st.p = 0.999; draw(); } else requestAnimationFrame(loop);
  return draw;
}

/* ================= laboratorio N = 2 ================= */
function classify(sim, x02) {
  const k0 = Math.floor(0.3 * sim.n);
  const on1 = onsets(sim.t, sim.x[0], -0.5, k0), on2 = onsets(sim.t, sim.x[1], -0.5, k0);
  const n1 = on1.length, n2 = on2.length, P1 = meanDiff(on1), P2 = meanDiff(on2);
  let reg;
  if (!n1 && !n2) reg = 'V';
  else if (!n2) reg = 'IV';
  else if (!n1) reg = '—';
  else {
    let sync = isFinite(P1) && isFinite(P2) && Math.abs(P1 - P2) / P1 < 0.01;
    if (sync) {
      const lags = [];
      for (const o of on1) { const d = on2.map(v => v - o).filter(v => v >= -0.5 * P1); if (d.length) lags.push(d[0]); }
      sync = lags.length >= 2 && (Math.max(...lags) - Math.min(...lags)) / P1 < 0.10;
    }
    reg = sync ? 'II' : x02 >= X0C ? 'III' : 'I';
  }
  return { reg, n1, n2, P1 };
}
function lab2() {
  const cx = $('#l2x'), cz = $('#l2z'), ia = $('#l2a'), ib = $('#l2b'), ik = $('#l2k'), pin = $('#l2pin');
  const st = { sim: null, p: 1, cur: null, stop: null };
  function compute(animate = false) {
    const a = +ia.value, b = +ib.value, K = +ik.value;
    st.sim = simulate({ x0: [a, b], nbrs: [[1], [0]], K, T: 60000, xi: [-1.5, -1.5], zi: [a, b], every: 30 });
    const r = classify(st.sim, b);
    $('#l2read').innerHTML =
      `<div><dt>Régimen</dt><dd class="epi">${r.reg}</dd></div>` +
      `<div><dt>Crisis región 1</dt><dd>${r.n1}</dd></div><div><dt>Crisis región 2</dt><dd>${r.n2}</dd></div>` +
      `<div><dt>Período foco \\(T_1\\)</dt><dd>${isFinite(r.P1) ? Math.round(r.P1).toLocaleString('es-CL') : '—'}</dd></div>`;
    tex($('#l2read'));
    // calibración del marcador sobre la figura del mapa (ejes en píxeles de la imagen 889×404)
    pin.style.left = (68 + (K + 0.05) / 4.1 * 814) / 889 * 100 + '%';
    pin.style.top = (35 + (clamp(b, 2, 4) - 2) / 2 * 315) / 404 * 100 + '%';
    pin.classList.toggle('off', Math.abs(a - 2.5) > 0.005);
    st.stop && st.stop();
    if (animate) st.stop = tween(900, u => { st.p = u; draw(); }); else { st.p = 1; draw(); }
  }
  function one(c, ys, yr, cols) {
    const { g, w, h } = fit(c), s = st.sim, { X, Y, m } = frame(g, w, h, [0, s.T], yr);
    const n = Math.max(2, Math.floor(st.p * s.n));
    ys.forEach((y, q) => { path(g, X, Y, s.t, y, 0, n); stroke(g, cols[q], 1.7, q ? [5, 3] : []); });
    if (st.cur != null) {
      const k = clamp(Math.round(st.cur * (s.n - 1)), 0, s.n - 1), px = X(s.t[k]);
      g.strokeStyle = rgba([42, 32, 24], .35); g.lineWidth = 1; g.beginPath(); g.moveTo(px, m.t); g.lineTo(px, h - m.b); g.stroke();
      ys.forEach((y, q) => dot(g, px, Y(y[k]), 4, cols[q], C.plate));
      g.fillStyle = C.ink; g.textAlign = px > w * 0.6 ? 'right' : 'left';
      g.fillText(`t=${Math.round(s.t[k])}  ${ys.map((y, q) => (q + 1) + ': ' + y[k].toFixed(2)).join('  ')}`, px + (px > w * 0.6 ? -8 : 8), m.t + 10);
    }
  }
  function draw() {
    if (!st.sim) return;
    one(cx, st.sim.x, [-2.2, 1.6], [C.accent, C.sage]);
    one(cz, st.sim.z, [2.2, 4.8], [C.accent, C.sage]);
  }
  [cx, cz].forEach(c => {
    c.addEventListener('pointermove', e => {
      const r = c.getBoundingClientRect();
      st.cur = clamp((e.clientX - r.left - 36) / (r.width - 46), 0, 1); draw();
    });
    c.addEventListener('pointerleave', () => { st.cur = null; draw(); });
  });
  const go = rafThrottle(() => compute());
  const u = [bindRange(ia, $('#l2ao'), 2, go), bindRange(ib, $('#l2bo'), 2, go), bindRange(ik, $('#l2ko'), 2, go)];
  [['I', 2.5, 2.2, 0.05], ['II', 2.5, 2.8, 1.5], ['III', 2.5, 3.1, 0.4], ['IV', 2.5, 3.6, 0.3], ['V', 2.5, 3.8, 2.5]].forEach(([n, a, b, K]) => {
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'btn ghost'; btn.textContent = n;
    btn.onclick = () => { ia.value = a; ib.value = b; ik.value = K; u.forEach(f => f()); compute(true); };
    $('#l2presets').append(btn);
  });
  compute(true);
  return draw;
}

/* ================= series compartidas: dibujo y cursor ================= */
function drawSeries(c, t, T, ys, yr, cols, o = {}) {
  const { g, w, h } = fit(c), { X, Y, m } = frame(g, w, h, [0, T], yr);
  const n = Math.max(2, Math.floor((o.p ?? 1) * t.length));
  g.save(); g.beginPath(); g.rect(m.l, m.t - 4, w - m.l - m.r, h - m.t - m.b + 8); g.clip();
  for (const y of o.ghost || []) { path(g, X, Y, t, y); stroke(g, rgba([118, 104, 90], .6), 1, [2, 3]); }
  for (const u of o.under || []) { path(g, X, Y, t, u.y, 0, n); stroke(g, u.color, 1.2); }
  ys.forEach((y, q) => { path(g, X, Y, t, y, 0, n); stroke(g, cols[q], o.lw ?? 1.7, o.dash && q ? [5, 3] : []); });
  g.restore();
  if (o.cur != null) {
    const k = clamp(Math.round(o.cur * (t.length - 1)), 0, t.length - 1), px = X(t[k]);
    g.strokeStyle = rgba([42, 32, 24], .35); g.lineWidth = 1; g.beginPath(); g.moveTo(px, m.t); g.lineTo(px, h - m.b); g.stroke();
    ys.forEach((y, q) => dot(g, px, Y(y[k]), 4, cols[q], C.plate));
    const right = px > w * 0.6;
    g.fillStyle = C.ink; g.textAlign = right ? 'right' : 'left';
    g.fillText(`t=${Math.round(t[k])}  ${ys.map((y, q) => (q + 1) + ': ' + y[k].toFixed(2)).join('  ')}`, px + (right ? -8 : 8), m.t + 10);
  }
}
function hoverSync(canvases, st, draw) {
  canvases.forEach(c => {
    c.addEventListener('pointermove', e => {
      const r = c.getBoundingClientRect();
      st.cur = clamp((e.clientX - r.left - 36) / (r.width - 46), 0, 1); draw();
    });
    c.addEventListener('pointerleave', () => { st.cur = null; draw(); });
  });
}
function span(arrs, pad = 0.2) {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrs) for (let k = 0; k < a.length; k += 4) { if (a[k] < lo) lo = a[k]; if (a[k] > hi) hi = a[k]; }
  return [lo - pad, hi + pad];
}
function presetButtons(host, items, onPick) {
  items.forEach(it => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'btn ghost'; b.innerHTML = it.label; b._v = it.v;
    b.onclick = () => onPick(it.v);
    host.append(b);
  });
}

/* ================= laboratorio x(t), z(t), h(x(t)) ================= */
function labH() {
  const cx = $('#lhx'), cz = $('#lhzc'), ch = $('#lhh'), ia = $('#lha'), ib = $('#lhb'), ik = $('#lhk');
  const st = { sim: null, h: [], p: 1, cur: null, stop: null, ic: 'rand', seed: 7, showZ: true, a: 2.5, b: 2.5 };
  function compute(animate = true) {
    const a = +ia.value, b = +ib.value, K = +ik.value;
    let xi = [-1.5, -1.5], zi = [a, b];
    if (st.ic === 'rand') { const r = mulberry32(st.seed); xi = [4 * (r() - .5), 4 * (r() - .5)]; zi = [3 + 2 * (r() - .5), 3 + 2 * (r() - .5)]; }
    st.sim = simulate({ x0: [a, b], nbrs: [[1], [0]], K, T: 30000, xi, zi, every: 15 });
    st.h = [st.sim.x[0].map(x => hEq8(x, a)), st.sim.x[1].map(x => hEq8(x, b))];
    st.a = a; st.b = b;
    $$('#lhpairs .btn').forEach(q => q.setAttribute('aria-pressed', String(Math.abs(q._v[0] - a) < 1e-9 && Math.abs(q._v[1] - b) < 1e-9)));
    $$('#lhks .btn').forEach(q => q.setAttribute('aria-pressed', String(Math.abs(q._v - K) < 1e-9)));
    st.stop && st.stop();
    if (animate) st.stop = tween(800, u => { st.p = u; draw(); }); else { st.p = 1; draw(); }
  }
  function draw() {
    if (!st.sim) return;
    const s = st.sim, cols = [C.accent, C.sage], o = { p: st.p, cur: st.cur, dash: true };
    drawSeries(cx, s.t, s.T, s.x, [-2.2, 1.6], cols, o);
    drawSeries(cz, s.t, s.T, s.z, span(s.z), cols, o);
    const lo = Math.min(st.a, st.b, ...(st.showZ ? [span(s.z, 0)[0]] : [])) - 0.4, hi = Math.max(st.a, st.b) + 3.4;
    drawSeries(ch, s.t, s.T, st.h, [lo, hi], cols, {
      ...o, lw: 2,
      under: st.showZ ? [{ y: s.z[0], color: rgba([197, 138, 42], .95) }, { y: s.z[1], color: rgba([197, 138, 42], .5) }] : [],
    });
  }
  const live = rafThrottle(() => compute(false));
  const ups = [bindRange(ia, $('#lhao'), 2, live), bindRange(ib, $('#lhbo'), 2, live), bindRange(ik, $('#lhko'), 2, live)];
  const sync = () => { ups.forEach(f => f()); compute(); };
  presetButtons($('#lhpairs'), [[2.5, 2.5], [2.8, 2.8], [2.0, 2.8], [2.0, 3.2]].map(v => ({ v, label: R`\((${v[0].toFixed(1)},\ ${v[1].toFixed(1)})\)` })),
    v => { ia.value = v[0]; ib.value = v[1]; sync(); });
  presetButtons($('#lhks'), [0, 0.2, 1.0].map(v => ({ v, label: R`\(K = ${v}\)` })), v => { ik.value = v; sync(); });
  $$('#lhic button').forEach(b => b.onclick = () => {
    $$('#lhic button').forEach(q => q.setAttribute('aria-pressed', String(q === b)));
    st.ic = b.dataset.v; compute();
  });
  $('#lhnew').onclick = () => {
    st.seed++; st.ic = 'rand';
    $$('#lhic button').forEach(q => q.setAttribute('aria-pressed', String(q.dataset.v === 'rand')));
    compute();
  };
  $('#lhz').onclick = e => { st.showZ = !st.showZ; e.currentTarget.setAttribute('aria-pressed', String(st.showZ)); draw(); };
  hoverSync([cx, cz, ch], st, draw);
  compute();
  return draw;
}

/* ================= laboratorio con ruido (Euler-Maruyama) ================= */
function labNoise() {
  const cx = $('#lnx'), cz = $('#lnz');
  const isx = $('#lnsx'), isz = $('#lnsz'), ik = $('#lnk'), ia = $('#lna'), ib = $('#lnb');
  const T = 50000, dt = 0.1, steps = 500000, every = 50, ns = steps / every + 1, sq = Math.sqrt(dt);
  const st = { seed: 11, W: null, wseed: -1, res: null, det: null, detKey: '', showDet: true, cur: null };

  // incrementos gaussianos fijos por semilla: al mover σ se deforma la misma realización
  function noise() {
    if (st.wseed === st.seed) return st.W;
    const r = mulberry32(st.seed * 7919 + 1), W = new Float32Array(4 * steps);
    for (let k = 0; k < W.length; k += 2) {
      const u = Math.sqrt(-2 * Math.log(1 - r())), v = 6.283185307179586 * r();
      W[k] = u * Math.cos(v); W[k + 1] = u * Math.sin(v);
    }
    st.W = W; st.wseed = st.seed;
    return W;
  }
  function em(a, b, K, sx, sz) {
    const W = noise(), t = new Float32Array(ns);
    const X1 = new Float32Array(ns), X2 = new Float32Array(ns), Z1 = new Float32Array(ns), Z2 = new Float32Array(ns);
    let x1 = -1.5, x2 = -1.5, z1 = a, z2 = b;
    X1[0] = x1; X2[0] = x2; Z1[0] = z1; Z2[0] = z2;
    const ax = sx * sq, az = sz * sq;
    for (let n = 1; n <= steps; n++) {
      const q = 4 * (n - 1);
      const d1 = (-x1 * x1 * x1 - 2 * x1 * x1 + 1 - z1 + I1) * dt + ax * W[q];
      const d2 = (-x2 * x2 * x2 - 2 * x2 * x2 + 1 - z2 + I1) * dt + ax * W[q + 1];
      const e1 = (hEq8(x1, a) - z1 - K * (x2 - x1)) / TAU0 * dt + az * W[q + 2];
      const e2 = (hEq8(x2, b) - z2 - K * (x1 - x2)) / TAU0 * dt + az * W[q + 3];
      x1 += d1; x2 += d2; z1 += e1; z2 += e2;
      if (n % every === 0) { const k = n / every; t[k] = n * dt; X1[k] = x1; X2[k] = x2; Z1[k] = z1; Z2[k] = z2; }
    }
    return { t, x: [X1, X2], z: [Z1, Z2] };
  }
  // conteo con histéresis, entra en crisis sobre −0.5 y sale bajo −1.0
  function crises(x) {
    let ict = x[0] > -0.5, n = 0, d = 0;
    for (let k = 1; k < x.length; k++) {
      if (!ict && x[k] > -0.5) { ict = true; n++; } else if (ict && x[k] < -1) ict = false;
      if (ict) d++;
    }
    return { n, duty: d / x.length };
  }
  function compute() {
    const a = +ia.value, b = +ib.value, K = +ik.value, sx = +isx.value, sz = +isz.value;
    const key = [a, b, K].join();
    if (key !== st.detKey) { st.det = em(a, b, K, 0, 0); st.detKey = key; }
    st.res = em(a, b, K, sx, sz);
    const c1 = crises(st.res.x[0]), c2 = crises(st.res.x[1]), d2 = crises(st.det.x[1]);
    const induced = c2.n > 0 && d2.n === 0;
    $('#lnread').innerHTML =
      `<div><dt>Crisis región 1</dt><dd>${c1.n}</dd></div><div><dt>Crisis región 2</dt><dd class="${c2.n ? 'epi' : 'sano'}">${c2.n}</dd></div>` +
      `<div><dt>Región 2 reclutada</dt><dd class="${c2.n ? 'epi' : 'sano'}">${c2.n ? (induced ? 'Sí, por ruido' : 'Sí') : 'No'}</dd></div>` +
      `<div><dt>Tiempo ictal región 2</dt><dd>${(100 * c2.duty).toFixed(1)} %</dd></div>`;
    draw();
  }
  function draw() {
    if (!st.res) return;
    const r = st.res, cols = [C.accent, C.sage];
    drawSeries(cx, r.t, T, r.x, [-2.5, 1.9], cols, { cur: st.cur, lw: 1.2, ghost: st.showDet ? st.det.x : [] });
    drawSeries(cz, r.t, T, r.z, span([...r.z, ...st.det.z]), cols, { cur: st.cur, lw: 1.3, ghost: st.showDet ? st.det.z : [] });
  }
  const go = rafThrottle(compute);
  bindRange(isx, $('#lnsxo'), 3, go); bindRange(isz, $('#lnszo'), 4, go);
  bindRange(ik, $('#lnko'), 2, go); bindRange(ia, $('#lnao'), 2, go); bindRange(ib, $('#lnbo'), 2, go);
  $('#lnnew').onclick = () => {
    st.seed++; st.detKey = ''; compute();
    if (M && !reduced) M.animate([cx, cz], { opacity: [0.35, 1] }, { duration: 0.18, ease: 'easeOut' });
  };
  $('#lndet').onclick = e => { st.showDet = !st.showDet; e.currentTarget.setAttribute('aria-pressed', String(st.showDet)); draw(); };
  hoverSync([cx, cz], st, draw);
  compute();
  return draw;
}

/* ================= laboratorio grafo dinámico ================= */
const NB_X0 = [2.3, 2.4, 3.1225, 3.4561, 3.2863, 3.3937, 3.166, 3.0363, 3.2328, 2.9383];
function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function lab3() {
  const cg = $('#g-graph'), cm = $('#g-mat'), ct = $('#g-ts'), tip = $('#g-tip'), busy = $('#g-busy');
  const iN = $('#g-n'), iF = $('#g-f'), iK = $('#g-k'), iS = $('#g-s'), iT = $('#g-time');
  const st = { dir: 'A', variant: 'eq8', sim: null, x0: [], nbrs: [], Ti: [], wmax: 0, K: 0.1, th: -0.5, fr: 0, playing: !reduced, vis: false, phase: 0, pos: [], hover: -1, ictal: -1 };

  function build() {
    const N = +iN.value, F = Math.min(+iF.value, N - 1), seed = +iS.value, K = +iK.value;
    let x0;
    if (N === 10 && F === 2 && seed === 42) x0 = NB_X0.slice();
    else { const r = mulberry32(seed), foci = [2.3, 2.4, 2.5, 2.2]; x0 = Array.from({ length: N }, (_, i) => i < F ? foci[i] : 2.9 + 0.6 * r()); }
    const nbrs = x0.map((xi, i) => x0.map((xj, j) => j).filter(j => j !== i && (st.dir === 'A' ? x0[j] < xi : x0[j] > xi)));
    const eq8 = st.variant === 'eq8';
    const sim = simulate(eq8
      ? { x0, nbrs, K, T: 20000, xi: x0.map(() => -1.5), zi: x0.slice(), h: hEq8, every: 100 }
      : { x0, nbrs, K, T: 5000, xi: x0.map(() => -1), zi: x0.map(() => 0), h: hNb, every: 25 });
    st.th = eq8 ? -0.5 : 1.0;
    // T_i: primer cruce sostenido (≥ 50 unidades de tiempo) sobre el umbral
    const Ti = sim.x.map(x => {
      let s = -1;
      for (let k = 0; k < sim.n; k++) {
        if (x[k] > st.th) { if (s < 0) s = k; if (sim.t[k] - sim.t[s] >= 50) return sim.t[s]; } else s = -1;
      }
      return Infinity;
    });
    let wmax = 0;
    for (let i = 0; i < N; i++) for (const j of nbrs[i]) for (let k = 0; k < sim.n; k += 2) wmax = Math.max(wmax, K * Math.abs(sim.x[j][k] - sim.x[i][k]));
    Object.assign(st, { sim, x0, nbrs, Ti, wmax, K, tipFor: -1 });
    iT.max = sim.n - 1;
    if (st.fr > sim.n - 1) st.fr = 0;

    const healthy = x0.map((v, i) => i).filter(i => x0[i] >= X0C), recd = healthy.filter(i => isFinite(Ti[i]));
    const firstH = recd.length ? Math.min(...recd.map(i => Ti[i])) : NaN;
    $('#g-read').innerHTML =
      `<div><dt>Aristas</dt><dd>${nbrs.reduce((a, b) => a + b.length, 0)}</dd></div>` +
      `<div><dt>Sanas reclutadas \\(R\\)</dt><dd class="epi">${healthy.length ? (recd.length / healthy.length).toFixed(2) : '—'}</dd></div>` +
      `<div><dt>Primera sana</dt><dd>${isFinite(firstH) ? R`\(t = ${Math.round(firstH)}\)` : '—'}</dd></div>` +
      `<div><dt>En crisis ahora</dt><dd id="g-now">0</dd></div>`;
    const order = x0.map((v, i) => i).sort((a, b) => Ti[a] - Ti[b] || a - b);
    $('#g-rec').innerHTML = order.map(i =>
      `<li data-i="${i}" class="${x0[i] < X0C ? 'epi' : ''}"><span class="dot">${i}</span><span>${R`\(${x0[i].toFixed(2)}\)`}</span>` +
      `<span class="bar"><i style="width:${isFinite(Ti[i]) ? Math.max(2, Ti[i] / sim.T * 100) : 0}%"></i></span>` +
      `<span>${isFinite(Ti[i]) ? R`\(${Math.round(Ti[i])}\)` : R`\(\infty\)`}</span></li>`).join('');
    $('#g-varnote').innerHTML = eq8
      ? R`\(h(x) = x_0 + 3\,\sigma\big(\tfrac{x+0.5}{0.1}\big)\), reposo inicial \(x_i(0) = -1.5,\ z_i(0) = x_{0,i}\). Crisis si \(x_i \gt -0.5\).`
      : R`\(h(x) = x_0 - 3 + 3\,\sigma\big(\tfrac{x-0.5}{0.1}\big)\), \(z_i(0) = 0\), en la variante del experimento base. Umbral \(x_i \gt 1.0\).`;
    ['#g-varnote', '#g-read', '#g-rec'].forEach(q => tex($(q)));
    st.tshown = -1;
    st.ictal = -1;
    draw();
  }
  let timer;
  const rebuild = () => {
    busy.classList.add('on');
    clearTimeout(timer);
    timer = setTimeout(() => { build(); busy.classList.remove('on'); }, 140);
  };

  function layout(w, h) {
    const N = st.x0.length, R = Math.min(w, h) / 2 - 36;
    st.pos = st.x0.map((_, i) => { const a = -Math.PI / 2 + i / N * 2 * Math.PI; return [w / 2 + R * Math.cos(a), h / 2 + R * Math.sin(a)]; });
    return N > 12 ? 13 : 17;
  }
  function drawGraph() {
    const { g, w, h } = fit(cg), s = st.sim, k = st.fr, N = st.x0.length, r = layout(w, h);
    g.clearRect(0, 0, w, h);
    const cxm = w / 2, cym = h / 2;
    for (let i = 0; i < N; i++) for (const j of st.nbrs[i]) {
      const W = st.K * Math.abs(s.x[j][k] - s.x[i][k]), u = st.wmax > 0 ? W / st.wmax : 0;
      const [x1, y1] = st.pos[j], [x2, y2] = st.pos[i];
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, qx = mx + (cxm - mx) * 0.22, qy = my + (cym - my) * 0.22;
      const d1 = Math.hypot(qx - x1, qy - y1) || 1, d2 = Math.hypot(qx - x2, qy - y2) || 1;
      const sx = x1 + (qx - x1) / d1 * r, sy = y1 + (qy - y1) / d1 * r, ex = x2 + (qx - x2) / d2 * (r + 5), ey = y2 + (qy - y2) / d2 * (r + 5);
      const hot = st.hover < 0 || st.hover === i || st.hover === j;
      const col = st.x0[j] < X0C ? ACC : [140, 112, 84], al = (0.07 + 0.83 * u) * (hot ? 1 : 0.15);
      g.beginPath(); g.moveTo(sx, sy); g.quadraticCurveTo(qx, qy, ex, ey);
      g.strokeStyle = rgba(col, al); g.lineWidth = 0.6 + 4.2 * u; g.stroke();
      const ang = Math.atan2(ey - qy, ex - qx), al2 = 4 + 5 * u;
      g.beginPath(); g.moveTo(ex, ey); g.lineTo(ex - al2 * Math.cos(ang - 0.45), ey - al2 * Math.sin(ang - 0.45)); g.lineTo(ex - al2 * Math.cos(ang + 0.45), ey - al2 * Math.sin(ang + 0.45)); g.closePath();
      g.fillStyle = rgba(col, Math.min(1, al + 0.1)); g.fill();
      if (u > 0.25 && hot && !reduced) {           // pulso que viaja de j hacia i
        const v = (st.phase * (0.5 + u) + (i * 7 + j * 3) % 10 / 10) % 1, a = 1 - v;
        dot(g, a * a * sx + 2 * a * v * qx + v * v * ex, a * a * sy + 2 * a * v * qy + v * v * ey, 1.5 + 2 * u, rgba(col, 0.9));
      }
    }
    let now = 0;
    for (let i = 0; i < N; i++) {
      const [x, y] = st.pos[i], xv = s.x[i][k], act = clamp((xv + 1.4) / 1.7, 0, 1), epi = st.x0[i] < X0C, ict = xv > st.th;
      if (ict) {
        now++;
        g.beginPath(); g.arc(x, y, r + 6 + (reduced ? 0 : 2 * Math.sin(st.phase * 12 + i)), 0, 6.2832);
        g.strokeStyle = rgba(ACC, .35); g.lineWidth = 2; g.stroke();
      }
      g.beginPath(); g.arc(x, y, r, 0, 6.2832);
      g.fillStyle = mix(CREAM, ACC, act); g.fill();
      g.lineWidth = st.hover === i ? 4 : 3; g.strokeStyle = epi ? C.accent : C.sage; g.stroke();
      g.fillStyle = act > 0.55 ? '#fff' : C.ink; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.font = `${r > 14 ? 15.5 : 13}px KaTeX_Main, "Times New Roman", serif`; g.fillText(i, x, y + 0.5); g.textBaseline = 'alphabetic';
    }
    if (now !== st.ictal) { st.ictal = now; const el = $('#g-now'); if (el) el.textContent = `${now} / ${N}`; }
  }
  function drawMat() {
    const { g, w, h } = fit(cm), s = st.sim, k = st.fr, N = st.x0.length;
    g.clearRect(0, 0, w, h);
    const size = Math.min(w - 30, h - 22), cell = size / N, ox = (w - size) / 2 + 8, oy = 16;
    g.font = '12.5px KaTeX_Main, "Times New Roman", serif'; g.fillStyle = C.muted; g.textAlign = 'center';
    for (let i = 0; i < N; i++) {
      if (N <= 12) { g.fillStyle = C.muted; g.fillText(i, ox + (i + .5) * cell, 11); g.fillText(i, ox - 9, oy + (i + .5) * cell + 3); }
      const set = new Set(st.nbrs[i]);
      for (let j = 0; j < N; j++) {
        const u = set.has(j) && st.wmax > 0 ? st.K * Math.abs(s.x[j][k] - s.x[i][k]) / st.wmax : 0;
        g.fillStyle = set.has(j) ? mix([240, 230, 214], ACC, u) : '#faf5ec';
        g.fillRect(ox + j * cell + .5, oy + i * cell + .5, cell - 1, cell - 1);
      }
    }
  }
  function drawTs() {
    const { g, w, h } = fit(ct), s = st.sim, N = st.x0.length;
    const { X, Y, m } = frame(g, w, h, [0, s.T], [-2.2, 1.6]);
    g.strokeStyle = rgba([42, 32, 24], .3); g.setLineDash([3, 4]); g.beginPath(); g.moveTo(m.l, Y(st.th)); g.lineTo(w - 10, Y(st.th)); g.stroke(); g.setLineDash([]);
    for (let i = N - 1; i >= 0; i--) {
      const epi = st.x0[i] < X0C, on = st.hover < 0 || st.hover === i;
      path(g, X, Y, s.t, s.x[i]); stroke(g, rgba(epi ? ACC : SAGE, on ? (epi ? 1 : .6) : .12), epi || st.hover === i ? 1.9 : 1.1);
    }
    const px = X(s.t[st.fr]);
    g.strokeStyle = C.ink; g.lineWidth = 1.5; g.beginPath(); g.moveTo(px, m.t); g.lineTo(px, h - m.b); g.stroke();
    dot(g, px, m.t, 4, C.ink);
  }
  function syncDom() {
    const t = st.sim.t[st.fr];
    iT.value = st.fr;
    iT.style.setProperty('--fill', st.fr / (st.sim.n - 1) * 100 + '%');
    if (st.tshown !== st.fr) {
      st.tshown = st.fr;
      const o = $('#g-tout');
      if (window.katex) katex.render(`t = ${Math.round(t)}`, o, { throwOnError: false }); else o.textContent = 't = ' + Math.round(t);
    }
    $$('#g-rec li').forEach(li => {
      const on = st.Ti[+li.dataset.i] <= t, was = li.classList.contains('on');
      if (on !== was) { li.classList.toggle('on', on); li.classList.toggle('pop', on && st.playing); }
    });
  }
  function draw() { if (!st.sim) return; drawGraph(); drawMat(); drawTs(); syncDom(); }

  const setPlay = v => {
    st.playing = v;
    $('#g-play').classList.toggle('is-playing', v);
  };
  let last, acc = 0;
  const loop = now => {
    if (st.vis && st.sim) {
      last ??= now;
      const dtm = Math.min(60, now - last);
      st.phase += dtm / 1000;
      if (st.playing) {
        acc += dtm / 16000 * (st.sim.n - 1);
        if (acc >= 1) { st.fr = (st.fr + Math.floor(acc)) % st.sim.n; acc %= 1; }
      }
      if (st.playing || !reduced) draw();
    }
    last = now;
    requestAnimationFrame(loop);
  };

  bindRange(iN, $('#g-no'), 0, rebuild); bindRange(iF, $('#g-fo'), 0, rebuild);
  bindRange(iK, $('#g-ko'), 3, rebuild); bindRange(iS, $('#g-so'), 0, rebuild);
  const seg = (id, key) => $$(`#${id} button`).forEach(b => b.onclick = () => {
    $$(`#${id} button`).forEach(o => o.setAttribute('aria-pressed', String(o === b)));
    st[key] = b.dataset.v; st.fr = 0; rebuild();
  });
  seg('g-dir', 'dir'); seg('g-var', 'variant');
  $('#g-play').onclick = () => setPlay(!st.playing);
  iT.addEventListener('input', () => { st.fr = +iT.value; setPlay(false); draw(); });
  const seek = e => { const r = ct.getBoundingClientRect(); st.fr = Math.round(clamp((e.clientX - r.left - 36) / (r.width - 46), 0, 1) * (st.sim.n - 1)); setPlay(false); draw(); };
  ct.addEventListener('pointerdown', e => { seek(e); ct.setPointerCapture(e.pointerId); ct.onpointermove = seek; });
  ct.addEventListener('pointerup', () => { ct.onpointermove = null; });
  cg.addEventListener('pointermove', e => {
    const b = cg.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
    let hit = -1;
    st.pos.forEach((p, i) => { if (Math.hypot(p[0] - x, p[1] - y) < 22) hit = i; });
    st.hover = hit;
    if (hit >= 0) {
      const T = st.Ti[hit];
      if (st.tipFor !== hit) {
        st.tipFor = hit;
        tip.innerHTML = `nodo ${hit} · ${R`\(x_{0,${hit}} = ${st.x0[hit].toFixed(3)}\)`}<br>recibe de ${st.nbrs[hit].length} · influye sobre ${st.nbrs.filter(n => n.includes(hit)).length}<br>${isFinite(T) ? R`\(T_{${hit}} = ${Math.round(T)}\)` : 'no reclutado'}`;
        tex(tip);
      }
      tip.style.left = clamp(x + 16, 0, b.width - 190) + 'px'; tip.style.top = (y + 34) + 'px';
    }
    tip.classList.toggle('on', hit >= 0);
    cg.style.cursor = hit >= 0 ? 'pointer' : 'default';
    if (reduced || !st.playing) draw();
  });
  cg.addEventListener('pointerleave', () => { st.hover = -1; tip.classList.remove('on'); draw(); });
  watchVisible($('#lab3'), v => { st.vis = v; });
  setPlay(st.playing);
  build();
  if (reduced) { st.fr = Math.floor(st.sim.n * 0.3); draw(); }
  requestAnimationFrame(loop);
  return draw;
}

/* ================= grafo dinámico 3D (Three.js) ================= */
function lab3d() {
  let onResize = () => {};
  loadThree().then(T => { onResize = buildGraph3D(T); }).catch(() => { $('#lab3d').hidden = true; });
  return () => onResize();
}

// disposición de resorte en el plano (Fruchterman-Reingold), equivalente al spring_layout usado para las figuras
function springLayout(N, edges, seed) {
  const r = mulberry32(seed), P = Array.from({ length: N }, () => [r() * 2 - 1, r() * 2 - 1]), k = Math.sqrt(4 / N);
  for (let it = 0, temp = 0.2; it < 120; it++, temp *= 0.97) {
    const D = P.map(() => [0, 0]);
    for (let a = 0; a < N; a++) for (let b = a + 1; b < N; b++) {
      const dx = P[a][0] - P[b][0], dy = P[a][1] - P[b][1], d = Math.hypot(dx, dy) || 1e-3, f = k * k / d;
      D[a][0] += dx / d * f; D[a][1] += dy / d * f; D[b][0] -= dx / d * f; D[b][1] -= dy / d * f;
    }
    for (const [a, b] of edges) {
      const dx = P[a][0] - P[b][0], dy = P[a][1] - P[b][1], d = Math.hypot(dx, dy) || 1e-3, f = d * d / k;
      D[a][0] -= dx / d * f; D[a][1] -= dy / d * f; D[b][0] += dx / d * f; D[b][1] += dy / d * f;
    }
    for (let a = 0; a < N; a++) {
      const d = Math.hypot(D[a][0], D[a][1]) || 1e-3, s = Math.min(d, temp) / d;
      P[a][0] += D[a][0] * s; P[a][1] += D[a][1] * s;
    }
  }
  // se estira cada eje para ocupar todo el plano: en este grafo casi todos los pares están conectados y el resorte los apiña
  const ext = q => { const v = P.map(p => p[q]), lo = Math.min(...v), hi = Math.max(...v); return [lo, hi - lo || 1]; };
  const [x0, wx] = ext(0), [y0, wy] = ext(1);
  return P.map(p => [(p[0] - x0) / wx * 2 - 1, (p[1] - y0) / wy * 2 - 1]);
}

// ---- paneles planos que acompañan al grafo 3D: los mismos de la versión 2D
function drawWMatrix(c, st) {
  const { g, w, h } = fit(c), s = st.sim, k = st.fr, N = st.x0.length;
  g.clearRect(0, 0, w, h);
  const size = Math.min(w - 30, h - 22), cell = size / N, ox = (w - size) / 2 + 8, oy = 16;
  g.font = '12.5px KaTeX_Main, "Times New Roman", serif'; g.fillStyle = C.muted; g.textAlign = 'center';
  for (let i = 0; i < N; i++) {
    if (N <= 12) { g.fillStyle = C.muted; g.fillText(i, ox + (i + .5) * cell, 11); g.fillText(i, ox - 9, oy + (i + .5) * cell + 3); }
    const set = new Set(st.nbrs[i]);
    for (let j = 0; j < N; j++) {
      const u = set.has(j) && st.wmax > 0 ? st.K * Math.abs(s.x[j][k] - s.x[i][k]) / st.wmax : 0;
      g.fillStyle = set.has(j) ? mix([240, 230, 214], ACC, u) : '#faf5ec';
      if (st.hover === i || st.hover === j) g.globalAlpha = 1; else g.globalAlpha = st.hover < 0 ? 1 : 0.35;
      g.fillRect(ox + j * cell + .5, oy + i * cell + .5, cell - 1, cell - 1);
    }
  }
  g.globalAlpha = 1;
}
function drawNodeSeries(c, st) {
  const { g, w, h } = fit(c), s = st.sim, N = st.x0.length;
  const { X, Y, m } = frame(g, w, h, [0, s.T], [-2.2, 1.6]);
  g.strokeStyle = rgba([42, 32, 24], .3); g.setLineDash([3, 4]); g.beginPath(); g.moveTo(m.l, Y(st.th)); g.lineTo(w - 10, Y(st.th)); g.stroke(); g.setLineDash([]);
  for (let i = N - 1; i >= 0; i--) {
    const epi = st.x0[i] < X0C, on = st.hover < 0 || st.hover === i;
    path(g, X, Y, s.t, s.x[i]); stroke(g, rgba(epi ? ACC : SAGE, on ? (epi ? 1 : .6) : .12), epi || st.hover === i ? 1.9 : 1.1);
  }
  const px = X(s.t[st.fr]);
  g.strokeStyle = C.ink; g.lineWidth = 1.5; g.beginPath(); g.moveTo(px, m.t); g.lineTo(px, h - m.b); g.stroke();
  dot(g, px, m.t, 4, C.ink);
}
function recruitTimes(sim, th) {            // primer cruce sostenido (≥ 50 unidades de tiempo) sobre el umbral
  return sim.x.map(x => {
    let s = -1;
    for (let k = 0; k < sim.n; k++) {
      if (x[k] > th) { if (s < 0) s = k; if (sim.t[k] - sim.t[s] >= 50) return sim.t[s]; } else s = -1;
    }
    return Infinity;
  });
}
function recruitList(el, st) {
  const { x0, Ti, sim } = st, order = x0.map((v, i) => i).sort((a, b) => Ti[a] - Ti[b] || a - b);
  el.innerHTML = order.map(i =>
    `<li data-i="${i}" class="${x0[i] < X0C ? 'epi' : ''}"><span class="dot">${i}</span><span>${R`\(${x0[i].toFixed(2)}\)`}</span>` +
    `<span class="bar"><i style="width:${isFinite(Ti[i]) ? Math.max(2, Ti[i] / sim.T * 100) : 0}%"></i></span>` +
    `<span>${isFinite(Ti[i]) ? R`\(${Math.round(Ti[i])}\)` : R`\(\infty\)`}</span></li>`).join('');
  tex(el);
}

function buildGraph3D(T) {
  const canvas = $('#g3-canvas'), stage = $('#g3-stage'), tip = $('#g3-tip'), busy = $('#g3-busy'), cm = $('#g3-mat'), ct = $('#g3-ts'), rec = $('#g3-rec');
  const iN = $('#g3-n'), iF = $('#g3-f'), iK = $('#g3-k'), iS = $('#g3-s'), iT = $('#g3-time'), iL = $('#g3-layers'), iP = $('#g3-pct');
  const axLab = $$('#g3-labels > span');
  const S = { x: 1.35, y: 0.9, z: 1.05 }, TRAIL = 220, VIEW = { yaw: -0.85, pitch: 0.36, dist: 5.6 };
  const st = { mode: 'estructura', dir: 'A', variant: 'eq8', sim: null, x0: [], nbrs: [], edges: [], Ti: [], lay: [], fr: 0, playing: !reduced, trails: true, spin: !reduced,
    ...VIEW, hover: -1, wmax: 0, K: 0.1, th: -0.5, xr: [-2, 1], zr: [2, 5], x0r: [2, 4], vis: false, blend: 1, from: null, tipKey: '', tshown: -1, drag: false, phase: 0, fit: 1 };

  const renderer = new T.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(34, 1, 0.1, 60);
  scene.add(new T.HemisphereLight(0xfff6ec, 0xa08468, 1.5));
  const sun = new T.DirectionalLight(0xffffff, 1.5);
  sun.position.set(2, 4, 3);
  scene.add(sun);

  // caja de ejes y rejilla del piso
  scene.add(new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(2 * S.x, 2 * S.y, 2 * S.z)), new T.LineBasicMaterial({ color: 0xcbb9a0 })));
  const gridPts = [];
  for (let q = 1; q < 6; q++) {
    const u = -1 + q / 3;
    gridPts.push(u * S.x, -S.y, -S.z, u * S.x, -S.y, S.z, -S.x, -S.y, u * S.z, S.x, -S.y, u * S.z);
  }
  const grid = new T.LineSegments(new T.BufferGeometry(), new T.LineBasicMaterial({ color: 0xe2d6c3 }));
  grid.geometry.setAttribute('position', new T.Float32BufferAttribute(gridPts, 3));
  scene.add(grid);

  // mismos colores que la versión 2D: borde por tipo de región, relleno por actividad, arista por origen
  const EPI = new T.Color('#b4502c'), SANO = new T.Color('#66754b'), CREAMc = new T.Color('#fffdf9'), BROWN = new T.Color('rgb(140,112,84)');
  const sphereGeo = new T.SphereGeometry(1, 24, 16), cylGeo = new T.CylinderGeometry(1, 1, 1, 8, 1, true), coneGeo = new T.ConeGeometry(1, 1, 10);
  const ringTex = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 128;
    const g = c.getContext('2d');
    g.strokeStyle = '#fff'; g.lineWidth = 13; g.beginPath(); g.arc(64, 64, 54, 0, 6.2832); g.stroke();
    return new T.CanvasTexture(c);
  })();
  const shared = new Set([sphereGeo, cylGeo, coneGeo]);
  const gDyn = new T.Group(), gLay = new T.Group();
  scene.add(gDyn, gLay);
  const clear = g => {
    for (const o of [...g.children]) {
      g.remove(o);
      if (o.geometry && !shared.has(o.geometry)) o.geometry.dispose();
      if (o.material) { if (o.material.map && o.material.map !== ringTex) o.material.map.dispose(); o.material.dispose(); }
    }
  };
  const ring = (color, opacity = 1) => new T.Sprite(new T.SpriteMaterial({ map: ringTex, color, transparent: true, opacity, depthWrite: false }));
  let nodes = [], rings = [], halos = [], labels = [], edgeObj = [], trails = [], plane = null;
  const P = [], UP = new T.Vector3(0, 1, 0), d = new T.Vector3(), dn = new T.Vector3(), v3 = new T.Vector3(), col = new T.Color();
  const nrm = (v, r) => (v - r[0]) / (r[1] - r[0]) * 2 - 1, activity = x => clamp((x + 1.4) / 1.7, 0, 1);

  function numSprite(n) {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d');
    g.font = '42px KaTeX_Main, "Times New Roman", serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = '#2a2018';
    g.fillText(n, 32, 34);
    const s = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(c), transparent: true, depthWrite: false }));
    s.scale.setScalar(0.2);
    return s;
  }

  // ---- simulación y objetos
  function build() {
    const N = +iN.value, F = Math.min(+iF.value, N - 1), seed = +iS.value, K = +iK.value, eq8 = st.variant === 'eq8';
    let x0 = NB_X0.slice();
    if (!(N === 10 && F === 2 && seed === 42)) { const r = mulberry32(seed), foci = [2.3, 2.4, 2.5, 2.2]; x0 = Array.from({ length: N }, (_, i) => i < F ? foci[i] : 2.9 + 0.6 * r()); }
    const nbrs = x0.map((xi, i) => x0.map((_, j) => j).filter(j => j !== i && (st.dir === 'A' ? x0[j] < xi : x0[j] > xi)));
    const sim = simulate(eq8
      ? { x0, nbrs, K, T: 20000, xi: x0.map(() => -1.5), zi: x0.slice(), h: hEq8, every: 100 }
      : { x0, nbrs, K, T: 5000, xi: x0.map(() => -1), zi: x0.map(() => 0), h: hNb, every: 25 });
    const edges = [], th = eq8 ? -0.5 : 1.0;
    nbrs.forEach((nb, i) => nb.forEach(j => edges.push([i, j])));          // [i, j]: j influye sobre i
    let wmax = 0;
    for (const [i, j] of edges) for (let k = 0; k < sim.n; k += 2) wmax = Math.max(wmax, K * Math.abs(sim.x[j][k] - sim.x[i][k]));
    const Ti = recruitTimes(sim, th);
    Object.assign(st, { sim, x0, nbrs, edges, Ti, wmax, K, th, xr: span(sim.x, 0.15), zr: span(sim.z, 0.15),
      x0r: [Math.min(...x0) - 0.1, Math.max(...x0) + 0.1], lay: springLayout(N, edges, 42), blend: 1, from: null, tshown: -1, tipKey: '' });
    if (st.fr > sim.n - 1) st.fr = 0;
    iT.max = sim.n - 1;

    clear(gDyn);
    const r = N > 12 ? 0.064 : 0.078;
    nodes = x0.map((v, i) => {
      const m = new T.Mesh(sphereGeo, new T.MeshStandardMaterial({ color: CREAMc.clone(), roughness: 0.5 }));
      m.userData = { i, r }; m.scale.setScalar(r);
      gDyn.add(m);
      return m;
    });
    rings = x0.map(v => { const s = ring(v < X0C ? EPI : SANO); s.scale.setScalar(r * 3.1); gDyn.add(s); return s; });
    halos = x0.map(() => { const s = ring(EPI, 0.35); gDyn.add(s); return s; });
    labels = x0.map((_, i) => { const s = numSprite(i); gDyn.add(s); return s; });
    edgeObj = edges.map(([, j]) => {
      const mat = new T.MeshBasicMaterial({ color: x0[j] < X0C ? EPI : BROWN, transparent: true, depthWrite: false });
      const cyl = new T.Mesh(cylGeo, mat), cone = new T.Mesh(coneGeo, mat), pulse = new T.Mesh(sphereGeo, new T.MeshBasicMaterial({ color: x0[j] < X0C ? EPI : BROWN }));
      gDyn.add(cyl, cone, pulse);
      return { cyl, cone, pulse, mat };
    });
    trails = x0.map(v => {
      const geo = new T.BufferGeometry();
      geo.setAttribute('position', new T.BufferAttribute(new Float32Array(TRAIL * 3), 3));
      const line = new T.Line(geo, new T.LineBasicMaterial({ color: v < X0C ? EPI : SANO, transparent: true, opacity: 0.5 }));
      line.frustumCulled = false;
      gDyn.add(line);
      return line;
    });
    P.length = 0;
    x0.forEach(() => P.push(new T.Vector3()));

    const healthy = x0.map((v, i) => i).filter(i => x0[i] >= X0C), recd = healthy.filter(i => isFinite(Ti[i]));
    const firstH = recd.length ? Math.min(...recd.map(i => Ti[i])) : NaN;
    $('#g3-read').innerHTML =
      `<div><dt>Aristas</dt><dd>${edges.length}</dd></div>` +
      `<div><dt>Sanas reclutadas ${R`\(R\)`}</dt><dd class="epi">${healthy.length ? (recd.length / healthy.length).toFixed(2) : '—'}</dd></div>` +
      `<div><dt>Primera sana</dt><dd>${isFinite(firstH) ? R`\(t = ${Math.round(firstH)}\)` : '—'}</dd></div>` +
      `<div><dt>En crisis ahora</dt><dd id="g3-now">0</dd></div>`;
    tex($('#g3-read'));
    recruitList(rec, st);
    buildLayers();
  }

  function target(i, k, mode, out) {
    const s = st.sim;
    if (mode === 'fases') return out.set(nrm(s.x[i][k], st.xr) * S.x, nrm(st.x0[i], st.x0r) * S.y, nrm(s.z[i][k], st.zr) * S.z);
    return out.set(st.lay[i][0] * S.x * 0.88, nrm(st.x0[i], st.x0r) * S.y, st.lay[i][1] * S.z * 0.88);
  }

  // ---- capas temporales: el grafo en n instantes, apilado según t
  function buildLayers() {
    clear(gLay);
    const s = st.sim, N = st.x0.length, n = +iL.value, pct = +iP.value, r = N > 12 ? 0.046 : 0.054;
    const ks = Array.from({ length: n }, (_, q) => Math.round(q * (s.n - 1) / (n - 1))), yOf = k => (k / (s.n - 1) * 2 - 1) * S.y;
    const xy = i => [st.lay[i][0] * S.x * 0.88, st.lay[i][1] * S.z * 0.88], seg = [], vert = [];
    for (const k of ks) {
      for (let i = 0; i < N; i++) {
        const m = new T.Mesh(sphereGeo, new T.MeshStandardMaterial({ color: CREAMc.clone().lerp(EPI, activity(s.x[i][k])), roughness: 0.5 }));
        m.position.set(xy(i)[0], yOf(k), xy(i)[1]); m.scale.setScalar(r);
        const b = ring(st.x0[i] < X0C ? EPI : SANO);
        b.position.copy(m.position); b.scale.setScalar(r * 3.1);
        gLay.add(m, b);
      }
      const ws = st.edges.map(([i, j]) => st.K * Math.abs(s.x[j][k] - s.x[i][k])), sorted = [...ws].sort((a, b) => a - b);
      const pos = pct / 100 * (sorted.length - 1), lo = Math.floor(pos), cut = sorted.length ? sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (pos - lo) : 0;
      st.edges.forEach(([i, j], q) => { if (ws[q] >= cut) seg.push(xy(j)[0], yOf(k), xy(j)[1], xy(i)[0], yOf(k), xy(i)[1]); });
    }
    for (let i = 0; i < N; i++) vert.push(xy(i)[0], -S.y, xy(i)[1], xy(i)[0], S.y, xy(i)[1]);
    const mk = (pts, color, opacity) => {
      const l = new T.LineSegments(new T.BufferGeometry(), new T.LineBasicMaterial({ color, transparent: true, opacity }));
      l.geometry.setAttribute('position', new T.Float32BufferAttribute(pts, 3));
      gLay.add(l);
    };
    mk(seg, 0x8c7a66, 0.55); mk(vert, 0xcbb9a0, 0.8);
    plane = new T.Mesh(new T.PlaneGeometry(2 * S.x, 2 * S.z), new T.MeshBasicMaterial({ color: EPI, transparent: true, opacity: 0.1, side: T.DoubleSide, depthWrite: false }));
    plane.rotation.x = -Math.PI / 2;
    gLay.add(plane);
  }

  // ---- estado del grafo en el cuadro k
  function update(k) {
    const s = st.sim, N = st.x0.length;
    gDyn.visible = st.mode !== 'capas'; gLay.visible = st.mode === 'capas';
    if (st.mode === 'capas') plane.position.y = (k / (s.n - 1) * 2 - 1) * S.y;
    let now = 0;
    for (let i = 0; i < N; i++) {
      target(i, k, st.mode === 'capas' ? 'estructura' : st.mode, P[i]);
      if (st.blend < 1 && st.from) P[i].lerpVectors(st.from[i], P[i], st.blend);
      const xv = s.x[i][k], ict = xv > st.th, m = nodes[i], r = m.userData.r;
      if (ict) now++;
      m.position.copy(P[i]); m.material.color.copy(CREAMc).lerp(EPI, activity(xv));
      rings[i].position.copy(P[i]); rings[i].scale.setScalar(r * (st.hover === i ? 3.7 : 3.1));
      halos[i].visible = ict; halos[i].position.copy(P[i]); halos[i].scale.setScalar(r * (4.5 + (reduced ? 0 : 0.35 * Math.sin(st.phase * 12 + i))));
      labels[i].position.copy(P[i]).y += r * 2.9;
    }
    st.edges.forEach(([i, j], q) => {
      const o = edgeObj[q], a = P[j], b = P[i], u = st.wmax > 0 ? st.K * Math.abs(s.x[j][k] - s.x[i][k]) / st.wmax : 0;
      const hot = st.hover < 0 || st.hover === i || st.hover === j;
      d.subVectors(b, a);
      const len = d.length(), ok = len > 0.02;
      o.cyl.visible = o.cone.visible = ok; o.pulse.visible = ok && u > 0.25 && hot && !reduced;
      if (!ok) return;
      dn.copy(d).divideScalar(len);
      const rad = 0.004 + 0.022 * u, hl = Math.min(0.1, len * 0.4);
      o.cyl.position.copy(a).addScaledVector(d, 0.5); o.cyl.quaternion.setFromUnitVectors(UP, dn); o.cyl.scale.set(rad, len, rad);
      o.cone.position.copy(a).addScaledVector(d, 0.8); o.cone.quaternion.copy(o.cyl.quaternion); o.cone.scale.set(rad * 2.4 + 0.012, hl, rad * 2.4 + 0.012);
      o.mat.opacity = (0.07 + 0.83 * u) * (hot ? 1 : 0.15);
      if (o.pulse.visible) {                      // pulso que viaja de j hacia i
        const v = (st.phase * (0.5 + u) + (i * 7 + j * 3) % 10 / 10) % 1;
        o.pulse.position.copy(a).addScaledVector(d, v); o.pulse.scale.setScalar(0.016 + 0.02 * u);
      }
    });
    const showTrails = st.trails && st.mode === 'fases' && st.blend >= 1;
    trails.forEach((line, i) => {
      line.visible = showTrails;
      if (!showTrails) return;
      const arr = line.geometry.attributes.position.array, k0 = Math.max(0, k - TRAIL + 1);
      for (let q = k0, m = 0; q <= k; q++, m += 3) { target(i, q, 'fases', v3); arr[m] = v3.x; arr[m + 1] = v3.y; arr[m + 2] = v3.z; }
      line.geometry.setDrawRange(0, k - k0 + 1);
      line.geometry.attributes.position.needsUpdate = true;
    });
    const nowEl = $('#g3-now');
    if (nowEl) nowEl.textContent = `${now} / ${N}`;
  }

  const AXES = {
    estructura: ['disposición 1', 'disposición 2', R`\(x_{0,i}\)`],
    capas: ['disposición 1', 'disposición 2', R`\(t\)`],
    fases: [R`\(x_i(t)\)`, R`\(z_i(t)\)`, R`\(x_{0,i}\)`],
  };
  const NOTES = {
    estructura: R`La altura es \(x_{0,i}\): los focos quedan abajo y las regiones sanas arriba.`,
    capas: 'Cada piso es el grafo en un instante. El plano marca el tiempo actual.',
    fases: R`Cada nodo está en \(\varphi_t(i) = (x_i(t),\ z_i(t),\ x_{0,i})\) y se mueve con la dinámica.`,
  };
  const ANCH = [new T.Vector3(0, -S.y, S.z + 0.3), new T.Vector3(S.x + 0.35, -S.y, 0), new T.Vector3(-S.x - 0.28, 0.1, -S.z)];
  function setMode(mode) {
    if (mode !== 'capas' && st.mode !== 'capas' && mode !== st.mode && st.sim) {      // los nodos viajan de una vista a la otra
      st.from = P.map(p => p.clone()); st.blend = 0;
      tween(650, u => { st.blend = u; if (reduced) draw(); });
    }
    st.mode = mode;
    $('#g3-modenote').innerHTML = NOTES[mode];
    axLab.forEach((el, q) => { el.innerHTML = AXES[mode][q]; });
    [$('#g3-modenote'), $('#g3-labels')].forEach(tex);
    $('#g3-capas').hidden = mode !== 'capas';
    $('#g3-trails').disabled = mode !== 'fases';
    st.hover = -1; tip.classList.remove('on');
    draw();
  }

  function draw() {
    if (!st.sim) return;
    update(st.fr);
    const cp = Math.cos(st.pitch);
    const dist = st.dist * st.fit;
    camera.position.set(dist * cp * Math.sin(st.yaw), dist * Math.sin(st.pitch), dist * cp * Math.cos(st.yaw));
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    const w = stage.clientWidth, h = stage.clientHeight;
    axLab.forEach((el, q) => {
      v3.copy(ANCH[q]).project(camera);
      el.style.transform = `translate(-50%, -50%) translate(${((v3.x + 1) / 2 * w).toFixed(1)}px, ${((1 - v3.y) / 2 * h).toFixed(1)}px)`;
    });
    drawWMatrix(cm, st); drawNodeSeries(ct, st);
    const t = st.sim.t[st.fr];
    if (st.tshown !== st.fr) {
      st.tshown = st.fr;
      iT.value = st.fr; iT.style.setProperty('--fill', st.fr / (st.sim.n - 1) * 100 + '%');
      const o = $('#g3-tout');
      if (window.katex) katex.render(`t = ${Math.round(t)}`, o, { throwOnError: false }); else o.textContent = 't = ' + Math.round(t);
    }
    $$('li', rec).forEach(li => {
      const on = st.Ti[+li.dataset.i] <= t, was = li.classList.contains('on');
      if (on !== was) { li.classList.toggle('on', on); li.classList.toggle('pop', on && st.playing); }
    });
    if (st.hover >= 0) showTip();
  }
  function showTip() {
    const i = st.hover, s = st.sim, k = st.fr, key = i + ':' + k, Ti = st.Ti[i];
    if (key !== st.tipKey) {
      st.tipKey = key;
      tip.innerHTML = `nodo ${i} · ${R`\(\varphi_t(${i}) = (${s.x[i][k].toFixed(2)},\ ${s.z[i][k].toFixed(2)},\ ${st.x0[i].toFixed(2)})\)`}<br>` +
        `recibe de ${st.nbrs[i].length} · influye sobre ${st.nbrs.filter(n => n.includes(i)).length}<br>${isFinite(Ti) ? R`\(T_{${i}} = ${Math.round(Ti)}\)` : 'no reclutado'}`;
      tex(tip);
    }
    v3.copy(P[i]).project(camera);
    tip.style.left = clamp((v3.x + 1) / 2 * stage.clientWidth + 14, 0, Math.max(0, stage.clientWidth - 250)) + 'px';
    tip.style.top = ((1 - v3.y) / 2 * stage.clientHeight + 16) + 'px';
  }

  // ---- interacción
  const ray = new T.Raycaster(), ndc = new T.Vector2();
  let lx = 0, ly = 0;
  canvas.addEventListener('pointerdown', e => { st.drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing'; });
  canvas.addEventListener('pointermove', e => {
    if (st.drag) {
      st.yaw -= (e.clientX - lx) * 0.009; st.pitch = clamp(st.pitch + (e.clientY - ly) * 0.007, -1.2, 1.35);
      lx = e.clientX; ly = e.clientY;
    } else if (st.mode !== 'capas' && st.sim) {
      const b = canvas.getBoundingClientRect();
      ndc.set((e.clientX - b.left) / b.width * 2 - 1, 1 - (e.clientY - b.top) / b.height * 2);
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObjects(nodes)[0];
      st.hover = hit ? hit.object.userData.i : -1;
      tip.classList.toggle('on', st.hover >= 0);
      canvas.style.cursor = st.hover >= 0 ? 'pointer' : 'grab';
    }
    draw();
  });
  const release = () => { st.drag = false; canvas.style.cursor = 'grab'; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', () => { if (!st.drag) { st.hover = -1; tip.classList.remove('on'); draw(); } });
  $('#g3-zin').onclick = () => { st.dist = clamp(st.dist * 0.85, 2.6, 9); draw(); };
  $('#g3-zout').onclick = () => { st.dist = clamp(st.dist / 0.85, 2.6, 9); draw(); };
  $('#g3-reset').onclick = () => { Object.assign(st, VIEW); draw(); };

  let timer;
  const rebuild = () => { busy.classList.add('on'); clearTimeout(timer); timer = setTimeout(() => { build(); busy.classList.remove('on'); draw(); }, 140); };
  bindRange(iN, $('#g3-no'), 0, rebuild); bindRange(iF, $('#g3-fo'), 0, rebuild);
  bindRange(iK, $('#g3-ko'), 3, rebuild); bindRange(iS, $('#g3-so'), 0, rebuild);
  bindRange(iL, $('#g3-layerso'), 0, () => { buildLayers(); draw(); });
  bindRange(iP, $('#g3-pcto'), 0, () => { buildLayers(); draw(); });
  const seg = (id, fn) => $$(`#${id} button`).forEach(b => b.onclick = () => {
    $$(`#${id} button`).forEach(o => o.setAttribute('aria-pressed', String(o === b)));
    fn(b.dataset.v);
  });
  seg('g3-mode', setMode);
  seg('g3-dir', v => { st.dir = v; st.fr = 0; rebuild(); });
  seg('g3-var', v => { st.variant = v; st.fr = 0; rebuild(); });
  const toggle = (id, key) => { $(id).onclick = e => { st[key] = !st[key]; e.currentTarget.setAttribute('aria-pressed', String(st[key])); draw(); }; $(id).setAttribute('aria-pressed', String(st[key])); };
  toggle('#g3-trails', 'trails'); toggle('#g3-spin', 'spin');
  const setPlay = v => { st.playing = v; $('#g3-play').classList.toggle('is-playing', v); };
  $('#g3-play').onclick = () => setPlay(!st.playing);
  iT.addEventListener('input', () => { st.fr = +iT.value; setPlay(false); draw(); });
  const seek = e => { const r = ct.getBoundingClientRect(); st.fr = Math.round(clamp((e.clientX - r.left - 36) / (r.width - 46), 0, 1) * (st.sim.n - 1)); setPlay(false); draw(); };
  ct.addEventListener('pointerdown', e => { seek(e); ct.setPointerCapture(e.pointerId); ct.onpointermove = seek; });
  ct.addEventListener('pointerup', () => { ct.onpointermove = null; });

  const size = () => {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    st.fit = Math.max(1, 1.42 / camera.aspect);        // en lienzos angostos la cámara se aleja para que quepa la caja
    draw();
  };
  let last = 0, acc = 0;
  const loop = now => {
    requestAnimationFrame(loop);
    if (!st.vis || document.hidden || !st.sim) { last = now; return; }
    const dt = Math.min(60, now - last || 16); last = now;
    st.phase += dt / 1000;
    if (st.playing) { acc += dt / 16000 * (st.sim.n - 1); if (acc >= 1) { st.fr = (st.fr + Math.floor(acc)) % st.sim.n; acc %= 1; } }
    if (st.spin && !st.drag) st.yaw += dt / 1000 * 0.11;
    if (!reduced || st.playing) draw();
  };
  watchVisible($('#lab3d'), v => { st.vis = v; });
  setPlay(st.playing);
  build();
  if (reduced) st.fr = Math.floor(st.sim.n * 0.3);
  setMode('estructura');
  size();
  requestAnimationFrame(loop);
  return size;
}

/* ================= dinámica normal: A = 1, comunidades de W (Three.js) ================= */
const DN_X0 = [2.3, 2.4, 3.12247881, 3.45605899, 3.28631907, 3.39365697, 3.16604852, 3.03634323, 3.23275087, 3.3965787];
const DN_K = { sub: 2.115, c: 2.35, ruido: 2.327, sup: 2.45 };
const DN_COL = ['#b4502c', '#66754b', '#c58a2a', '#5d7a8a', '#7a4a6a', '#8c7054', '#a0436a', '#3f6b5a', '#b08a5a', '#55606e'];

// punto fijo de reposo en la rama izquierda (brentq en [-3, -4/3])
function restX(x0) {
  let a = -3, b = -4 / 3;
  const f = x => cubic(x) - hEq8(x, x0);
  for (let q = 0; q < 60; q++) { const m = (a + b) / 2; (f(a) * f(m) <= 0) ? b = m : a = m; }
  return (a + b) / 2;
}

// Euler-Maruyama de la red completa; guarda cada 5 unidades de tiempo después del transitorio
function simDN({ K, sx, sz, T, seed, x0 = DN_X0, dt = 0.1, every = 50, trans = 5000 }) {
  const N = x0.length, n = Math.round(T / dt), nt = Math.round(trans / dt), ns = Math.floor((n - nt) / every);
  const x = new Float64Array(N), z = new Float64Array(N), on = new Uint8Array(N), crises = new Int32Array(N);
  for (let i = 0; i < N; i++) { x[i] = x0[i] > X0C ? restX(x0[i]) : -1.5; z[i] = hEq8(x[i], x0[i]); on[i] = x[i] > -0.5; }
  const X = x0.map(() => new Float32Array(ns)), Z = x0.map(() => new Float32Array(ns)), S = x0.map(() => new Uint8Array(ns));
  const r = mulberry32(seed), ax = sx * Math.sqrt(dt), az = sz * Math.sqrt(dt), KN = K / N, noisy = sx > 0 || sz > 0;
  let spare = NaN;
  const gauss = () => {      // Box-Muller; la misma semilla da la misma realización al mover σ
    if (spare === spare) { const g = spare; spare = NaN; return g; }
    const u = Math.sqrt(-2 * Math.log(1 - r())), v = 6.283185307179586 * r();
    spare = u * Math.sin(v);
    return u * Math.cos(v);
  };
  for (let k = 1; k <= n; k++) {
    let sum = 0;
    for (let i = 0; i < N; i++) sum += x[i];
    for (let i = 0; i < N; i++) {
      const xi = x[i];
      let dx = (-xi * xi * xi - 2 * xi * xi + 1 - z[i] + I1) * dt;
      let dz = (hEq8(xi, x0[i]) - z[i] - KN * (sum - N * xi)) * dt / TAU0;     // sum_j (x_j - x_i) = sum - N x_i
      if (noisy) { dx += ax * gauss(); dz += az * gauss(); }
      x[i] = xi + dx; z[i] += dz;
      const up = x[i] > -0.5;
      if (up && !on[i] && k > nt) crises[i]++;
      on[i] = up || (on[i] && x[i] >= -1) ? 1 : 0;
    }
    if (k > nt && (k - nt) % every === 0) {
      const q = (k - nt) / every - 1;
      if (q < ns) for (let i = 0; i < N; i++) { X[i][q] = x[i]; Z[i][q] = z[i]; S[i][q] = on[i]; }
    }
  }
  return { N, n: ns, t: Float32Array.from({ length: ns }, (_, q) => (q + 1) * every * dt), T: ns * every * dt, x: X, z: Z, s: S, crises };
}

// ---- comunidades: modularidad, Louvain reducido y consenso
function modQ(W, lab, N) {
  const k = new Float64Array(N);
  let m2 = 0;
  for (let i = 0; i < N; i++) { for (let j = 0; j < N; j++) k[i] += W[i * N + j]; m2 += k[i]; }
  if (!m2) return 0;
  let q = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (lab[i] === lab[j]) q += W[i * N + j] - k[i] * k[j] / m2;
  return q / m2;
}
function louvainLite(W, N, r) {
  const lab = Int32Array.from({ length: N }, (_, i) => i);
  let best = modQ(W, lab, N), moved = true;
  while (moved) {                 // movimientos locales en orden aleatorio
    moved = false;
    const order = [...lab.keys()].sort(() => r() - 0.5);
    for (const i of order) {
      const own = lab[i];
      let bc = own, bq = best;
      for (const c of new Set(lab)) {
        if (c === own) continue;
        lab[i] = c;
        const q = modQ(W, lab, N);
        if (q > bq + 1e-12) { bq = q; bc = c; }
      }
      lab[i] = bc;
      if (bc !== own) { best = bq; moved = true; }
    }
  }
  for (;;) {                      // fusión de comunidades completas, como la agregación de Louvain
    const cs = [...new Set(lab)];
    let bq = best, pair = null;
    for (let a = 0; a < cs.length; a++) for (let b = a + 1; b < cs.length; b++) {
      const tmp = lab.map(c => c === cs[b] ? cs[a] : c), q = modQ(W, tmp, N);
      if (q > bq + 1e-12) { bq = q; pair = [cs[a], cs[b]]; }
    }
    if (!pair) break;
    for (let i = 0; i < N; i++) if (lab[i] === pair[1]) lab[i] = pair[0];
    best = bq;
  }
  return lab;
}
function consensusDN(W, N, runs = 20) {
  const P = new Float32Array(N * N);
  for (let s = 0; s < runs; s++) {
    const lab = louvainLite(W, N, mulberry32(s + 1));
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (lab[i] === lab[j]) P[i * N + j] += 1 / runs;
  }
  // componentes conexas del grafo P_ij > 0.5
  const lab = new Int32Array(N).fill(-1);
  let nc = 0;
  for (let i = 0; i < N; i++) {
    if (lab[i] >= 0) continue;
    const stack = [i]; lab[i] = nc;
    while (stack.length) { const a = stack.pop(); for (let b = 0; b < N; b++) if (lab[b] < 0 && P[a * N + b] > 0.5) { lab[b] = nc; stack.push(b); } }
    nc++;
  }
  return { lab, coms: Array.from({ length: nc }, (_, c) => [...lab.keys()].filter(i => lab[i] === c)) };
}

// W en ventanas deslizantes, comunidades de consenso y disposición en el plano que sigue a la anterior
function windowsDN(sim, win, step = 2500) {
  const N = sim.N, nv = Math.min(sim.n, Math.round(win / 5)), ns = Math.max(1, Math.round(step / 5)), out = [];
  let pos = Array.from({ length: N }, (_, i) => [Math.cos(i / N * 6.2832), Math.sin(i / N * 6.2832)]);
  for (let a = 0; a + nv <= sim.n; a += ns) {
    const mu = new Float64Array(N), sd = new Float64Array(N), W = new Float32Array(N * N);
    for (let i = 0; i < N; i++) {
      const z = sim.z[i];
      let m = 0, v = 0;
      for (let q = a; q < a + nv; q++) m += z[q];
      m /= nv;
      for (let q = a; q < a + nv; q++) v += (z[q] - m) ** 2;
      mu[i] = m; sd[i] = Math.sqrt(v);
    }
    let wlo = Infinity, whi = 0;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      let c = 0;
      const zi = sim.z[i], zj = sim.z[j];
      for (let q = a; q < a + nv; q++) c += (zi[q] - mu[i]) * (zj[q] - mu[j]);
      const w = sd[i] > 1e-12 && sd[j] > 1e-12 ? Math.max(0, c / (sd[i] * sd[j])) : 0;
      W[i * N + j] = W[j * N + i] = w;
      wlo = Math.min(wlo, w); whi = Math.max(whi, w);
    }
    const { lab, coms } = consensusDN(W, N);
    const frac = coms.map(c => { let n = 0; for (let q = a; q < a + nv; q++) { let m = 0; for (const i of c) m += sim.s[i][q]; if (m / c.length > 0.5) n++; } return n / nv; });
    pos = layoutDN(W, lab, N, pos, wlo, whi, out.length ? 60 : 300);
    out.push({ a, b: a + nv, c: a + nv / 2, W, whi, lab, coms, frac, Q: modQ(W, lab, N), pos });
  }
  return out;
}
function layoutDN(W, lab, N, start, lo, hi, iters) {
  const P = start.map(p => p.slice()), k = 0.75, span = hi - lo || 1;
  for (let it = 0, temp = iters > 100 ? 0.25 : 0.08; it < iters; it++, temp *= 0.985) {
    const D = P.map(() => [0, 0]);
    for (let a = 0; a < N; a++) for (let b = a + 1; b < N; b++) {
      const dx = P[a][0] - P[b][0], dy = P[a][1] - P[b][1], d = Math.hypot(dx, dy) || 1e-3;
      const w = ((W[a * N + b] - lo) / span) ** 2 + (lab[a] === lab[b] ? 0.8 : 0);
      const f = k * k / d - w * d * d / k;        // repulsión de todos, atracción por W y por comunidad
      D[a][0] += dx / d * f; D[a][1] += dy / d * f; D[b][0] -= dx / d * f; D[b][1] -= dy / d * f;
    }
    for (let a = 0; a < N; a++) {
      D[a][0] -= 0.08 * P[a][0]; D[a][1] -= 0.08 * P[a][1];
      const d = Math.hypot(D[a][0], D[a][1]) || 1e-3, s = Math.min(d, temp) / d;
      P[a][0] += D[a][0] * s; P[a][1] += D[a][1] * s;
    }
  }
  let cx = 0, cy = 0;
  P.forEach(p => { cx += p[0] / N; cy += p[1] / N; });
  const R = Math.max(...P.map(p => Math.hypot(p[0] - cx, p[1] - cy))) || 1;
  return P.map(p => [(p[0] - cx) / R, (p[1] - cy) / R]);
}

function labDN() {
  let onResize = () => {};
  loadThree().then(T => { onResize = buildDN(T); }).catch(() => { $('#labdn').hidden = true; });
  return () => onResize();
}

function buildDN(T) {
  const canvas = $('#dn-canvas'), stage = $('#dn-stage'), tip = $('#dn-tip'), busy = $('#dn-busy'), cm = $('#dn-mat'), ct = $('#dn-ts'), comsEl = $('#dn-coms');
  const iK = $('#dn-k'), iX = $('#dn-sx'), iZ = $('#dn-sz'), iW = $('#dn-w'), iP = $('#dn-p'), iT = $('#dn-time');
  const axLab = $$('#dn-labels > span');
  const S = { x: 1.35, y: 0.95, z: 1.05 }, VIEW = { yaw: -0.7, pitch: 0.42, dist: 5.4 }, N = DN_X0.length;
  const ORD = DN_X0.map((v, i) => i).sort((a, b) => DN_X0[a] - DN_X0[b]);
  const st = { sim: null, wins: [], fr: 0, playing: !reduced, spin: !reduced, ...VIEW, hover: -1, vis: false, T: 60000, seed: 7, drag: false, phase: 0, fit: 1, tshown: -1, comKey: '', tipKey: '' };

  const renderer = new T.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(34, 1, 0.1, 60);
  scene.add(new T.HemisphereLight(0xfff6ec, 0xa08468, 1.5));
  const sun = new T.DirectionalLight(0xffffff, 1.5);
  sun.position.set(2, 4, 3);
  scene.add(sun);
  scene.add(new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(2 * S.x, 2 * S.y, 2 * S.z)), new T.LineBasicMaterial({ color: 0xcbb9a0 })));
  const yOf = x => clamp((x + 2.2) / 3.8, 0, 1) * 2 * S.y - S.y;
  // plano del umbral de crisis x = -0.5
  const thr = new T.Mesh(new T.PlaneGeometry(2 * S.x, 2 * S.z), new T.MeshBasicMaterial({ color: 0xb4502c, transparent: true, opacity: 0.06, side: T.DoubleSide, depthWrite: false }));
  thr.rotation.x = -Math.PI / 2; thr.position.y = yOf(-0.5);
  scene.add(thr);
  scene.add(new T.LineSegments(new T.EdgesGeometry(thr.geometry), new T.LineBasicMaterial({ color: 0xd9a58c })).rotateX(-Math.PI / 2).translateZ(yOf(-0.5)));

  const EPI = new T.Color('#b4502c'), CREAMc = new T.Color('#fffdf9'), BROWN = new T.Color('rgb(140,112,84)');
  const COLS = DN_COL.map(c => new T.Color(c));
  const sphereGeo = new T.SphereGeometry(1, 24, 16), cylGeo = new T.CylinderGeometry(1, 1, 1, 8, 1, true), discGeo = new T.CircleGeometry(1, 48);
  const ringTex = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 128;
    const g = c.getContext('2d');
    g.strokeStyle = '#fff'; g.lineWidth = 13; g.beginPath(); g.arc(64, 64, 54, 0, 6.2832); g.stroke();
    return new T.CanvasTexture(c);
  })();
  const ring = (color, opacity = 1) => new T.Sprite(new T.SpriteMaterial({ map: ringTex, color, transparent: true, opacity, depthWrite: false }));
  const numSprite = (n, color) => {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d');
    g.font = '42px KaTeX_Main, "Times New Roman", serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = color;
    g.fillText(n, 32, 34);
    const s = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(c), transparent: true, depthWrite: false }));
    s.scale.setScalar(0.2);
    return s;
  };
  const r0 = 0.075;
  const nodes = DN_X0.map((_, i) => { const m = new T.Mesh(sphereGeo, new T.MeshStandardMaterial({ color: CREAMc.clone(), roughness: 0.5 })); m.userData.i = i; m.scale.setScalar(r0); scene.add(m); return m; });
  const rings = DN_X0.map(() => { const s = ring(COLS[0]); s.scale.setScalar(r0 * 3.1); scene.add(s); return s; });
  const halos = DN_X0.map(() => { const s = ring(EPI, 0.35); scene.add(s); return s; });
  const labels = DN_X0.map((v, i) => { const s = numSprite(i, v < X0C ? '#8f3a1f' : '#2a2018'); scene.add(s); return s; });
  const discs = DN_X0.map(() => {
    const m = new T.Mesh(discGeo, new T.MeshBasicMaterial({ color: COLS[0], transparent: true, opacity: 0.16, depthWrite: false, side: T.DoubleSide }));
    m.rotation.x = -Math.PI / 2; scene.add(m); return m;
  });
  const pairs = [];
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const mat = new T.MeshBasicMaterial({ color: BROWN.clone(), transparent: true, depthWrite: false }), m = new T.Mesh(cylGeo, mat);
    scene.add(m); pairs.push({ i, j, m, mat });
  }
  const stems = new T.LineSegments(new T.BufferGeometry(), new T.LineBasicMaterial({ color: 0xbfae98, transparent: true, opacity: 0.8 }));
  stems.geometry.setAttribute('position', new T.BufferAttribute(new Float32Array(N * 6), 3));
  stems.frustumCulled = false;
  scene.add(stems);
  const P = DN_X0.map(() => new T.Vector3()), UP = new T.Vector3(0, 1, 0), d = new T.Vector3(), v3 = new T.Vector3();
  const activity = x => clamp((x + 1.4) / 1.7, 0, 1);

  // ventana a la que pertenece el cuadro k: la de centro más cercano, y la interpolación de posiciones entre centros
  function winAt(k) {
    const W = st.wins;
    let q = 0;
    while (q + 1 < W.length && W[q + 1].c <= k) q++;
    const nxt = W[Math.min(q + 1, W.length - 1)], u = nxt === W[q] ? 0 : clamp((k - W[q].c) / (nxt.c - W[q].c), 0, 1);
    const near = u > 0.5 && nxt !== W[q] ? q + 1 : q;
    return { a: W[q], b: nxt, u: k < W[0].c ? 0 : u, w: W[near], q: near };
  }
  const comColor = (win, c) => (win.coms[c].includes(0) ? 0 : 1 + (c - (win.lab[0] < c ? 1 : 0))) % DN_COL.length;   // la comunidad del foco siempre en terracota

  function simulateNow() {
    const K = +iK.value, sx = +iX.value, sz = +iZ.value;
    st.sim = simDN({ K, sx, sz, T: st.T, seed: st.seed });
    st.K = K; st.sx = sx; st.sz = sz;
    if (st.fr > st.sim.n - 1) st.fr = 0;
    iT.max = st.sim.n - 1;
    windows();
    const cr = st.sim.crises, sanas = DN_X0.map((v, i) => i).filter(i => DN_X0[i] >= X0C), rec = sanas.filter(i => cr[i] > 0);
    $('#dn-read').innerHTML =
      `<div><dt>${R`\(K / K_c\)`}</dt><dd>${(K / DN_K.c).toFixed(3)}</dd></div>` +
      `<div><dt>Crisis por nodo del foco</dt><dd class="epi">${cr[0]} · ${cr[1]}</dd></div>` +
      `<div><dt>Sanas con crisis</dt><dd class="${rec.length ? 'epi' : 'sano'}">${rec.length} / ${sanas.length}</dd></div>` +
      `<div><dt>Crisis sanas (total)</dt><dd>${rec.reduce((s, i) => s + cr[i], 0)}</dd></div>` +
      `<div><dt>Comunidades ahora</dt><dd id="dn-nc">—</dd></div>` +
      `<div><dt>Modularidad ${R`\(Q\)`}</dt><dd id="dn-q">—</dd></div>`;
    tex($('#dn-read'));
  }
  function windows() { st.wins = windowsDN(st.sim, +iW.value); st.comKey = ''; }

  function update(k) {
    const s = st.sim, { a, b, u, w } = winAt(k);
    let now = 0;
    for (let i = 0; i < N; i++) {
      const px = a.pos[i][0] + (b.pos[i][0] - a.pos[i][0]) * u, pz = a.pos[i][1] + (b.pos[i][1] - a.pos[i][1]) * u;
      P[i].set(px * S.x * 0.82, yOf(s.x[i][k]), pz * S.z * 0.82);
      const ict = s.s[i][k] === 1, c = comColor(w, w.lab[i]);
      if (ict) now++;
      nodes[i].position.copy(P[i]); nodes[i].material.color.copy(CREAMc).lerp(EPI, activity(s.x[i][k]));
      rings[i].position.copy(P[i]); rings[i].material.color.copy(COLS[c]); rings[i].scale.setScalar(r0 * (st.hover === i ? 3.8 : 3.1));
      halos[i].visible = ict; halos[i].position.copy(P[i]); halos[i].scale.setScalar(r0 * (4.6 + (reduced ? 0 : 0.35 * Math.sin(st.phase * 12 + i))));
      labels[i].position.copy(P[i]).y += r0 * 2.9;
      const arr = stems.geometry.attributes.position.array;
      arr.set([P[i].x, P[i].y, P[i].z, P[i].x, -S.y, P[i].z], i * 6);
    }
    stems.geometry.attributes.position.needsUpdate = true;
    // sombras de comunidad en el piso
    discs.forEach((m, c) => {
      const com = w.coms[c];
      m.visible = !!com;
      if (!com) return;
      let cx = 0, cz = 0;
      for (const i of com) { cx += P[i].x / com.length; cz += P[i].z / com.length; }
      const rad = Math.max(...com.map(i => Math.hypot(P[i].x - cx, P[i].z - cz))) + 0.17;
      m.position.set(cx, -S.y + 0.003 + c * 0.001, cz); m.scale.setScalar(rad); m.material.color.copy(COLS[comColor(w, c)]);
      m.material.opacity = w.frac[c] > 0 && com.filter(i => s.s[i][k]).length / com.length > 0.5 ? 0.32 : 0.15;
    });
    // aristas sobre el percentil elegido de W en la ventana
    const ws = pairs.map(p => w.W[p.i * N + p.j]).sort((x, y) => x - y), pos = +iP.value / 100 * (ws.length - 1), lo = Math.floor(pos);
    const cut = ws[lo] + ((ws[Math.min(lo + 1, ws.length - 1)] - ws[lo]) * (pos - lo));
    for (const p of pairs) {
      const val = w.W[p.i * N + p.j], show = val > 0 && val >= cut, hot = st.hover < 0 || st.hover === p.i || st.hover === p.j;
      p.m.visible = show;
      if (!show) continue;
      d.subVectors(P[p.j], P[p.i]);
      const len = d.length(), uu = w.whi > 0 ? val / w.whi : 0, same = w.lab[p.i] === w.lab[p.j];
      p.m.position.copy(P[p.i]).addScaledVector(d, 0.5);
      p.m.quaternion.setFromUnitVectors(UP, d.divideScalar(len || 1));
      const rad = 0.003 + 0.016 * uu * uu;
      p.m.scale.set(rad, len, rad);
      p.mat.color.copy(same ? COLS[comColor(w, w.lab[p.i])] : BROWN);
      p.mat.opacity = (same ? 0.25 + 0.6 * uu : 0.1 + 0.3 * uu) * (hot ? 1 : 0.15);
    }
    return { now, w };
  }

  function drawMat(w) {
    const { g, w: cw, h } = fit(cm);
    g.clearRect(0, 0, cw, h);
    const size = Math.min(cw - 40, h - 22), cell = size / N, ox = (cw - size) / 2 + 10, oy = 16;
    g.font = '12.5px KaTeX_Main, "Times New Roman", serif'; g.textAlign = 'center';
    ORD.forEach((i, r) => {
      g.fillStyle = DN_X0[i] < X0C ? '#8f3a1f' : C.muted;
      g.fillText(i, ox + (r + .5) * cell, 11); g.fillText(i, ox - 16, oy + (r + .5) * cell + 4);
      g.fillStyle = DN_COL[comColor(w, w.lab[i])]; g.fillRect(ox - 8, oy + r * cell + 1, 4, cell - 2);
      ORD.forEach((j, c) => {
        const u = i === j ? 0 : w.W[i * N + j] / (w.whi || 1);
        g.fillStyle = i === j ? '#faf5ec' : mix([240, 230, 214], ACC, u);
        g.globalAlpha = st.hover < 0 || st.hover === i || st.hover === j ? 1 : 0.35;
        g.fillRect(ox + c * cell + .5, oy + r * cell + .5, cell - 1, cell - 1);
      });
    });
    g.globalAlpha = 1;
  }
  function drawTs(win) {
    const s = st.sim, { g, w, h } = fit(ct), { X, Y, m } = frame(g, w, h, [0, s.T], [-2.2, 1.6]);
    g.fillStyle = rgba([197, 138, 42], .13);
    g.fillRect(X(s.t[win.a]), m.t, X(s.t[win.b - 1]) - X(s.t[win.a]), h - m.t - m.b);
    g.strokeStyle = rgba([42, 32, 24], .3); g.setLineDash([3, 4]); g.beginPath(); g.moveTo(m.l, Y(-0.5)); g.lineTo(w - 10, Y(-0.5)); g.stroke(); g.setLineDash([]);
    const stepK = Math.max(1, Math.floor(s.n / (w * 1.5)));
    for (let q = N - 1; q >= 0; q--) {
      const i = ORD[N - 1 - q], epi = DN_X0[i] < X0C, on = st.hover < 0 || st.hover === i;
      g.beginPath();
      for (let k = 0; k < s.n; k += stepK) { const px = X(s.t[k]), py = Y(s.x[i][k]); k ? g.lineTo(px, py) : g.moveTo(px, py); }
      stroke(g, rgba(epi ? ACC : SAGE, on ? (epi ? 1 : .55) : .1), epi || st.hover === i ? 1.7 : 1);
    }
    const px = X(s.t[st.fr]);
    g.strokeStyle = C.ink; g.lineWidth = 1.5; g.beginPath(); g.moveTo(px, m.t); g.lineTo(px, h - m.b); g.stroke();
    dot(g, px, m.t, 4, C.ink);
  }
  function drawComs(w, q) {
    const s = st.sim, k = st.fr, act = DN_X0.map((_, i) => s.s[i][k]), key = q + ':' + act.join('');
    if (key === st.comKey) return;
    st.comKey = key;
    comsEl.innerHTML = w.coms.map((c, ci) => {
      const inC = c.filter(i => act[i]).length / c.length > 0.5;
      return `<li class="${inC ? 'crisis' : ''}"><span class="sw" style="background:${DN_COL[comColor(w, ci)]}"></span>` +
        `<span class="mem">${c.map(i => `<b class="${DN_X0[i] < X0C ? 'epi' : ''} ${act[i] ? 'on' : ''}">${i}</b>`).join('')}</span>` +
        `<span class="st">${inC ? 'en crisis' : (100 * w.frac[ci]).toFixed(1) + ' % en crisis'}</span></li>`;
    }).join('');
    const nc = $('#dn-nc'), qq = $('#dn-q');
    if (nc) nc.textContent = w.coms.length;
    if (qq) qq.textContent = w.Q.toFixed(3);
  }

  function draw() {
    if (!st.sim || !st.wins.length) return;
    const { w } = update(st.fr), q = winAt(st.fr).q;
    const cp = Math.cos(st.pitch), dist = st.dist * st.fit;
    camera.position.set(dist * cp * Math.sin(st.yaw), dist * Math.sin(st.pitch), dist * cp * Math.cos(st.yaw));
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    const sw = stage.clientWidth, sh = stage.clientHeight;
    [new T.Vector3(0, -S.y, S.z + 0.3), new T.Vector3(-S.x - 0.3, 0.1, -S.z)].forEach((p, n) => {
      v3.copy(p).project(camera);
      axLab[n].style.transform = `translate(-50%, -50%) translate(${((v3.x + 1) / 2 * sw).toFixed(1)}px, ${((1 - v3.y) / 2 * sh).toFixed(1)}px)`;
    });
    drawMat(w); drawTs(w); drawComs(w, q);
    if (st.tshown !== st.fr) {
      st.tshown = st.fr;
      iT.value = st.fr; iT.style.setProperty('--fill', st.fr / (st.sim.n - 1) * 100 + '%');
      const o = $('#dn-tout'), t = Math.round(st.sim.t[st.fr]);
      if (window.katex) katex.render(`t = ${t}`, o, { throwOnError: false }); else o.textContent = 't = ' + t;
    }
    if (st.hover >= 0) showTip(w);
  }
  function showTip(w) {
    const i = st.hover, s = st.sim, k = st.fr, key = i + ':' + k;
    if (key !== st.tipKey) {
      st.tipKey = key;
      tip.innerHTML = `nodo ${i} · ${DN_X0[i] < X0C ? 'foco' : 'sana'} · ${R`\(x_0 = ${DN_X0[i].toFixed(3)}\)`}<br>` +
        `${R`\(x_${i}(t) = ${s.x[i][k].toFixed(2)}\)`} · ${s.s[i][k] ? 'en crisis' : 'fuera de crisis'}<br>` +
        `comunidad con ${w.coms[w.lab[i]].length} nodos · ${s.crises[i]} crisis`;
      tex(tip);
    }
    v3.copy(P[i]).project(camera);
    tip.style.left = clamp((v3.x + 1) / 2 * stage.clientWidth + 14, 0, Math.max(0, stage.clientWidth - 250)) + 'px';
    tip.style.top = ((1 - v3.y) / 2 * stage.clientHeight + 16) + 'px';
  }

  // ---- interacción
  const ray = new T.Raycaster(), ndc = new T.Vector2();
  let lx = 0, ly = 0;
  canvas.addEventListener('pointerdown', e => { st.drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing'; });
  canvas.addEventListener('pointermove', e => {
    if (st.drag) {
      st.yaw -= (e.clientX - lx) * 0.009; st.pitch = clamp(st.pitch + (e.clientY - ly) * 0.007, -1.2, 1.35);
      lx = e.clientX; ly = e.clientY;
    } else if (st.sim) {
      const b = canvas.getBoundingClientRect();
      ndc.set((e.clientX - b.left) / b.width * 2 - 1, 1 - (e.clientY - b.top) / b.height * 2);
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObjects(nodes)[0];
      st.hover = hit ? hit.object.userData.i : -1;
      tip.classList.toggle('on', st.hover >= 0);
      canvas.style.cursor = st.hover >= 0 ? 'pointer' : 'grab';
    }
    draw();
  });
  const release = () => { st.drag = false; canvas.style.cursor = 'grab'; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', () => { if (!st.drag) { st.hover = -1; tip.classList.remove('on'); draw(); } });
  $('#dn-zin').onclick = () => { st.dist = clamp(st.dist * 0.85, 2.6, 9); draw(); };
  $('#dn-zout').onclick = () => { st.dist = clamp(st.dist / 0.85, 2.6, 9); draw(); };
  $('#dn-reset').onclick = () => { Object.assign(st, VIEW); draw(); };

  let timer;
  const later = fn => { busy.classList.add('on'); clearTimeout(timer); timer = setTimeout(() => { fn(); busy.classList.remove('on'); draw(); }, 160); };
  const rebuild = () => later(simulateNow);
  const syncMode = () => $$('#dn-mode button').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.v === 'rui') === (+iX.value > 0 || +iZ.value > 0))));
  const syncKs = () => $$('#dn-ks .btn').forEach(b => b.setAttribute('aria-pressed', String(Math.abs(b._v - +iK.value) < 1e-9)));
  const upK = bindRange(iK, $('#dn-ko'), 3, () => { syncKs(); rebuild(); });
  const upX = bindRange(iX, $('#dn-sxo'), 3, () => { syncMode(); rebuild(); });
  const upZ = bindRange(iZ, $('#dn-szo'), 5, () => { syncMode(); rebuild(); });
  bindRange(iW, $('#dn-wo'), 0, () => later(windows));
  bindRange(iP, $('#dn-po'), 0, draw);
  presetButtons($('#dn-ks'), [['sub', R`K_{\rm sub}`], ['ruido', R`K_{\rm ruido}`], ['c', R`K_c`], ['sup', R`K_{\rm sup}`]].map(([k, l]) => ({ v: DN_K[k], label: R`\(${l}\)` })),
    v => { iK.value = v; upK(); syncKs(); rebuild(); });
  tex($('#dn-ks'));
  $$('#dn-mode button').forEach(b => b.onclick = () => {
    const noisy = b.dataset.v === 'rui';
    iX.value = noisy ? 0.02 : 0; iZ.value = noisy ? 0.00075 : 0; iK.value = noisy ? DN_K.ruido : DN_K.sup;
    upX(); upZ(); upK(); syncMode(); syncKs(); st.fr = 0; rebuild();
  });
  $$('#dn-t button').forEach(b => b.onclick = () => {
    $$('#dn-t button').forEach(o => o.setAttribute('aria-pressed', String(o === b)));
    st.T = +b.dataset.v; st.fr = 0; rebuild();
  });
  $('#dn-new').onclick = () => { st.seed++; rebuild(); };
  $('#dn-spin').onclick = e => { st.spin = !st.spin; e.currentTarget.setAttribute('aria-pressed', String(st.spin)); };
  $('#dn-spin').setAttribute('aria-pressed', String(st.spin));
  const setPlay = v => { st.playing = v; $('#dn-play').classList.toggle('is-playing', v); };
  $('#dn-play').onclick = () => setPlay(!st.playing);
  iT.addEventListener('input', () => { st.fr = +iT.value; setPlay(false); draw(); });
  const seek = e => { const r = ct.getBoundingClientRect(); st.fr = Math.round(clamp((e.clientX - r.left - 36) / (r.width - 46), 0, 1) * (st.sim.n - 1)); setPlay(false); draw(); };
  ct.addEventListener('pointerdown', e => { seek(e); ct.setPointerCapture(e.pointerId); ct.onpointermove = seek; });
  ct.addEventListener('pointerup', () => { ct.onpointermove = null; });

  const size = () => {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    st.fit = Math.max(1, 1.42 / camera.aspect);
    draw();
  };
  let last = 0, acc = 0;
  const loop = now => {
    requestAnimationFrame(loop);
    if (!st.vis || document.hidden || !st.sim) { last = now; return; }
    const dt = Math.min(60, now - last || 16); last = now;
    st.phase += dt / 1000;
    if (st.playing) { acc += dt / 1000 * st.sim.n / 40; if (acc >= 1) { st.fr = (st.fr + Math.floor(acc)) % st.sim.n; acc %= 1; } }   // 40 s por recorrido
    if (st.spin && !st.drag) st.yaw += dt / 1000 * 0.1;
    if (!reduced || st.playing) draw();
  };
  axLab[0].textContent = 'disposición según W';
  axLab[1].innerHTML = R`\(x_i(t)\)`;
  tex($('#dn-labels'));
  watchVisible($('#labdn'), v => { st.vis = v; });
  setPlay(st.playing);
  syncKs(); syncMode();
  simulateNow();
  size();
  requestAnimationFrame(loop);
  return size;
}

const CL_SEED = 26;                    // realización con ruido que tiene una cascada completa
/* ================= clusters del grafo dinámico (Three.js) ================= */
// métodos de comunidades sobre W; cada uno devuelve etiquetas 0..k-1
function cnmDN(W, N) {                 // Greedy (CNM): fusiones que más suben Q, desde nodos sueltos
  const lab = Int32Array.from({ length: N }, (_, i) => i);
  let best = modQ(W, lab, N);
  for (;;) {
    const cs = [...new Set(lab)];
    let bq = best, pair = null;
    for (let a = 0; a < cs.length; a++) for (let b = a + 1; b < cs.length; b++) {
      const q = modQ(W, lab.map(c => c === cs[b] ? cs[a] : c), N);
      if (q > bq + 1e-12) { bq = q; pair = [cs[a], cs[b]]; }
    }
    if (!pair) return lab;
    for (let i = 0; i < N; i++) if (lab[i] === pair[1]) lab[i] = pair[0];
    best = bq;
  }
}
function lpaDN(W, N, r) {               // propagación de etiquetas asíncrona y ponderada
  const lab = Int32Array.from({ length: N }, (_, i) => i);
  for (let it = 0; it < 100; it++) {
    let changed = false;
    for (const i of [...lab.keys()].sort(() => r() - 0.5)) {
      const acc = new Map();
      for (let j = 0; j < N; j++) if (j !== i && W[i * N + j] > 0) acc.set(lab[j], (acc.get(lab[j]) || 0) + W[i * N + j]);
      if (!acc.size) continue;
      const mx = Math.max(...acc.values()), best = [...acc].filter(([, v]) => v >= mx - 1e-12).map(([c]) => c);
      if (best.includes(lab[i])) continue;
      lab[i] = best[Math.floor(r() * best.length)]; changed = true;
    }
    if (!changed) break;
  }
  return lab;
}
function jacobiEig(A, N) {              // vectores propios de una matriz simétrica pequeña
  const a = Float64Array.from(A), V = new Float64Array(N * N);
  for (let i = 0; i < N; i++) V[i * N + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < N; p++) for (let q = p + 1; q < N; q++) off += a[p * N + q] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < N; p++) for (let q = p + 1; q < N; q++) {
      const apq = a[p * N + q];
      if (Math.abs(apq) < 1e-15) continue;
      const th = (a[q * N + q] - a[p * N + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < N; k++) {
        const akp = a[k * N + p], akq = a[k * N + q];
        a[k * N + p] = c * akp - s * akq; a[k * N + q] = s * akp + c * akq;
      }
      for (let k = 0; k < N; k++) {
        const apk = a[p * N + k], aqk = a[q * N + k];
        a[p * N + k] = c * apk - s * aqk; a[q * N + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < N; k++) {
        const vkp = V[k * N + p], vkq = V[k * N + q];
        V[k * N + p] = c * vkp - s * vkq; V[k * N + q] = s * vkp + c * vkq;
      }
    }
  }
  const idx = [...Array(N).keys()].sort((i, j) => a[i * N + i] - a[j * N + j]);
  return { val: idx.map(i => a[i * N + i]), vec: idx.map(i => Array.from({ length: N }, (_, k) => V[k * N + i])) };
}
function spectralDN(W, N, r) {
  const d = Array.from({ length: N }, (_, i) => { let s = 0; for (let j = 0; j < N; j++) s += W[i * N + j]; return Math.max(s, 1e-12); });
  const L = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) L[i * N + j] = (i === j ? 1 : 0) - W[i * N + j] / Math.sqrt(d[i] * d[j]);
  const { val, vec } = jacobiEig(L, N), head = val.slice(0, Math.floor(N / 2) + 1);
  let k = 0, gap = -Infinity;
  for (let q = 0; q + 1 < head.length; q++) if (head[q + 1] - head[q] > gap) { gap = head[q + 1] - head[q]; k = q; }
  k = Math.max(2, k + 1);
  const U = Array.from({ length: N }, (_, i) => { const row = vec.slice(0, k).map(v => v[i]), n = Math.hypot(...row) || 1e-12; return row.map(v => v / n); });
  let best = null, bestI = Infinity;
  for (let init = 0; init < 10; init++) {              // k-medias con siembra k-means++
    const C = [U[Math.floor(r() * N)].slice()];
    while (C.length < k) {
      const dd = U.map(u => Math.min(...C.map(c => u.reduce((s, v, q) => s + (v - c[q]) ** 2, 0)))), tot = dd.reduce((s, v) => s + v, 0);
      let x = r() * tot, q = 0;
      while (q < N - 1 && (x -= dd[q]) > 0) q++;
      C.push(U[q].slice());
    }
    const lab = new Int32Array(N);
    for (let it = 0; it < 50; it++) {
      let ch = false;
      U.forEach((u, i) => {
        let bc = 0, bd = Infinity;
        C.forEach((c, q) => { const s = u.reduce((t, v, m) => t + (v - c[m]) ** 2, 0); if (s < bd) { bd = s; bc = q; } });
        if (lab[i] !== bc || it === 0) { ch = ch || lab[i] !== bc; lab[i] = bc; }
      });
      C.forEach((c, q) => { const mem = U.filter((_, i) => lab[i] === q); if (mem.length) c.forEach((_, m) => { c[m] = mem.reduce((s, u) => s + u[m], 0) / mem.length; }); });
      if (!ch && it) break;
    }
    const inertia = U.reduce((s, u, i) => s + u.reduce((t, v, m) => t + (v - C[lab[i]][m]) ** 2, 0), 0);
    if (inertia < bestI) { bestI = inertia; best = lab; }
  }
  return best;
}
// reordena etiquetas: comunidades ordenadas por su menor miembro
function partition(lab, N) {
  const coms = [...new Set(lab)].map(c => [...Array(N).keys()].filter(i => lab[i] === c)).sort((a, b) => a[0] - b[0]);
  const out = new Int32Array(N);
  coms.forEach((c, q) => c.forEach(i => { out[i] = q; }));
  return { lab: out, coms };
}
function ariDN(a, b, N) {              // índice de Rand ajustado
  const key = new Map(), ca = new Map(), cb = new Map(), c2 = n => n * (n - 1) / 2;
  for (let i = 0; i < N; i++) { const k = a[i] + ',' + b[i]; key.set(k, (key.get(k) || 0) + 1); ca.set(a[i], (ca.get(a[i]) || 0) + 1); cb.set(b[i], (cb.get(b[i]) || 0) + 1); }
  const sij = [...key.values()].reduce((s, n) => s + c2(n), 0), sa = [...ca.values()].reduce((s, n) => s + c2(n), 0), sb = [...cb.values()].reduce((s, n) => s + c2(n), 0);
  const exp = sa * sb / c2(N), mx = (sa + sb) / 2;
  return mx === exp ? 1 : (sij - exp) / (mx - exp);
}
function corrW(sim, a, nv) {
  const N = sim.N, mu = new Float64Array(N), sd = new Float64Array(N), W = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    const z = sim.z[i];
    let m = 0, v = 0;
    for (let q = a; q < a + nv; q++) m += z[q];
    m /= nv;
    for (let q = a; q < a + nv; q++) v += (z[q] - m) ** 2;
    mu[i] = m; sd[i] = Math.sqrt(v);
  }
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    let c = 0;
    const zi = sim.z[i], zj = sim.z[j];
    for (let q = a; q < a + nv; q++) c += (zi[q] - mu[i]) * (zj[q] - mu[j]);
    W[i * N + j] = W[j * N + i] = sd[i] > 1e-12 && sd[j] > 1e-12 ? Math.max(0, c / (sd[i] * sd[j])) : 0;
  }
  return W;
}
// disposición de resortes en 3D según W (análoga a nx.spring_layout con peso), con arranque en la anterior
function spring3D(W, N, start, iters) {
  const P = start.map(p => p.slice()), k = 0.55;
  for (let it = 0, temp = iters > 100 ? 0.2 : 0.07; it < iters; it++, temp *= 0.985) {
    const D = P.map(() => [0, 0, 0]);
    for (let a = 0; a < N; a++) for (let b = a + 1; b < N; b++) {
      const dv = [0, 1, 2].map(q => P[a][q] - P[b][q]), d = Math.hypot(...dv) || 1e-3;
      const f = k * k / d - 1.6 * W[a * N + b] ** 2 * d * d / k;
      for (let q = 0; q < 3; q++) { D[a][q] += dv[q] / d * f; D[b][q] -= dv[q] / d * f; }
    }
    for (let a = 0; a < N; a++) {
      for (let q = 0; q < 3; q++) D[a][q] -= 0.06 * P[a][q];
      const d = Math.hypot(...D[a]) || 1e-3, s = Math.min(d, temp) / d;
      for (let q = 0; q < 3; q++) P[a][q] += D[a][q] * s;
    }
  }
  const c = [0, 1, 2].map(q => P.reduce((s, p) => s + p[q], 0) / N), R = Math.max(...P.map(p => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]))) || 1;
  return P.map(p => p.map((v, q) => (v - c[q]) / R));
}

function labCL() {
  let onResize = () => {};
  loadThree().then(T => { onResize = buildCL(T); }).catch(() => { $('#labcl').hidden = true; });
  return () => onResize();
}

function buildCL(T) {
  const canvas = $('#cl-canvas'), stage = $('#cl-stage3'), tip = $('#cl-tip'), busy = $('#cl-busy'), cm = $('#cl-mat'), ct = $('#cl-ts'), comsEl = $('#cl-coms');
  const iK = $('#cl-k'), iZ = $('#cl-sz'), iN = $('#cl-n'), iT = $('#cl-time');
  let X0 = DN_X0, N = X0.length, REF = Int32Array.from(X0, v => v < X0C ? 0 : 1);
  const VIEW = { yaw: -0.6, pitch: 0.38, dist: 5.2 }, STEP = 200;    // paso de 1 000
  const METHODS = [
    { id: 'consenso', label: 'Consenso', f: W => consensusDN(W, N, N > 12 ? 20 : 50).lab },
    { id: 'louvain', label: 'Louvain', f: W => louvainLite(W, N, mulberry32(42)) },
    { id: 'cnm', label: 'Greedy (CNM)', f: W => cnmDN(W, N) },
    { id: 'lpa', label: 'Propagación de etiquetas', f: W => lpaDN(W, N, mulberry32(42)) },
    { id: 'esp', label: 'Espectral', f: W => spectralDN(W, N, mulberry32(42)) },
  ];
  const SCEN = [
    { id: 'sub', label: R`\(K_{\rm sub}\)`, K: DN_K.sub, sz: 0, T: 60000, note: 'Sin ruido: solo el foco tiene crisis.' },
    { id: 'sup', label: R`\(K_{\rm sup}\)`, K: DN_K.sup, sz: 0, T: 60000, note: 'Sin ruido: cascadas completas cada ~16 000.' },
    { id: 'ruido', label: R`\(K_{\rm ruido}\) con ruido`, K: DN_K.ruido, sz: 7.5e-4, T: 100000, seed: CL_SEED, note: 'Una realización con ruido en que el ruido dispara una cascada completa.' },
  ];
  const st = { sim: null, method: 'consenso', fr: 0, playing: !reduced, spin: !reduced, ...VIEW, hover: -1, vis: false, T: 60000, seed: 3, drag: false, phase: 0, fit: 1,
    cache: new Map(), cur: null, curKey: '', targets: null, tshown: -1, tipKey: '' };

  const renderer = new T.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(34, 1, 0.1, 60);
  scene.add(new T.HemisphereLight(0xfff6ec, 0xa08468, 1.6));
  const sun = new T.DirectionalLight(0xffffff, 1.4);
  sun.position.set(2, 4, 3);
  scene.add(sun);
  const SC = 1.25, COLS = DN_COL.map(c => new T.Color(c)), GREY = new T.Color('#cbbfae'), CREAMc = new T.Color('#fffdf9');
  const sphereGeo = new T.SphereGeometry(1, 28, 18), cylGeo = new T.CylinderGeometry(1, 1, 1, 8, 1, true);
  const numSprite = (n, color) => {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d');
    g.font = '42px KaTeX_Main, "Times New Roman", serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = color;
    g.fillText(n, 32, 34);
    const s = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(c), transparent: true, depthWrite: false }));
    s.scale.setScalar(0.21);
    return s;
  };
  const r0 = 0.085;
  // x0 para N regiones: foco (2.3, 2.4) y sanas U[2.9, 3.5] sobre 2.95; con N = 10 es el vector del estudio
  const x0For = n => {
    if (n === DN_X0.length) return DN_X0;
    const r = mulberry32(42);
    return Array.from({ length: n }, (_, i) => { if (i < 2) return [2.3, 2.4][i]; let v; do v = 2.9 + 0.6 * r(); while (v < 2.95); return v; });
  };
  function setN() {
    X0 = x0For(+iN.value); N = X0.length; REF = Int32Array.from(X0, v => v < X0C ? 0 : 1);
    st.cache.clear(); st.curKey = ''; makeObjects();
  }
  let nodes = [], labels = [], bubbles = [], pairs = [], P = [];
  const UP = new T.Vector3(0, 1, 0), d = new T.Vector3(), v3 = new T.Vector3(), tgt = new T.Vector3();
  // objetos de la escena para el N actual
  function makeObjects() {
    for (const o of [...nodes, ...bubbles, ...pairs.map(q => q.m)]) { scene.remove(o); o.material.dispose(); }
    for (const o of labels) { scene.remove(o); o.material.map.dispose(); o.material.dispose(); }
    nodes = X0.map((_, i) => { const m = new T.Mesh(sphereGeo, new T.MeshStandardMaterial({ color: CREAMc.clone(), roughness: 0.45 })); m.userData.i = i; m.scale.setScalar(r0); scene.add(m); return m; });
    labels = X0.map((v, i) => { const s = numSprite(i, v < X0C ? '#8f3a1f' : '#2a2018'); scene.add(s); return s; });
    bubbles = X0.map(() => { const m = new T.Mesh(sphereGeo, new T.MeshStandardMaterial({ color: COLS[0], transparent: true, opacity: 0.13, depthWrite: false, roughness: 0.9 })); scene.add(m); return m; });
    pairs = [];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const mat = new T.MeshBasicMaterial({ color: GREY.clone(), transparent: true, depthWrite: false }), m = new T.Mesh(cylGeo, mat); scene.add(m); pairs.push({ i, j, m, mat }); }
    P = X0.map((_, i) => new T.Vector3(Math.cos(i), Math.sin(i * 1.7) * 0.3, Math.sin(i)).multiplyScalar(0.6));
    st.targets = null; st.hover = -1;
  }
  const colOf = (part, c) => part.coms[c].includes(0) ? 0 : 1 + (c - (part.lab[0] < c ? 1 : 0));

  const nv = () => Math.min(st.sim.n, 800);          // ventana fija de 4 000
  const startOf = fr => { const n = nv(); return clamp(Math.round((fr - n / 2) / STEP) * STEP, 0, st.sim.n - n); };
  function analyse(a) {
    const key = st.method + ':' + a + ':' + nv();
    if (st.cache.has(key)) return st.cache.get(key);
    const W = corrW(st.sim, a, nv()), m = METHODS.find(q => q.id === st.method), part = partition(m.f(W), N);
    let whi = 0;
    for (let q = 0; q < N * N; q++) whi = Math.max(whi, W[q]);
    const n = nv(), crisis = part.coms.map(c => { let k = 0; for (let q = a; q < a + n; q++) { let s = 0; for (const i of c) s += st.sim.s[i][q]; if (s / c.length > 0.5) k++; } return k / n; });
    const res = { a, W, whi, ...part, Q: modQ(W, part.lab, N), ari: ariDN(part.lab, REF, N), crisis };
    st.cache.set(key, res);
    return res;
  }
  function refresh() {
    const a = startOf(st.fr), cur = analyse(a), key = st.method + ':' + a + ':' + nv();
    if (key === st.curKey) return;
    st.curKey = key; st.cur = cur;
    st.targets = spring3D(cur.W, N, st.targets || P.map(p => [p.x / SC, p.y / SC, p.z / SC]), st.targets ? 70 : 300);
    const sanas = X0.map((v, i) => i).filter(i => X0[i] >= X0C), n = nv();
    let simul = 0, recS = 0;
    for (let q = a; q < a + n; q++) { let s = 0; for (let i = 0; i < N; i++) s += st.sim.s[i][q]; simul = Math.max(simul, s); }
    for (const i of sanas) { for (let q = a; q < a + n; q++) if (st.sim.s[i][q]) { recS++; break; } }
    $('#cl-read').innerHTML =
      `<div><dt>Comunidades</dt><dd>${cur.coms.length}</dd></div>` +
      `<div><dt>Modularidad ${R`\(Q\)`}</dt><dd>${cur.Q.toFixed(3)}</dd></div>` +
      `<div><dt>ARI vs {foco, sanas}</dt><dd>${cur.ari.toFixed(2)}</dd></div>` +
      `<div><dt>En crisis a la vez (máx.)</dt><dd class="${simul > 2 ? 'epi' : ''}">${simul}</dd></div>` +
      `<div><dt>Sanas con crisis en la ventana</dt><dd class="${recS ? 'epi' : 'sano'}">${recS} / ${sanas.length}</dd></div>`;
    tex($('#cl-read'));
    comsEl.innerHTML = cur.coms.map((c, q) =>
      `<li class="${cur.crisis[q] > 0 ? 'crisis' : ''}"><span class="sw" style="background:${DN_COL[colOf(cur, q) % DN_COL.length]}"></span>` +
      `<span class="mem">${c.map(i => `<b class="${X0[i] < X0C ? 'epi' : ''}">${i}</b>`).join('')}</span>` +
      `<span class="st">${(100 * cur.crisis[q]).toFixed(1)} % en crisis</span></li>`).join('');
    if (M && !reduced) M.animate(comsEl, { opacity: [0.4, 1] }, { duration: 0.25 });
  }

  function update(dt) {
    const cur = st.cur, k = Math.min(1, dt * 0.005);
    for (let i = 0; i < N; i++) {
      tgt.set(...st.targets[i]).multiplyScalar(SC);
      P[i].lerp(tgt, reduced ? 1 : k);
      const c = colOf(cur, cur.lab[i]) % DN_COL.length, ict = st.sim.s[i][st.fr];
      nodes[i].position.copy(P[i]); nodes[i].material.color.copy(COLS[c]).lerp(CREAMc, ict ? 0 : 0.25);
      nodes[i].material.emissive?.copy(COLS[c]).multiplyScalar(ict ? 0.35 + (reduced ? 0 : 0.2 * Math.sin(st.phase * 10 + i)) : 0);
      nodes[i].scale.setScalar(r0 * (st.hover === i ? 1.35 : 1) * (ict ? 1.25 : 1));
      labels[i].position.copy(P[i]).y += r0 * 2.6;
    }
    bubbles.forEach((b, q) => {
      const com = cur.coms[q];
      b.visible = !!com;
      if (!com) return;
      v3.set(0, 0, 0);
      for (const i of com) v3.addScaledVector(P[i], 1 / com.length);
      b.position.copy(v3);
      b.scale.setScalar(Math.max(...com.map(i => P[i].distanceTo(v3))) + 0.2);
      b.material.color.copy(COLS[colOf(cur, q) % DN_COL.length]);
      b.material.opacity = cur.crisis[q] > 0 ? 0.2 : 0.11;
    });
    for (const p of pairs) {
      const w = cur.W[p.i * N + p.j], u = cur.whi > 0 ? w / cur.whi : 0, same = cur.lab[p.i] === cur.lab[p.j], hot = st.hover < 0 || st.hover === p.i || st.hover === p.j;
      p.m.visible = u > 0.05;
      if (!p.m.visible) continue;
      d.subVectors(P[p.j], P[p.i]);
      const len = d.length();
      p.m.position.copy(P[p.i]).addScaledVector(d, 0.5);
      p.m.quaternion.setFromUnitVectors(UP, d.divideScalar(len || 1));
      const rad = 0.002 + 0.02 * u * u;
      p.m.scale.set(rad, len, rad);
      p.mat.color.copy(same ? COLS[colOf(cur, cur.lab[p.i]) % DN_COL.length] : GREY);
      p.mat.opacity = (same ? 0.2 + 0.7 * u : 0.08 + 0.3 * u) * (hot ? 1 : 0.15);
    }
  }

  function drawMat() {
    const cur = st.cur, { g, w, h } = fit(cm);
    g.clearRect(0, 0, w, h);
    const ord = cur.coms.flatMap(c => [...c].sort((a, b) => X0[a] - X0[b])), size = Math.min(w - 40, h - 22), cell = size / N, ox = (w - size) / 2 + 10, oy = 16;
    g.font = '12.5px KaTeX_Main, "Times New Roman", serif'; g.textAlign = 'center';
    ord.forEach((i, r) => {
      g.fillStyle = X0[i] < X0C ? '#8f3a1f' : C.muted;
      g.fillText(i, ox + (r + .5) * cell, 11); g.fillText(i, ox - 14, oy + (r + .5) * cell + 4);
      ord.forEach((j, c) => {
        const u = i === j ? 0 : cur.W[i * N + j] / (cur.whi || 1);
        g.fillStyle = i === j ? '#faf5ec' : mix([240, 230, 214], ACC, u);
        g.globalAlpha = st.hover < 0 || st.hover === i || st.hover === j ? 1 : 0.35;
        g.fillRect(ox + c * cell + .5, oy + r * cell + .5, cell - 1, cell - 1);
      });
    });
    g.globalAlpha = 1;
    let s0 = 0;
    cur.coms.forEach((c, q) => {
      g.strokeStyle = DN_COL[colOf(cur, q) % DN_COL.length]; g.lineWidth = 2.5;
      g.strokeRect(ox + s0 * cell, oy + s0 * cell, c.length * cell, c.length * cell);
      s0 += c.length;
    });
  }
  function drawTs() {
    const s = st.sim, { g, w, h } = fit(ct), { X, Y, m } = frame(g, w, h, [0, s.T], [-2.2, 1.6]), a = startOf(st.fr), b = a + nv() - 1;
    g.fillStyle = rgba([197, 138, 42], .16);
    g.fillRect(X(s.t[a]), m.t, X(s.t[b]) - X(s.t[a]), h - m.t - m.b);
    g.strokeStyle = rgba([42, 32, 24], .3); g.setLineDash([3, 4]); g.beginPath(); g.moveTo(m.l, Y(-0.5)); g.lineTo(w - 10, Y(-0.5)); g.stroke(); g.setLineDash([]);
    const stepK = Math.max(1, Math.floor(s.n / (w * 1.5)));
    for (let i = N - 1; i >= 0; i--) {
      const epi = X0[i] < X0C, on = st.hover < 0 || st.hover === i;
      g.beginPath();
      for (let k = 0; k < s.n; k += stepK) { const px = X(s.t[k]), py = Y(s.x[i][k]); k ? g.lineTo(px, py) : g.moveTo(px, py); }
      stroke(g, rgba(epi ? ACC : SAGE, on ? (epi ? 1 : .55) : .1), epi || st.hover === i ? 1.6 : 1);
    }
    const px = X(s.t[st.fr]);
    g.strokeStyle = C.ink; g.lineWidth = 1.5; g.beginPath(); g.moveTo(px, m.t); g.lineTo(px, h - m.b); g.stroke();
  }

  let lastDraw = 0;
  function draw(now = performance.now()) {
    if (!st.sim) return;
    refresh();
    update(now - lastDraw || 16); lastDraw = now;
    const cp = Math.cos(st.pitch), dist = st.dist * st.fit;
    camera.position.set(dist * cp * Math.sin(st.yaw), dist * Math.sin(st.pitch), dist * cp * Math.cos(st.yaw));
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    if (st.tshown !== st.fr) {
      st.tshown = st.fr;
      drawMat(); drawTs();
      iT.value = st.fr; iT.style.setProperty('--fill', st.fr / (st.sim.n - 1) * 100 + '%');
      const o = $('#cl-tout'), a = startOf(st.fr), txt = `t \\in [${Math.round(st.sim.t[a])},\\ ${Math.round(st.sim.t[a + nv() - 1])}]`;
      if (window.katex) katex.render(txt, o, { throwOnError: false }); else o.textContent = txt;
    }
    if (st.hover >= 0) showTip();
  }
  const redraw2d = () => { st.tshown = -1; draw(); };
  function showTip() {
    const i = st.hover, cur = st.cur, key = i + ':' + st.curKey;
    if (key !== st.tipKey) {
      st.tipKey = key;
      const nb = [...Array(N).keys()].filter(j => j !== i).sort((a, b) => cur.W[i * N + b] - cur.W[i * N + a])[0];
      tip.innerHTML = `nodo ${i} · ${X0[i] < X0C ? 'foco' : 'sana'} · ${R`\(x_0 = ${X0[i].toFixed(3)}\)`}<br>` +
        `comunidad ${cur.lab[i] + 1} de ${cur.coms.length}<br>mayor ${R`\(W_{${i}j}\)`}: con ${nb} (${cur.W[i * N + nb].toFixed(2)})`;
      tex(tip);
    }
    v3.copy(P[i]).project(camera);
    tip.style.left = clamp((v3.x + 1) / 2 * stage.clientWidth + 14, 0, Math.max(0, stage.clientWidth - 250)) + 'px';
    tip.style.top = ((1 - v3.y) / 2 * stage.clientHeight + 16) + 'px';
  }

  // ---- simulación y momentos de la cascada
  function simulateNow() {
    const K = +iK.value, sz = +iZ.value;
    st.sim = simDN({ K, sx: sz > 0 ? 0.02 : 0, sz, T: st.T, seed: st.seed, x0: X0 });
    st.cache.clear(); st.curKey = '';
    if (st.fr > st.sim.n - 1) st.fr = 0;
    iT.max = st.sim.n - 1;
    stageBtns();
    redraw2d();
  }
  // antes: última ventana sin crisis sanas previa a la de más regiones en crisis a la vez; después: la primera tranquila siguiente
  function stages() {
    const s = st.sim, n = nv(), sanas = X0.map((v, i) => i).filter(i => X0[i] >= X0C), wins = [];
    for (let a = 0; a + n <= s.n; a += STEP) {
      let simul = 0, rec = 0;
      for (let q = a; q < a + n; q++) { let t = 0; for (let i = 0; i < N; i++) t += s.s[i][q]; simul = Math.max(simul, t); }
      for (const i of sanas) for (let q = a; q < a + n; q++) if (s.s[i][q]) { rec++; break; }
      wins.push({ a, simul, rec });
    }
    let wd = 0;
    wins.forEach((w, q) => { if (w.simul > wins[wd].simul) wd = q; });
    const quiet = wins.map((w, q) => w.rec === 0 ? q : -1).filter(q => q >= 0);
    const pick = q => q == null || q < 0 ? null : wins[q].a + Math.round(n / 2);
    return wins[wd].rec ? { antes: pick(quiet.filter(q => q < wd).pop()), durante: pick(wd), despues: pick(quiet.find(q => q > wd)) } : {};
  }
  function stageBtns() {
    st.stages = stages();
    $$('#cl-stage button').forEach(b => { b.disabled = st.stages[b.dataset.v] == null; b.setAttribute('aria-pressed', 'false'); });
  }

  // ---- interacción
  const ray = new T.Raycaster(), ndc = new T.Vector2();
  let lx = 0, ly = 0;
  canvas.addEventListener('pointerdown', e => { st.drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing'; });
  canvas.addEventListener('pointermove', e => {
    if (st.drag) {
      st.yaw -= (e.clientX - lx) * 0.009; st.pitch = clamp(st.pitch + (e.clientY - ly) * 0.007, -1.2, 1.35);
      lx = e.clientX; ly = e.clientY;
    } else if (st.sim) {
      const b = canvas.getBoundingClientRect();
      ndc.set((e.clientX - b.left) / b.width * 2 - 1, 1 - (e.clientY - b.top) / b.height * 2);
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObjects(nodes)[0], h = hit ? hit.object.userData.i : -1;
      if (h !== st.hover) { st.hover = h; st.tshown = -1; }
      tip.classList.toggle('on', st.hover >= 0);
      canvas.style.cursor = st.hover >= 0 ? 'pointer' : 'grab';
    }
    draw();
  });
  const release = () => { st.drag = false; canvas.style.cursor = 'grab'; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', () => { if (!st.drag && st.hover >= 0) { st.hover = -1; tip.classList.remove('on'); redraw2d(); } });
  $('#cl-zin').onclick = () => { st.dist = clamp(st.dist * 0.85, 2.6, 9); draw(); };
  $('#cl-zout').onclick = () => { st.dist = clamp(st.dist / 0.85, 2.6, 9); draw(); };
  $('#cl-reset').onclick = () => { Object.assign(st, VIEW); draw(); };

  let timer;
  const rebuild = () => { busy.classList.add('on'); clearTimeout(timer); timer = setTimeout(() => { simulateNow(); busy.classList.remove('on'); }, 160); };
  const pressed = (host, test) => $$(`${host} .btn`).forEach(b => b.setAttribute('aria-pressed', String(test(b._v))));
  const upK = bindRange(iK, $('#cl-ko'), 3, () => { pressed('#cl-esc', () => false); rebuild(); });
  const upZ = bindRange(iZ, $('#cl-szo'), 5, () => { pressed('#cl-esc', () => false); st.T = +iZ.value > 0 ? 100000 : 60000; rebuild(); });
  bindRange(iN, $('#cl-no'), 0, () => { setN(); rebuild(); });
  presetButtons($('#cl-met'), METHODS.map(m => ({ v: m.id, label: m.label })), v => { st.method = v; pressed('#cl-met', q => q === v); st.curKey = ''; redraw2d(); });
  presetButtons($('#cl-esc'), SCEN.map(s => ({ v: s.id, label: s.label })), v => {
    const s = SCEN.find(q => q.id === v);
    iK.value = s.K; iZ.value = s.sz; upK(); upZ(); st.T = s.T; if (s.seed) st.seed = s.seed;
    pressed('#cl-esc', q => q === v); $('#cl-escnote').textContent = s.note; st.fr = 0; rebuild();
  });
  $$('#cl-met .btn, #cl-esc .btn').forEach(b => { b.style.fontFamily = 'var(--font-tex)'; });
  tex($('#cl-esc'));
  pressed('#cl-met', q => q === st.method);
  $$('#cl-stage button').forEach(b => b.onclick = () => {
    const fr = st.stages[b.dataset.v];
    if (fr == null) return;
    $$('#cl-stage button').forEach(o => o.setAttribute('aria-pressed', String(o === b)));
    st.fr = fr; setPlay(false); redraw2d();
  });
  $('#cl-new').onclick = () => { st.seed++; pressed('#cl-esc', () => false); rebuild(); };
  $('#cl-spin').onclick = e => { st.spin = !st.spin; e.currentTarget.setAttribute('aria-pressed', String(st.spin)); };
  $('#cl-spin').setAttribute('aria-pressed', String(st.spin));
  const setPlay = v => { st.playing = v; $('#cl-play').classList.toggle('is-playing', v); };
  $('#cl-play').onclick = () => setPlay(!st.playing);
  iT.addEventListener('input', () => { st.fr = +iT.value; setPlay(false); draw(); });
  const seek = e => { const r = ct.getBoundingClientRect(); st.fr = Math.round(clamp((e.clientX - r.left - 36) / (r.width - 46), 0, 1) * (st.sim.n - 1)); setPlay(false); draw(); };
  ct.addEventListener('pointerdown', e => { seek(e); ct.setPointerCapture(e.pointerId); ct.onpointermove = seek; });
  ct.addEventListener('pointerup', () => { ct.onpointermove = null; });

  const size = () => {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    st.fit = Math.max(1, 1.3 / camera.aspect);
    redraw2d();
  };
  let last = 0, acc = 0;
  const loop = now => {
    requestAnimationFrame(loop);
    if (!st.vis || document.hidden || !st.sim) { last = now; return; }
    const dt = Math.min(60, now - last || 16); last = now;
    st.phase += dt / 1000;
    if (st.playing) { acc += dt / 1000 * st.sim.n / 45; if (acc >= 1) { st.fr = (st.fr + Math.floor(acc)) % st.sim.n; acc %= 1; } }   // 45 s por recorrido
    if (st.spin && !st.drag) st.yaw += dt / 1000 * 0.1;
    draw(now);
  };
  makeObjects();
  watchVisible($('#labcl'), v => { st.vis = v; });
  setPlay(st.playing);
  $$('#cl-esc .btn')[1].click();
  clearTimeout(timer); simulateNow(); busy.classList.remove('on');
  size();
  requestAnimationFrame(loop);
  return size;
}

/* ================= láminas de figuras ================= */
// tamaño en píxeles de cada figura: reserva su espacio antes de que cargue, para que la página no salte
const FIG_DIM = {"DN_grafo":[1700,708],"DN_W_sub":[1200,1000],"DN_W_sup":[1200,1000],"DN_W_ruido":[1200,1000],"DN_tiempo_sin":[1700,1569],"DN_tiempo_con":[1700,1569],"DN_series_ruido":[1700,915],"DN_resumen":[1699,546],"CL_sub":[1700,316],"CL_sup":[1700,316],"CL_ruido":[1700,316],"CL_sin_antes":[1700,948],"CL_sin_durante":[1700,948],"CL_sin_despues":[1700,948],"CL_cascada_ruido":[1700,1569],"CL_rui_antes":[1700,948],"CL_rui_durante":[1700,948],"CL_rui_despues":[1700,948],"G3_48_0":[702,690],"G3_50_0":[671,690],"G3_50_1":[671,690],"G3_50_2":[671,690],"G3_52_0":[797,790],"G_11_0":[989,690],"G_22_0":[1189,490],"G_22_1":[1189,490],"G_22_2":[1189,490],"G_22_3":[1189,490],"G_25_0":[989,490],"G_29_0":[1202,590],"G_32_0":[671,590],"G_34_0":[1790,495],"G_37_0":[690,490],"G_37_1":[690,490],"G_45_1":[690,490],"P_12_0":[554,455],"P_18_0":[1987,590],"P_19_0":[1989,440],"P_21_0":[1498,1769],"P_23_0":[717,474],"P_24_0":[726,474],"P_25_0":[695,474],"P_26_0":[780,551],"P_36_0":[1589,1651],"P_38_0":[1589,1651],"P_40_0":[1603,1651],"P_46_0":[1590,1110],"P_46_1":[1590,1110],"P_46_2":[1590,1110],"P_48_0":[1589,1150],"P_48_1":[1589,1150],"P_48_2":[1590,1150],"P_50_0":[1289,590],"P_50_1":[1289,590],"P_50_2":[1289,590],"P_51_0":[1289,590],"P_51_1":[1289,590],"P_54_0":[1289,889],"P_54_1":[1289,889],"P_54_2":[1289,889],"P_54_3":[1289,889],"P_54_4":[1289,890],"P_56_0":[1362,760],"P_59_0":[890,409],"P_60_0":[570,409],"P_61_0":[1073,409],"P_64_0":[1303,1030],"P_65_0":[614,434],"P_67_1":[1489,440],"P_67_2":[1489,440],"P_69_0":[690,409],"P_70_1":[1389,440],"P_9_0":[556,457]};
const F = (id, label, cap) => ({ src: `assets/fig/${id}.png`, label, cap, w: FIG_DIM[id][0], h: FIG_DIM[id][1] });
const PLATES = {
  n1: [
    { t: 'Permitividad', figs: [F('P_9_0', '', R`Función \(h(x)\) para cinco valores de \(x_0\): un escalón suave de altura \(3\) centrado en \(x = -0.5\) que \(x_0\) desplaza verticalmente.`)] },
    { t: 'Nullclinas', figs: [F('P_12_0', '', R`Nullclina cúbica \(\dot x = 0\) y nullclina sigmoidal \(\dot z = 0\). Al subir \(x_0\) el cruce pasa de la rama media a la rama izquierda estable.`)] },
    { t: 'Series de tiempo', wide: true, figs: [F('P_18_0', '', R`\(x(t)\) y \(z(t)\) para \(x_0 \in \{2.0,\ 2.5,\ 2.8,\ 3.0,\ 3.5\}\): oscilaciones de relajación bajo el umbral, reposo sobre él.`)] },
    { t: 'Plano de fase', wide: true, figs: [F('P_19_0', '', R`Trayectorias sobre las nullclinas: ciclo límite para \(x_0 \lt 2.91\) y convergencia al punto fijo para \(x_0 \gt 2.91\).`)] },
    { t: 'Zona crítica', wide: true, figs: [F('P_21_0', '', R`Barrido fino \(x_0 \in \{2.89,\ \dots,\ 2.92\}\) desde la misma condición inicial \((x, z) = (-1.2,\ 3.0)\): el intervalo entre crisis se alarga hasta que la oscilación desaparece.`)] },
    { t: 'Período', figs: [F('P_23_0', '', R`Período \(T(x_0)\) medido entre máximos: crece de \(\sim 4\,850\) a más de \(20\,000\) al acercarse a \(x_0 = 2.91\).`)] },
    { t: 'Frecuencia', figs: [F('P_24_0', '', R`Frecuencia de crisis por dos métodos, \(f = 1/T\) y conteo de máximos. Cae continuamente a cero en el umbral.`)] },
    { t: 'Amplitud', figs: [F('P_25_0', '', R`Amplitud \(\max x - \min x\) en régimen permanente: se mantiene alta y cae de golpe.`)] },
    { t: 'Bifurcación', figs: [F('P_26_0', '', R`Diagrama de bifurcación: máximos de \(x(t)\), puntos fijos estables (negro) e inestables (rojo) en función de \(x_0\).`)] },
  ],
  n2series: [
    { t: 'Excitabilidades iguales', wide: true, figs: [
      F('P_36_0', R`\(K = 0.01\)`, R`Regiones idénticas con \(x_{0,1} = x_{0,2} \in \{2.0,\ \dots,\ 3.5\}\) y condiciones iniciales aleatorias. Con \(K = 0.01\) cada una conserva su fase.`),
      F('P_38_0', R`\(K = 0.5\)`, R`Con \(K = 0.5\) las dos regiones se sincronizan tras un transitorio.`),
      F('P_40_0', R`\(K = 5\)`, R`Con \(K = 5\) la sincronía es casi inmediata.`)] },
    { t: 'Foco + vecina', wide: true, figs: [
      F('P_46_0', R`\(K = 0\)`, R`Pares \((x_{0,1},\ x_{0,2}) \in \{(2.0, 3.2),\ (2.5, 3.2),\ (2.8, 3.2),\ (2.0, 2.8)\}\). Sin acoplamiento la región sana no hace crisis.`),
      F('P_46_1', R`\(K = 0.2\)`, R`Acoplamiento débil \(K = 0.2\): aparecen reclutamientos parciales.`),
      F('P_46_2', R`\(K = 1.0\)`, R`Acoplamiento fuerte \(K = 1.0\): la región sana es arrastrada en cada crisis.`)] },
    { t: 'Onda rectangular', wide: true, figs: [
      F('P_48_0', R`\(K = 0\)`, R`\(z_i(t)\) junto a la onda rectangular \(h(x_i(t))\) que lo fuerza, para pares simétricos y uno asimétrico.`),
      F('P_48_1', R`\(K = 0.2\)`, R`Mismos pares con \(K = 0.2\).`),
      F('P_48_2', R`\(K = 1.0\)`, R`Mismos pares con \(K = 1.0\).`)] },
    { t: 'Par simétrico', figs: [
      F('P_50_0', R`\(K = 0\)`, R`Dos regiones idénticas, \(x_0 = (2.5,\ 2.5)\): \(x_i(t)\) y \(h(x_i(t))\).`),
      F('P_50_1', R`\(K = 0.2\)`, R`\(x_0 = (2.5,\ 2.5)\) con \(K = 0.2\).`),
      F('P_50_2', R`\(K = 1.0\)`, R`\(x_0 = (2.5,\ 2.5)\) con \(K = 1.0\).`)] },
    { t: 'Par asimétrico', figs: [
      F('P_51_0', R`\(K = 0\)`, R`\(x_0 = (2.0,\ 2.8)\): dos regiones epileptogénicas con frecuencias propias distintas.`),
      F('P_51_1', R`\(K = 1.0\)`, R`\(x_0 = (2.0,\ 2.8)\) con \(K = 1.0\): la más lenta sigue a la más rápida.`)] },
    { t: 'C.I. aleatorias', wide: true, figs: [
      F('P_54_0', R`\((2.5,\ 2.5),\ K = 0\)`, R`\(x_i(t)\), \(z_i(t)\) y \(h(x_i(t))\) con condiciones iniciales aleatorias fijas, para comparar entre valores de \(K\).`),
      F('P_54_1', R`\((2.5,\ 2.5),\ K = 0.2\)`, R`Mismas condiciones iniciales, \(K = 0.2\).`),
      F('P_54_2', R`\((2.5,\ 2.5),\ K = 1.0\)`, R`Mismas condiciones iniciales, \(K = 1.0\).`),
      F('P_54_3', R`\((2.0,\ 2.8),\ K = 0\)`, R`Par asimétrico \(x_0 = (2.0,\ 2.8)\) sin acoplamiento.`),
      F('P_54_4', R`\((2.0,\ 2.8),\ K = 1.0\)`, R`Par asimétrico \(x_0 = (2.0,\ 2.8)\) con \(K = 1.0\).`)] },
  ],
  n2maps: [
    { t: 'Actividad ictal', wide: true, figs: [F('P_56_0', '', R`Fracción del tiempo en estado ictal en el plano \((x_{0,1},\ x_{0,2})\) para \(K \in \{0,\ 0.2,\ 1,\ 3\}\). Al crecer \(K\) la frontera deja de ser \(x_0 = 2.91\) por región y pasa a depender de ambas.`)] },
    { t: 'Regímenes', wide: true, figs: [F('P_59_0', '', R`Mapa de regímenes I a V en \((K,\ x_{0,2})\) con \(x_{0,1} = 2.5\): reproduce la figura 5A de Proix et al.`)] },
    { t: 'Ciclo de trabajo', figs: [F('P_60_0', '', R`Fracción de tiempo ictal de la región 2 en \((K,\ x_{0,2})\).`)] },
    { t: 'Duración', figs: [F('P_61_0', '', R`Duración media de las crisis en cada región en \((K,\ x_{0,2})\).`)] },
  ],
  n2noise: [
    { t: 'Series con ruido', figs: [F('P_64_0', '', R`\(x_0 = (2.5,\ 3.5)\), \(K = 0.5\), \(\sigma_x = 0.02\) y \(\sigma_z\) creciente: el ruido en la permitividad dispara crisis en la región sana.`)] },
    { t: 'Probabilidad', figs: [F('P_65_0', '', R`Probabilidad de que la región 2 haga al menos una crisis (\(30\) réplicas por punto) en función de \(\sigma_z\), para \(K \in \{0,\ 0.3,\ 0.6,\ 0.9\}\).`)] },
  ],
  n2scale: [
    { t: 'Campo medio', wide: true, figs: [F('P_67_1', '', R`Campo medio \(\bar x(t)\) para \(N = 2, \dots, 100\) con \(K \cdot N = 1\), desde condiciones casi sincrónicas y aleatorias.`)] },
    { t: 'Sincronía', wide: true, figs: [F('P_67_2', '', R`Dispersión transversal \(\sigma_\perp^2(t)\) y su mediana en función de \(N\) para cuatro configuraciones.`)] },
    { t: 'Exponente', figs: [F('P_69_0', '', R`Exponente transversal \(\lambda_\perp\) sobre la órbita sincrónica en función de \(K \cdot N\), para \(x_0 = 2.5\) y \(x_0 = 2.8\).`)] },
    { t: 'Fronteras', figs: [F('P_70_1', '', R`Regímenes para \(N = 2\) y fronteras superpuestas para \(N \in \{2,\ 4,\ 10,\ 50\}\) con \(c = K \cdot N\).`)] },
  ],
  grafo3d: [
    { t: 'Capas temporales', figs: [F('G3_52_0', '', R`Cinco instantes apilados según \(t\). El color de cada nodo es \(x_i(t)\) y en cada capa se dibujan las aristas con \(W_{ij}\) sobre el percentil \(60\).`)] },
    { t: 'Espacio de fases', figs: [
      F('G3_50_0', R`\(t = 0.1\,T\)`, R`Cada nodo en \((x_i,\ z_i,\ x_{0,i})\) al \(10\,\%\) de la simulación. El color y el grosor de las aristas siguen \(W_{ij}(t)\).`),
      F('G3_50_1', R`\(t = 0.5\,T\)`, R`El mismo grafo a mitad de la simulación.`),
      F('G3_50_2', R`\(t = 0.9\,T\)`, R`El mismo grafo al \(90\,\%\) de la simulación.`)] },
    { t: 'Grafo estructural 3D', figs: [F('G3_48_0', '', R`Grafo \(A\) con la disposición en el plano y la excitabilidad \(x_{0,i}\) como altura. En rojo los nodos con \(x_0 \lt 2.91\).`)] },
  ],
  clusters: [
    { t: 'Clusters por método', wide: true, figs: [
      F('CL_sub', R`\(K_{\rm sub}\)`, R`Las cinco particiones sobre la disposición de resortes de \(W\) y \(W\) reordenada por el consenso, con \(K_{\rm sub}\): el foco aislado y las sanas en un bloque.`),
      F('CL_sup', R`\(K_{\rm sup}\)`, R`Con \(K_{\rm sup}\): la región 7 queda con el foco y la propagación de etiquetas junta todo en una comunidad.`),
      F('CL_ruido', R`\(K_{\rm ruido}\)`, R`Con ruido y \(K_{\rm ruido}\): de nuevo \(\{0,1\}\) y \(\{2,\dots,9\}\) en los cinco métodos.`)] },
    { t: 'Cascada sin ruido', wide: true, figs: [
      F('CL_sin_antes', 'Antes', R`\(t \in [7\,001,\ 11\,001]\): el foco se separa, \(\{0\}\) y \(\{1,\dots,9\}\).`),
      F('CL_sin_durante', 'Durante', R`\(t \in [9\,001,\ 13\,001]\), nueve regiones en crisis a la vez: \(\{0,1\}\), \(\{2,7\}\) y \(\{3,4,5,6,8,9\}\).`),
      F('CL_sin_despues', 'Después', R`\(t \in [14\,001,\ 18\,001]\): la red vuelve a ser una sola comunidad.`)] },
    { t: 'Cascada con ruido', wide: true, figs: [
      F('CL_rui_antes', 'Antes', R`Realización 18 con \(K_{\rm ruido}\), \(t \in [78\,001,\ 82\,001]\): \(\{0,1\}\) y \(\{2,\dots,9\}\).`),
      F('CL_rui_durante', 'Durante', R`\(t \in [80\,001,\ 84\,001]\): las primeras reclutadas se unen al foco, \(\{0,1,2,7\}\) y \(\{3,4,5,6,8,9\}\).`),
      F('CL_rui_despues', 'Después', R`\(t \in [85\,001,\ 89\,001]\): \(\{0,1,3,4,5,6,8,9\}\) y \(\{2,7\}\).`),
      F('CL_cascada_ruido', 'En el tiempo', R`Comunidades en ventanas de \(4\,000\) con paso \(1\,000\) para la realización con la cascada: estado de crisis, pertenencia a la comunidad del foco, fracción en crisis y tamaños.`)] },
  ],
  dinamica: [
    { t: 'Grafo completo', wide: true, figs: [F('DN_grafo', '', R`Matriz de adyacencia \(A_{ij} = 1\) y su grafo: una sola componente conexa. En rojo el foco \((x_0 = 2.3,\ 2.4)\).`)] },
    { t: 'Matriz dinámica', wide: true, figs: [
      F('DN_W_sub', R`\(K_{\rm sub}\)`, R`\(W_{ij} = \max(\operatorname{corr}(z_i, z_j), 0)\) con \(K_{\rm sub} = 2.115\): un bloque \(\{0,1\}\) y otro \(\{2,\dots,9\}\).`),
      F('DN_W_sup', R`\(K_{\rm sup}\)`, R`\(W\) con \(K_{\rm sup} = 2.45\); el consenso da \(\{0,1,7\}\) y \(\{2,3,4,5,6,8,9\}\).`),
      F('DN_W_ruido', R`\(K_{\rm ruido}\)`, R`\(W\) con ruido y \(K_{\rm ruido} = 2.327\) \((T = 300\,000)\): de nuevo \(\{0,1\}\) y \(\{2,\dots,9\}\).`)] },
    { t: 'Comunidades en el tiempo', wide: true, figs: [
      F('DN_tiempo_sin', 'Sin ruido', R`\(K_{\rm sup}\), \(T = 150\,000\), ventanas de \(4\,000\) con paso \(1\,000\): (a) estado de crisis, (b) pertenencia a la comunidad del foco, (c) fracción de la ventana con la comunidad en crisis y (d) tamaño de la comunidad del foco.`),
      F('DN_tiempo_con', 'Con ruido', R`Los mismos paneles con ruido y \(K_{\rm ruido}\): solo la región 7 se une al foco, en las ventanas con sus crisis.`)] },
    { t: 'Series con ruido', wide: true, figs: [F('DN_series_ruido', '', R`\(x_i(t)\) y estado de crisis con \(K_{\rm ruido}\), \(\sigma_x = 0.02\), \(\sigma_z = 7.5	imes 10^{-4}\): el foco tiene \(74\) crisis por nodo y la región 7, tres.`)] },
    { t: 'Realizaciones', wide: true, figs: [F('DN_resumen', '', R`\(20\) realizaciones con \(K = 0\) contra \(20\) con \(K_{\rm ruido}\): probabilidad de quedar en la comunidad del foco y fracción del tiempo con la comunidad en crisis, por región sana.`)] },
  ],
  grafo: [
    { t: 'Grafo estructural', figs: [F('G_11_0', '', R`Grafo \(A\) para \(N = 10\): en rojo los nodos con \(x_0 \lt 2.91\), en azul el resto. Etiquetas: índice \(i\) y \(x_{0,i}\).`)] },
    { t: 'Grafo dinámico', figs: [F('G_34_0', '', R`Pesos \(W_{ij}(t)\) en tres instantes de la simulación: \(t = 0\), \(t = T/2\) y \(t = 0.9\,T\).`)] },
    { t: 'Series', wide: true, figs: [
      F('G_22_2', 'Todas', R`\(x_i(t)\) de las diez regiones con \(K = 0.1\): salto inicial a la rama alta y decaimiento lento.`),
      F('G_22_0', 'Epileptogénicas', R`Regiones \(0\) y \(1\) \((x_0 = 2.3,\ 2.4)\).`),
      F('G_22_1', 'Sanas', R`Regiones \(2\) a \(9\).`),
      F('G_22_3', 'Actividad media', R`Observables colectivos \(m_N(t)\), \(m_{\rm epi}(t)\) y \(m_{\rm sano}(t)\).`)] },
    { t: 'Espacio-tiempo', wide: true, figs: [F('G_29_0', '', R`Mapa espacio-tiempo \(X_{i,k} = x_i(t_k)\).`)] },
    { t: 'Influencia', figs: [F('G_32_0', '', R`Matriz de influencia acumulada \(I_{ij} = \int_0^T W_{ij}(t)\,dt\).`)] },
    { t: 'Reclutamiento', figs: [
      F('G_25_0', R`\(T_i\)`, R`Tiempos de reclutamiento \(T_i\) por región (umbral \(x_{\rm th} = 1.0\), duración mínima \(\Delta t = 50\)).`),
      F('G_45_1', R`\(R(K)\)`, R`Fracción de regiones sanas reclutadas \(R(K)\).`)] },
    { t: 'Centralidad', wide: true, figs: [
      F('G_37_0', R`\(C_B\) vs \(H^{dyn,out}\)`, R`Centralidad de intermediación \(C_B(i)\) contra hub dinámico de salida \(H_i^{dyn,out}\).`),
      F('G_37_1', R`\(C_B\) vs \(T_i\)`, R`Centralidad de intermediación \(C_B(i)\) contra tiempo de reclutamiento \(T_i\).`)] },
  ],
};
let M = null;   // módulo Motion, si cargó
function plates() {
  const box = $('#lightbox'), bimg = $('img', box), bcap = $('p', box);
  const close = () => {
    if (!box.open || box.dataset.closing) return;
    if (!M || reduced) { box.close(); return; }
    box.dataset.closing = '1';
    M.animate(box, { opacity: [1, 0], scale: [1, 0.97] }, { duration: 0.15, ease: [0.23, 1, 0.32, 1] }).then(() => {
      box.close();
      delete box.dataset.closing;
      box.style.opacity = ''; box.style.transform = ''; box.style.scale = '';
    });
  };
  $('button', box).onclick = close;
  box.addEventListener('click', e => { if (e.target === box) close(); });
  box.addEventListener('cancel', e => { e.preventDefault(); close(); });     // tecla Escape
  $$('[data-plates]').forEach(host => {
    PLATES[host.dataset.plates].forEach((p, idx) => {
      const fig = document.createElement('figure');
      fig.className = 'plate' + (p.wide ? ' wide' : '');
      fig.dataset.reveal = ''; fig.dataset.delay = p.wide ? 0 : (idx % 2) * 0.08;
      let cur = 0;
      const tabs = p.figs.length > 1 ? `<div class="plate-tabs" role="tablist">${p.figs.map((f, k) => `<button type="button" role="tab" aria-selected="${k === 0}">${f.label}</button>`).join('')}</div>` : '';
      fig.innerHTML = `${tabs}<button type="button" class="plate-img" aria-label="Ampliar figura: ${p.t}"><img loading="lazy" alt="Figura: ${p.t}" src="${p.figs[0].src}" width="${p.figs[0].w}" height="${p.figs[0].h}"></button><figcaption><b>${p.t}</b><span>${p.figs[0].cap}</span></figcaption>`;
      const img = $('img', fig), cap = $('figcaption span', fig);
      $$('.plate-tabs button', fig).forEach((b, k) => b.onclick = () => {
        if (k === cur) return;
        const dirn = k > cur ? 1 : -1; cur = k;
        $$('.plate-tabs button', fig).forEach((o, q) => o.setAttribute('aria-selected', String(q === k)));
        img.width = p.figs[k].w; img.height = p.figs[k].h; img.src = p.figs[k].src; cap.innerHTML = p.figs[k].cap; tex(cap);
        if (M && !reduced) M.animate(img, { opacity: [0, 1], x: [14 * dirn, 0] }, { duration: 0.35, ease: [0.22, 1, 0.36, 1] });
      });
      $('.plate-img', fig).onclick = () => {
        bimg.src = p.figs[cur].src; bimg.alt = 'Figura: ' + p.t; bcap.innerHTML = p.figs[cur].cap; tex(bcap);
        box.showModal();
        if (M && !reduced) M.animate(bimg, { opacity: [0, 1], scale: [0.94, 1] }, { type: 'spring', stiffness: 260, damping: 26 });
      };
      host.append(fig);
    });
  });
}

/* ================= navegación ================= */
function nav() {
  const links = $$('#navlist a'), map = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    const cur = map.get(e.target.id), ul = $('#navlist');
    links.forEach(a => a.setAttribute('aria-current', String(a === cur)));
    // en pantallas angostas la lista se desplaza sola para que el enlace activo quede a la vista
    const li = cur.parentElement;
    ul.scrollTo({ left: li.offsetLeft - ul.offsetLeft - (ul.clientWidth - li.offsetWidth) / 2, behavior: reduced ? 'auto' : 'smooth' });
  }), { rootMargin: '-45% 0px -50% 0px' });
  map.forEach((_, id) => io.observe(document.getElementById(id)));
}

/* ================= Motion: entradas, scroll y contadores ================= */
function motion(m) {
  M = m;
  const { animate, inView, scroll, stagger } = m, ease = [0.22, 1, 0.36, 1];
  const root = document.documentElement;
  if (reduced) { root.classList.remove('pre'); return; }
  root.classList.add('js-motion');
  animate('[data-hero]', { opacity: [0, 1], y: [26, 0] }, { duration: 0.9, delay: stagger(0.1), ease });
  root.classList.remove('pre');
  const elOf = a => a instanceof Element ? a : a.target;   // v11 entrega la entrada del observer; v12, el elemento
  inView('[data-reveal]', arg => {
    const el = elOf(arg);
    animate(el, { opacity: [0, 1], y: [14, 0] }, { duration: 0.5, delay: +(el.dataset.delay || 0), ease });
  }, { margin: '0px 0px -8% 0px' });
  // manchas de color que derivan lentamente y se desplazan con el scroll
  [['.b1', [0, 90, -50, 0], [0, 70, 120, 0], 46], ['.b2', [0, -110, -30, 0], [0, -60, 80, 0], 58], ['.b3', [0, 70, -90, 0], [0, -90, -20, 0], 52]]
    .forEach(([sel, x, y, duration]) => animate(sel, { x, y, scale: [1, 1.14, 0.94, 1] }, { duration, repeat: Infinity, ease: 'easeInOut' }));
  scroll(animate('#bgDrift', { y: ['0vh', '-18vh'] }, { ease: 'linear' }));
  scroll(animate('#brain', { y: [0, 90], opacity: [1, 0.15] }, { ease: 'linear' }), { target: $('.hero'), offset: ['start start', 'end start'] });
  scroll(animate('#progress', { scaleX: [0, 1] }, { ease: 'linear' }));
  $$('.num').forEach(n => scroll(animate(n, { y: [26, -26] }, { ease: 'linear' }), { target: n.closest('.sec-head'), offset: ['start end', 'end start'] }));
  inView('[data-count]', arg => {
    const el = elOf(arg);
    const to = +el.dataset.count, dec = +el.dataset.dec, suf = el.dataset.suffix || '';
    const fmt = v => (dec ? v.toFixed(dec) : Math.round(v).toLocaleString('es-CL')) + suf;
    animate(0, to, { duration: 1.4, ease: 'easeOut', onUpdate: v => { el.textContent = fmt(v); } });
  }, { margin: '0px 0px -10% 0px' });
}

/* ================= arranque ================= */
function boot() {
  background(); plates(); nav();
  const redraws = [brain3d(), hero(), lab1(), lab2(), labH(), labNoise(), lab3(), lab3d(), labDN(), labCL()];
  let rt;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => redraws.forEach(f => f()), 120); });
  tex(document.body);
  // los ejes de los gráficos usan la tipografía de LaTeX: se redibujan cuando termina de cargar
  if (document.fonts) document.fonts.load('13px KaTeX_Main').then(() => redraws.forEach(f => f())).catch(() => {});
  import('https://cdn.jsdelivr.net/npm/motion@11.18.2/+esm').then(motion).catch(() => document.documentElement.classList.remove('pre'));
  setTimeout(() => document.documentElement.classList.remove('pre'), 2500);
}
document.readyState === 'loading' ? addEventListener('DOMContentLoaded', boot) : boot();
})();
