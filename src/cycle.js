// Cardiac cycle engine (Wiggers diagram).  Everything — shape keys, ECG trace, blood flow, heart
// sounds — is derived from the same phase variable, so they stay locked together at any heart rate.

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const sstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
const gauss = (x, mu, sd) => Math.exp(-0.5 * ((x - mu) / sd) ** 2);

export function timings(hr) {
  const T = 60 / hr;
  const k = clamp(Math.sqrt(T), 0.6, 1.12);
  const PR = clamp(0.19 - 0.00055 * hr, 0.09, 0.2);
  const QT = clamp(0.4 * Math.sqrt(T), 0.2, 0.5);
  const tA0 = 0.045;                    // atrial mechanical onset (after P onset)
  const dA = Math.min(0.2, 1.3 * (PR + 0.02));
  const tV0 = PR + 0.045;               // ventricular mechanical onset (just after R)
  const IVC = 0.05 * k;                 // isovolumetric contraction
  const xES = QT - 0.025;               // end of ejection, measured from tV0
  const xAVO = xES + 0.07 * k;          // mitral / tricuspid opening
  const xAS = T - PR;                   // atrial systole expressed in ventricular time
  return { T, k, PR, QT, tA0, dA, tV0, IVC, xES, xAVO, xAS };
}

function atrial(tau, c) {
  // quick contraction, slower relaxation
  let s = (tau - c.tA0) / c.dA;
  if (s < 0) s = (tau + c.T - c.tA0) / c.dA;      // tail of the previous beat (never at rest rates)
  if (s < 0 || s > 1) return 0;
  return sstep(0, 0.42, s) * (1 - sstep(0.42, 1, s));
}

function ventricular(x, c) {
  if (x < c.IVC) return 0.07 * sstep(0, c.IVC, x);
  if (x < c.xES) {
    const e = (x - c.IVC) / (c.xES - c.IVC);
    return 0.07 + 0.93 * (1 - (1 - e) ** 2.4);
  }
  if (x < c.xAVO) return 1 - 0.06 * sstep(c.xES, c.xAVO, x);
  const tauF = 0.07 * c.k;
  const F = 0.1 + 0.84 * Math.exp(-(x - c.xAVO) / tauF);
  const xs = Math.max(c.xAS, c.xAVO + 0.01);
  const xe = Math.max(xs + 0.03, Math.min(c.T - 0.002, xs + 0.6 * c.dA));
  return F * (1 - sstep(xs, xe, x));
}

/** ECG lead II (mV) at time tau (s) after P-wave onset. */
export function ecg(tau, c) {
  let v = 0;
  for (let k = -1; k <= 1; k++) {
    const t = tau + k * c.T;
    const q = c.PR;
    const s = c.QT / 0.4;
    v += 0.13 * gauss(t, 0.05, 0.02);
    v += -0.09 * gauss(t, q + 0.02, 0.0075);
    v += 1.0 * gauss(t, q + 0.04, 0.0095);
    v += -0.21 * gauss(t, q + 0.062, 0.009);
    const tT = q + c.QT - 0.085 * s;
    v += 0.29 * gauss(t, tT, (t < tT ? 0.058 : 0.036) * s);
  }
  return v;
}

/**
 * State of the heart at `phase` in [0, 1) of the current beat (0 = P-wave onset).
 */
export function cardiacState(phase, hr) {
  const c = timings(hr);
  const tau = phase * c.T;
  let x = tau - c.tV0;
  if (x < 0) x += c.T;

  const a = atrial(tau, c);
  const vc = ventricular(x, c);
  const eps = 0.004;
  const dV = (ventricular(Math.min(x + eps, c.T - 1e-4), c) - ventricular(Math.max(x - eps, 0), c)) / (2 * eps);
  const eject = clamp(dV / 7.0, 0, 1.2);
  const fill = clamp(-dV / 10.0, 0, 1.2);

  // atrioventricular valves
  let av;
  if (x < 0.04) av = 0.5 * (1 - sstep(0, 0.04, x));
  else if (x < c.xAVO) av = 0;
  else {
    const o1 = sstep(c.xAVO, c.xAVO + 0.05, x);
    const lvl = 0.45 + 0.55 * Math.exp(-Math.max(0, x - c.xAVO - 0.05) / 0.12);
    av = o1 * Math.min(1, lvl + 0.5 * a);
  }
  // semilunar valves
  let sl = 0;
  if (x >= c.IVC && x < c.xES + 0.03) {
    const e = clamp((x - c.IVC) / (c.xES - c.IVC), 0, 1);
    sl = sstep(c.IVC, c.IVC + 0.03, x) * (1 - sstep(c.xES - 0.012, c.xES + 0.022, x)) * (0.84 + 0.16 * (1 - e));
  }
  // atria: active systole minus passive reservoir filling during ventricular systole
  const reservoir = x < c.xAVO ? sstep(c.IVC, c.xES, x) : Math.exp(-(x - c.xAVO) / 0.06);
  const as = a - 0.32 * reservoir;

  let name;
  if (x >= 0.0 && x < c.IVC) name = 'Contracción isovolumétrica';
  else if (x < c.xES) name = (x - c.IVC) / (c.xES - c.IVC) < 0.42 ? 'Eyección rápida' : 'Eyección reducida';
  else if (x < c.xAVO) name = 'Relajación isovolumétrica';
  else if (a > 0.06) name = 'Sístole auricular';
  else if (x < c.xAVO + 0.14 * c.k) name = 'Llenado rápido';
  else name = 'Diástasis';

  return {
    c, tau, x, as, a, vc, av, sl, eject, fill, name,
    systole: x < c.xES,
    ecg: ecg(tau, c),
    lvVolume: 124 - 72 * vc + 4 * a,            // mL (EDV ~125, ESV ~52, EF ~58 %)
    s1At: c.tV0 + 0.03,                          // AV closure  -> first heart sound
    s2At: c.tV0 + c.xES + 0.012,                 // semilunar closure -> second heart sound
    rAt: c.PR + 0.04,
  };
}
