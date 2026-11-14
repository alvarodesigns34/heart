// Synthesised phonocardiogram: S1 ("lub", AV-valve closure) and S2 ("dub", semilunar closure),
// plus an optional bedside-monitor blip on the R wave.  Pure WebAudio, no samples.

export class HeartAudio {
  constructor() {
    this.ctx = null;
    this.heart = false;
    this.monitor = false;
  }

  ensure() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      this.ctx = new AC();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.9;
      const comp = this.ctx.createDynamicsCompressor();
      this.master.connect(comp).connect(this.ctx.destination);
      // short band-limited noise buffer for the "tissue" component of the sounds
      const n = Math.floor(this.ctx.sampleRate * 0.25);
      this.noise = this.ctx.createBuffer(1, n, this.ctx.sampleRate);
      const d = this.noise.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return true;
  }

  thump(f0, f1, dur, gain, partials) {
    if (!this.ctx || !this.heart) return;
    const t = this.ctx.currentTime + 0.005;
    const out = this.ctx.createGain();
    out.gain.setValueAtTime(0.0001, t);
    out.gain.exponentialRampToValueAtTime(gain, t + 0.012);
    out.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    const lp = this.ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 420;
    out.connect(lp).connect(this.master);
    for (const [mult, g] of partials) {
      const o = this.ctx.createOscillator();
      o.type = 'sine';
      o.frequency.setValueAtTime(f0 * mult, t);
      o.frequency.exponentialRampToValueAtTime(f1 * mult, t + dur);
      const og = this.ctx.createGain();
      og.gain.value = g;
      o.connect(og).connect(out);
      o.start(t);
      o.stop(t + dur + 0.02);
    }
    const ns = this.ctx.createBufferSource();
    ns.buffer = this.noise;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = f0 * 2.2;
    bp.Q.value = 0.9;
    const ng = this.ctx.createGain();
    ng.gain.value = 0.5;
    ns.connect(bp).connect(ng).connect(out);
    ns.start(t);
    ns.stop(t + dur);
  }

  s1() {
    this.thump(62, 38, 0.15, 0.9, [[1, 1], [2, 0.55], [3.1, 0.22]]);
  }

  s2() {
    this.thump(92, 62, 0.085, 0.62, [[1, 1], [2, 0.6], [3.2, 0.3]]);
  }

  beep() {
    if (!this.ctx || !this.monitor) return;
    const t = this.ctx.currentTime + 0.005;
    const o = this.ctx.createOscillator();
    o.type = 'sine';
    o.frequency.value = 880;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.07, t + 0.006);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.085);
    o.connect(g).connect(this.master);
    o.start(t);
    o.stop(t + 0.1);
  }
}
