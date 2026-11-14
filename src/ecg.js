// Sweep-style ECG monitor.  The trace is sampled from the same analytic lead-II function that
// drives the 3D heart, at 2 ms sub-steps, so the QRS complex is never aliased by the frame rate.

export class EcgMonitor {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.pxPerSec = 150;
    this.x = 0;
    this.lastY = null;
    this.color = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#b4e36b';
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.canvas.getBoundingClientRect();
    if (!r.width) return;
    this.w = Math.round(r.width * dpr);
    this.h = Math.round(r.height * dpr);
    this.canvas.width = this.w;
    this.canvas.height = this.h;
    this.dpr = dpr;
    this.x = 0;
    this.lastY = null;
    this.ctx.clearRect(0, 0, this.w, this.h);
  }

  yOf(mv) {
    return this.h * 0.74 - mv * this.h * 0.56;
  }

  /** advance by simulated time `dt`; `sample(f)` returns the ECG value at fraction f of this step */
  advance(dt, sample) {
    if (!this.w || dt <= 0) return;
    const ctx = this.ctx;
    const steps = Math.max(1, Math.ceil(dt / 0.002));
    const dx = (dt * this.pxPerSec * this.dpr) / steps;
    ctx.lineWidth = 1.6 * this.dpr;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = this.color;
    ctx.shadowColor = this.color;
    ctx.shadowBlur = 6 * this.dpr;
    for (let i = 1; i <= steps; i++) {
      const y = this.yOf(sample(i / steps));
      let nx = this.x + dx;
      // erase a gap ahead of the pen
      const gap = 26 * this.dpr;
      ctx.shadowBlur = 0;
      ctx.clearRect(Math.floor(this.x) + 1, 0, Math.ceil(dx) + gap, this.h);
      if (this.x + gap > this.w) ctx.clearRect(0, 0, this.x + gap - this.w, this.h);
      ctx.shadowBlur = 6 * this.dpr;
      if (this.lastY !== null) {
        ctx.beginPath();
        ctx.moveTo(this.x, this.lastY);
        ctx.lineTo(nx, y);
        ctx.stroke();
      }
      if (nx >= this.w) {
        nx = 0;
        this.lastY = null;
      } else {
        this.lastY = y;
      }
      this.x = nx;
    }
  }
}
