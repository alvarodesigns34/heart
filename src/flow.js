import * as THREE from 'three';

const sstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/**
 * Dual haemodynamics.  Each particle rides one anatomical route
 *   vein -> atrium -> AV valve -> ventricle -> semilunar valve -> artery
 * Its route parameter u is integrated on the CPU from the cardiac-cycle flow rates (so it brakes
 * against closed valves and surges in systole); its 3D position is resolved on the GPU from a
 * half-float lookup texture holding the route centre-line, radius and transport frame.
 */
export class BloodFlow {
  constructor(routes, toWorld, worldScale, perRoute = 2600) {
    const W = 256;
    const R = routes.length;
    const data = new Uint16Array(W * R * 3 * 4);
    const hf = THREE.DataUtils.toHalfFloat;
    this.routes = routes.map((rt, ri) => {
      const pts = rt.pts.map((p) => toWorld(p));
      const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
      const len = curve.getLength();
      // radius as a function of arclength fraction
      const arc = [0];
      for (let i = 1; i < pts.length; i++) arc.push(arc[i - 1] + pts[i].distanceTo(pts[i - 1]));
      const radAt = (u) => {
        const s = u * arc[arc.length - 1];
        let i = 1;
        while (i < arc.length - 1 && arc[i] < s) i++;
        const f = (s - arc[i - 1]) / Math.max(1e-6, arc[i] - arc[i - 1]);
        return (rt.radius[i - 1] * (1 - f) + rt.radius[i] * f) * worldScale;
      };
      let n = new THREE.Vector3(0.31, 0.55, 0.77);
      const b = new THREE.Vector3();
      for (let i = 0; i < W; i++) {
        const u = i / (W - 1);
        const p = curve.getPointAt(u);
        const t = curve.getTangentAt(u);
        n.addScaledVector(t, -n.dot(t)).normalize();        // parallel transport
        b.crossVectors(t, n);
        const rows = [[p.x, p.y, p.z, radAt(u)], [n.x, n.y, n.z, 0], [b.x, b.y, b.z, 0]];
        rows.forEach((row, k) => {
          const o = ((ri * 3 + k) * W + i) * 4;
          for (let c = 0; c < 4; c++) data[o + c] = hf(row[c]);
        });
      }
      return { ...rt, len, oxy: rt.kind === 'oxy' };
    });
    const lut = new THREE.DataTexture(data, W, R * 3, THREE.RGBAFormat, THREE.HalfFloatType);
    lut.minFilter = lut.magFilter = THREE.LinearFilter;
    lut.wrapS = lut.wrapT = THREE.ClampToEdgeWrapping;
    lut.needsUpdate = true;

    const N = perRoute * R;
    this.N = N;
    this.u = new Float32Array(N);
    this.speed = new Float32Array(N);
    this.routeOf = new Uint8Array(N);
    const aRoute = new Float32Array(N);
    const aOff = new Float32Array(N * 3);
    const aGate = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      const r = i % R;
      this.routeOf[i] = r;
      this.u[i] = Math.random();
      this.speed[i] = 0.72 + Math.random() * 0.56;
      aRoute[i] = r;
      aOff[i * 3] = Math.sqrt(Math.random());           // radial position (uniform over the disc)
      aOff[i * 3 + 1] = Math.random() * Math.PI * 2;
      aOff[i * 3 + 2] = 0.75 + Math.random() * 0.6;      // size jitter
      const rt = this.routes[r];
      aGate.set([rt.u_vein, rt.u_av, rt.u_sl, rt.oxy ? 1 : 0], i * 4);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    geo.setAttribute('aRoute', new THREE.BufferAttribute(aRoute, 1));
    this.uAttr = new THREE.BufferAttribute(this.u, 1);
    this.uAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aU', this.uAttr);
    geo.setAttribute('aOff', new THREE.BufferAttribute(aOff, 3));
    geo.setAttribute('aGate', new THREE.BufferAttribute(aGate, 4));
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 50);

    this.material = new THREE.ShaderMaterial({
      clipping: true,
      uniforms: {
        uLut: { value: lut }, uRows: { value: R * 3 }, uW: { value: W },
        uVent: { value: 1 }, uAtr: { value: 1 }, uSize: { value: 0.95 }, uPixelRatio: { value: 1 },
        uOxy: { value: new THREE.Color(0.86, 0.035, 0.03) }, uDeoxy: { value: new THREE.Color(0.17, 0.006, 0.075) },
      },
      vertexShader: /* glsl */ `
        #include <clipping_planes_pars_vertex>
        uniform sampler2D uLut; uniform float uRows, uW, uVent, uAtr, uSize, uPixelRatio;
        attribute float aRoute, aU; attribute vec3 aOff; attribute vec4 aGate;
        varying float vOxy; varying float vShade;
        void main() {
          float uu = (clamp(aU, 0.0, 1.0) * (uW - 1.0) + 0.5) / uW;
          vec4 pr = texture2D(uLut, vec2(uu, (aRoute * 3.0 + 0.5) / uRows));
          vec3 n = texture2D(uLut, vec2(uu, (aRoute * 3.0 + 1.5) / uRows)).xyz;
          vec3 b = texture2D(uLut, vec2(uu, (aRoute * 3.0 + 2.5) / uRows)).xyz;
          float s = 1.0;
          if (aU > aGate.x && aU <= aGate.y) s = uAtr; else if (aU > aGate.y && aU <= aGate.z) s = uVent;
          vec3 p = pr.xyz + (cos(aOff.y) * n + sin(aOff.y) * b) * (pr.w * aOff.x * s);
          vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mvPosition;
          gl_PointSize = uSize * aOff.z * uPixelRatio / -mvPosition.z;
          vOxy = aGate.w; vShade = aOff.z;
          #include <clipping_planes_vertex>
        }`,
      fragmentShader: /* glsl */ `
        #include <clipping_planes_pars_fragment>
        uniform vec3 uOxy, uDeoxy; varying float vOxy; varying float vShade;
        void main() {
          #include <clipping_planes_fragment>
          vec2 q = gl_PointCoord * 2.0 - 1.0;
          float r2 = dot(q, q);
          if (r2 > 1.0) discard;
          float z = sqrt(1.0 - r2);
          vec3 base = mix(uDeoxy, uOxy, vOxy);
          float lit = 0.45 + 0.75 * max(0.0, dot(normalize(vec3(-0.4, 0.5, 0.75)), vec3(q.x, -q.y, z)));
          float spec = pow(max(0.0, dot(normalize(vec3(-0.3, 0.45, 0.85)), vec3(q.x, -q.y, z))), 26.0);
          gl_FragColor = vec4(base * lit * (0.9 + 0.3 * vShade) + vec3(spec) * 0.55, 1.0);
          #include <colorspace_fragment>
        }`,
    });
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
    this.points.renderOrder = 30;
    this.smoothEject = 0;
  }

  update(dt, st) {
    if (!this.points.visible || dt <= 0) return;
    this.smoothEject += (st.eject - this.smoothEject) * Math.min(1, dt * 9);
    const sys = st.systole ? 1 : 0;
    const vVein = 0.34 + 0.38 * sys + 0.5 * st.fill;
    const vAtr = 0.10 + 1.9 * st.fill * st.av + 0.55 * Math.max(0, st.a);
    const vVentIn = 0.07 + 1.5 * st.fill;
    const vVentOut = 0.05 + 3.4 * st.eject;
    const vArt = 0.22 + 3.3 * this.smoothEject;
    const avOpen = sstep(0.05, 0.3, st.av);
    const slOpen = sstep(0.05, 0.3, st.sl);
    const u = this.u;
    for (let i = 0; i < this.N; i++) {
      const rt = this.routes[this.routeOf[i]];
      let x = u[i];
      let v;
      if (x < rt.u_vein) v = vVein;
      else if (x < rt.u_av) v = vAtr;
      else if (x < rt.u_sl) {
        const f = (x - rt.u_av) / (rt.u_sl - rt.u_av);
        v = f < 0.55 ? vVentIn * (1 - f) + vVentOut * 0.6 : vVentOut;
        if (f < 0.55) v += vVentOut * 0.5 * f;
      } else v = vArt;
      let nx = x + (v * this.speed[i] * dt) / rt.len;
      // valves: a closed leaflet is a wall — the stream decelerates and piles up in front of it
      if (x < rt.u_av && nx >= rt.u_av - 0.004) {
        if (avOpen < 0.5) nx = Math.min(nx, rt.u_av - 0.004 - 0.02 * (this.speed[i] - 0.72));
        else nx = x + (nx - x) * avOpen;
      } else if (x < rt.u_sl && nx >= rt.u_sl - 0.004) {
        if (slOpen < 0.5) nx = Math.min(nx, rt.u_sl - 0.004 - 0.025 * (this.speed[i] - 0.72));
        else nx = x + (nx - x) * slOpen;
      }
      if (nx < x) nx = x;
      if (nx >= 1) nx = Math.random() * 0.02;
      u[i] = nx;
    }
    this.uAttr.needsUpdate = true;
    this.material.uniforms.uVent.value = 1 - 0.34 * st.vc;
    this.material.uniforms.uAtr.value = 1 - 0.16 * Math.max(0, st.a);
  }
}
