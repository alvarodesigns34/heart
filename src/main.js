import '@fontsource/newsreader/400.css';
import '@fontsource/newsreader/400-italic.css';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-mono/400.css';
import './styles.css';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { SSAOPass } from 'three/addons/postprocessing/SSAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { cardiacState, ecg as ecgAt, timings } from './cycle.js';
import { surgicalEnvironment } from './environment.js';
import { Dissection, tissueTexture } from './dissection.js';
import { BloodFlow } from './flow.js';
import { Pins } from './pins.js';
import { EcgMonitor } from './ecg.js';
import { HeartAudio } from './audio.js';

const BASE = import.meta.env.BASE_URL;
const WORLD = 10;                       // glTF metres -> scene units (1 unit = 10 cm)
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const params = new URLSearchParams(location.search);

// ------------------------------------------------------------------------------------------
// Renderer / scene
// ------------------------------------------------------------------------------------------
const canvas = $('#scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, stencil: true, powerPreference: 'high-performance' });
renderer.setClearColor(0x07080b, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.02;
renderer.localClippingEnabled = true;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080b);
scene.environment = surgicalEnvironment(renderer);
scene.environmentIntensity = 0.85;

const camera = new THREE.PerspectiveCamera(30, 1, 0.3, 40);
camera.position.set(0.5, 0.45, 5.2);
scene.add(camera);

// the light rig travels with the observer, like a head-mounted surgical lamp
const key = new THREE.DirectionalLight(0xffe3c6, 2.5);       // warm key
key.position.set(-2.6, 3.0, 2.2);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.5;
key.shadow.camera.far = 14;
Object.assign(key.shadow.camera, { left: -1.7, right: 1.7, top: 1.9, bottom: -1.5 });
key.shadow.bias = -0.0006;
key.shadow.normalBias = 0.012;
key.shadow.radius = 5;
const rim = new THREE.DirectionalLight(0x8fbcff, 3.4);       // cold back light: makes the serosa glisten
rim.position.set(3.2, 1.6, -4.2);
const fill = new THREE.DirectionalLight(0xbfd0ff, 0.35);
fill.position.set(3.0, -0.6, 2.0);
const lightTarget = new THREE.Object3D();
lightTarget.position.set(0, 0, -5.2);
camera.add(key, rim, fill, lightTarget);
key.target = rim.target = fill.target = lightTarget;
scene.add(new THREE.HemisphereLight(0x8ea0bd, 0x1c0e0c, 0.22));

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 1.1;
controls.maxDistance = 10;
controls.rotateSpeed = 0.75;
controls.zoomSpeed = 0.8;
controls.target.set(0, 0, 0);

// ------------------------------------------------------------------------------------------
// Loading
// ------------------------------------------------------------------------------------------
const loaderEl = $('#loader');
const setProgress = (f, note) => {
  $('#loader-pct').textContent = String(Math.round(f * 100));
  $('#loader-bar').style.width = (f * 100).toFixed(1) + '%';
  if (note) $('#loader-note').textContent = note;
};

async function loadAll() {
  const meta = await (await fetch(`${BASE}models/heart_meta.json`)).json();
  setProgress(0.03);
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  const gltf = await new Promise((resolve, reject) => {
    loader.load(`${BASE}models/human_heart.glb`, resolve, (e) => {
      const total = e.lengthComputable && e.total ? e.total : meta.glb_bytes || 30e6;
      setProgress(0.03 + 0.9 * Math.min(1, e.loaded / total));
    }, reject);
  });
  setProgress(0.95, 'Compilando materiales');
  return { meta, gltf };
}

// ------------------------------------------------------------------------------------------
// Materials
// ------------------------------------------------------------------------------------------
const dissection = new Dissection(scene);
const maxAniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());

function wet(src, o = {}) {
  for (const t of [src.map, src.normalMap, src.aoMap, src.roughnessMap]) if (t) t.anisotropy = maxAniso;
  const m = new THREE.MeshPhysicalMaterial({
    color: src.color ? src.color.clone() : new THREE.Color(1, 1, 1),
    map: src.map || null,
    normalMap: src.normalMap || null,
    aoMap: src.aoMap || null,
    roughnessMap: src.roughnessMap || null,
    roughness: src.roughness ?? 0.4,
    metalness: 0,
    vertexColors: !!src.vertexColors,
    clearcoat: 0.85,
    clearcoatRoughness: 0.12,
    ior: 1.4,
    envMapIntensity: 1.0,
    clippingPlanes: dissection.active,
    clipShadows: true,
    ...o.props,
  });
  if (m.normalMap) m.normalScale.setScalar(o.normalScale ?? 1.0);
  if (m.aoMap) m.aoMapIntensity = o.ao ?? 1.0;
  // cheap subsurface cue: light bleeding through thin tissue at grazing angles
  const glow = new THREE.Color(...(o.glow || [0.1, 0.02, 0.015]));
  m.onBeforeCompile = (shader) => {
    shader.uniforms.uGlow = { value: glow };
    shader.fragmentShader = shader.fragmentShader
      .replace('uniform vec3 emissive;', 'uniform vec3 emissive;\nuniform vec3 uGlow;')
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        float sssF = pow(1.0 - saturate(dot(normalize(vViewPosition), normal)), 2.6);
        totalEmissiveRadiance += uGlow * sssF * (0.35 + diffuseColor.rgb);`);
  };
  m.customProgramCacheKey = () => 'wet' + glow.getHexString();
  return m;
}

const L = { myocardium: [], fat: [], coronary: [], valves: [] };
const layerOn = { myocardium: true, fat: true, coronary: true, valves: true, flow: true, pins: true };
let myoMode = 'solid';
const morphMeshes = [];
const prepass = [];        // {mesh, beauty, normal}

function setupModel(gltf, meta) {
  const root = gltf.scene;
  const c = meta.axes.center;
  const holder = new THREE.Group();
  holder.scale.setScalar(WORLD);
  holder.position.set(-c[0] * WORLD, -c[1] * WORLD, -c[2] * WORLD);
  holder.add(root);
  scene.add(holder);

  root.traverse((o) => {
    if (!o.isMesh) return;
    const n = o.name;
    let mat;
    if (n.startsWith('Myocardium')) {
      mat = wet(o.material, { normalScale: 1.35, glow: [0.16, 0.02, 0.015] });
      L.myocardium.push(o);
    } else if (n.startsWith('Epicardial')) {
      mat = wet(o.material, { normalScale: 1.2, glow: [0.30, 0.17, 0.04], props: { sheen: 0.5, sheenRoughness: 0.5, sheenColor: new THREE.Color(1.0, 0.86, 0.5) } });
      L.fat.push(o);
    } else if (n.startsWith('Coronary')) {
      mat = wet(o.material, { glow: n.includes('Vein') ? [0.03, 0.03, 0.12] : [0.2, 0.01, 0.01], props: { roughness: 0.3 } });
      L.coronary.push(o);
    } else {
      mat = wet(o.material, { glow: [0.34, 0.2, 0.15], props: { clearcoat: 0.55, clearcoatRoughness: 0.2, sheen: 0.6, sheenRoughness: 0.45, sheenColor: new THREE.Color(1, 0.9, 0.82), roughness: 0.34 } });
      L.valves.push(o);
    }
    o.material = mat;
    o.castShadow = true;
    o.receiveShadow = true;
    o.frustumCulled = false;              // bounds change with the morph targets
    o.renderOrder = 20;
    o.userData.normalMat = new THREE.MeshNormalMaterial({ clippingPlanes: dissection.active });
    prepass.push({ mesh: o, beauty: mat, normal: o.userData.normalMat });
    if (o.morphTargetDictionary) morphMeshes.push(o);
  });
  return holder;
}

function applyMorphs(st) {
  for (const m of morphMeshes) {
    const d = m.morphTargetDictionary;
    const inf = m.morphTargetInfluences;
    if (d.atrial_systole !== undefined) inf[d.atrial_systole] = st.as;
    if (d.ventricular_contraction !== undefined) inf[d.ventricular_contraction] = st.vc;
    if (d.av_valves_open !== undefined) inf[d.av_valves_open] = st.av;
    if (d.semilunar_valves_open !== undefined) inf[d.semilunar_valves_open] = st.sl;
  }
}

// ------------------------------------------------------------------------------------------
// Post-processing: SSAO (with a stencil-aware normal pre-pass) -> bloom -> ACES output
// ------------------------------------------------------------------------------------------
class HeartSSAOPass extends SSAOPass {
  renderOverride(rendererIn, _overrideMaterial, renderTarget, clearColor, clearAlpha) {
    const hidden = [];
    for (const p of prepass) {
      if (p.beauty.transparent && p.mesh.visible) {
        p.mesh.visible = false;
        hidden.push(p.mesh);
      }
      p.mesh.material = p.normal;
    }
    scene.traverse((o) => {
      if (o.userData.cap) o.material = o.userData.normal;
      else if ((o.isPoints || o.isLine || o.userData.noPrepass) && o.visible) {
        o.visible = false;
        hidden.push(o);
      }
    });
    const bg = scene.background;
    scene.background = null;
    rendererIn.getClearColor(this.originalClearColor);
    const a = rendererIn.getClearAlpha();
    const auto = rendererIn.autoClear;
    rendererIn.setRenderTarget(renderTarget);
    rendererIn.autoClear = false;
    rendererIn.setClearColor(clearColor, clearAlpha);
    rendererIn.clear(true, true, true);
    rendererIn.render(scene, camera);
    rendererIn.autoClear = auto;
    rendererIn.setClearColor(this.originalClearColor, a);
    scene.background = bg;
    for (const p of prepass) p.mesh.material = p.beauty;
    scene.traverse((o) => {
      if (o.userData.cap) o.material = o.userData.beauty;
    });
    for (const o of hidden) o.visible = true;
  }
}

let composer, ssaoPass, bloomPass;
const quality = { tier: params.get('q') || 'high', dpr: Math.min(window.devicePixelRatio || 1, 2) };

function buildComposer() {
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  const rt = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: 4, stencilBuffer: true });
  composer = new EffectComposer(renderer, rt);
  composer.addPass(new RenderPass(scene, camera));
  ssaoPass = new HeartSSAOPass(scene, camera, size.x, size.y, 24);
  ssaoPass.kernelRadius = 0.11;
  ssaoPass.minDistance = 0.0006;
  ssaoPass.maxDistance = 0.012;
  composer.addPass(ssaoPass);
  bloomPass = new UnrealBloomPass(new THREE.Vector2(size.x, size.y), 0.16, 0.55, 0.92);
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());
}

function applyQuality() {
  const t = quality.tier;
  quality.dpr = t === 'high' ? Math.min(window.devicePixelRatio || 1, 2) : t === 'medium' ? Math.min(window.devicePixelRatio || 1, 1.25) : 1;
  ssaoPass.enabled = t !== 'low';
  bloomPass.enabled = t !== 'low';
  renderer.shadowMap.enabled = t !== 'low';
  key.castShadow = t !== 'low';
  resize();
}

function resize() {
  const w = canvas.clientWidth || window.innerWidth;
  const h = canvas.clientHeight || window.innerHeight;
  renderer.setPixelRatio(quality.dpr);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // keep the whole specimen in frame on tall (mobile) viewports
  camera.fov = camera.aspect < 0.8 ? 42 : 30;
  camera.updateProjectionMatrix();
  if (composer) {
    composer.setPixelRatio(quality.dpr);
    composer.setSize(w, h);
  }
  if (flow) flow.material.uniforms.uPixelRatio.value = (h * quality.dpr) / (2 * Math.tan((camera.fov * Math.PI) / 360)) / 100;
}
window.addEventListener('resize', resize);

// ------------------------------------------------------------------------------------------
// App state
// ------------------------------------------------------------------------------------------
let flow = null;
let pins = null;
let meta = null;
const audio = new HeartAudio();
const monitor = new EcgMonitor($('#ecg'));
const sim = { hr: 72, phase: 0.62, speed: 1, paused: false, state: cardiacState(0.62, 72) };
const toWorld = (p) => new THREE.Vector3((p[0] - meta.axes.center[0]) * WORLD, (p[1] - meta.axes.center[1]) * WORLD, (p[2] - meta.axes.center[2]) * WORLD);

// ---- camera choreography -------------------------------------------------------------------
const tween = { active: false };
function flyTo(dir, dist, target = new THREE.Vector3(), ms = 900) {
  const d0 = camera.position.clone().sub(controls.target);
  tween.fromDir = d0.clone().normalize();
  tween.toDir = dir.clone().normalize();
  tween.fromDist = d0.length();
  tween.toDist = dist;
  tween.fromTarget = controls.target.clone();
  tween.toTarget = target.clone();
  tween.t0 = performance.now();
  tween.ms = matchMedia('(prefers-reduced-motion: reduce)').matches ? 1 : ms;
  tween.active = true;
}
function stepTween(now) {
  if (!tween.active) return;
  const t = Math.min(1, (now - tween.t0) / tween.ms);
  const e = 1 - Math.pow(1 - t, 4);
  const q = new THREE.Quaternion().setFromUnitVectors(tween.fromDir, tween.toDir);
  const dir = tween.fromDir.clone().applyQuaternion(new THREE.Quaternion().slerp(q, e));
  const dist = tween.fromDist + (tween.toDist - tween.fromDist) * e;
  controls.target.lerpVectors(tween.fromTarget, tween.toTarget, e);
  camera.position.copy(controls.target).addScaledVector(dir, dist);
  if (t >= 1) tween.active = false;
}
controls.addEventListener('start', () => {
  tween.active = false;
  $$('#views button').forEach((b) => b.classList.remove('is-active'));
});

const fitDist = () => (camera.aspect < 0.8 ? 5.7 : 4.9);
let VIEWS = {};

// ---- dissection planes ---------------------------------------------------------------------
const planeDefs = {};
function setPlane(id, { enabled, offset, invert }) {
  const def = planeDefs[id];
  const el = $(`.plane[data-plane="${id}"]`);
  if (enabled !== undefined) {
    def.enabled = enabled;
    $('.switch', el).checked = enabled;
    el.classList.toggle('is-on', enabled);
  }
  if (offset !== undefined) {
    const range = $('input[type=range]', el);
    range.value = offset;
    def.offset = (def.mmToWorld || 0.01) * Number(offset) - (def.origin || 0);
    $('output', el).textContent = `${Math.round(Number(offset))} mm`;
  }
  if (invert !== undefined) {
    def.invert = invert;
    $('.flip', el).setAttribute('aria-pressed', String(invert));
  }
  dissection.refresh();
}

function setMyocardium(mode) {
  myoMode = mode;
  $$('#myo-mode button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === mode)));
  for (const m of L.myocardium) {
    m.visible = mode !== 'hidden';
    const mat = m.material;
    const x = mode === 'xray';
    mat.transparent = x;
    mat.opacity = x ? 0.2 : 1;
    mat.depthWrite = !x;
    mat.side = THREE.FrontSide;
    m.renderOrder = x ? 40 : 20;
    m.castShadow = !x;
    mat.needsUpdate = true;
  }
  layerOn.myocardium = mode !== 'hidden';
  dissection.setLayerVisible('myocardium', mode === 'solid');
}

function setLayer(name, on) {
  layerOn[name] = on;
  const box = $(`.switch[data-layer="${name}"]`);
  if (box) box.checked = on;
  if (L[name]) for (const m of L[name]) m.visible = on;
  if (name === 'flow' && flow) flow.points.visible = on;
  if (name === 'pins' && pins) pins.enabled = on;
  if (name === 'fat' || name === 'coronary') dissection.setLayerVisible(name, on);
}

function preset(name) {
  $$('#presets button').forEach((b) => b.classList.toggle('is-active', b.dataset.preset === name));
  $$('#views button').forEach((b) => b.classList.remove('is-active'));
  for (const id of Object.keys(planeDefs)) setPlane(id, { enabled: false, invert: false });
  if (pins) pins.select(null);
  const ax = new THREE.Vector3(...meta.axes.long_axis);
  if (name === 'whole') {
    setMyocardium('solid');
    setLayer('fat', true);
    setLayer('coronary', true);
    flyTo(VIEWS.anterior, fitDist());
  } else if (name === 'four') {
    setMyocardium('solid');
    setPlane('coronal', { enabled: true, offset: 0 });
    flyTo(new THREE.Vector3(...meta.axes.four_chamber_normal), fitDist() * 0.82, toWorld(meta.axes.lv_base).multiplyScalar(0.5));
  } else if (name === 'short') {
    setMyocardium('solid');
    setPlane('transverse', { enabled: true, offset: 33 });
    flyTo(ax.clone().negate().add(new THREE.Vector3(0, 0.12, 0.1)), fitDist() * 0.72, toWorld(meta.axes.lv_base).lerp(toWorld(meta.axes.apex), 0.45));
  } else if (name === 'valves') {
    setMyocardium('solid');
    setPlane('transverse', { enabled: true, offset: -8 });
    flyTo(ax.clone().negate().add(new THREE.Vector3(0, 0.05, 0.04)), fitDist() * 0.6, toWorld(meta.axes.valve_plane_point));
  } else if (name === 'vessels') {
    setMyocardium('xray');
    setLayer('fat', false);
    setLayer('coronary', true);
    flyTo(VIEWS.anterior, fitDist());
  }
}

// ------------------------------------------------------------------------------------------
// Boot
// ------------------------------------------------------------------------------------------
async function boot() {
  resize();
  let data;
  try {
    data = await loadAll();
  } catch (err) {
    console.error(err);
    loaderEl.classList.add('is-error');
    $('#loader-note').textContent = 'No se pudo cargar el modelo. Recarga la página.';
    return;
  }
  meta = data.meta;
  setupModel(data.gltf, meta);

  // --- section planes (kept half-space normals)
  const ax = new THREE.Vector3(...meta.axes.long_axis).normalize();
  const fc = new THREE.Vector3(...meta.axes.four_chamber_normal).normalize();
  const mm = (WORLD * meta.scale) / 0.92 / 10;         // 1 mm of the real organ in scene units
  planeDefs.coronal = dissection.addPlane('coronal', toWorld(meta.axes.four_chamber_point), fc.clone().negate());
  planeDefs.sagittal = dissection.addPlane('sagittal', new THREE.Vector3(0.12, 0, 0), new THREE.Vector3(-1, 0, 0));
  planeDefs.transverse = dissection.addPlane('transverse', toWorld(meta.axes.lv_base), ax.clone());
  planeDefs.coronal.mmToWorld = -mm;
  planeDefs.sagittal.mmToWorld = -mm;
  planeDefs.transverse.mmToWorld = mm;

  const capNormal = new THREE.MeshNormalMaterial();
  const muscleCap = new THREE.MeshPhysicalMaterial({ map: tissueTexture('muscle'), color: 0xffffff, roughness: 0.42, metalness: 0, clearcoat: 0.7, clearcoatRoughness: 0.18 });
  const fatCap = new THREE.MeshPhysicalMaterial({ map: tissueTexture('fat'), color: 0xffffff, roughness: 0.4, clearcoat: 0.6, clearcoatRoughness: 0.2 });
  const artCap = new THREE.MeshPhysicalMaterial({ color: 0x5a0d0a, roughness: 0.35, clearcoat: 0.7 });
  dissection.addLayer('myocardium', L.myocardium, { beauty: muscleCap, normal: capNormal });
  dissection.addLayer('fat', L.fat, { beauty: fatCap, normal: capNormal });
  dissection.addLayer('coronary', L.coronary, { beauty: artCap, normal: capNormal });
  dissection.refresh();

  // --- haemodynamics
  flow = new BloodFlow(meta.routes, toWorld, WORLD, quality.tier === 'low' ? 1200 : 2600);
  flow.material.clippingPlanes = dissection.active;
  scene.add(flow.points);

  // --- pins
  pins = new Pins($('#pins'), meta.pins, toWorld, WORLD, showCard);

  VIEWS = {
    anterior: new THREE.Vector3(0.1, 0.1, 1),
    posterior: new THREE.Vector3(-0.08, 0.16, -1),
    left: new THREE.Vector3(1, 0.06, 0.12),
    right: new THREE.Vector3(-1, 0.06, 0.12),
    superior: new THREE.Vector3(0, 1, 0.22),
    inferior: new THREE.Vector3(-0.15, -1, -0.3),
    apex: ax.clone().add(new THREE.Vector3(0, 0.05, 0.1)),
  };

  buildComposer();
  applyQuality();
  wireUi();
  camera.position.copy(VIEWS.anterior.clone().normalize().multiplyScalar(fitDist()));
  applyMorphs(sim.state);
  composer.render();                       // compile programs before revealing
  setProgress(1, 'Listo');
  requestAnimationFrame(() => loaderEl.classList.add('is-done'));
  window.__heart = {
    sim, dissection, preset, setPlane, setMyocardium, setLayer, flyTo, VIEWS, quality, applyQuality, renderer, ready: true,
    // test hooks: freeze the loop, then step it deterministically (used by the headless checks)
    freeze(v = true) { debug.frozen = v; },
    step(n = 1, ms = 16) {
      debug.frozen = false;
      for (let i = 0; i < n; i++) {
        debug.skipRender = i < n - 1;
        tween.t0 -= ms;
        frame(last + ms);
      }
      debug.skipRender = false;
      debug.frozen = true;
    },
  };
  last = performance.now();
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------------------------------
// UI
// ------------------------------------------------------------------------------------------
const LAYER_NAMES = { myocardium: 'Miocardio', coronary: 'Circulación coronaria', valves: 'Aparato valvular', fat: 'Tejido adiposo' };
function showCard(item) {
  const card = $('#card');
  if (!item) {
    card.hidden = true;
    return;
  }
  $('#card-layer').textContent = LAYER_NAMES[item.layer] || '';
  $('#card-title').textContent = item.label;
  $('#card-desc').textContent = item.desc;
  card.hidden = false;
  card.style.animation = 'none';
  void card.offsetWidth;
  card.style.animation = '';
}

function wireUi() {
  $$('.switch[data-layer]').forEach((box) => box.addEventListener('change', () => setLayer(box.dataset.layer, box.checked)));
  $$('#myo-mode button').forEach((b) => b.addEventListener('click', () => setMyocardium(b.dataset.mode)));
  $$('.plane').forEach((el) => {
    const id = el.dataset.plane;
    $('.switch', el).addEventListener('change', (e) => {
      setPlane(id, { enabled: e.target.checked });
      $$('#presets button').forEach((b) => b.classList.remove('is-active'));
    });
    $('input[type=range]', el).addEventListener('input', (e) => setPlane(id, { offset: e.target.value }));
    $('.flip', el).addEventListener('click', () => setPlane(id, { invert: !planeDefs[id].invert }));
    setPlane(id, { offset: $('input[type=range]', el).value });
  });
  $$('#views button').forEach((b) => b.addEventListener('click', () => {
    $$('#views button').forEach((x) => x.classList.toggle('is-active', x === b));
    flyTo(VIEWS[b.dataset.view], fitDist());
  }));
  $$('#presets button').forEach((b) => b.addEventListener('click', () => preset(b.dataset.preset)));

  const bpmOut = $('#bpm');
  $('#bpm-range').addEventListener('input', (e) => {
    sim.hr = Number(e.target.value);
    bpmOut.textContent = String(sim.hr);
  });
  $('#play').addEventListener('click', (e) => {
    sim.paused = !sim.paused;
    e.currentTarget.setAttribute('aria-pressed', String(sim.paused));
    e.currentTarget.setAttribute('aria-label', sim.paused ? 'Reanudar el latido' : 'Pausar el latido');
  });
  $$('#speed button').forEach((b) => b.addEventListener('click', () => {
    sim.speed = Number(b.dataset.speed);
    $$('#speed button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
  }));
  $('#sound').addEventListener('click', (e) => {
    if (!audio.ensure()) return;
    audio.heart = !audio.heart;
    e.currentTarget.setAttribute('aria-pressed', String(audio.heart));
  });
  $('#beep').addEventListener('click', (e) => {
    if (!audio.ensure()) return;
    audio.monitor = !audio.monitor;
    e.currentTarget.setAttribute('aria-pressed', String(audio.monitor));
  });
  $('#card-close').addEventListener('click', () => pins.select(null));
  $('#card-focus').addEventListener('click', () => {
    const it = pins.selected;
    if (!it) return;
    const dir = it.interior ? camera.position.clone().sub(controls.target) : it.n.clone().add(new THREE.Vector3(0, 0.1, 0.25));
    flyTo(dir, 2.6, it.world.clone().multiplyScalar(0.75));
  });
  const drawer = $('#drawer-toggle');
  drawer.addEventListener('click', () => {
    const open = $('#panel-left').classList.toggle('is-open');
    drawer.setAttribute('aria-expanded', String(open));
  });
  // click on empty space clears the selection
  let downAt = null;
  canvas.addEventListener('pointerdown', (e) => (downAt = [e.clientX, e.clientY]));
  canvas.addEventListener('pointerup', (e) => {
    if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 4 && pins.selected) pins.select(null);
  });
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') {
      e.preventDefault();
      $('#play').click();
    } else if (e.key === 'Escape') pins.select(null);
  });
}

// ------------------------------------------------------------------------------------------
// Frame loop
// ------------------------------------------------------------------------------------------
let last = 0;
let frames = 0;
let acc = 0;
const ui = { phase: '', vol: -1, av: null, sl: null };
const crossed = (a, b, t, T) => (a <= b ? a < t && t <= b : t > a || t <= b) && T > 0;

const debug = { frozen: false, skipRender: false };
function frame(now) {
  requestAnimationFrame(frame);
  if (debug.frozen) {
    last = now;
    return;
  }
  const dtReal = Math.min(0.1, (now - last) / 1000);
  last = now;
  const dt = sim.paused ? 0 : dtReal * sim.speed;

  const p0 = sim.phase;
  const dph = (dt * sim.hr) / 60;
  sim.phase = (p0 + dph) % 1;
  const st = (sim.state = cardiacState(sim.phase, sim.hr));
  applyMorphs(st);

  if (dt > 0) {
    const c = st.c;
    monitor.advance(dt, (f) => ecgAt((((p0 + dph * f) % 1) + 1) % 1 * c.T, c));
    const t0 = p0 * c.T;
    const t1 = sim.phase * c.T;
    if (crossed(t0, t1, st.s1At % c.T, c.T)) audio.s1();
    if (crossed(t0, t1, st.s2At % c.T, c.T)) audio.s2();
    if (crossed(t0, t1, st.rAt % c.T, c.T)) audio.beep();
  }
  flow.update(dt, st);

  stepTween(now);
  controls.update();
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  pins.update({
    camera, width: w, height: h, state: st, dissection, layers: layerOn,
    interiorVisible: dissection.anyEnabled || myoMode !== 'solid',
  });

  if (st.name !== ui.phase) $('#phase').textContent = ui.phase = st.name;
  const vol = Math.round(st.lvVolume);
  if (vol !== ui.vol) $('#vol').textContent = String((ui.vol = vol));
  const avO = st.av > 0.12;
  if (avO !== ui.av) {
    ui.av = avO;
    const el = $('#av-state');
    el.textContent = avO ? 'abiertas' : 'cerradas';
    el.classList.toggle('is-open', avO);
  }
  const slO = st.sl > 0.12;
  if (slO !== ui.sl) {
    ui.sl = slO;
    const el = $('#sl-state');
    el.textContent = slO ? 'abiertas' : 'cerradas';
    el.classList.toggle('is-open', slO);
  }

  if (!debug.skipRender) composer.render();

  // adaptive quality: step down once if the device cannot hold ~40 fps
  if (frames < 150 && !params.get('q')) {
    frames++;
    if (frames > 30) acc += dtReal;
    if (frames === 150) {
      const avg = acc / 120;
      if (avg > 0.05) quality.tier = 'low';
      else if (avg > 0.026) quality.tier = 'medium';
      if (quality.tier !== 'high') applyQuality();
    }
  }
}

boot();
