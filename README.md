# Corazón humano · atlas vivo 3D

Corazón humano adulto (~300 g, 12 × 8,5 × 6 cm) modelado **procedimentalmente** en Blender
(headless, `bpy`) a partir de campos de distancia con signo, y un visor WebGL (Three.js + Vite)
con ciclo cardíaco fisiológico, ECG y ruidos cardíacos sincronizados, hemodinámica de partículas
y disección real con *stencil cap*.

**Demo:** https://alvarodesigns34.github.io/heart/

---

## Pipeline de Blender (`scripts/blender/`)

```bash
blender --background --python scripts/blender/run_pipeline.py -- \
        [--stages 01,02,03,04,05] [--voxel 0.04] [--samples 96] [--res 1024] [--no-render]
```

Cada etapa se ejecuta en su propio proceso de Blender y deja sus resultados en `$HEART_BUILD`
(por defecto `/tmp/heart_build`), así que cualquiera puede repetirse por separado.
Requiere `scipy` y `scikit-image` en el Python de Blender (el orquestador los instala si faltan).

| Script | Qué construye |
|---|---|
| `heartlib.py` | Marco anatómico (eje largo del VI, eje septal, eje diafragmático), landmarks valvulares, caja de herramientas SDF (round-cones, uniones/sustracciones suaves, ruido de ondas planas), campos de desplazamiento fisiológico compartidos. |
| `01_chambers_myocardium.py` | Cuatro cavidades como volúmenes SDF. **VI** en bala con pared de 12–13 mm; **VD** en semiluna con pared libre de 4 mm abrazando un tabique que se abomba hacia él; infundíbulo liso; aurículas con orejuela derecha triangular de base ancha y orejuela izquierda larga y festoneada. Relieve endocárdico volumétrico: trabéculas carnosas (malla esponjosa fina en VI, gruesa en VD), 2 papilares en VI y 3 en VD, banda moderadora, músculos pectíneos, *crista terminalis*, fosa oval con limbo. |
| `02_great_vessels.py` | Aorta (3 senos de Valsalva, cayado, tronco braquiocefálico → subclavia + carótida derechas, carótida común y subclavia izquierdas, aorta descendente), tronco pulmonar y ramas, ligamento arterioso, cavas (válvula de Eustaquio) y cuatro venas pulmonares, todos con pared real. Composición booleana `envolvente \ pool sanguíneo`, muñones planos, relajación gaussiana, limpieza de componentes y **voxel remesh** (marching cubes a 0,4 mm, ~3,3 M triángulos → diezmado cuádrico a 750 k). Malla *manifold* y sin costuras. |
| `03_valves_chordae.py` | Mitral (valva anterior + posterior festoneada P1-P2-P3, anillo en silla de montar), tricúspide (anterior, posterior, septal), aórtica y pulmonar (3 sigmoideas en nido de golondrina con lúnulas y nódulos de Arancio/Morgagni). Valvas sólidas de grosor variable ajustadas por *ray-casting* al orificio real y proyectadas fuera del tejido en ambas poses. Cuerdas tendinosas en árbol de 1.º→2.º→3.er orden (Bézier) desde los ápices papilares al borde libre y zona rugosa, más cuerdas septales directas. Shape keys `av_valves_open` y `semilunar_valves_open`. |
| `04_coronaries_epicardial_fat.py` | Árbol coronario 3D *shrink-wrapped* (proyección de Newton sobre el SDF epicárdico + relajación al fondo del surco): tronco común → LAD (diagonales, septales, ramas al VD) + LCx (marginales obtusas, posterolateral); RCA (cono, nodo sinusal, marginales agudas, PDA en la cruz). Venas magna, media, menor, posterior del VI, marginal izquierda y anteriores, drenando en el seno coronario. Uniones por voxel remesh. Grasa epicárdica lobulada a lo largo de los surcos envolviendo parcialmente los vasos. |
| `05_materials_bake_export.py` | Materiales PBR procedurales (`heartmat.py`): estriación helicoidal de fibras de Torrent-Guasp (hélice levógira subepicárdica / dextrógira subendocárdica), red capilar superficial, moteado, tono violáceo de cavidades derechas, SSS. Atlas UV + horneado Cycles de **Albedo / Normal / Roughness / AO** (4K), shape keys `atrial_systole` y `ventricular_contraction` (acortamiento base-ápex, engrosamiento parietal, torsión helicoidal del ápex, fuelle del VD) en todas las mallas a partir del mismo campo de desplazamiento → sin interpenetración. Exporta glTF y metadatos del visor (pines, rutas de flujo, planos). |
| `run_pipeline.py` | Orquestador + renders diagnósticos Cycles 1024² en `/tmp/renders/` (anterior, posterior, corte de 4 cámaras, plano valvular). |

`scripts/optimize-glb.mjs` cuantiza y comprime con **Meshopt** geometría y morph targets y
re-codifica las texturas a WebP → `public/models/human_heart.glb` (< 45 MB).

## Visor web (`src/`)

- **Shading**: `MeshPhysicalMaterial` (clearcoat 0.85 / 0.12, mapas horneados, *sheen* y
  translucidez en grasa y valvas), entorno PMREM de quirófano procedural, luz clave cálida y
  contraluz fría que viajan con la cámara, sombras suaves. Post-procesado `EffectComposer`:
  SSAO (con pre-pase de normales consciente del corte), Unreal Bloom sutil y tone mapping ACES.
- **Ciclo cardíaco** (`cycle.js`): diagrama de Wiggers paramétrico en función de la FC
  (PR y QT dependientes de la frecuencia): sístole auricular → contracción isovolumétrica →
  eyección rápida/reducida → relajación isovolumétrica → llenado rápido → diástasis.
  ECG derivación II, volumen del VI, estado valvular y fonocardiograma (S1/S2 sintetizados con
  WebAudio + pitido opcional en la onda R) derivan de la misma fase.
- **Hemodinámica**: ~26 000 partículas GPU en 10 rutas anatómicas (cavas → AD → VD → pulmonares;
  venas pulmonares → AI → VI → aorta y troncos supraaórticos), aceleran en sístole y se frenan
  contra las válvulas cerradas.
- **Disección**: planos coronal (4 cámaras), sagital y transversal (eje corto) con
  *stencil buffer cap* texturizado como músculo/grasa cortados.
- **HUD**: capas (miocardio sólido/translúcido/oculto, grasa, coronarias, válvulas, flujo,
  pines), 7 vistas, 5 preparaciones, ~55 pines anatómicos con ficha, FC 40–180 lpm,
  pausa y cámara lenta. Calidad adaptativa (`?q=high|medium|low`).

```bash
npm install
npm run dev        # desarrollo
npm run build      # producción -> dist/
npm run preview    # sirve dist/ en /heart/
```

El despliegue en GitHub Pages lo hace `.github/workflows/deploy.yml` en cada *push* a `main`.
