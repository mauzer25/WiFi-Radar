#!/usr/bin/env python3
"""
WiFi Radar — Real-time 3D Matrix-style WiFi network visualizer.

Scans nearby WiFi networks and displays them as glowing nodes in 3D space.
Uses Pearson correlation on RSSI time-series to infer physical proximity
between access points.

Usage:
    python3 wifi_radar.py          # Demo mode (works everywhere)
    python3 wifi_radar.py --live   # Real WiFi scanning (macOS only)
"""

import argparse
import asyncio
import http.server
import json
import math
import random
import threading
import time
import webbrowser
from collections import defaultdict, deque

import numpy as np

# Try importing CoreWLAN for real WiFi scanning
try:
    from CoreWLAN import CWWiFiClient
    HAS_COREWLAN = True
except ImportError:
    HAS_COREWLAN = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HTTP_PORT = 8088
WS_PORT = 8089
SCAN_INTERVAL = 4          # seconds between scans
HISTORY_LENGTH = 30        # number of scans to keep per BSSID
MIN_SAMPLES = 5            # minimum co-observed scans for correlation
CORRELATION_EDGE_THRESHOLD = 0.5
LAYOUT_ITERATIONS = 50

# ---------------------------------------------------------------------------
# WiFi Scanning
# ---------------------------------------------------------------------------

def scan_wifi_real():
    """Scan WiFi networks using macOS CoreWLAN."""
    client = CWWiFiClient.sharedWiFiClient()
    iface = client.interface()
    if iface is None:
        print("[scan] No WiFi interface found")
        return None
    # Retry up to 5 times on "Resource busy" errors
    networks = None
    for _ in range(5):
        nets, error = iface.scanForNetworksWithName_error_(None, None)
        if error:
            if 'Resource busy' in str(error):
                time.sleep(1.0)
                continue
            print(f"[scan] CoreWLAN error: {error}")
            return None
        networks = nets
        break
    if networks is None or len(networks) == 0:
        print("[scan] No networks returned — check Location Services permissions")
        return None
    results = []
    # Sort networks by (channel, rssi) for stable synthetic ID assignment
    net_list = sorted(networks, key=lambda n: (
        n.wlanChannel().channelNumber() if n.wlanChannel() else 0,
        n.rssiValue(),
    ))
    channel_counters = {}
    for net in net_list:
        bssid = net.bssid()
        ssid = net.ssid()
        channel_num = net.wlanChannel().channelNumber() if net.wlanChannel() else 0
        rssi = int(net.rssiValue())
        if bssid is None:
            idx = channel_counters.get(channel_num, 0)
            channel_counters[channel_num] = idx + 1
            bssid = f"XX:XX:XX:{channel_num:03d}:{idx:03d}:00"
        results.append({
            'bssid': str(bssid),
            'ssid': str(ssid or f'Ch{channel_num}'),
            'rssi': rssi,
            'channel': int(channel_num),
            'band': '5GHz' if channel_num > 14 else '2.4GHz',
        })
    no_loc = any(net.bssid() is None for net in networks)
    if no_loc:
        print(f"[scan] Found {len(results)} networks (no BSSID — Location Services not granted)")
    else:
        print(f"[scan] Found {len(results)} networks")
    return results


def scan_wifi_demo():
    """Generate synthetic WiFi data with correlated clusters."""
    t = time.time()
    drift1 = math.sin(t * 0.5) * 5
    drift2 = math.sin(t * 0.3) * 8
    drift3 = math.cos(t * 0.7) * 4

    clusters = [
        # Cluster 1: Home network (3 APs, highly correlated)
        ("AA:BB:CC:00:00:01", "HomeNet",      6, -45 + drift1 + random.gauss(0, 1.5)),
        ("AA:BB:CC:00:00:02", "HomeNet_5G",  36, -50 + drift1 + random.gauss(0, 1.5)),
        ("AA:BB:CC:00:00:03", "HomeNet_6E", 149, -55 + drift1 + random.gauss(0, 1.5)),
        # Cluster 2: Neighbor (2 APs, correlated with each other)
        ("DD:EE:FF:00:00:01", "Neighbor",     1, -65 + drift2 + random.gauss(0, 1)),
        ("DD:EE:FF:00:00:02", "Neighbor_5G", 44, -70 + drift2 + random.gauss(0, 1)),
        # Cluster 3: Office
        ("77:88:99:00:00:01", "Office",       6, -60 + drift3 + random.gauss(0, 2)),
        ("77:88:99:00:00:02", "Office_5G",   48, -58 + drift3 + random.gauss(0, 2)),
        # Independent APs
        ("11:22:33:00:00:01", "CoffeeShop",  11, -75 + random.gauss(0, 4)),
        ("44:55:66:00:00:01", "FreeWiFi",     6, -80 + random.gauss(0, 3)),
        ("55:66:77:00:00:01", "Guest",        3, -72 + random.gauss(0, 5)),
    ]
    return [{
        'bssid': bssid, 'ssid': ssid,
        'rssi': max(-95, min(-20, int(rssi))),
        'channel': ch,
        'band': '5GHz' if ch > 14 else '2.4GHz',
    } for bssid, ssid, ch, rssi in clusters]


# ---------------------------------------------------------------------------
# RSSI History & Pearson Correlation
# ---------------------------------------------------------------------------

rssi_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
scan_counter = 0
latest_ssid = {}  # bssid -> last seen ssid/channel/band


def record_scan(results):
    global scan_counter
    scan_counter += 1
    seen = set()
    for ap in results:
        b = ap['bssid']
        rssi_history[b].append((scan_counter, ap['rssi']))
        latest_ssid[b] = ap
        seen.add(b)
    for b in list(rssi_history.keys()):
        if b not in seen:
            rssi_history[b].append((scan_counter, None))


def compute_correlations():
    bssids = [b for b, h in rssi_history.items() if len(h) >= MIN_SAMPLES]
    n = len(bssids)
    if n < 2:
        return bssids, np.eye(max(n, 1))

    all_scans = set()
    for b in bssids:
        for idx, _ in rssi_history[b]:
            all_scans.add(idx)
    scan_list = sorted(all_scans)
    scan_to_col = {s: i for i, s in enumerate(scan_list)}

    matrix = np.full((n, len(scan_list)), np.nan)
    for i, b in enumerate(bssids):
        for idx, rssi in rssi_history[b]:
            if rssi is not None:
                matrix[i, scan_to_col[idx]] = rssi

    corr = np.zeros((n, n))
    np.fill_diagonal(corr, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            mask = ~np.isnan(matrix[i]) & ~np.isnan(matrix[j])
            if mask.sum() >= MIN_SAMPLES:
                xi, xj = matrix[i, mask], matrix[j, mask]
                if np.std(xi) > 0 and np.std(xj) > 0:
                    r = np.corrcoef(xi, xj)[0, 1]
                    if not np.isnan(r):
                        corr[i, j] = corr[j, i] = r
    return bssids, corr


# ---------------------------------------------------------------------------
# Force-Directed 3D Layout
# ---------------------------------------------------------------------------

prev_positions = {}


def compute_layout(bssids, corr):
    global prev_positions
    n = len(bssids)
    if n == 0:
        prev_positions = {}
        return {}

    pos = np.array([
        prev_positions.get(b, np.random.randn(3) * 10.0) for b in bssids
    ], dtype=float)

    for _ in range(LAYOUT_ITERATIONS):
        forces = np.zeros((n, 3))
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[j] - pos[i]
                dist = max(np.linalg.norm(diff), 0.1)
                direction = diff / dist

                # Repulsion: always push apart (stronger when close)
                repulsion = -3.0 / (dist * dist + 0.1)
                forces[i] += direction * repulsion
                forces[j] -= direction * repulsion

                # Attraction: only for correlated pairs
                if corr[i, j] > CORRELATION_EDGE_THRESHOLD:
                    target = 2.0 + (1.0 - corr[i, j]) * 4.0
                    attraction = (dist - target) * 0.05
                    forces[i] += direction * attraction
                    forces[j] -= direction * attraction
                else:
                    # Non-correlated: push to target distance
                    target = 6.0 + (1.0 - max(corr[i, j], 0)) * 6.0
                    f = (dist - target) * 0.02
                    forces[i] += direction * f
                    forces[j] -= direction * f

            forces[i] -= pos[i] * 0.005  # weak centering

        pos += np.clip(forces, -2.0, 2.0) * 0.3

    prev_positions = {bssids[i]: pos[i].copy() for i in range(n)}
    return {bssids[i]: pos[i].tolist() for i in range(n)}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

connected_clients = set()
event_loop = None


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WiFi Radar</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #000; overflow: hidden; font-family: 'Courier New', monospace; }

  #hud {
    position: absolute; top: 0; left: 0; right: 0; padding: 16px 20px;
    color: #00ff41; font-size: 13px; pointer-events: none; z-index: 10;
    text-shadow: 0 0 8px #00ff41;
  }
  #hud .title { font-size: 22px; font-weight: bold; letter-spacing: 6px; }
  #hud .stats { margin-top: 4px; opacity: 0.7; font-size: 12px; }

  #tooltip {
    display: none; position: absolute; padding: 12px 16px;
    background: rgba(0,20,0,0.92); border: 1px solid #00ff41;
    color: #00ff41; font-size: 13px; pointer-events: none; z-index: 20;
    border-radius: 6px; white-space: nowrap;
    box-shadow: 0 0 20px rgba(0,255,65,0.4);
    line-height: 1.5;
  }

  #info-panel {
    position: absolute; top: 70px; right: 20px; width: 280px;
    background: rgba(0,15,0,0.88); border: 1px solid #00ff4180;
    color: #00ff41; font-size: 12px; z-index: 15;
    border-radius: 8px; padding: 0;
    box-shadow: 0 0 30px rgba(0,255,65,0.15);
    max-height: calc(100vh - 100px); overflow-y: auto;
  }
  #info-panel .panel-title {
    padding: 12px 14px; border-bottom: 1px solid #00ff4140;
    font-size: 13px; font-weight: bold; letter-spacing: 2px;
    text-shadow: 0 0 6px #00ff41;
  }
  .net-item {
    padding: 8px 14px; border-bottom: 1px solid #00ff4115;
    cursor: pointer; transition: background 0.2s;
    display: flex; justify-content: space-between; align-items: center;
  }
  .net-item:hover { background: rgba(0,255,65,0.08); }
  .net-item.selected { background: rgba(0,255,65,0.15); border-left: 3px solid #00ff41; }
  .net-item .name { font-weight: bold; font-size: 12px; }
  .net-item .meta { opacity: 0.6; font-size: 10px; }
  .net-item .rssi-bar {
    width: 50px; height: 6px; background: #001a00; border-radius: 3px;
    overflow: hidden; margin-top: 3px;
  }
  .net-item .rssi-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }

  #controls {
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 10px; z-index: 15;
  }
  #controls button {
    background: rgba(0,20,0,0.8); border: 1px solid #00ff4160;
    color: #00ff41; padding: 8px 16px; font-family: 'Courier New', monospace;
    font-size: 11px; cursor: pointer; border-radius: 4px;
    transition: all 0.2s; letter-spacing: 1px;
  }
  #controls button:hover { background: rgba(0,255,65,0.15); border-color: #00ff41; }
  #controls button.active { background: rgba(0,255,65,0.2); border-color: #00ff41; }
</style>
</head>
<body>
<div id="hud">
  <div class="title">WIFI RADAR</div>
  <div class="stats" id="stats">INITIALIZING...</div>
</div>
<div id="tooltip"></div>
<div id="info-panel">
  <div class="panel-title">NETWORKS</div>
  <div id="net-list"></div>
</div>
<div id="controls">
  <button id="btn-rotate" class="active" onclick="toggleRotate()">AUTO-ROTATE</button>
  <button id="btn-labels" class="active" onclick="toggleLabels()">LABELS</button>
  <button id="btn-edges" class="active" onclick="toggleEdges()">EDGES</button>
  <button onclick="resetCamera()">RESET VIEW</button>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ---- Scene ----
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.012);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 300);
camera.position.set(0, 12, 30);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ReinhardToneMapping;
renderer.toneMappingExposure = 1.5;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.3;
controls.maxDistance = 80;
controls.minDistance = 5;

// ---- Post Processing ----
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(
  new THREE.Vector2(innerWidth, innerHeight), 1.5, 0.6, 0.8
);
composer.addPass(bloom);

// ---- Lights ----
scene.add(new THREE.AmbientLight(0x003300, 0.8));
const dirLight = new THREE.DirectionalLight(0x00ff41, 0.3);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// ---- Grid ----
const grid = new THREE.GridHelper(60, 60, 0x004400, 0x001a00);
grid.position.y = -8;
scene.add(grid);

// ---- Digital Rain ----
const RAIN_COUNT = 2000;
const rainGeo = new THREE.BufferGeometry();
const rainPos = new Float32Array(RAIN_COUNT * 3);
const rainVel = new Float32Array(RAIN_COUNT);
for (let i = 0; i < RAIN_COUNT; i++) {
  rainPos[i * 3]     = (Math.random() - 0.5) * 60;
  rainPos[i * 3 + 1] = Math.random() * 40 - 8;
  rainPos[i * 3 + 2] = (Math.random() - 0.5) * 60;
  rainVel[i] = 0.02 + Math.random() * 0.05;
}
rainGeo.setAttribute('position', new THREE.BufferAttribute(rainPos, 3));
const rainMat = new THREE.PointsMaterial({
  color: 0x00ff41, size: 0.05, transparent: true, opacity: 0.2,
  blending: THREE.AdditiveBlending, depthWrite: false,
});
scene.add(new THREE.Points(rainGeo, rainMat));

function animateRain() {
  for (let i = 0; i < RAIN_COUNT; i++) {
    rainPos[i * 3 + 1] -= rainVel[i];
    if (rainPos[i * 3 + 1] < -8) {
      rainPos[i * 3 + 1] = 35;
      rainPos[i * 3]     = (Math.random() - 0.5) * 60;
      rainPos[i * 3 + 2] = (Math.random() - 0.5) * 60;
    }
  }
  rainGeo.attributes.position.needsUpdate = true;
}

// ---- State ----
let showLabels = true;
let showEdges = true;
let selectedId = null;
let latestData = { nodes: [], edges: [] };

// ---- Node Management ----
const nodeGroup = new THREE.Group();
scene.add(nodeGroup);
const nodeMeshes = {};   // id -> { mesh, label, ring, data, phaseOffset }
const edgeGroup = new THREE.Group();
scene.add(edgeGroup);

// Band colors
function bandColor(band, rssi) {
  const strength = THREE.MathUtils.clamp((rssi + 90) / 55, 0, 1);
  if (band === '5GHz') {
    return new THREE.Color(0.0, 0.6 + strength * 0.4, 0.8 + strength * 0.2);  // cyan
  }
  return new THREE.Color(0.0, 0.7 + strength * 0.3, 0.2 + strength * 0.3);  // green
}

function rssiToRadius(rssi) {
  return THREE.MathUtils.clamp(THREE.MathUtils.mapLinear(rssi, -90, -30, 0.3, 1.0), 0.25, 1.2);
}

function makeLabel(text, rssi) {
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 96;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 32px Courier New';
  ctx.fillStyle = '#00ff41';
  ctx.textAlign = 'center';
  ctx.shadowColor = '#00ff41';
  ctx.shadowBlur = 10;
  ctx.fillText(text.slice(0, 20), 256, 36);
  ctx.font = '22px Courier New';
  ctx.fillStyle = '#00ff4199';
  ctx.shadowBlur = 4;
  ctx.fillText(`${rssi} dBm`, 256, 72);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.9, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(4.5, 0.85, 1);
  return sprite;
}

function makeRing(radius, color) {
  const geo = new THREE.RingGeometry(radius * 2.0, radius * 2.3, 32);
  const mat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.0, side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

function updateScene(data) {
  latestData = data;
  const { nodes, edges } = data;
  const seen = new Set();

  for (const node of nodes) {
    seen.add(node.id);
    let entry = nodeMeshes[node.id];
    const radius = rssiToRadius(node.rssi);
    const color = bandColor(node.band, node.rssi);

    if (!entry) {
      const geo = new THREE.SphereGeometry(radius, 24, 24);
      const mat = new THREE.MeshStandardMaterial({
        color, emissive: color, emissiveIntensity: 1.8,
        transparent: true, opacity: 0.9, roughness: 0.2, metalness: 0.4,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData.nodeId = node.id;

      // Outer glow shell
      const glowGeo = new THREE.SphereGeometry(radius * 2.5, 16, 16);
      const glowMat = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0.06,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.BackSide,
      });
      mesh.add(new THREE.Mesh(glowGeo, glowMat));

      // Point light per node
      const light = new THREE.PointLight(color, 0.5, 8);
      mesh.add(light);

      const label = makeLabel(node.ssid || node.id.slice(-8), node.rssi);
      label.position.y = radius + 1.0;
      mesh.add(label);

      // Selection ring
      const ring = makeRing(radius, color);
      mesh.add(ring);

      nodeGroup.add(mesh);
      entry = { mesh, label, ring, data: node, phaseOffset: Math.random() * Math.PI * 2 };
      nodeMeshes[node.id] = entry;
    }

    entry.data = node;

    // Smooth position update
    const target = new THREE.Vector3(...node.pos);
    entry.mesh.position.lerp(target, 0.06);

    // Update color
    const col = bandColor(node.band, node.rssi);
    entry.mesh.material.color.copy(col);
    entry.mesh.material.emissive.copy(col);

    // Label visibility
    entry.label.visible = showLabels;

    // Selection highlight
    const isSelected = node.id === selectedId;
    entry.ring.material.opacity = isSelected ? 0.5 : 0.0;
  }

  // Remove stale nodes
  for (const [id, entry] of Object.entries(nodeMeshes)) {
    if (!seen.has(id)) {
      nodeGroup.remove(entry.mesh);
      delete nodeMeshes[id];
      if (selectedId === id) selectedId = null;
    }
  }

  // Update edges
  while (edgeGroup.children.length) edgeGroup.remove(edgeGroup.children[0]);
  if (showEdges) {
    for (const edge of edges) {
      const a = nodeMeshes[edge.from], b = nodeMeshes[edge.to];
      if (!a || !b) continue;
      const points = [a.mesh.position.clone(), b.mesh.position.clone()];
      const geo = new THREE.BufferGeometry().setFromPoints(points);

      const isHighlighted = (selectedId && (edge.from === selectedId || edge.to === selectedId));
      const opacity = isHighlighted
        ? Math.max(0.4, edge.correlation * 0.8)
        : Math.max(0.08, (edge.correlation - 0.3) * 0.5);
      const lineColor = isHighlighted ? 0x00ffaa : 0x00ff41;

      const mat = new THREE.LineBasicMaterial({
        color: lineColor, transparent: true, opacity,
        blending: THREE.AdditiveBlending, depthWrite: false,
      });
      edgeGroup.add(new THREE.Line(geo, mat));
    }
  }

  // HUD stats
  document.getElementById('stats').textContent =
    `${nodes.length} ACCESS POINTS  //  ${edges.length} CORRELATIONS  //  SCAN #${data.scan || 0}`;

  // Update side panel
  updateNetList(nodes);
}

// ---- Network list panel ----
function updateNetList(nodes) {
  const list = document.getElementById('net-list');
  const sorted = [...nodes].sort((a, b) => b.rssi - a.rssi);
  list.innerHTML = sorted.map(n => {
    const strength = Math.max(0, Math.min(100, ((n.rssi + 90) / 55) * 100));
    const color = n.band === '5GHz' ? '#00ccff' : '#00ff41';
    const sel = n.id === selectedId ? 'selected' : '';
    return `<div class="net-item ${sel}" data-id="${n.id}">
      <div>
        <div class="name" style="color:${color}">${n.ssid || '(Hidden)'}</div>
        <div class="meta">Ch ${n.channel} / ${n.band} / ${n.rssi} dBm</div>
        <div class="rssi-bar"><div class="rssi-fill" style="width:${strength}%;background:${color}"></div></div>
      </div>
    </div>`;
  }).join('');

  list.querySelectorAll('.net-item').forEach(el => {
    el.addEventListener('click', () => {
      const id = el.dataset.id;
      selectedId = (selectedId === id) ? null : id;
      if (selectedId) flyToNode(selectedId);
      updateScene(latestData);
    });
  });
}

// ---- Fly camera to node ----
function flyToNode(id) {
  const entry = nodeMeshes[id];
  if (!entry) return;
  const pos = entry.mesh.position;
  controls.target.copy(pos);
}

// ---- Click to select in 3D ----
const raycaster = new THREE.Raycaster();
raycaster.params.Mesh = { threshold: 0.5 };
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');

renderer.domElement.addEventListener('click', (e) => {
  mouse.x = (e.clientX / innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const meshes = Object.values(nodeMeshes).map(n => n.mesh);
  const hits = raycaster.intersectObjects(meshes);
  if (hits.length > 0) {
    const id = hits[0].object.userData.nodeId;
    if (id) {
      selectedId = (selectedId === id) ? null : id;
      if (selectedId) flyToNode(selectedId);
      updateScene(latestData);
    }
  } else {
    selectedId = null;
    updateScene(latestData);
  }
});

renderer.domElement.addEventListener('mousemove', (e) => {
  mouse.x = (e.clientX / innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const meshes = Object.values(nodeMeshes).map(n => n.mesh);
  const hits = raycaster.intersectObjects(meshes);
  if (hits.length > 0) {
    const hit = hits[0].object;
    const entry = Object.values(nodeMeshes).find(n => n.mesh === hit);
    if (entry) {
      const d = entry.data;
      renderer.domElement.style.cursor = 'pointer';
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 16) + 'px';
      tooltip.style.top = (e.clientY + 16) + 'px';

      // Find correlations for this node
      const corrs = latestData.edges
        .filter(e => e.from === d.id || e.to === d.id)
        .map(e => {
          const otherId = e.from === d.id ? e.to : e.from;
          const other = latestData.nodes.find(n => n.id === otherId);
          return `${other?.ssid || '?'} (${(e.correlation * 100).toFixed(0)}%)`;
        });
      const corrText = corrs.length > 0
        ? `<br><span style="opacity:0.6">Correlated: ${corrs.join(', ')}</span>`
        : '';

      tooltip.innerHTML =
        `<b>${d.ssid || '(Hidden)'}</b><br>` +
        `BSSID: ${d.id}<br>` +
        `Signal: ${d.rssi} dBm<br>` +
        `Channel: ${d.channel} (${d.band})` +
        corrText;
    }
  } else {
    renderer.domElement.style.cursor = 'default';
    tooltip.style.display = 'none';
  }
});

// ---- Controls ----
window.toggleRotate = () => {
  controls.autoRotate = !controls.autoRotate;
  document.getElementById('btn-rotate').classList.toggle('active');
};
window.toggleLabels = () => {
  showLabels = !showLabels;
  document.getElementById('btn-labels').classList.toggle('active');
  for (const entry of Object.values(nodeMeshes)) entry.label.visible = showLabels;
};
window.toggleEdges = () => {
  showEdges = !showEdges;
  document.getElementById('btn-edges').classList.toggle('active');
  updateScene(latestData);
};
window.resetCamera = () => {
  camera.position.set(0, 12, 30);
  controls.target.set(0, 0, 0);
  selectedId = null;
  updateScene(latestData);
};

// ---- WebSocket ----
let scanCount = 0;
function connectWS() {
  const ws = new WebSocket(`ws://${location.hostname}:""" + str(WS_PORT) + r"""`);
  ws.onmessage = (e) => {
    scanCount++;
    updateScene(JSON.parse(e.data));
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
  ws.onerror = () => {};
}
connectWS();

// ---- Resize ----
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
});

// ---- Animate ----
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  animateRain();

  const t = performance.now() * 0.001;
  for (const entry of Object.values(nodeMeshes)) {
    // Pulse glow
    const pulse = 1.5 + Math.sin(t * 2.0 + entry.phaseOffset) * 0.5;
    entry.mesh.material.emissiveIntensity = pulse;

    // Gentle float
    entry.mesh.position.y += Math.sin(t * 1.2 + entry.phaseOffset) * 0.002;

    // Rotate selection ring
    if (entry.ring) entry.ring.rotation.z = t * 0.5;
  }

  composer.render();
}
animate();
</script>
</body>
</html>"""


class HTTPHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def log_message(self, *args):
        pass


async def ws_handler(websocket):
    connected_clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        connected_clients.discard(websocket)


async def broadcast(payload):
    if connected_clients:
        msg = json.dumps(payload)
        await asyncio.gather(
            *[c.send(msg) for c in connected_clients],
            return_exceptions=True,
        )


def scanner_loop(scan_fn, loop):
    """Background thread: scan WiFi, compute correlations, broadcast."""
    scan_num = 0
    last_payload = {'nodes': [], 'edges': [], 'scan': 0}
    while True:
        try:
            scan_num += 1
            results = scan_fn()
            # If scan failed (None), re-broadcast last known data
            if results is None:
                last_payload['scan'] = scan_num
                asyncio.run_coroutine_threadsafe(broadcast(last_payload), loop)
                time.sleep(SCAN_INTERVAL)
                continue
            nodes = []
            edges = []

            if results:
                record_scan(results)
                bssids, corr = compute_correlations()
                positions = compute_layout(bssids, corr)

                for b in bssids:
                    if b in positions and b in latest_ssid:
                        ap = latest_ssid[b]
                        nodes.append({
                            'id': b,
                            'ssid': ap['ssid'],
                            'rssi': ap['rssi'],
                            'channel': ap['channel'],
                            'band': ap['band'],
                            'pos': positions[b],
                        })

                for i, b1 in enumerate(bssids):
                    for j, b2 in enumerate(bssids):
                        if i < j and corr[i, j] > CORRELATION_EDGE_THRESHOLD:
                            edges.append({
                                'from': b1, 'to': b2,
                                'correlation': round(float(corr[i, j]), 3),
                            })

            payload = {'nodes': nodes, 'edges': edges, 'scan': scan_num}
            last_payload = payload
            asyncio.run_coroutine_threadsafe(broadcast(payload), loop)

        except Exception as e:
            print(f"[scan error] {e}")

        time.sleep(SCAN_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description='WiFi Radar — 3D network visualizer')
    parser.add_argument('--live', action='store_true',
                        help='Use real WiFi scanning (macOS only, requires CoreWLAN + Location Services)')
    parser.add_argument('--port', type=int, default=HTTP_PORT,
                        help=f'HTTP port (default {HTTP_PORT})')
    args = parser.parse_args()

    if args.live:
        if HAS_COREWLAN:
            scan_fn = scan_wifi_real
            print("[mode] Live — scanning with CoreWLAN")
            print("[info] If no networks appear, enable Location Services for Terminal:")
            print("       System Settings > Privacy & Security > Location Services")
        else:
            scan_fn = scan_wifi_demo
            print("[warn] CoreWLAN not available — falling back to demo mode")
            print("[info] For live scanning on macOS: pip3 install pyobjc-framework-CoreWLAN")
    else:
        scan_fn = scan_wifi_demo
        print("[mode] Demo — using synthetic WiFi data")
        print("[info] Use --live for real WiFi scanning (macOS only)")

    # Start HTTP server
    httpd = http.server.HTTPServer(('127.0.0.1', args.port), HTTPHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # Asyncio event loop for WebSocket
    loop = asyncio.new_event_loop()

    # Start scanner thread
    threading.Thread(target=scanner_loop, args=(scan_fn, loop), daemon=True).start()

    url = f'http://127.0.0.1:{args.port}'
    print(f"[server] Running at {url}")
    print("[server] Press Ctrl+C to stop\n")
    webbrowser.open(url)

    async def serve_ws():
        try:
            import websockets
        except ImportError:
            print("[error] websockets not installed. Run: pip3 install websockets")
            return
        async with websockets.serve(ws_handler, '127.0.0.1', WS_PORT):
            await asyncio.Future()

    try:
        loop.run_until_complete(serve_ws())
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")


if __name__ == '__main__':
    main()
