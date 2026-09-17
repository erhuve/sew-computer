import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { apiResponse, ApiError } from '../lib/api';
import { inspectPrivateGlb } from '../../../../packages/contracts/inspection-display';

type Piece = { instanceId: string; templateId: string; role: string };
type Display = { select: (instanceId: string) => void; reset: () => void; rotate: (angle: number) => void; zoom: (factor: number) => void; wireframe: (visible: boolean) => void };

export default function GarmentViewport({ path, patternDigest, selected, onSelect, onUnavailable }: {
  path: string; patternDigest: string; selected: string | null; onSelect: (templateId: string) => void; onUnavailable: (error: ApiError) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const display = useRef<Display | null>(null);
  const selectCallback = useRef(onSelect);
  selectCallback.current = onSelect;
  const unavailableCallback = useRef(onUnavailable);
  unavailableCallback.current = onUnavailable;
  const [pieces, setPieces] = useState<Piece[]>([]);
  const [instance, setInstance] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [wireframe, setWireframe] = useState(false);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    const container = host.current!;
    const controller = new AbortController();
    let disposed = false;
    let cleanup = () => {};
    setError(''); setLoading(true); setPieces([]); setInstance(''); setWireframe(false);
    const load = async () => {
      const response = await apiResponse(path, { signal: controller.signal });
      const bytes = await response.arrayBuffer();
      if (disposed) return;
      const metadata = inspectPrivateGlb(bytes);
      if (metadata.extras.patternDigest !== patternDigest) throw new Error('The 3D file does not match this saved pattern.');
      setPieces(metadata.meshes.map(mesh => mesh.extras));
      const manager = new THREE.LoadingManager();
      manager.setURLModifier(() => { throw new Error('Private 3D files cannot load external resources.'); });
      const loaded = await new GLTFLoader(manager).parseAsync(bytes, '');
      const scene = new THREE.Scene();
      scene.background = new THREE.Color('#f3f1e9');
      scene.add(loaded.scene);
      const meshes: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>[] = [];
      loaded.scene.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        const oldMaterials = Array.isArray(object.material) ? object.material : [object.material];
        oldMaterials.forEach(material => material.dispose());
        object.geometry.computeVertexNormals();
        object.material = new THREE.MeshStandardMaterial({ color: '#b4c7bb', roughness: 0.9, side: THREE.DoubleSide });
        meshes.push(object as THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>);
      });
      const disposeMeshes = () => meshes.forEach(mesh => { mesh.geometry.dispose(); mesh.material.dispose(); });
      if (disposed) { disposeMeshes(); return; }
      let renderer: THREE.WebGLRenderer;
      try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false }); }
      catch { disposeMeshes(); throw new Error('3D rendering is unavailable on this device. Your pattern and piece list remain available.'); }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      const canvas = renderer.domElement;
      canvas.setAttribute('aria-label', 'Pattern-derived 3D placement. Drag to rotate; use the controls to rotate or zoom with a keyboard.');
      canvas.setAttribute('role', 'img');
      container.appendChild(canvas);
      const camera = new THREE.PerspectiveCamera(38, 1, 0.001, 100);
      const controls = new OrbitControls(camera, canvas);
      controls.enableDamping = false;
      controls.enablePan = true;
      controls.minDistance = 0.03;
      controls.maxDistance = 80;
      scene.add(new THREE.HemisphereLight('#ffffff', '#657468', 2.4));
      const light = new THREE.DirectionalLight('#ffffff', 2);
      light.position.set(2, 3, 4); scene.add(light);
      const bounds = new THREE.Box3().setFromObject(loaded.scene);
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      let frame = 0;
      const render = () => {
        if (!frame && !disposed) frame = requestAnimationFrame(() => { frame = 0; if (!disposed) renderer.render(scene, camera); });
      };
      const reset = () => {
        const distance = Math.max(size.y, size.x / camera.aspect, 0.1) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.2;
        camera.position.copy(center).add(new THREE.Vector3(0, 0, distance));
        controls.target.copy(center); controls.update(); render();
      };
      const resize = () => {
        const width = Math.max(1, container.clientWidth), height = Math.max(1, container.clientHeight);
        renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix(); render();
      };
      const observer = new ResizeObserver(resize); observer.observe(container);
      controls.addEventListener('change', render);
      const select = (instanceId: string) => {
        meshes.forEach(mesh => mesh.material.color.set(mesh.userData.instanceId === instanceId ? '#b96439' : '#b4c7bb'));
        render();
      };
      const ray = new THREE.Raycaster();
      let pointerStart = [0, 0];
      const down = (event: PointerEvent) => { pointerStart = [event.clientX, event.clientY]; };
      const pick = (event: PointerEvent) => {
        if (Math.hypot(event.clientX - pointerStart[0]!, event.clientY - pointerStart[1]!) > 5) return;
        const rect = canvas.getBoundingClientRect();
        ray.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1), camera);
        const hit = ray.intersectObjects(meshes)[0]?.object;
        if (hit) { setInstance(hit.userData.instanceId); select(hit.userData.instanceId); selectCallback.current(hit.userData.templateId); }
      };
      const lost = (event: Event) => { event.preventDefault(); setError('The 3D display was interrupted. Reload the display to recover; your saved pattern is unchanged.'); };
      canvas.addEventListener('pointerdown', down); canvas.addEventListener('pointerup', pick); canvas.addEventListener('webglcontextlost', lost);
      display.current = {
        select, reset,
        rotate: angle => { camera.position.sub(controls.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), angle).add(controls.target); controls.update(); render(); },
        zoom: factor => { const offset = camera.position.clone().sub(controls.target); offset.setLength(THREE.MathUtils.clamp(offset.length() * factor, controls.minDistance, controls.maxDistance)); camera.position.copy(controls.target).add(offset); controls.update(); render(); },
        wireframe: visible => { meshes.forEach(mesh => { mesh.material.wireframe = visible; }); render(); },
      };
      resize(); reset();
      setLoading(false);
      cleanup = () => {
        display.current = null; cancelAnimationFrame(frame); observer.disconnect(); controls.dispose();
        canvas.removeEventListener('pointerdown', down); canvas.removeEventListener('pointerup', pick); canvas.removeEventListener('webglcontextlost', lost);
        disposeMeshes(); renderer.dispose(); renderer.forceContextLoss(); canvas.remove();
      };
    };
    void load().catch(reason => {
      if (disposed) return;
      setLoading(false); setError(reason instanceof Error ? reason.message : 'Unable to open the 3D display.');
      if (reason instanceof ApiError && [401, 403, 404].includes(reason.status)) unavailableCallback.current(reason);
    });
    return () => { disposed = true; controller.abort(); cleanup(); };
  }, [path, patternDigest, reload]);

  useEffect(() => {
    const currentPiece = pieces.find(piece => piece.instanceId === instance);
    if (currentPiece?.templateId === selected) return;
    const next = pieces.find(piece => piece.templateId === selected)?.instanceId ?? '';
    setInstance(next); display.current?.select(next);
  }, [selected, pieces, instance]);

  return <div className="garment-viewport">
    <div className="three-d-controls" aria-label="3D camera controls">
      <button onClick={() => display.current?.reset()} disabled={loading || !!error}>Reset view</button>
      <button aria-label="Rotate 3D left" onClick={() => display.current?.rotate(-Math.PI / 8)} disabled={loading || !!error}>↶</button>
      <button aria-label="Rotate 3D right" onClick={() => display.current?.rotate(Math.PI / 8)} disabled={loading || !!error}>↷</button>
      <button aria-label="Zoom 3D in" onClick={() => display.current?.zoom(0.8)} disabled={loading || !!error}>+</button>
      <button aria-label="Zoom 3D out" onClick={() => display.current?.zoom(1.25)} disabled={loading || !!error}>−</button>
      <button aria-pressed={wireframe} onClick={() => { setWireframe(!wireframe); display.current?.wireframe(!wireframe); }} disabled={loading || !!error}>Mesh edges</button>
    </div>
    <div className="three-d-canvas" ref={host} data-testid="three-d-canvas" />
    {loading && <p role="status">Opening private 3D geometry…</p>}
    {error && <div role="alert" className="alert error">{error} <button onClick={() => setReload(value => value + 1)}>Reload display</button></div>}
    <label className="three-d-piece-picker">Physical piece
      <select aria-label="Physical piece" value={instance} onChange={event => {
        const piece = pieces.find(item => item.instanceId === event.target.value);
        setInstance(event.target.value); display.current?.select(event.target.value);
        if (piece) onSelect(piece.templateId);
      }}>
        <option value="">Select a piece</option>
        {pieces.map(piece => <option key={piece.instanceId} value={piece.instanceId}>{piece.instanceId} · {piece.role}</option>)}
      </select>
    </label>
    {instance && <p className="fineprint">Source pattern: <strong>{pieces.find(piece => piece.instanceId === instance)?.templateId}</strong>. Selection carries over to the Pattern view.</p>}
  </div>;
}
