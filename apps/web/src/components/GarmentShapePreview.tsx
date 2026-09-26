import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { apiResponse, ApiError } from '../lib/api';
import { GarmentPreviewSchema } from '../../../../packages/contracts/garment-preview';
import './garment-shape.css';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

type Piece = { instanceId: string; templateId: string; role: string; restXY: number[][]; positions: number[][]; triangles: number[][]; boundaryLoops: number[][] };
type Shape = { profile: string; patternDigest: string; acceptedSimulation: boolean; pieces: Piece[]; iterations: {edgeStrainP95: number; edgeStrainMax: number; seamGapMaxMm: number}[]; buttons: {position: number[]; normal: number[]; templateId: string}[] };

/** The viewer displays saved worker positions; source rest coordinates remain separate. */
export default function GarmentShapePreview({ id, path, sha256, patternDigest, color, onSelect, onUnavailable }: {id?: string; path?: string; sha256?: string; patternDigest: string; color: string; onSelect: (id: string) => void; onUnavailable?: (error: ApiError) => void}) {
  const host = useRef<HTMLDivElement>(null);
  const appearance = useRef(color); appearance.current = color;
  const materials = useRef<THREE.MeshPhysicalMaterial[]>([]);
  const controlsRef = useRef<{ reset: () => void; back: () => void; render: () => void } | null>(null);
  const pickCallback = useRef(onSelect); pickCallback.current = onSelect;
  const unavailable=useRef(onUnavailable); unavailable.current=onUnavailable;
  const [retry,setRetry]=useState(0);
  const [pieces,setPieces]=useState<Piece[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [diagnostic, setDiagnostic] = useState<Shape['iterations'][number] | null>(null);
  useEffect(() => {
    let disposed = false, cleanup = () => {};
    const abort = new AbortController(); setError(''); setLoading(true); setDiagnostic(null);
    const load = async () => {
      let bytes:ArrayBuffer,expected=sha256;
      if(path) {
        const response=await apiResponse(path,{signal:abort.signal});
        bytes=await response.arrayBuffer();
      } else {
        if(!id||!/^(long|short)-(clean|frill)$/.test(id))throw new Error('Unsupported preview.');
        const [response,manifestResponse]=await Promise.all([fetch(`/demo-fixtures/${id}-shape.json`,{signal:abort.signal,credentials:'omit'}),fetch('/demo-fixtures/shape-catalog.json',{signal:abort.signal,credentials:'omit'})]);
        if(!response.ok||!manifestResponse.ok)throw new Error('The garment preview is unavailable.');
        bytes=await response.arrayBuffer();
        const manifest=await manifestResponse.json();expected=manifest.files?.[`${id}-shape.json`]?.sha256;
      }
      if(bytes.byteLength>16*1024*1024)throw new Error('Invalid garment preview file.');
      const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
      if(sha!==expected)throw new Error('The preview file does not match its saved generation.');
      const decoded=JSON.parse(new TextDecoder().decode(bytes));
      const data=(path?GarmentPreviewSchema.parse(decoded):decoded) as Shape;
      if(!['sew-guided-cloth-preview/1','sew-guided-cloth-preview/2'].includes(data.profile)||data.patternDigest!==patternDigest||data.acceptedSimulation!==false||!Array.isArray(data.pieces)||data.pieces.length>64)throw new Error('The garment preview does not match its pattern.');
      for (const piece of data.pieces) {
        if (piece.positions.length > 30000 || piece.restXY.length !== piece.positions.length || piece.positions.some(p => p.length !== 3 || p.some(v => !Number.isFinite(v) || Math.abs(v) > 5)) || piece.triangles.some(t => t.length !== 3 || t.some(v => !Number.isInteger(v) || v < 0 || v >= piece.positions.length))) throw new Error('Invalid preview geometry.');
      }
      if (disposed) return;
      setPieces(data.pieces);
      setDiagnostic(data.iterations.at(-1) ?? null);
      let renderer:THREE.WebGLRenderer;
      try { renderer=new THREE.WebGLRenderer({antialias:true,alpha:true}); } catch { throw new Error('3D rendering is unavailable on this device. Your pattern is still available.'); }
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = .92;
      const canvas = renderer.domElement; canvas.setAttribute('role','img'); canvas.setAttribute('aria-label','Pattern-derived garment shape preview. Drag to rotate.');
      host.current!.appendChild(canvas);
      cleanup=()=>{renderer.dispose();renderer.forceContextLoss();canvas.remove();};
      const lost=(event:Event)=>{event.preventDefault();setError('The display was interrupted. Reload to continue.');};
      canvas.addEventListener('webglcontextlost',lost);
      const scene = new THREE.Scene();
      const pmrem = new THREE.PMREMGenerator(renderer); const room = new RoomEnvironment(); const environment = pmrem.fromScene(room, .06); scene.environment = environment.texture; room.dispose(); pmrem.dispose();
      const group = new THREE.Group(); scene.add(group);
      const fabric = new THREE.MeshPhysicalMaterial({color: appearance.current, roughness: .92, metalness: 0, sheen: .45, sheenRoughness: .85, sheenColor: '#e6e3dd', side: THREE.DoubleSide, envMapIntensity: .12});
      materials.current = [fabric];
      const weave = new Uint8Array(128*128*4);
      for(let y=0;y<128;y++) for(let x=0;x<128;x++) {const value=128+25*Math.sin(x*Math.PI/2)+25*Math.sin(y*Math.PI/2)+8*Math.sin((x+y)*2.3);const k=(y*128+x)*4;weave[k]=weave[k+1]=weave[k+2]=value;weave[k+3]=255;}
      const texture = new THREE.DataTexture(weave,128,128); texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.repeat.set(45,45);texture.needsUpdate=true;texture.magFilter=THREE.LinearFilter;texture.minFilter=THREE.LinearMipmapLinearFilter;texture.generateMipmaps=true;
      fabric.bumpMap=texture;fabric.bumpScale=.00010;
      const pickable: THREE.Mesh[]=[];
      const seamMaterial=new THREE.LineBasicMaterial({color:'#7c766a',transparent:true,opacity:.2});
      for (const piece of data.pieces) {
        const geometry = new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(piece.positions.flat(),3));geometry.setAttribute('uv',new THREE.Float32BufferAttribute(piece.restXY.flat(),2));geometry.setIndex(piece.triangles.flat());geometry.computeVertexNormals();
        const mesh=new THREE.Mesh(geometry,fabric);mesh.userData.templateId=piece.templateId;mesh.castShadow=true;mesh.receiveShadow=true;group.add(mesh);pickable.push(mesh);
        if(piece.role==='shell') for(const indices of piece.boundaryLoops) { const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(indices.map(i=>new THREE.Vector3(...piece.positions[i] as [number,number,number]))),seamMaterial);group.add(line); }
      }
      const buttonMaterial=new THREE.MeshPhysicalMaterial({color:'#ece7dc',roughness:.33,metalness:0,clearcoat:.5});
      for(const button of data.buttons) {
        const b=new THREE.Mesh(new THREE.CylinderGeometry(.0042,.0042,.0018,20),buttonMaterial);b.position.fromArray(button.position);const normal=new THREE.Vector3().fromArray(button.normal);b.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),normal.normalize());b.castShadow=true;group.add(b);
      }
      scene.add(new THREE.HemisphereLight('#ffffff','#b7ad9c',.6));
      const key=new THREE.DirectionalLight('#fff7e8',2);key.position.set(-1.2,1.8,2.2);key.castShadow=true;key.shadow.mapSize.set(2048,2048);key.shadow.camera.left=-1.5;key.shadow.camera.right=1.5;key.shadow.camera.top=1.5;key.shadow.camera.bottom=-1.5;key.shadow.normalBias=.003;key.shadow.bias=-.0001;key.shadow.radius=4;scene.add(key);
      const fill=new THREE.DirectionalLight('#e7edf5',.7);fill.position.set(1,.2,-1.2);scene.add(fill);
      const bounds=new THREE.Box3().setFromObject(group);const center=bounds.getCenter(new THREE.Vector3());const size=bounds.getSize(new THREE.Vector3());
      key.target.position.copy(center);scene.add(key.target);
      const floor=new THREE.Mesh(new THREE.PlaneGeometry(20,20),new THREE.ShadowMaterial({opacity:.065}));floor.rotation.x=-Math.PI/2;floor.position.y=bounds.min.y-.055;floor.receiveShadow=true;scene.add(floor);
      const camera=new THREE.PerspectiveCamera(33,1,.01,30);const controls=new OrbitControls(camera,canvas);controls.target.copy(center);controls.enablePan=false;controls.minDistance=.5;controls.maxDistance=4;
      let frame=0;
      const render=()=>{if(!frame&&!disposed)frame=requestAnimationFrame(()=>{frame=0;if(!disposed)renderer.render(scene,camera);});};
      const reset=(back=false)=>{const distance=Math.max(size.y,size.x/camera.aspect)/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2)))*1.17+size.z*.6;camera.position.copy(center).add(new THREE.Vector3(back?-.15:.20,.06,back?-1:1).normalize().multiplyScalar(distance));controls.target.copy(center);controls.update();render();};
      const resize=()=>{const el=host.current!;renderer.setSize(el.clientWidth,el.clientHeight,false);camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix();reset();};
      controls.addEventListener('change',render);const observer=new ResizeObserver(resize);observer.observe(host.current!);
      const ray=new THREE.Raycaster();let pointer=[0,0];const down=(event:PointerEvent)=>{pointer=[event.clientX,event.clientY];};
      const pick=(event:PointerEvent)=>{if(Math.hypot(event.clientX-pointer[0]!,event.clientY-pointer[1]!)>5)return;const r=canvas.getBoundingClientRect();ray.setFromCamera(new THREE.Vector2((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1),camera);const hit=ray.intersectObjects(pickable)[0];if(hit)pickCallback.current(hit.object.userData.templateId);};
      canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointerup',pick);
      controlsRef.current={reset:()=>reset(),back:()=>reset(true),render};resize();setLoading(false);
      cleanup=()=>{canvas.removeEventListener('webglcontextlost',lost);cancelAnimationFrame(frame);observer.disconnect();controls.dispose();canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointerup',pick);scene.traverse(obj=>{if(obj instanceof THREE.Mesh||obj instanceof THREE.Line)obj.geometry.dispose();});fabric.dispose();buttonMaterial.dispose();seamMaterial.dispose();(floor.material as THREE.Material).dispose();environment.dispose();texture.dispose();key.shadow.dispose();renderer.dispose();renderer.forceContextLoss();canvas.remove();controlsRef.current=null;materials.current=[];};
    };
    void load().catch(reason=>{cleanup();cleanup=()=>{};if(!disposed){setError(String(reason.message||reason));setLoading(false);if(reason instanceof ApiError&&[401,403,404].includes(reason.status))unavailable.current?.(reason);}});
    return ()=>{disposed=true;abort.abort();cleanup();};
  },[id,path,sha256,patternDigest,retry]);
  useEffect(()=>{materials.current.forEach(m=>m.color.set(color));controlsRef.current?.render();},[color]);
  return <div className="demo-shape"><div className="demo-drawing-controls"><button disabled={loading || !!error} onClick={()=>controlsRef.current?.reset()}>Front</button><button disabled={loading || !!error} onClick={()=>controlsRef.current?.back()}>Back</button><span>Drag to rotate · scroll to zoom</span></div><div className="demo-shape-stage" ref={host}/>{loading&&<p role="status">Preparing garment preview…</p>}{error&&<p role="alert">{error} <button onClick={()=>setRetry(value=>value+1)}>Reload display</button></p>}<p className="demo-drawing-note">Pattern-derived preview · assumed cotton · fit unverified</p><details className="shape-source"><summary>Find a pattern piece</summary><select aria-label="Source pattern piece" defaultValue="" onChange={event=>onSelect(event.target.value)}><option value="" disabled>Choose a piece</option>{pieces.filter(piece=>piece.role==='shell').map(piece=><option key={piece.instanceId} value={piece.templateId}>{piece.templateId.replaceAll('_',' ')}</option>)}</select></details>{diagnostic && <details className="demo-shape-details"><summary>About this approximation</summary><p>Shape guides and elastic constraints bend and stretch the display mesh; the original pattern dimensions are unchanged. This is a posed preview, not a prediction of how fabric will fit.</p><p>95th-percentile edge deformation: {(diagnostic.edgeStrainP95*100).toFixed(1)}%. Maximum: {(diagnostic.edgeStrainMax*100).toFixed(1)}%. Maximum sampled seam gap: {diagnostic.seamGapMaxMm.toFixed(1)} mm. Body and self-contact are unchecked; allowances and interfacing are omitted.</p></details>}</div>;
}
