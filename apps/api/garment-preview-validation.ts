import { canonical, type PatternGeometry } from '../../packages/contracts';
import type { Inspection } from '../../packages/contracts/assembly';
import { GarmentPreviewSchema } from '../../packages/contracts/garment-preview';
import { validateInspectionRest } from './inspection-validation';
import { ApiError } from './validation';

/** A display pose is allowed to deform; its source mesh and identity are not. */
export function validateGarmentPreview(bytes:Uint8Array,pattern:PatternGeometry,patternDigest:string,constructionDigest:string) {
  if(!(bytes instanceof Uint8Array)||bytes.length>16*1024*1024)throw new ApiError(422,'Garment preview budget exceeded');
  const value=GarmentPreviewSchema.parse(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes)));
  if(value.patternDigest!==patternDigest||value.constructionDigest!==constructionDigest)throw new ApiError(422,'Garment preview source mismatch');
  const templates=new Map<string,Inspection['templates'][number]>(),instances:Inspection['instances']=[];
  let vertices=0;
  for(const piece of value.pieces) {
    const panel=pattern.panels.find(panel=>panel.id===piece.templateId);
    if(!panel?.draft||piece.positions.length!==piece.restXY.length||piece.triangles.length!==piece.restTriangles.length)throw new ApiError(422,'Garment preview inventory mismatch');
    vertices+=piece.positions.length;
    const template={templateId:piece.templateId,restPositions:piece.restXY.map(([x,y])=>[x*1000,y*1000] as [number,number]),triangles:piece.restTriangles,sourceWeights:piece.sourceWeights};
    const previous=templates.get(piece.templateId);
    if(previous&&canonical(previous)!==canonical(template))throw new ApiError(422,'Garment preview facing source mismatch');
    templates.set(piece.templateId,template);
    instances.push({id:piece.instanceId,templateId:piece.templateId,role:piece.role,mirrorX:piece.mirrorX,sourceGrainline:panel.draft.grainline});
    for(let i=0;i<piece.triangles.length;i++) {
      const face=piece.triangles[i]!,source=piece.restTriangles[i]!;
      if(canonical([...face].sort((a,b)=>a-b))!==canonical([...source].sort((a,b)=>a-b)))throw new ApiError(422,'Garment preview topology changed');
    }
    const edges=new Map<string,number>();
    for(const face of piece.restTriangles)for(let i=0;i<3;i++) {
      const key=[face[i]!,face[(i+1)%3]!].sort((a,b)=>a-b).join(':');edges.set(key,(edges.get(key)??0)+1);
    }
    const boundary=new Set([...edges].filter(([,count])=>count===1).map(([key])=>key)),seen=new Set<string>();
    for(const loop of piece.boundaryLoops)for(let i=1;i<loop.length;i++) {
      const key=[loop[i-1]!,loop[i]!].sort((a,b)=>a-b).join(':');
      if(!boundary.has(key)||seen.has(key))throw new ApiError(422,'Garment preview boundary mismatch');
      seen.add(key);
    }
    if(seen.size!==boundary.size)throw new ApiError(422,'Garment preview boundary missing');
  }
  if(vertices!==value.sourceVertices||vertices>60000)throw new ApiError(422,'Garment preview vertex budget exceeded');
  validateInspectionRest({instances,templates:[...templates.values()]},pattern);
  for(const panel of pattern.panels) {
    const expected=panel.draft?.marks.filter(mark=>mark.kind==='button').length??0;
    if(value.buttons.filter(button=>button.templateId===panel.id).length!==expected)throw new ApiError(422,'Garment preview hardware mismatch');
  }
  if(value.buttons.some(button=>!templates.has(button.templateId)||Math.abs(Math.hypot(...button.normal)-1)>1e-6))throw new ApiError(422,'Garment preview hardware invalid');
  return value;
}
