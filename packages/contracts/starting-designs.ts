import {defaultPanelDress} from './dress';
import {defaultCustomPattern} from './custom-pattern';
import {emptyDocument,assumed} from './index';
import {applySample,withPreviewSizing} from './sizing';
import {defaultShirtDesign,defaultSkirtDesign} from './design';

export type StartingFamily='shirt'|'dress'|'skirt'|'custom';
/** A deliberate owner-selected approximation; original intent and unsupported details survive. */
export function simplifiedPreviewDocument(doc:ReturnType<typeof emptyDocument>,family:'shirt'|'dress'|'skirt') {
  const base=startingDocument(family);
  const result=withPreviewSizing({...doc,garment:{...base.garment,...(doc.garment.appearance===undefined?{}:{appearance:doc.garment.appearance})}});
  result.requirements=doc.requirements.map(row=>row.status==='supported'&&row.feature!=='material'?{...row,status:'unresolved' as const}:row);
  result.requirements.push({id:`simplified-preview-${crypto.randomUUID()}`,text:`Preview as ${base.brief}`,feature:'body',status:'supported',note:'Owner explicitly chose this simplified construction for a preview. Original design requests remain above; this does not implement unsupported details.'});
  return result;
}
/** Explicitly chosen synthetic starting sizes; never silently applied to existing projects. */
export function startingDocument(family:StartingFamily) {
  if(family==='custom'){const doc=emptyDocument('Your custom pattern','Original garment with owner-authored sewing pieces.');doc.garment={family:'custom',length:{state:'not-applicable'},ease:{state:'not-applicable'},flare:1,design:structuredClone(defaultCustomPattern)};return doc;}
  const description={custom:'Original garment with editable sewing pieces and explicit edge connections.',shirt:'Relaxed woven button shirt.',dress:'Woven dress with a separate bodice, waist seam and flared skirt.',skirt:'Woven skirt with an elastic waist and a gently flared hem.'}[family];
  const doc=applySample(emptyDocument(`Your ${family}`,description),2);
  doc.sizeLabel='Sample M · estimates';
  doc.garment={family,length:assumed(family==='shirt'?650:family==='dress'?1000:650),ease:assumed(80),flare:family==='shirt'?1:1.35,design:structuredClone({shirt:defaultShirtDesign,dress:defaultPanelDress,skirt:defaultSkirtDesign,custom:defaultCustomPattern}[family])};
  doc.requirements=[{id:'starting-shape',text:description,status:'supported',feature:'body',note:'Chosen editable starting shape. Synthetic sizing; fit is unverified.'}];
  return doc;
}
