import {emptyDocument,assumed} from './index';
import {applySample} from './sizing';
import {defaultShirtDesign,defaultDressDesign,defaultSkirtDesign} from './design';

export type StartingFamily='shirt'|'dress'|'skirt';
/** Explicitly chosen synthetic starting sizes; never silently applied to existing projects. */
export function startingDocument(family:StartingFamily) {
  const description={shirt:'Relaxed woven button shirt.',dress:'Relaxed woven dress with short sleeves and a gently flared hem.',skirt:'Woven skirt with an elastic waist and a gently flared hem.'}[family];
  const doc=applySample(emptyDocument(`Your ${family}`,description),2);
  doc.sizeLabel='Sample M · estimates';
  doc.garment={family,length:assumed(family==='shirt'?650:family==='dress'?1000:650),ease:assumed(80),flare:family==='shirt'?1:1.35,design:structuredClone({shirt:defaultShirtDesign,dress:defaultDressDesign,skirt:defaultSkirtDesign}[family])};
  doc.requirements=[{id:'starting-shape',text:description,status:'supported',feature:'body',note:'Chosen editable starting shape. Synthetic sizing; fit is unverified.'}];
  return doc;
}
