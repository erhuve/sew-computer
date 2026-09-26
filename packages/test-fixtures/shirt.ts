import { assumed, emptyDocument } from '../contracts';
import type { ShirtDesign } from '../contracts/design';

export const shirtDesign: ShirtDesign = {
  block:'relaxed-drop-shoulder', sleeves:'long', sleeveLengthMm:550,
  cuff:'button', cuffCircumferenceMm:220, cuffDepthMm:55,
  collar:'stand-and-fall', collarStandMm:30, collarFallMm:60,
  opening:'buttons', placketWidthMm:30, buttonSpacingMm:80,
  hem:'curved-back-tail', tailExtensionMm:100,
  frill:'front-opening', frillWidthMm:35, frillFullness:1.8,
  seamAllowanceMm:10,
  rationale:'Relaxed woven button-up with a longer curved back hem and narrow gathered front frills. All dimensions are synthetic design assumptions for testing.',
};
export function shirtDocument() {
  const doc = emptyDocument('Synthetic complete shirt', 'White button-up with practical tails and tasteful frills.');
  doc.body={height:assumed(1700),bust:assumed(960),waist:assumed(760),hip:assumed(1000),shoulder:assumed(400)};
  doc.garment={family:'shirt',length:assumed(650),ease:assumed(100),flare:1,design:structuredClone(shirtDesign)};
  doc.requirements = (['body','sleeves','cuffs','collar','front-opening','tails','frills'] as const).map(feature => ({id:feature,text:feature,status:'supported' as const,note:'Synthetic fixture',feature}));
  return {...doc,garment:{...doc.garment,design:structuredClone(shirtDesign)}};
}
