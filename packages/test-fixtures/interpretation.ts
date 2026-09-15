import { assumed } from '../contracts';
import type { Interpretation } from '../contracts/interpretation';

export const interpretationFixture:Interpretation={
  summary:'A relaxed blue linen sleeveless top with a round neckline. Contrast embroidery remains a design note, not generated geometry.',
  garment:{family:'shirt',length:assumed(600),ease:assumed(80),flare:1.1},
  requirements:[{text:'Sleeveless round-neck top',status:'supported',note:'The current shirt adapter supports this base shape.'},{text:'Blue linen with contrast embroidery',status:'unsupported',note:'Fabric, color and embroidery are preserved in the BOM and brief; geometry does not implement them.'}],
  bom:[{name:'Blue linen',category:'fabric',specification:'Blue linen; weight and shrinkage to confirm',placement:'Main body',quantity:'',source:'AI suggestion'}],
  poms:[{name:'Finished chest circumference',method:'Measure around finished chest level without stretching',target:{state:'unknown'},tolerance:{state:'unknown'},size:'Owner size',note:'Measure a sample; body plus ease is not a verified POM.'}],
  construction:[{operation:'Join shoulder and side seams',note:'Confirm seam allowances and finish on a toile before cutting final linen.'}],
  questions:['Confirm neckline finishing, seam allowances and whether an opening is needed to pass over the head.'],
};
