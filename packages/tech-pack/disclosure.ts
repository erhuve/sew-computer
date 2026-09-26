import type {GarmentDocument,Disclosure} from '../contracts';
/** Custom outlines are geometry, and artwork handles are private reference data. */
export function disclosedGarment(garment:GarmentDocument['garment'],disclosure:Disclosure){
  const output=structuredClone(garment);
  if(!disclosure.includePatterns&&output.design?.block==='custom-pattern')output.design=null;
  if(!disclosure.includeReferences&&output.appearance?.print?.assetId)output.appearance.print.assetId=null;
  return output;
}
