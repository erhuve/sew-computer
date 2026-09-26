export {};
import { interpretationFixture } from '../../packages/test-fixtures/interpretation';
import {defaultShirtDesign} from '../../packages/contracts/design';
import {startingDocument} from '../../packages/contracts/starting-designs';
import { shirtDocument } from '../../packages/test-fixtures/shirt';
const model=Bun.serve({hostname:'127.0.0.1',port:0,fetch:async request=>{
  const body = await request.text();
  if(body.includes('slow-interpretation-fixture'))await new Promise(resolve=>setTimeout(resolve,6000));
  const result = structuredClone(interpretationFixture);
  result.garment.design={...defaultShirtDesign,sleeves:'none',cuff:'none',collar:'none',opening:'none',frill:'none'};
  if (body.includes('complete-shirt-fixture')) {
    const shirt=shirtDocument();
    result.garment={...result.garment,design:shirt.garment.design!};
    result.requirements=shirt.requirements.map(({id,...row})=>({...row,feature:row.feature!}));
    result.summary='Relaxed white button-up with curved back tails and gathered front frills.';
    result.questions=[];
  }
  for(const family of ['dress','skirt'] as const)if(body.includes(`complete-${family}-fixture`)) {
    const doc=startingDocument(family);
    result.garment=doc.garment as typeof result.garment;
    result.requirements=doc.requirements.map(({id,...row})=>({...row,feature:row.feature!}));
    result.summary=`Relaxed woven ${family} with editable construction.`;result.questions=[];
  }
  if (body.includes('unsupported-tailcoat-fixture')) result.garment = {family:'none',length:{state:'unknown'},ease:{state:'unknown'},flare:1,design:null};
  return Response.json({choices:[{finish_reason:'stop',message:{content:JSON.stringify(result)}}],usage:{prompt_tokens:100,completion_tokens:200}});
}});
delete process.env.SEW_CODEX_AUTH_FILE;
process.env.SEW_AI_BASE_URL=`http://127.0.0.1:${model.port}`;
process.env.SEW_AI_API_KEY='test-provider-only';
process.env.SEW_AI_MODEL='browser-fixture';
let handler: (request: Request) => Response | Promise<Response> = () => new Response('Starting', {status:503});
const listener = Bun.serve({hostname:'127.0.0.1',port:0,fetch:request=>handler(request)});
process.env.SEW_ALLOWED_ORIGINS = `http://127.0.0.1:${listener.port}`;
const {default:application} = await import('../../apps/web/server');
handler = request => {
  const headers = new Headers(request.headers);
  headers.delete('cookie');
  headers.delete('authorization');
  return application.fetch(new Request(request, { headers }));
};
console.log(process.env.SEW_ALLOWED_ORIGINS);
