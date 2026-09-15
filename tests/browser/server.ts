export {};
import { interpretationFixture } from '../../packages/test-fixtures/interpretation';
const model=Bun.serve({hostname:'127.0.0.1',port:0,fetch:()=>Response.json({choices:[{finish_reason:'stop',message:{content:JSON.stringify(interpretationFixture)}}],usage:{prompt_tokens:100,completion_tokens:200}})});
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
