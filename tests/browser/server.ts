export {};
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
