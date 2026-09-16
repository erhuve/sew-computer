import {test,expect} from './fixture';
import {mkdir,readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {PDFDocument} from 'pdf-lib';
import sharp from 'sharp';
const evidence=resolve(import.meta.dirname,'../../docs/verification/manual-prototype');
test('cookie-stripping proxy retains login on reload and revokes it on logout',async({studio})=>{
 await studio.login();
 await studio.page.reload();
 await expect(studio.page.getByRole('heading',{name:'What will you make?'})).toBeVisible();
 await studio.call('POST','/auth/logout');
 await studio.page.reload();
 await expect(studio.page.getByLabel('Owner access key',{exact:true})).toBeVisible();
 expect(await studio.page.evaluate(async()=>{const response=await fetch('/api/projects');return response.status;})).toBe(401);
});
async function create(studio:any,title='The everyday overshirt'){
 await studio.login();const page=studio.page;
 await page.getByRole('button',{name:'New garment',exact:true}).first().click();
 await page.getByLabel('Garment name',{exact:true}).fill(title);
 await page.getByLabel('Your idea',{exact:true}).fill('A boxy top with an asymmetric collar. Preserve the unsupported detail.');
 await page.getByRole('button',{name:'Create garment',exact:false}).click();
 await expect(page.getByRole('button',{name:'Save & generate',exact:true})).toBeVisible();
}
async function generate(studio:any){const page=studio.page;
 await page.getByRole('button',{name:'Shape & body',exact:true}).click();
 await page.getByRole('combobox',{name:'Geometry family',exact:true}).selectOption('shirt');
 await page.getByText('Try an explicitly synthetic example',{exact:true}).click();
 await page.getByRole('button',{name:'Use these assumed values'}).click();
 await page.getByRole('button',{name:'Save & generate',exact:true}).click();
 await expect(page.locator('.pattern-stage svg[role="img"]')).toBeVisible({timeout:30000});
 await expect(page.locator('.pattern-stage g[role="button"]')).toHaveCount(4);
}
test('unfinished design explains missing inputs and recovers to real generation',async({studio})=>{
 await create(studio);
 await studio.page.getByRole('button',{name:'Save & generate',exact:true}).click();
 await expect(studio.page.getByText(/Pattern generation failed: No garment family selected/)).toBeVisible();
 await expect(studio.page.getByText(/Open Design and interpret your brief/)).toBeVisible();
 await generate(studio);
 await studio.page.reload();
 await studio.page.getByRole('button',{name:/The everyday overshirt Updated/}).click();
 await expect(studio.page.locator('.pattern-stage svg[role="img"]')).toBeVisible();
 await expect(studio.page.getByText(/Pattern generation failed:/)).toHaveCount(0);
});
test('private login and real CPU generation produce useful above-fold panels',async({studio})=>{
 await create(studio);await generate(studio);
 const {page}=studio;
 const stage=await page.locator('.pattern-stage').boundingBox();expect(stage!.y).toBeLessThan(240);expect(stage!.height).toBeGreaterThan(320);expect(stage!.y+stage!.height).toBeLessThanOrEqual(900);
 await page.getByRole('button',{name:'Zoom in',exact:true}).click();await expect(page.locator('.canvas-tools output')).toHaveText('125%');
 await page.locator('.pattern-stage g[role="button"]').first().focus();await page.keyboard.press('Enter');await expect(page.locator('.selected-panel')).toBeVisible();
 await page.getByRole('button',{name:'Fit pattern to view',exact:true}).click();
 await mkdir(evidence,{recursive:true});await page.screenshot({path:resolve(evidence,'desktop-pattern.png')});
 await page.getByRole('button',{name:'Idea',exact:true}).click();await expect(page.getByRole('textbox',{name:'The idea',exact:true})).toHaveValue(/asymmetric collar/);
});
test('mobile puts the actual pattern above long authoring fields without horizontal overflow',async({studio})=>{
 await studio.page.setViewportSize({width:390,height:844});await create(studio);await generate(studio);
 const page=studio.page,stage=await page.locator('.pattern-stage').boundingBox();expect(stage!.y).toBeLessThan(220);expect(stage!.y+stage!.height).toBeLessThan(650);
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 await mkdir(evidence,{recursive:true});await page.screenshot({path:resolve(evidence,'mobile-pattern.png')});
});
test('manual tech pack exports before geometry exists, preserves privacy and downloads a real seven-section PDF',async({studio})=>{
 await create(studio,'Original garment — no preset required');const page=studio.page;
 await page.getByRole('button',{name:'Materials',exact:true}).click();await page.getByRole('button',{name:'Add material or trim',exact:true}).click();await page.getByLabel('Material 1',{exact:true}).fill('Cotton twill — sourcing undecided');
 await page.getByRole('button',{name:'Revisions & export',exact:true}).click();await page.getByRole('button',{name:'Save new revision',exact:true}).click();
 await page.getByRole('button',{name:'Export draft',exact:true}).first().click();
 await expect(page.getByLabel('Include private body inputs',{exact:true})).not.toBeChecked();
 await page.getByRole('button',{name:'Build review package',exact:true}).click();
 const link=page.getByRole('link',{name:/tech-pack.pdf/});await expect(link).toBeVisible();
 const download=page.waitForEvent('download');await link.click();const file=await download;await mkdir(evidence,{recursive:true});await file.saveAs(resolve(evidence,'sample-tech-pack.pdf'));
 const pdf=await PDFDocument.load(await readFile(resolve(evidence,'sample-tech-pack.pdf')));expect(pdf.getPageCount()).toBeGreaterThanOrEqual(7);
 const manifestLink=page.getByRole('link',{name:/manifest.json/});const href=await manifestLink.getAttribute('href');const manifest=await studio.call('GET',href!.replace(/^\/api/,''));
 expect(manifest.sections.overview.title).toBe('Original garment — no preset required');expect(manifest.sections.bodyInputs).toBeUndefined();expect(manifest.sections.bom[0].name).toContain('Cotton twill');
 await page.screenshot({path:resolve(evidence,'export.png')});
});
test('authenticated sanitized reference upload remains private and is not a simulated garment',async({studio})=>{
 await create(studio);const page=studio.page;
 await page.getByRole('button',{name:'References',exact:true}).click();
 const image=await sharp({create:{width:400,height:400,channels:3,background:'#f0eadf'}}).png().toBuffer();
 const uploaded=page.waitForResponse(response=>response.url().endsWith('/references')&&response.request().method()==='POST'&&response.status()===201);
 await page.locator('input[type=file]').first().setInputFiles({name:'original-sketch.png',mimeType:'image/png',buffer:image});
 await expect(page.locator('.reference-thumb')).toBeVisible();
 await page.getByRole('tab',{name:'Idea & references',exact:true}).click();await expect(page.locator('.reference-board img')).toBeVisible();
 await expect.poll(()=>page.locator('.reference-board img').evaluate((image:HTMLImageElement)=>image.naturalWidth)).toBe(400);
 const response=await uploaded,asset=await response.json();
 const src=response.url()+'/'+asset.id;page.once('dialog',dialog=>dialog.accept());await page.getByRole('button',{name:'Sign out',exact:true}).click();
 await expect(page.getByLabel('Owner access key',{exact:true})).toBeVisible();
 const status=await page.evaluate(async src=>(await fetch(src!)).status,src);expect(status).toBe(401);
});
test('two-tab save conflicts retain local work and allow an explicit reload',async({studio})=>{
 await create(studio);const page=studio.page;
 const {projects}=await studio.call('GET','/projects'),id=projects[0].id;const state=await studio.call('GET',`/projects/${id}`);
 await studio.call('PUT',`/projects/${id}/draft`,{expectedVersion:state.draft.version,expectedRevisionId:state.draft.baseRevisionId,document:{...state.draft.document,title:'Remote draft'}});
 await page.getByLabel('Garment name',{exact:true}).fill('Local work kept');
 await expect(page.getByText('Another tab changed this draft.',{exact:true})).toBeVisible();
 await expect(page.getByLabel('Garment name',{exact:true})).toHaveValue('Local work kept');
 await page.getByRole('button',{name:'Load saved version',exact:true}).click();await expect(page.getByLabel('Garment name',{exact:true})).toHaveValue('Remote draft');
});
test('native dialog traps focus and returns it on Escape',async({studio})=>{
 await studio.login();const page=studio.page;const trigger=page.getByRole('button',{name:'New garment',exact:true}).first();await trigger.click();
 await expect(page.locator('dialog')).toBeVisible();await page.keyboard.press('Escape');await expect(page.locator('dialog')).not.toBeVisible();await expect(trigger).toBeFocused();
});

test('slow revision saves retain subsequent local edits and reload them after autosave',async({studio})=>{
 await create(studio);const {page}=studio;
 let release!:()=>void;
 const paused=new Promise<void>(resolve=>{release=resolve;});
 let started!:()=>void;
 const requested=new Promise<void>(resolve=>{started=resolve;});
 await page.route('**/api/projects/*/revisions',async route=>{started();await paused;await route.continue();});
 await page.getByRole('button',{name:'Save & generate',exact:true}).click();
 await requested;
 await page.getByRole('textbox',{name:'The idea',exact:true}).fill('Retain the new collar edit made while saving.');
 release();
 await expect(page.getByRole('textbox',{name:'The idea',exact:true})).toHaveValue('Retain the new collar edit made while saving.');
 await expect.poll(async()=>{const {projects}=await studio.call('GET','/projects');return (await studio.call('GET',`/projects/${projects[0].id}`)).draft.document.brief;}).toBe('Retain the new collar edit made while saving.');
});

test('slow image uploads retain edits made in another authoring section',async({studio})=>{
 await create(studio);const {page}=studio;
 let release!:()=>void;
 const paused=new Promise<void>(resolve=>{release=resolve;});
 let started!:()=>void;
 const requested=new Promise<void>(resolve=>{started=resolve;});
 await page.route('**/api/projects/*/references',async route=>{started();await paused;await route.continue();});
 await page.getByRole('button',{name:'References',exact:true}).click();
 const buffer=await sharp({create:{width:32,height:32,channels:3,background:'#eee'}}).png().toBuffer();
 await page.locator('input[type=file]').first().setInputFiles({name:'sketch.png',mimeType:'image/png',buffer});
 await requested;
 await page.getByRole('button',{name:'Idea',exact:true}).click();
 await page.getByRole('textbox',{name:'The idea',exact:true}).fill('Keep this edit while the sketch uploads.');
 release();
 await expect.poll(async()=>{const {projects}=await studio.call('GET','/projects');return (await studio.call('GET',`/projects/${projects[0].id}`)).draft.document.views.length;}).toBe(1);
 await expect(page.getByRole('textbox',{name:'The idea',exact:true})).toHaveValue('Keep this edit while the sketch uploads.');
});

test('disclosure changes invalidate both pending and completed download links',async({studio})=>{
 await create(studio);const {page}=studio;
 await page.getByRole('button',{name:'Revisions & export',exact:true}).click();
 await page.getByRole('button',{name:'Save new revision',exact:true}).click();
 await page.getByRole('button',{name:'Export draft',exact:true}).first().click();
 let release!:()=>void;
 const paused=new Promise<void>(resolve=>{release=resolve;});
 let started!:()=>void;
 const requested=new Promise<void>(resolve=>{started=resolve;});
 await page.route('**/api/projects/*/exports',async route=>{started();await paused;await route.continue();});
 await page.getByLabel('Include private body inputs',{exact:true}).check();
 await page.getByRole('button',{name:'Build review package',exact:true}).click();
 await requested;
 await page.getByLabel('Include private body inputs',{exact:true}).uncheck();
 release();
 await expect(page.getByRole('button',{name:'Build review package',exact:true})).toBeEnabled();
 await expect(page.getByRole('link',{name:/tech-pack.pdf/})).toHaveCount(0);
 await page.unroute('**/api/projects/*/exports');
 await page.getByRole('button',{name:'Build review package',exact:true}).click();
 await expect(page.getByRole('link',{name:/tech-pack.pdf/})).toBeVisible();
 await page.getByLabel('Include private body inputs',{exact:true}).check();
 await expect(page.getByRole('link',{name:/tech-pack.pdf/})).toHaveCount(0);
});

test('populated garment exports real patterns and incorporates a bounded maker correction',async({studio})=>{
 await create(studio,'Cotton overshirt handoff');const {page}=studio;
 await page.getByRole('button',{name:'Add a requirement',exact:true}).click();
 await page.getByRole('textbox',{name:'Requirement 1',exact:true}).fill('Preserve asymmetric collar intent; current adapter cannot realize it.');
 await page.getByRole('combobox',{name:'Engine coverage (owner assessment)',exact:true}).selectOption('unsupported');
 await page.getByRole('button',{name:'Materials',exact:true}).click();
 for(const name of ['Cotton twill','Corozo buttons']){
   await page.getByRole('button',{name:'Add material or trim',exact:true}).click();
   await page.getByRole('textbox',{name:/^Material \d+$/}).last().fill(name);
 }
 await page.getByRole('combobox',{name:'Category',exact:true}).last().selectOption('trim');
 await page.getByRole('button',{name:'Measurements',exact:true}).click();
 for(const [index,name] of ['Flat chest','Back length'].entries()){
   await page.getByRole('button',{name:'Add point of measure',exact:true}).click();
   await page.getByRole('textbox',{name:`Point of measure ${index+1}`,exact:true}).fill(name);
   await page.getByRole('textbox',{name:'Measurement method',exact:true}).last().fill(index===0?'Across relaxed garment below armhole':'High shoulder point to back hem');
 }
 await page.getByRole('combobox',{name:'POM 1 target status',exact:true}).selectOption('known');
 await page.getByRole('spinbutton',{name:'POM 1 target value',exact:true}).fill('540');
 await page.getByRole('button',{name:'Construction',exact:true}).click();
 await page.getByRole('button',{name:'Add construction step',exact:true}).click();
 await page.getByRole('textbox',{name:'Construction step 1',exact:true}).fill('Collar attachment');
 await page.getByRole('textbox',{name:'Construction questions / details',exact:true}).fill('Confirm facing and edge finish with maker.');
 await page.getByRole('button',{name:'Add a callout',exact:true}).click();
 await page.getByRole('textbox',{name:'Callout 1',exact:true}).fill('Resolve collar construction before cutting.');
 await page.getByRole('button',{name:'References',exact:true}).click();
 const buffer=await sharp({create:{width:100,height:100,channels:3,background:'#e6dfd3'}}).png().toBuffer();
 for(const role of ['front','back']){
   await page.locator('input[type=file]').first().setInputFiles({name:`${role}-fixture.png`,mimeType:'image/png',buffer});
   await expect(page.locator('.reference-thumb')).toHaveCount(role==='front'?1:2);
   await page.getByRole('combobox',{name:'View',exact:true}).last().selectOption(role);
   await page.getByRole('combobox',{name:'Evidence type',exact:true}).last().selectOption('sketch');
   await page.getByRole('textbox',{name:'Caption / source',exact:true}).last().fill('Synthetic color swatch used to exercise the upload flow; not a garment flat.');
 }
 await generate(studio);
 await page.getByRole('button',{name:'Revisions & export',exact:true}).click();
 await page.getByRole('textbox',{name:'Reported reviewer (optional)',exact:true}).fill('Test maker');
 await page.getByRole('textbox',{name:'Feedback',exact:true}).fill('Change main fabric to lightweight linen.');
 await page.getByRole('button',{name:'Record note',exact:true}).click();
 await expect(page.getByRole('textbox',{name:'Feedback',exact:true})).toHaveValue('');
 await page.getByRole('button',{name:'Export draft',exact:true}).first().click();
 await page.getByLabel('Include generated pattern references',{exact:true}).check();
 await page.getByLabel('Include private photos, sketches and technical views',{exact:true}).check();
 await page.getByRole('button',{name:'Build review package',exact:true}).click();
 const manifestLink=page.getByRole('link',{name:/manifest.json/});await expect(manifestLink).toBeVisible();
 const href=await manifestLink.getAttribute('href');
 const manifest=await studio.call('GET',href!.replace(/^\/api/,''));
 expect(manifest.sections.bom).toHaveLength(2);expect(manifest.sections.finishedMeasurements).toHaveLength(2);
 expect(manifest.sections.patternInventory.artifacts).toHaveLength(5);expect(manifest.sections.review.comments).toHaveLength(1);
 expect(manifest.sections.bodyInputs).toBeUndefined();expect(manifest.sections.requirements[0].status).toBe('unsupported');
 const download=page.waitForEvent('download');await page.getByRole('link',{name:/tech-pack.pdf/}).click();
 await (await download).saveAs(resolve(evidence,'populated-tech-pack.pdf'));
 manifest.sections.bom[0].name='Lightweight linen';manifest.sections.materials[0].name='Lightweight linen';
 await page.keyboard.press('Escape');await page.getByRole('button',{name:'Revisions & export',exact:true}).click();
 await page.locator('dialog input[type=file]').setInputFiles({name:'maker-feedback.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(manifest))});
 await expect(page.locator('.import-change')).toHaveCount(1);
 await page.getByRole('button',{name:'Apply reviewed changes',exact:true}).click();
 await expect(page.locator('dialog')).not.toBeVisible();
 await page.getByRole('button',{name:'Materials',exact:true}).click();
 await expect(page.getByRole('textbox',{name:'Material 1',exact:true})).toHaveValue('Lightweight linen');
 await page.getByRole('button',{name:'Revisions & export',exact:true}).click();
 await page.getByRole('button',{name:'Save new revision',exact:true}).click();
 await expect(page.locator('.revision-list article')).toHaveCount(2);
 const {projects}=await studio.call('GET','/projects');const state=await studio.call('GET',`/projects/${projects[0].id}`);
 expect(state.draft.document.views).toHaveLength(2);expect(state.draft.document.body.bust.value).toBe(920);
});
