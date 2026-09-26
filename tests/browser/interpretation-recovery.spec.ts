import {test,expect} from './fixture';
import {mkdir} from 'node:fs/promises';
import {resolve} from 'node:path';
const evidence=resolve(import.meta.dirname,'../../docs/verification/interpretation-recovery');

test('a failed design keeps its prompt and offers one-click retry through a real persisted job',async({studio})=>{
  const {page}=studio;await studio.login();await mkdir(evidence,{recursive:true});
  const brief='retry-interpretation-fixture — a blue linen top';
  await page.getByLabel('What are you imagining?',{exact:true}).fill(brief);
  await page.getByRole('button',{name:'Design with AI',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('Your design didn’t finish');
  await expect(page.getByLabel('Describe your garment',{exact:true})).toHaveValue(brief);
  await expect(page.getByRole('button',{name:'Try again',exact:true})).toBeEnabled();
  await page.screenshot({path:resolve(evidence,'retry-desktop.png')});
  const project=(await studio.call('GET','/projects')).projects[0],path=`/projects/${project.id}`;
  const failed=await studio.call('GET',`${path}/interpretations/latest`);
  const before=await studio.call('GET',path);
  expect(failed.status).toBe('failed');expect(before.project.headRevisionId).toBeNull();
  await page.reload();await page.getByRole('button',{name:new RegExp(project.title)}).click();
  await page.setViewportSize({width:390,height:844});
  await expect(page.getByRole('button',{name:'Try again',exact:true})).toBeInViewport();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:resolve(evidence,'retry-mobile.png')});
  await page.getByRole('button',{name:'Try again',exact:true}).click();
  await expect(page.getByRole('button',{name:'See garment',exact:true})).toBeEnabled();
  await expect(page.getByRole('alert')).toHaveCount(0);
  const succeeded=await studio.call('GET',`${path}/interpretations/latest`),after=await studio.call('GET',path);
  expect(succeeded.status).toBe('succeeded');expect(succeeded.id).not.toBe(failed.id);
  expect(after.draft.document).toEqual(before.draft.document);expect(after.project.headRevisionId).toBeNull();
});

test('stream phase and elapsed time make a slow attempt visible and it can still be cancelled',async({studio})=>{
  const {page}=studio;await studio.login();
  // The isolated HTTP fixture is non-streaming. Supply Codex progress metadata
  // on its real running job to exercise the shared browser presentation.
  await page.route('**/interpretations/latest',async route=>{
    const response=await route.fetch(),job=await response.json();
    if(job?.status==='running'){job.phase='writing';job.startedAt=new Date(Date.now()-65000).toISOString();}
    await route.fulfill({response,json:job});
  });
  await page.getByLabel('What are you imagining?',{exact:true}).fill('slow-interpretation-fixture — a blue linen top');
  await page.getByRole('button',{name:'Design with AI',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Writing your design…'})).toBeVisible();
  await expect(page.getByText('Still working.',{exact:false})).toBeVisible();
  await expect(page.getByLabel('Time spent on this attempt')).toContainText('1:');
  await mkdir(evidence,{recursive:true});await page.screenshot({path:resolve(evidence,'progress-desktop.png')});
  await page.getByRole('button',{name:'Cancel interpretation',exact:true}).click();
  await expect(page.getByText('Interpretation cancelled. Your draft is unchanged.',{exact:true})).toBeVisible();
  await expect(page.getByLabel('Describe your garment',{exact:true})).toHaveValue('slow-interpretation-fixture — a blue linen top');
});
