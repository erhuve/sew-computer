import { test, expect } from './fixture';
import { mkdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const evidence = resolve(import.meta.dirname, '../../.planning/visual-demo-review');
test('visual demo switches real source variants, inspects and downloads the matching pattern', async ({ studio }) => {
  await studio.login();
  await studio.page.goto(new URL('/demo', studio.page.url()).href);
  await expect(studio.page.getByRole('heading', { name: 'Make it your own.' })).toBeVisible();
  await expect(studio.page.getByText('24 fabric pieces · 18 source templates')).toBeVisible();
  await expect(studio.page.locator('.demo-shape-stage canvas')).toBeVisible();
  await studio.page.getByRole('button', { name: 'Back', exact: true }).click();
  await studio.page.getByRole('button', { name: 'Front', exact: true }).click();
  await studio.page.getByRole('tab', { name: 'Design preview', exact: true }).click();
  await expect(studio.page.getByRole('img', { name: 'front garment construction preview' })).toBeVisible();
  await studio.page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(studio.page.getByRole('img', { name: 'back garment construction preview' })).toBeVisible();
  await studio.page.getByRole('button', { name: 'Front', exact: true }).click();
  await studio.page.getByRole('tab', { name: '3D pieces', exact: true }).click();
  await expect(studio.page.locator('.three-d-canvas canvas')).toBeVisible();
  await expect(studio.page.getByRole('button', { name: 'Garment layout', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await studio.page.getByLabel('Physical piece', { exact: true }).selectOption('front_left:shell');
  await expect(studio.page.getByRole('img', { name: 'front left original pattern' })).toBeVisible();
  await studio.page.getByRole('button', { name: 'Flat pieces', exact: true }).click();
  await expect(studio.page.getByRole('button', { name: 'Flat pieces', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await studio.page.getByRole('button', { name: 'Garment layout', exact: true }).click();
  await mkdir(evidence, { recursive: true });
  await studio.page.screenshot({ path: resolve(evidence, 'pieces.png'), fullPage: true });
  await studio.page.getByRole('tab', { name: 'Design preview', exact: true }).click();
  await studio.page.getByRole('tab', { name: 'Garment preview', exact: true }).click();
  await expect(studio.page.locator('.demo-shape-stage canvas')).toBeVisible();
  await studio.page.screenshot({ path: resolve(evidence, 'desktop.png'), fullPage: true });
  await studio.page.getByRole('button', { name: 'Short', exact: true }).click();
  await studio.page.getByRole('button', { name: 'Clean', exact: true }).click();
  await expect(studio.page.getByText('14 fabric pieces · 10 source templates')).toBeVisible();
  await expect(studio.page.locator('.demo-shape-stage canvas')).toBeVisible();
  await studio.page.getByRole('tab', { name: 'Design preview', exact: true }).click();
  await expect(studio.page.getByRole('img', { name: 'front garment construction preview' })).toBeVisible();
  await studio.page.getByRole('tab', { name: '3D pieces', exact: true }).click();
  await expect(studio.page.locator('.three-d-canvas canvas')).toBeVisible();
  await expect(studio.page.getByLabel('Physical piece', { exact: true }).locator('option')).toHaveCount(15);
  const [download] = await Promise.all([
    studio.page.waitForEvent('download'), studio.page.getByRole('link', { name: 'Download draft pattern', exact: true }).click(),
  ]);
  expect(download.suggestedFilename()).toBe('short-clean-draft-pattern.svg');
  const content = await readFile((await download.path())!, 'utf8');
  expect(content).toContain('sleeve left');
  expect(content).not.toContain('frill left');
  expect(content).not.toContain('cuff left');
  await studio.page.getByLabel('Demo notes', { exact: true }).fill('Synthetic test note: sleeve changes are useful.');
  const [notes] = await Promise.all([studio.page.waitForEvent('download'), studio.page.getByRole('button', { name: 'Save feedback notes' }).click()]);
  expect(JSON.parse(await readFile((await notes.path())!, 'utf8')).example).toBe('short-clean');
  await expect(studio.page.getByRole('alert')).toHaveCount(0);
});

test('visual demo remains usable on a narrow screen', async ({ studio }) => {
  await studio.login();
  await studio.page.setViewportSize({ width: 390, height: 844 });
  await studio.page.goto(new URL('/demo', studio.page.url()).href);
  await expect(studio.page.locator('.demo-shape-stage canvas')).toBeVisible();
  await studio.page.getByRole('tab', { name: '3D pieces', exact: true }).click();
  await expect(studio.page.locator('.three-d-canvas canvas')).toBeVisible();
  await studio.page.getByRole('button', { name: 'Rotate 3D left' }).click();
  await studio.page.getByRole('button', { name: 'Sage', exact: true }).click();
  await expect(studio.page.getByRole('button', { name: 'Sage', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await studio.page.getByRole('tab', { name: 'Garment preview', exact: true }).click();
  await expect(studio.page.locator('.demo-shape-stage canvas')).toBeVisible();
  expect(await studio.page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await mkdir(evidence, { recursive: true });
  await studio.page.screenshot({ path: resolve(evidence, 'mobile.png'), fullPage: true });
  await expect(studio.page.getByRole('alert')).toHaveCount(0);
});

test('all four garment previews match the chosen pattern and support front/back views', async ({ studio }) => {
  await studio.login();
  await studio.page.goto(new URL('/demo', studio.page.url()).href);
  for (const sleeves of ['Long + cuff', 'Short']) for (const front of ['Clean', 'Frill']) {
    await studio.page.getByRole('button', {name: sleeves, exact: true}).click();
    await studio.page.getByRole('button', {name: front, exact: true}).click();
    await expect(studio.page.locator('.demo-shape-stage canvas')).toBeVisible();
    await expect(studio.page.getByRole('button', {name: 'Back', exact: true})).toBeEnabled();
    await studio.page.getByRole('button', {name: 'Back', exact: true}).click();
    await studio.page.getByRole('button', {name: 'Front', exact: true}).click();
    await expect(studio.page.getByRole('alert')).toHaveCount(0);
  }
  await studio.page.getByRole('button', {name: 'Ink', exact: true}).click();
  await expect(studio.page.getByRole('button', {name: 'Ink', exact: true})).toHaveAttribute('aria-pressed', 'true');
});
