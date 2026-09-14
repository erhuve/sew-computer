import { test, expect } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createApi } from './index';

test('header sessions survive stripped cookies and preserve origin checks and revocation', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'sew-auth-'));
  const origin = 'https://studio.example';
  const api = createApi({ dataDir: directory, allowedOrigins: [origin], authKey: 'test-owner-key' });
  try {
    const login = await api.request('/auth/login', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', 'X-Sew-Session-Transport': 'header' }, body: JSON.stringify({ key: 'test-owner-key' }) });
    expect(login.status).toBe(204);
    const token = login.headers.get('X-Sew-Session')!;
    expect(token).toMatch(/^[A-Za-z0-9_-]{48}$/);
    expect(login.headers.get('Cache-Control')).toBe('no-store');
    const headers = { 'X-Sew-Session': token, Origin: origin };
    expect((await api.request('/projects', { headers })).status).toBe(200);
    expect((await api.request('/projects')).status).toBe(401);
    expect((await api.request('/projects', { headers: { 'X-Sew-Session': 'invalid' } })).status).toBe(401);
    expect((await api.request('/auth/logout', { method: 'POST', headers: { ...headers, Origin: 'https://other.example' } })).status).toBe(403);
    expect((await api.request('/auth/logout', { method: 'POST', headers })).status).toBe(204);
    expect((await api.request('/projects', { headers })).status).toBe(401);
  } finally {
    api.close();
    rmSync(directory, { recursive: true, force: true });
  }
});
