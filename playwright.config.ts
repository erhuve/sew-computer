import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/browser',
  timeout: 90000,
  expect: {timeout: 12000},
  workers: 2,
  use: {
    actionTimeout:15000,
    viewport: {width:1440,height:900},
    screenshot:'only-on-failure',
    trace:'retain-on-failure',
    launchOptions:{executablePath:process.env.CHROMIUM_PATH},
  },
});
