import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const root = fileURLToPath(new URL('../', import.meta.url));
const url = 'http://127.0.0.1:5173/';
const identity = `curriculum-dashboard:${root}`;
let server;
let stopping = false;

function openBrowser() {
  const browser = spawn('explorer.exe', [`${url}#overview`], { stdio: 'ignore', windowsHide: true });
  browser.on('error', () => console.log(`Open ${url} in your browser.`));
  browser.unref();
}

async function stop() {
  if (stopping) return;
  stopping = true;
  await server?.close();
  console.log('\nDashboard stopped.');
  process.exit(0);
}

process.once('SIGINT', stop);
process.once('SIGTERM', stop);

try {
  server = await createServer({
    root,
    server: { host: '127.0.0.1', port: 5173, strictPort: true, open: false },
    plugins: [{
      name: 'cs-session-identity',
      configureServer(instance) {
        instance.middlewares.use('/__cs', (req, res) => {
          if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
          res.setHeader('Content-Type', 'text/plain');
          res.end(identity);
        });
      },
    }],
  });
  await server.listen();
  console.log(`\nCurriculum dashboard: ${url}`);
  console.log('Running only in this session. Press Ctrl+C to stop.\n');
  openBrowser();
} catch (error) {
  await server?.close();
  let existing;
  try { existing = await fetch(`${url}__cs`, { signal: AbortSignal.timeout(1500) }).then(response => response.text()); } catch { /* No matching running dashboard. */ }
  if (existing === identity) {
    console.log('The dashboard is already running in another Command Prompt. Opening it.');
    console.log('Stop it with Ctrl+C in the Command Prompt that started it.');
    openBrowser();
  } else {
    console.error(`Could not start the dashboard: ${error.message}`);
    console.error('If port 5173 is in use, stop that server and run cs --web again.');
    process.exitCode = 1;
  }
}
