import { defineConfig } from 'vite';
import { loadCurriculum, loadExampleCurriculum } from './server/data.js';

function readOnlyCurriculum() {
  function middleware(server) {
    server.middlewares.use('/data/curriculum.json', async (req, res) => {
      if (req.method !== 'GET' && req.method !== 'HEAD') { res.statusCode = 405; res.end('Read-only'); return; }
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.setHeader('Cache-Control', 'no-store');
      try { res.end(JSON.stringify(await loadCurriculum())); }
      catch { res.statusCode = 500; res.end(JSON.stringify({ error: 'Could not read curriculum files. Check that state.json and state.schema.json contain valid JSON.' })); }
    });
  }
  return {
    name: 'read-only-curriculum',
    configureServer: middleware,
    configurePreviewServer: middleware,
    async generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'data/curriculum.json', source: JSON.stringify(await loadExampleCurriculum()) });
    },
  };
}

export default defineConfig({ base: './', plugins: [readOnlyCurriculum()], server: { port: 5173, strictPort: true }, preview: { port: 4173, strictPort: true } });
