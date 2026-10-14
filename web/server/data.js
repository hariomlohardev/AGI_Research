import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const projectRoot = fileURLToPath(new URL('../../', import.meta.url));
const rootDocuments = ['progress.md', 'growth-notes.md', 'struggle-log.md', 'discipline-log.md', 'bonus-watches.md', 'project-history.md', 'STATE_SCHEMA.md'];

// Explicitly read only: no file writes and no request-controlled filesystem paths.
export async function loadCurriculum(root = projectRoot) {
  const [state, schema] = await Promise.all(['state.json', 'state.schema.json'].map(async name => JSON.parse(await readFile(path.join(root, name), 'utf8'))));
  const documents = [];
  async function addDocument(relative) {
    try {
      documents.push({ path: relative.replaceAll('\\', '/'), content: await readFile(path.join(root, relative), 'utf8') });
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  async function walk(relative) {
    let entries;
    try { entries = await readdir(path.join(root, relative), { withFileTypes: true }); }
    catch (error) { if (error.code === 'ENOENT') return; throw error; }
    for (const entry of entries) {
      // Symlinks and code/dependency directories are intentionally not traversed.
      if (entry.isDirectory() && /^(week_\d+|day_\d+|final_project|github_contribution|micro_projects|[a-z0-9]+(?:-[a-z0-9]+)+)$/.test(entry.name)) await walk(path.join(relative, entry.name));
      if (entry.isFile() && entry.name.endsWith('.md')) await addDocument(path.join(relative, entry.name));
    }
  }
  await Promise.all(rootDocuments.map(addDocument));
  for (const month of Object.keys(state.months)) if (/^\d+$/.test(month)) await walk(`month_${month}`);
  documents.sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));
  return { state, schema, documents, loadedAt: new Date().toISOString() };
}
