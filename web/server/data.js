import { readFile, readdir, lstat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const projectRoot = fileURLToPath(new URL('../../', import.meta.url));
const rootDocuments = ['progress.md', 'growth-notes.md', 'struggle-log.md', 'discipline-log.md', 'bonus-watches.md', 'project-history.md', 'STATE_SCHEMA.md'];
const MAX_PREVIEW_BYTES = 1024 * 1024;
const excludedFolders = new Set(['__pycache__', 'node_modules', 'data', 'checkpoints', 'runs', 'artifacts', 'scratch', 'results']);

async function readOptional(root, relative) {
  try {
    const info = await lstat(path.join(root, relative));
    if (!info.isFile() || info.size > MAX_PREVIEW_BYTES) return null;
    return await readFile(path.join(root, relative), 'utf8');
  } catch (error) { if (error.code !== 'ENOENT') throw error; return null; }
}

function validPlan(plan, month) {
  const identities = new Set(), globalDays = new Set();
  return plan?.month === Number(month) && plan.layout_version === 2 && Array.isArray(plan.days) && plan.days.length > 0 && plan.days.every(day => {
    if (!day || typeof day !== 'object') return false;
    const key = day.week + '-' + day.day_in_week;
    if (![day.week, day.day_in_week, day.global_day].every(n => Number.isInteger(n) && n > 0) || typeof day.label !== 'string' || typeof day.rest !== 'boolean' || identities.has(key) || globalDays.has(day.global_day)) return false;
    identities.add(key); globalDays.add(day.global_day); return true;
  });
}

// Local read-only data; no request-controlled paths or notebook execution.
export async function loadCurriculum(root = projectRoot) {
  const [state, schema] = await Promise.all(['state.json', 'state.schema.json'].map(async name => JSON.parse(await readFile(path.join(root, name), 'utf8'))));
  const documents = [], files = [], plans = [];
  const directories = await readdir(root, { withFileTypes: true });
  const monthNames = new Set(Object.keys(state.months).filter(month => /^[1-9]\d*$/.test(month)));
  for (const entry of directories) {
    const match = /^month_([1-9]\d*)$/.exec(entry.name);
    if (!entry.isDirectory() || !match) continue;
    const raw = await readOptional(root, entry.name + '/curriculum.json');
    if (!raw) continue;
    try {
      const plan = JSON.parse(raw);
      if (validPlan(plan, match[1])) {
        plans.push({ month: plan.month, layout_version: plan.layout_version, days: plan.days, roadmap: entry.name + '/docs/roadmap.md' });
        monthNames.add(match[1]);
      }
    } catch (error) { if (!(error instanceof SyntaxError)) throw error; }
  }
  async function addDocument(relative) {
    const content = await readOptional(root, relative);
    if (content !== null) documents.push({ path: relative.split(path.sep).join('/'), content, kind: 'markdown' });
  }
  async function entries(relative) {
    try {
      if (!(await lstat(path.join(root, relative))).isDirectory()) return [];
      return await readdir(path.join(root, relative), { withFileTypes: true });
    } catch (error) { if (error.code !== 'ENOENT') throw error; return []; }
  }
  async function walkDocs(relative) {
    for (const entry of await entries(relative)) {
      const allowed = /^(docs|days|reports|week_\d+|day_\d+|final_project|github_contribution|micro_projects)$/.test(entry.name);
      const micro = path.basename(relative) === 'micro_projects' && /^[a-z0-9]+(?:[-_][a-z0-9]+)*$/.test(entry.name);
      if (entry.isDirectory() && (allowed || micro)) await walkDocs(path.join(relative, entry.name));
      if (entry.isFile() && entry.name.endsWith('.md')) await addDocument(path.join(relative, entry.name));
    }
  }
  async function walkFiles(relative) {
    for (const entry of await entries(relative)) {
      const next = path.join(relative, entry.name);
      if (entry.isDirectory() && /^[A-Za-z_][\w-]*$/.test(entry.name) && !excludedFolders.has(entry.name)) await walkFiles(next);
      if (!entry.isFile() || !/\.(py|ipynb)$/.test(entry.name)) continue;
      const filePath = next.split(path.sep).join('/');
      const content = await readOptional(root, next);
      const kind = entry.name.endsWith('.ipynb') ? 'notebook' : /\/tests\//.test(filePath) ? 'test' : 'python';
      if (content === null) { files.push({ path: filePath, kind, content: '', error: 'This file is too large for a preview. Open it in your editor.' }); continue; }
      if (kind !== 'notebook') { files.push({ path: filePath, kind, content }); continue; }
      try {
        const notebook = JSON.parse(content);
        if (notebook.nbformat !== 4 || !Array.isArray(notebook.cells) || notebook.cells.some(cell => !cell || typeof cell !== 'object')) throw new SyntaxError('Unsupported notebook');
        const cells = notebook.cells.filter(cell => ['markdown', 'code', 'raw'].includes(cell.cell_type)).map(cell => ({
          cell_type: cell.cell_type,
          source: Array.isArray(cell.source) ? cell.source.join('') : typeof cell.source === 'string' ? cell.source : '',
        }));
        files.push({ path: filePath, kind, cells, content: cells.map(cell => cell.source).join('\n\n') });
      } catch (error) {
        if (!(error instanceof SyntaxError)) throw error;
        files.push({ path: filePath, kind, content: '', error: 'This notebook could not be read. Finish saving it in your editor, then refresh.' });
      }
    }
  }
  await Promise.all(rootDocuments.map(addDocument));
  for (const month of monthNames) {
    const base = 'month_' + month;
    await walkDocs(base);
    if (Number(month) >= 3) {
      await walkFiles(base + '/notebooks');
      await walkFiles(base + '/src');
      await walkFiles(base + '/tests');
    } else {
      for (const week of await entries(base)) if (week.isDirectory() && /^week_\d+$/.test(week.name)) {
        for (const day of await entries(base + '/' + week.name)) if (day.isDirectory() && /^day_\d+$/.test(day.name)) {
          const dayRoot = base + '/' + week.name + '/' + day.name;
          await walkFiles(dayRoot + '/code');
          for (const micro of await entries(dayRoot + '/micro_projects')) if (micro.isDirectory() && /^[a-z0-9]+(?:[-_][a-z0-9]+)*$/.test(micro.name)) await walkFiles(dayRoot + '/micro_projects/' + micro.name + '/code');
        }
      }
    }
  }
  for (const list of [documents, files]) list.sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));
  plans.sort((a, b) => a.month - b.month);
  return { state, schema, documents, files, plans, stateSource: 'local', loadedAt: new Date().toISOString() };
}

// Public builds use an example only; local plans/code/progress are never bundled.
export async function loadExampleCurriculum(root = projectRoot) {
  const [state, schema] = await Promise.all(['tests/fixtures/valid_state.json', 'state.schema.json'].map(async name => JSON.parse(await readFile(path.join(root, name), 'utf8'))));
  return { state, schema, documents: [], files: [], plans: [], stateSource: 'example', loadedAt: new Date().toISOString() };
}
