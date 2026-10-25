import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { loadCurriculum, loadExampleCurriculum } from '../server/data.js';
import { curriculumView, flatten, dayWorkspace, pairedResource } from '../src/model.js';

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'curriculum-workspace-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const state = JSON.parse(await readFile(new URL('../../tests/fixtures/valid_state.json', import.meta.url), 'utf8'));
  async function put(name, content) {
    await mkdir(path.dirname(path.join(root, name)), { recursive: true });
    await writeFile(path.join(root, name), typeof content === 'string' ? content : JSON.stringify(content));
  }
  await put('state.json', state);
  await put('state.schema.json', await readFile(new URL('../../state.schema.json', import.meta.url), 'utf8'));
  const plan = { month: 3, layout_version: 2, days: [
    { week: 1, day_in_week: 8, global_day: 8, label: 'Independent MNIST classifier', rest: false },
    { week: 2, day_in_week: 7, global_day: 15, label: 'Protected rest', rest: true },
  ] };
  await put('month_3/curriculum.json', plan);
  return { root, state, plan, put };
}

test('prepared Month 3 is discoverable without starting it or editing state', async t => {
  const { root, state, put } = await fixture(t);
  await put('month_3/docs/roadmap.md', '# Approved plan');
  const before = await readFile(path.join(root, 'state.json'), 'utf8');
  const data = await loadCurriculum(root);
  assert.deepEqual(data.state, state);
  assert.equal(data.plans[0].month, 3);
  assert.ok(data.documents.some(item => item.path === 'month_3/docs/roadmap.md'));
  const view = curriculumView(data.state, data.plans);
  assert.equal(view.current_month, state.current_month);
  assert.equal(view.months['3'].prepared, true);
  const rows = flatten(view).filter(day => day.month === '3');
  assert.equal(rows[0].day_in_week, 8);
  assert.deepEqual(rows.map(day => day.status), ['pending', 'rest']);
  assert.ok(!rows.some(day => day.status === 'current' || day.status === 'done'));
  assert.deepEqual(data.state, state);
  assert.equal(await readFile(path.join(root, 'state.json'), 'utf8'), before);
});

test('actual recorded progress always takes precedence over a prepared plan', async t => {
  const { state, plan } = await fixture(t);
  state.months['3'] = { weeks: { 1: { days: { 8: { status: 'done', day_in_week: 8, global_day: 8 } } } } };
  const before = JSON.stringify(state);
  const view = curriculumView(state, [plan]);
  assert.equal(view.months['3'], state.months['3']);
  assert.equal(JSON.stringify(state), before);
  assert.equal(view.months['3'].prepared, undefined);
});

test('separate notebook, Python and tests are grouped with notes and paired correctly', async t => {
  const { root, put } = await fixture(t);
  const note = 'month_3/docs/days/week_1/day_8/learn.md';
  const notebook = 'month_3/notebooks/week_1/day_8/mnist.ipynb';
  const source = 'month_3/src/month3/week_1/day_8/mnist.py';
  const tests = 'month_3/tests/week_1/day_8/test_mnist.py';
  await put(note, '# MNIST');
  await put(notebook, { nbformat: 4, metadata: { widgets: 'PRIVATE_WIDGET' }, cells: [
    { cell_type: 'markdown', source: ['# Mechanism\n', 'Predict first.'] },
    { cell_type: 'code', source: ['def model():\n', '    pass'], outputs: [{ text: 'PRIVATE_STDOUT' }], execution_count: 7 },
  ] });
  await put(source, 'def model():\n    pass\n');
  await put(tests, 'def test_shape():\n    assert True\n');
  await put('month_3/notebooks/week_1/day_80/other.ipynb', { nbformat: 4, cells: [] });
  await put('month_3/src/month3/data/leak.py', 'PRIVATE_DATA');
  await put('month_3/checkpoints/leak.py', 'PRIVATE_WEIGHTS');
  await put('.claude/private.py', 'PRIVATE_ASSISTANT');
  const data = await loadCurriculum(root);
  const view = curriculumView(data.state, data.plans);
  const day = flatten(view).find(item => item.month === '3' && item.day_in_week === 8);
  const resources = [...data.documents, ...data.files];
  const groups = dayWorkspace(resources, day);
  assert.deepEqual(groups.map(group => group.items.map(item => item.path)), [[note], [notebook], [source], [tests]]);
  assert.equal(pairedResource(resources, data.files.find(item => item.path === notebook)).path, source);
  assert.equal(pairedResource(resources, data.files.find(item => item.path === source)).path, notebook);
  assert.deepEqual(data.files.find(item => item.path === notebook).cells[1], { cell_type: 'code', source: 'def model():\n    pass' });
  assert.ok(!JSON.stringify(data).includes('PRIVATE_'));
});

test('legacy code and tests remain available without moving old coursework', async t => {
  const { root, put } = await fixture(t);
  await put('month_1/week_1/day_1/code/example.py', 'def example():\n    return 1');
  await put('month_1/week_1/day_1/code/tests/test_example.py', 'def test_example():\n    pass');
  const data = await loadCurriculum(root);
  const day = { month: '1', week: '1', day_in_week: 1, folder: 'month_1/week_1/day_1' };
  const groups = dayWorkspace(data.files, day);
  assert.equal(groups.find(g => g.kind === 'python').items.length, 1);
  assert.equal(groups.find(g => g.kind === 'test').items.length, 1);
});

test('a partially saved notebook and an invalid plan do not break other materials', async t => {
  const { root, put } = await fixture(t);
  await put('month_4/curriculum.json', { month: 4, layout_version: 2, days: [null] });
  await put('month_3/notebooks/week_1/day_8/broken.ipynb', '{');
  await put('month_3/notebooks/week_1/day_8/null_cell.ipynb', { nbformat: 4, cells: [null] });
  await put('month_3/docs/roadmap.md', '# Still readable');
  const data = await loadCurriculum(root);
  assert.equal(data.plans.length, 1);
  assert.equal(data.files.filter(item => item.error).length, 2);
  assert.ok(data.documents.some(item => item.content === '# Still readable'));
});

test('public build payload never includes local plans, source files or notebook cells', async t => {
  const { root, state, put } = await fixture(t);
  await put('tests/fixtures/valid_state.json', state);
  await put('month_3/notebooks/private.ipynb', { nbformat: 4, cells: [{ cell_type: 'code', source: 'PRIVATE_LOCAL_CODE' }] });
  const data = await loadExampleCurriculum(root);
  assert.deepEqual(data.files, []);
  assert.deepEqual(data.plans, []);
  assert.deepEqual(data.documents, []);
  assert.equal(data.stateSource, 'example');
  assert.ok(!JSON.stringify(data).includes('PRIVATE_LOCAL_CODE'));
});
