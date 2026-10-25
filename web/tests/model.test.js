import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { flatten, summarize, studyStreak, dateKey, needsAttention, dayDocuments, finalProjectFolder } from '../src/model.js';
import { loadCurriculum, loadExampleCurriculum } from '../server/data.js';

const day = (status, extra = {}) => ({ status, time_spent_minutes: null, review_history: { times_reviewed: 0 }, ...extra });
test('streak spans months, skips rest and current days, and stops at a real gap', () => {
  assert.equal(studyStreak([day('done'), day('rest'), day('done'), day('current'), day('pending')]), 2);
  assert.equal(studyStreak([day('done'), day('pending'), day('done'), day('rest'), day('done')]), 2);
  assert.equal(studyStreak([day('current')]), 0);
});
test('rest does not inflate completion; missing time is not a recorded zero', () => {
  const result = summarize([day('done'), day('rest'), day('current', { time_spent_minutes: 0 }), day('done', { time_spent_minutes: 90 })]);
  assert.equal(result.total, 3); assert.equal(result.done, 2); assert.equal(result.percent, 67);
  assert.equal(result.recorded, 2); assert.equal(result.minutes, 90);
  assert.equal(summarize([]).percent, 0);
});
test('due dates and struggle gate use the actual schema rules', () => {
  const days = [day('done', { confidence: 'struggled' }), day('done', { has_flags: true, review_history: { times_reviewed: 1 } }), day('done', { srs: { next_review_due: '2026-10-14' } }), day('done', { srs: { next_review_due: '2026-10-15' } }), day('current', { has_flags: true })];
  assert.equal(summarize(days, '2026-10-14').due.length, 1);
  assert.equal(summarize(days).attention.length, 1);
  assert.equal(needsAttention(days[1]), false);
});
test('dates use India timezone, independent of server timezone', () => {
  assert.equal(dateKey(new Date('2026-10-13T20:00:00Z')), '2026-10-14');
});
test('flattening keeps month ordering and distinguishes month-local day numbers', () => {
  const month = days => ({ weeks: { 1: { days } } });
  const rows = flatten({ months: { 2: month({ 1: { global_day: 1, day_in_week: 1 } }), 1: month({ 2: { global_day: 2, day_in_week: 2 }, 1: { global_day: 1, day_in_week: 1 } }) } });
  assert.deepEqual(rows.map(row => row.id), ['1-1-1', '1-1-2', '2-1-1']);
  assert.equal(rows[2].folder, 'month_2/week_1/day_1');
});
test('adapter supports both layouts without requiring personal state in a clone', async t => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'curriculum-layout-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const fixture = JSON.parse(await readFile(new URL('../../tests/fixtures/valid_state.json', import.meta.url), 'utf8'));
  fixture.months['3'] = { weeks: { '1': { days: { '8': { global_day: 8, day_in_week: 8, status: 'current' } } } } };
  const before = JSON.stringify(fixture);
  await writeFile(path.join(root, 'state.json'), before);
  await writeFile(path.join(root, 'state.schema.json'), await readFile(new URL('../../state.schema.json', import.meta.url)));
  const docs = ['progress.md', 'month_1/week_1/day_1/learn.md', 'month_3/docs/days/week_1/day_8/learn.md', 'month_3/docs/final_project/structure.md'];
  for (const name of [...docs, 'month_3/src/month3/private.md', '.claude/private.md']) {
    await mkdir(path.dirname(path.join(root, name)), { recursive: true });
    await writeFile(path.join(root, name), '# fixture');
  }
  const data = await loadCurriculum(root);
  assert.deepEqual(data.state, fixture);
  assert.ok(data.schema.properties.months);
  assert.deepEqual(data.documents.map(doc => doc.path).sort(), docs.sort());
  const current = flatten(fixture).find(day => day.month === '3');
  assert.equal(current.folder, 'month_3/docs/days/week_1/day_8');
  assert.equal(dayDocuments(data.documents, current).length, 1);
  assert.equal(finalProjectFolder('3'), 'month_3/docs/final_project/');
  assert.equal(finalProjectFolder('2'), 'month_2/final_project/');
  assert.equal(await readFile(path.join(root, 'state.json'), 'utf8'), before);
});

test('public bundle data contains only the sanitized fixture and no private documents', async () => {
  const example = await loadExampleCurriculum();
  const fixture = JSON.parse(await readFile(new URL('../../tests/fixtures/valid_state.json', import.meta.url), 'utf8'));
  assert.deepEqual(example.state, fixture);
  assert.equal(example.stateSource, 'example');
  assert.deepEqual(example.documents, []);
});
