import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { flatten, summarize, studyStreak, dateKey, needsAttention } from '../src/model.js';
import { loadCurriculum } from '../server/data.js';

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
test('live adapter reads actual state/schema and exposes only learning markdown', async () => {
  const before = await readFile(new URL('../../state.json', import.meta.url), 'utf8');
  const data = await loadCurriculum();
  assert.deepEqual(data.state, JSON.parse(before));
  assert.ok(data.schema.properties.months);
  assert.ok(data.documents.some(doc => doc.path === 'progress.md'));
  assert.ok(data.documents.every(doc => doc.path.endsWith('.md') && !doc.path.includes('/code/') && !doc.path.includes('.claude/')));
  assert.equal(await readFile(new URL('../../state.json', import.meta.url), 'utf8'), before);
});
