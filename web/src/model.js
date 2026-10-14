export const dateKey = (date = new Date()) => new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata', year: 'numeric', month: '2-digit', day: '2-digit' }).format(date);
export const pretty = value => String(value ?? '').replaceAll(/[-_]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
export const dayTitle = day => day.topic || (day.status === 'rest' ? 'A little room to recharge' : `Day ${day.global_day} · Topic not recorded`);
export function flatten(state) {
  return Object.entries(state.months).sort(([a], [b]) => Number(a) - Number(b)).flatMap(([month, data]) =>
    Object.entries(data.weeks).flatMap(([week, details]) => Object.entries(details.days).map(([key, day]) => ({ ...day, month, week, key, id: `${month}-${week}-${key}`, folder: `month_${month}/week_${week}/day_${day.day_in_week}` }))).sort((a, b) => a.global_day - b.global_day));
}
export function studyStreak(days) {
  let index = days.findLastIndex(day => day.status === 'done');
  let count = 0;
  for (; index >= 0; index--) {
    if (days[index].status === 'rest') continue;
    if (days[index].status !== 'done') break;
    count++;
  }
  return count;
}
export const needsAttention = day => day.status === 'done' && (day.has_flags || day.confidence === 'struggled') && day.review_history?.times_reviewed === 0;
export function summarize(days, today = dateKey()) {
  const study = days.filter(day => day.status !== 'rest');
  const done = study.filter(day => day.status === 'done');
  const recorded = days.filter(day => day.time_spent_minutes != null);
  return {
    total: study.length, done: done.length, percent: study.length ? Math.round(done.length / study.length * 100) : 0,
    minutes: recorded.reduce((sum, day) => sum + day.time_spent_minutes, 0), recorded: recorded.length,
    due: done.filter(day => day.srs?.next_review_due && day.srs.next_review_due <= today),
    attention: days.filter(needsAttention),
    reviews: days.reduce((sum, day) => sum + (day.review_history?.times_reviewed || 0), 0),
  };
}
export function dayDocuments(documents, day) {
  return documents.filter(doc => doc.path.startsWith(`${day.folder}/`));
}
