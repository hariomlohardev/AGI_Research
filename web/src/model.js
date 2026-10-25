export const dateKey = (date = new Date()) => new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata', year: 'numeric', month: '2-digit', day: '2-digit' }).format(date);
export const pretty = value => String(value ?? '').replaceAll(/[-_]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
export const dayTitle = day => day.topic || (day.status === 'rest' ? 'A little room to recharge' : `Day ${day.global_day} · Topic not recorded`);
export function flatten(state) {
  return Object.entries(state.months).sort(([a], [b]) => Number(a) - Number(b)).flatMap(([month, data]) =>
    Object.entries(data.weeks).flatMap(([week, details]) => Object.entries(details.days).map(([key, day]) => ({ ...day, month, week, key, id: `${month}-${week}-${key}`, folder: `month_${month}/${Number(month) >= 3 ? 'docs/days/' : ''}week_${week}/day_${day.day_in_week}` }))).sort((a, b) => a.global_day - b.global_day));
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

export const finalProjectFolder = month => `month_${month}/${Number(month) >= 3 ? 'docs/' : ''}final_project/`;

// View-only preparation: source plans never become recorded progress here.
export function curriculumView(state, plans = []) {
  const months = { ...state.months };
  for (const plan of plans) {
    const key = String(plan.month);
    if (months[key]) continue;
    const weeks = {};
    for (const item of plan.days) {
      const week = String(item.week), day = String(item.day_in_week);
      weeks[week] ??= { done_when: 'Follow the source roadmap for this learning week.', review_status: 'pending', days: {} };
      weeks[week].days[day] = { ...item, topic: item.label, status: item.rest ? 'rest' : 'pending',
        mode: 'guided', tags: [], has_flags: false, confidence: null, difficulty: null,
        time_spent_minutes: null, expected_minutes: null, confusion_notes: [], micro_projects: [],
        github_contribution: null, review_history: { times_reviewed: 0, last_reviewed: null },
        srs: { next_review_due: null }, prepared: true };
    }
    months[key] = { language: 'python', skill_level: 'intermediate', prepared: true, roadmap: plan.roadmap, weeks };
  }
  return { ...state, months };
}

export const resourceKind = resource => resource.kind || 'markdown';
export const resourceLabels = { markdown: 'Notes', notebook: 'Notebooks', python: 'Python code', test: 'Tests' };

export function dayWorkspace(resources, day) {
  const legacy = 'month_' + day.month + '/week_' + day.week + '/day_' + day.day_in_week;
  const suffix = '/week_' + day.week + '/day_' + day.day_in_week + '/';
  const prefixes = Number(day.month) >= 3
    ? [day.folder + '/', 'month_' + day.month + '/notebooks' + suffix,
      'month_' + day.month + '/src/month' + day.month + suffix, 'month_' + day.month + '/tests' + suffix]
    : [legacy + '/'];
  const selected = resources.filter(item => prefixes.some(prefix => item.path.startsWith(prefix)));
  return Object.entries(resourceLabels).map(([kind, label]) => ({ kind, label, items: selected.filter(item => resourceKind(item) === kind) }));
}

export function pairedResource(resources, resource) {
  let match, target;
  if ((match = /^month_(\d+)\/notebooks\/(.+)\.ipynb$/.exec(resource.path))) target = 'month_' + match[1] + '/src/month' + match[1] + '/' + match[2] + '.py';
  else if ((match = /^month_(\d+)\/src\/month\d+\/(.+)\.py$/.exec(resource.path))) target = 'month_' + match[1] + '/notebooks/' + match[2] + '.ipynb';
  return resources.find(item => item.path === target) || null;
}
