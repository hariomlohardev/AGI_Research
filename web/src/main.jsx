import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowUpRight, ArrowRight, BookOpen, Check, ChevronRight, Circle, Clock3, Code2, FileText, Flame, Flower2, FolderOpen, GitPullRequest, LayoutDashboard, Leaf, Library, LoaderCircle, Moon, RefreshCw, Search, ShieldCheck, Sparkles, Sprout, Sun, Target, Trophy, X } from 'lucide-react';
import { dateKey, dayDocuments, dayTitle, flatten, needsAttention, pretty, studyStreak, summarize } from './model.js';
import './styles.css';
import DocumentPage from './DocumentPage.jsx';

const navigation = [
  ['Overview', LayoutDashboard], ['Curriculum', BookOpen], ['Reviews', RefreshCw], ['Projects', Code2], ['Library', Library],
];
const badgeIcons = { 'first-day-done': Sprout, 'five-day-streak': Flame, comeback: Sparkles, 'first-micro-project': Code2, 'flawless-week': Trophy, 'month-complete': Trophy, 'first-oss-pr': GitPullRequest };
const docName = path => pretty(path.split('/').at(-1).replace('.md', ''));
const safeLink = value => /^https?:\/\//i.test(value || '') ? value : undefined;
const documentFromHash = () => new URLSearchParams(location.hash.slice(1)).get('document');

function Status({ status }) { return <span className={`status ${status}`}>{status === 'done' && <Check size={12} />}{status === 'current' && <span className="live-dot" />}{status === 'current' ? 'In progress' : pretty(status)}</span>; }
function Empty({ icon: Icon = Leaf, title, children }) { return <div className="empty"><Icon size={32} strokeWidth={1.5} /><h3>{title}</h3><p>{children}</p></div>; }
function Command({ value }) {
  const [copied, setCopied] = useState(false);
  return <button className="command" onClick={async () => { try { await navigator.clipboard.writeText(value); setCopied(true); setTimeout(() => setCopied(false), 1800); } catch { setCopied(false); } }} title={`Copy ${value} to use in your curriculum assistant`}>{copied ? <Check size={14} /> : <Code2 size={14} />}{copied ? 'Copied' : value}</button>;
}
function App() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(() => navigation.some(([name]) => name.toLowerCase() === location.hash.slice(1)) ? pretty(location.hash.slice(1)) : 'Overview');
  const [month, setMonth] = useState('');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [selectedDay, setSelectedDay] = useState(null);
  const [selectedDoc, updateSelectedDoc] = useState(documentFromHash);
  const [dark, setDark] = useState(() => { try { return localStorage.getItem('bloom-theme') === 'dark'; } catch { return false; } });
  const dialog = useRef(null);
  const focusBeforeDialog = useRef(null);
  const request = useRef(null);

  async function refresh() {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}data/curriculum.json`, { cache: 'no-store', signal: controller.signal });
      if (!response.ok) throw new Error('Unable to read your curriculum. Check your local server and JSON files, then retry.');
      const next = await response.json();
      if (!next.state?.months || !Array.isArray(next.documents)) throw new Error('The curriculum data has an unexpected format.');
      setPayload(next); setMonth(old => next.state.months[old] ? old : String(next.state.current_month)); setError('');
    } catch (e) { if (e.name !== 'AbortError') setError(e.message); }
    finally { if (request.current === controller) setLoading(false); }
  }
  useEffect(() => { refresh(); const timer = setInterval(refresh, 30000); return () => { clearInterval(timer); request.current?.abort(); }; }, []);
  useEffect(() => { document.documentElement.dataset.theme = dark ? 'dark' : 'light'; try { localStorage.setItem('bloom-theme', dark ? 'dark' : 'light'); } catch { /* Theme still works without storage. */ } }, [dark]);
  useEffect(() => {
    const handler = () => {
      updateSelectedDoc(documentFromHash());
      const value = pretty(location.hash.slice(1));
      if (navigation.some(([name]) => name === value)) setPage(value);
    };
    addEventListener('hashchange', handler);
    addEventListener('popstate', handler);
    return () => { removeEventListener('hashchange', handler); removeEventListener('popstate', handler); };
  }, []);
  const isDialogOpen = Boolean(selectedDay && !selectedDoc);
  useEffect(() => {
    if (isDialogOpen && !dialog.current?.open) { focusBeforeDialog.current = document.activeElement; dialog.current?.showModal(); }
    if (!isDialogOpen && dialog.current?.open) { dialog.current.close(); focusBeforeDialog.current?.focus(); }
    document.body.style.overflow = isDialogOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [isDialogOpen]);
  function go(name) { setPage(name); setQuery(''); setFilter('all'); location.hash = name.toLowerCase(); window.scrollTo({ top: 0, behavior: 'smooth' }); }
  function closeDialog() { setSelectedDay(null); setSelectedDoc(null); }
  function setSelectedDoc(path) {
    if (path && path !== selectedDoc) history.pushState({ bloomDocument: true }, '', `#document=${encodeURIComponent(path)}`);
    updateSelectedDoc(path);
  }
  function backFromDocument() {
    if (history.state?.bloomDocument) history.back();
    else { updateSelectedDoc(null); go('Library'); }
  }

  if (!payload) return <div className="boot"><Flower2 size={50} /><h1>Room to grow.</h1><p>{error || 'Gathering your learning journey…'}</p>{error && <button className="primary" onClick={refresh}>Try again</button>}</div>;
  const { state, documents, schema } = payload;
  const all = flatten(state);
  const days = all.filter(day => day.month === month);
  const current = all.find(day => Number(day.month) === state.current_month && day.global_day === state.current_day);
  const stats = summarize(days);
  const overall = summarize(all);
  const monthData = state.months[month];
  const currentStats = summarize(all.filter(day => Number(day.month) === state.current_month));
  const reviewDays = days.filter(day => day.status === 'done' && (day.confidence === 'shaky' || needsAttention(day) || stats.due.some(due => due.id === day.id))).sort((a, b) => (a.srs?.next_review_due || '9999').localeCompare(b.srs?.next_review_due || '9999'));
  const projects = days.flatMap(day => (day.micro_projects || []).map(project => ({ ...project, day })));
  const contributions = days.filter(day => day.github_contribution);
  const selected = selectedDay && all.find(day => day.id === selectedDay);
  const doc = selectedDoc && documents.find(item => item.path === selectedDoc);
  const match = day => `${dayTitle(day)} ${day.tags.join(' ')} week ${day.week} day ${day.global_day}`.toLowerCase().includes(query.toLowerCase());
  const searchDays = days.filter(match).filter(day => filter === 'all' || day.status === filter);

  if (selectedDoc) return <DocumentPage key={selectedDoc} document={doc} path={selectedDoc} documents={documents} onBack={backFromDocument} onOpen={setSelectedDoc} dark={dark} onToggleTheme={() => setDark(!dark)} error={error} />;

  const DayRow = ({ day, compact = false }) => <button className={`day-row ${compact ? 'compact' : ''}`} onClick={() => { setSelectedDoc(null); setSelectedDay(day.id); }}>
    <span className={`day-icon ${day.status}`}>{day.status === 'done' ? <Check size={18} /> : day.status === 'rest' ? <Moon size={18} /> : day.status === 'current' ? <ArrowRight size={18} /> : <span>{day.global_day}</span>}</span>
    <span className="day-copy"><small>WEEK {day.week} · DAY {day.day_in_week}</small><strong>{dayTitle(day)}</strong>{!compact && <span>{day.tags.length ? day.tags.join(' · ') : pretty(day.mode)}{day.confidence && ` · ${pretty(day.confidence)} confidence`}</span>}</span>
    <Status status={day.status} /><ChevronRight size={18} className="row-arrow" />
  </button>;

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#overview" onClick={event => { event.preventDefault(); go('Overview'); }}><span className="brand-mark"><Flower2 size={27} /></span>bloom<span className="brand-period">.</span></a>
      <div className="space-label">YOUR LEARNING SPACE</div>
      <nav aria-label="Main navigation">{navigation.map(([name, Icon]) => <button key={name} aria-label={name} title={name} className={`nav-item ${page === name ? 'active' : ''}`} onClick={() => go(name)} aria-current={page === name ? 'page' : undefined}><Icon size={21} /><span>{name}</span>{name === 'Reviews' && stats.due.length > 0 && <span className="nav-count">{stats.due.length}</span>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="grow-note"><Sprout size={28} /><strong>A little every day.<br />A lot over time.</strong><p>This is what progress looks like.</p></div><div className="profile"><span className="avatar">Y</span><div><strong>Your journey</strong><small>{pretty(monthData.skill_level)} learner</small></div><Leaf size={18} /></div></div>
    </aside>

    <main>
      <header className="topbar"><div className="breadcrumb">My workspace <ChevronRight size={14} /><strong>{page}</strong></div><div className="top-actions"><span className="read-only"><ShieldCheck size={15} /> Read-only</span><button className="icon-button" onClick={refresh} aria-label="Refresh curriculum" title="Refresh curriculum"><RefreshCw size={18} className={loading ? 'spin' : ''} /></button><button className="icon-button" onClick={() => setDark(!dark)} aria-label={dark ? 'Use light theme' : 'Use dark theme'}>{dark ? <Sun size={19} /> : <Moon size={19} />}</button><span className="avatar small">Y</span></div></header>
      <div className="page-content">
        {error && <div className="error" role="alert">{error} Showing the last successfully loaded data.</div>}
        <section className="page-heading"><div><div className="eyebrow"><span className="live-dot" /> YOUR NEXT CHAPTER STARTS HERE</div><h1>{page === 'Overview' ? 'Look at you grow.' : page === 'Curriculum' ? 'One day at a time.' : page === 'Reviews' ? 'Make it stick.' : page === 'Projects' ? 'Turn knowing into doing.' : 'Your little library.'}</h1><p>{page === 'Overview' ? 'Every concept, every small win. It all adds up.' : page === 'Curriculum' ? 'Your learning path, with room for the work and the rest.' : page === 'Reviews' ? 'Revisit the ideas that deserve a little more attention.' : page === 'Projects' ? 'Real practice. Useful things. A record of what you’re building.' : 'All your learning materials and reflections, together.'}</p></div><label className="month-picker"><span className="sr-only">Select month</span><BookOpen size={17} /><select value={month} onChange={event => setMonth(event.target.value)}>{Object.keys(state.months).map(key => <option key={key} value={key}>Month {key}</option>)}</select></label></section>

        {page === 'Overview' && <>
          <div className="overview-top"><section className="hero"><div className="hero-copy"><div className="hero-label"><Sparkles size={15} /> KEEP YOUR MOMENTUM</div><h2>Small steps.<br /><span>Real progress.</span></h2><p>You’ve completed {stats.done} study days this month.<br />{stats.total - stats.done ? 'Your next idea is waiting for you.' : 'Take a moment to appreciate how far you’ve come.'}</p><button className="primary" onClick={() => current && setSelectedDay(current.id)} disabled={!current}>Continue learning <ArrowUpRight size={19} /></button></div><div className="progress-flower" aria-label={`${stats.percent}% of month complete`}><svg viewBox="0 0 240 240" className="flower-shape" aria-hidden="true"><path d="M120 14C145-8 167 12 169 33C201 20 220 44 208 69C240 78 244 109 221 124C245 149 225 177 201 177C207 207 180 226 158 210C145 240 115 242 101 216C74 238 48 220 50 194C17 202 2 175 18 153C-8 134 0 106 24 97C1 70 20 45 46 49C43 18 74 4 94 25C102 13 111 12 120 14Z" /></svg><div className="ring" style={{ '--progress': `${stats.percent}%` }}><div><strong>{stats.percent}<span>%</span></strong><small>MONTH COMPLETE</small></div></div><span className="flower-spark spark-one">✦</span><span className="flower-spark spark-two">✧</span></div><div className="hero-bottom"><span><span className="live-dot" /> Growing at your own pace</span><span>{stats.total - stats.done} study days to go <ArrowRight size={13} /></span></div></section>
          <section className="current-card"><div className="section-kicker"><span className="live-dot" /> UP NEXT <span className="pill">{current?.mode || 'guided'}</span></div>{current ? <><p className="current-location">MONTH {current.month} <span>/</span> WEEK {current.week} <span>/</span> DAY {current.day_in_week}</p><h2>{current.topic || 'Your next learning day'}</h2><p>{current.topic ? 'Pick up where you left off and keep building your understanding.' : 'Your current day is ready to explore. Its topic hasn’t been recorded in your curriculum yet.'}</p><div className="current-meta"><Code2 size={16} />{pretty(state.months[current.month].language)}<span>·</span>{dayDocuments(documents, current).length} materials available</div><button className="outline-button" onClick={() => setSelectedDay(current.id)}>Open day {current.global_day}<ArrowRight size={17} /></button></> : <Empty title="A fresh chapter">No current day is recorded.</Empty>}</section></div>
          <section className="stats-grid" aria-label="Learning statistics">
            {[{ icon: Check, value: stats.done, suffix: `/ ${stats.total}`, label: 'Study days completed', note: 'A little further every day', color: 'lime' }, { icon: Flame, value: studyStreak(all), suffix: 'days', label: 'Study-day streak', note: 'Across all recorded months', color: 'peach' }, { icon: Clock3, value: stats.recorded ? (stats.minutes / 60).toFixed(1) : '—', suffix: stats.recorded ? 'hrs' : '', label: 'Recorded study time', note: `${stats.recorded} days with time logged`, color: 'lavender' }, { icon: Trophy, value: state.badges_unlocked.length, suffix: 'earned', label: 'Milestones unlocked', note: 'Your wins, worth keeping', color: 'blue' }].map(({ icon: Icon, ...item }) => <article className="stat-card" key={item.label}><span className={`stat-icon ${item.color}`}><Icon size={19} /></span><div className="stat-number">{item.value}<span>{item.suffix}</span></div><strong>{item.label}</strong><small>{item.note}</small></article>)}
          </section>
          <div className="overview-bottom"><section className="panel journey-panel"><div className="section-heading"><div><h2>Your month, unfolding</h2><p>One square. One step forward.</p></div><button className="text-button" onClick={() => go('Curriculum')}>View curriculum <ArrowUpRight size={16} /></button></div><div className="week-grid">{Object.entries(monthData.weeks).map(([week, info]) => { const list = days.filter(day => day.week === week); return <div className="week-column" key={week}><div className="week-label">WEEK {week}<span>{list.filter(day => day.status === 'done').length}/{list.filter(day => day.status !== 'rest').length}</span></div><div className="day-squares">{list.map(day => <button key={day.id} className={`day-square ${day.status}`} title={`${dayTitle(day)} · ${pretty(day.status)}`} aria-label={`Week ${week}, day ${day.day_in_week}, ${day.status}`} onClick={() => setSelectedDay(day.id)}>{day.status === 'done' ? <Check size={17} /> : day.status === 'rest' ? <Moon size={15} /> : day.day_in_week}</button>)}</div><div className={`week-review ${info.review_status === 'done' ? 'reviewed' : ''}`}><span className="live-dot" />{info.review_status === 'done' ? 'Week reviewed' : 'Review pending'}</div></div>; })}</div><div className="legend"><span><i className="done" />Completed</span><span><i className="current" />In progress</span><span><i className="pending" />Coming up</span><span><i className="rest" />Rest day</span></div></section>
          <section className="panel review-card"><div className="section-heading"><h2>A little reflection</h2><span className="round-icon"><RefreshCw size={17} /></span></div><div className="review-number">{stats.due.length}<span>scheduled reviews due</span></div><p>{stats.attention.length ? `${stats.attention.length} completed days have flags or struggles without a review yet.` : 'Revisiting what you’ve learned helps it stay with you.'}</p><button className="text-button" onClick={() => go('Reviews')}>Make room for review <ArrowRight size={16} /></button></section></div>
          <section className="milestones"><div><span className="eyebrow">LITTLE WINS, BIG FEELINGS</span><h2>Collected along the way</h2></div><div className="badge-list">{state.badges_unlocked.length ? state.badges_unlocked.map(badge => { const Icon = badgeIcons[badge.id] || Trophy; return <div className="badge" key={badge.id} title={`Unlocked ${badge.date}`}><span><Icon size={22} /></span><div><strong>{pretty(badge.id)}</strong><small>{badge.date}</small></div></div>; }) : <p>Your first milestone is still ahead.</p>}</div></section>
        </>}

        {page === 'Curriculum' && <><div className="toolbar"><label className="search"><Search size={18} /><input aria-label="Search curriculum" placeholder="Find a topic, tag, or day…" value={query} onChange={event => setQuery(event.target.value)} /></label><div className="segments" aria-label="Filter days">{['all', 'done', 'current', 'pending', 'rest'].map(status => <button key={status} aria-pressed={filter === status} className={filter === status ? 'selected' : ''} onClick={() => setFilter(status)}>{status === 'all' ? 'All days' : pretty(status)}</button>)}</div></div>{searchDays.length ? Object.entries(monthData.weeks).map(([week, data]) => { const list = searchDays.filter(day => day.week === week); return list.length > 0 && <section className="panel curriculum-week" key={week}><div className="section-heading"><h2>Week {week}</h2><Status status={data.review_status === 'done' ? 'done' : 'pending'} /></div><p className="week-goal">{data.done_when}</p>{list.map(day => <DayRow day={day} key={day.id} />)}</section>; }) : <Empty icon={Search} title="No days found">Try another topic or filter.</Empty>}</>}

        {page === 'Reviews' && <><div className={`notice ${currentStats.attention.length >= 3 ? 'attention' : ''}`}><ShieldCheck size={22} /><div><strong>{currentStats.attention.length >= 3 ? 'Time to strengthen your foundation' : 'Understanding is the goal'}</strong><p>{currentStats.attention.length >= 3 ? `${currentStats.attention.length} days in your current month have unresolved struggle. Complete a spaced-review pass to reduce the backlog before generating another day.` : 'Scheduled reviews, flagged questions, and shaky concepts are collected here.'}</p></div><Command value="/spaced-review" /></div><div className="mini-stats"><span><strong>{stats.due.length}</strong>Scheduled & due</span><span><strong>{stats.attention.length}</strong>Unresolved struggle</span><span><strong>{stats.reviews}</strong>Review sessions</span></div><section className="panel"><div className="section-heading"><h2>Your review queue</h2><span className="muted">Month {month}</span></div>{reviewDays.length ? reviewDays.map(day => <div className="review-item" key={day.id}><DayRow day={day} compact /><div className="review-tags">{day.has_flags && <span className="tag warning">Flagged questions</span>}{day.confidence && <span className="tag">{pretty(day.confidence)} confidence</span>}<span className="tag">{day.review_history.times_reviewed} reviews</span>{day.srs.next_review_due && <span className={`tag ${day.srs.next_review_due <= dateKey() ? 'warning' : ''}`}>Due {day.srs.next_review_due}</span>}</div></div>) : <Empty title="A little breathing room">No due reviews or unresolved struggles in this month.</Empty>}</section></>}

        {page === 'Projects' && <><section className="project-hero"><div><span className="eyebrow">BUILD SOMETHING THAT’S YOURS</span><h2>Knowledge, put to work.</h2><p>Micro-projects, a final project, and open-source contributions.</p></div><Code2 size={76} strokeWidth={1} /></section><div className="project-grid"><section className="panel"><div className="section-heading"><h2>Final project</h2><FolderOpen size={21} /></div>{documents.filter(item => item.path.startsWith(`month_${month}/final_project/`)).map(item => <button className="document-row" key={item.path} onClick={() => setSelectedDoc(item.path)}><FileText size={20} /><span>{docName(item.path)}</span><ArrowUpRight size={17} /></button>)}{!documents.some(item => item.path.startsWith(`month_${month}/final_project/`)) && <Empty title="The bigger picture is ahead">Final-project materials will appear here when available.</Empty>}</section><section className="panel"><div className="section-heading"><h2>Micro-projects</h2><Command value="/micro-project" /></div>{projects.length ? projects.map(project => <button className="document-row" key={`${project.day.id}-${project.slug}`} onClick={() => setSelectedDay(project.day.id)}><Code2 size={20} /><span>{project.name}<small>Day {project.day.global_day} · {project.created}</small></span><Status status={project.status} /></button>) : <Empty icon={Sprout} title="Space for your next idea">No micro-projects recorded for this month yet.</Empty>}</section></div><section className="panel"><div className="section-heading"><h2>Open-source contributions</h2><Command value="/github-issues" /></div>{contributions.length ? contributions.map(day => <div className="contribution" key={day.id}><GitPullRequest size={23} /><div><h3>{day.github_contribution.issue_title}</h3><p>{day.github_contribution.repo} · {pretty(day.github_contribution.status)}</p><a href={safeLink(day.github_contribution.issue_url)} target="_blank" rel="noreferrer">View issue ↗</a>{safeLink(day.github_contribution.pr_url) && <a href={day.github_contribution.pr_url} target="_blank" rel="noreferrer">View pull request ↗</a>}</div></div>) : <Empty icon={GitPullRequest} title="Your first contribution starts with curiosity">When you choose an issue, your contribution and PR review status will appear here.</Empty>}</section></>}

        {page === 'Library' && <><div className="toolbar"><label className="search wide"><Search size={18} /><input aria-label="Search library" placeholder="Search notes, resources, or their contents…" value={query} onChange={event => setQuery(event.target.value)} /></label><span className="muted">{documents.length} documents across all months</span></div><div className="library-grid">{documents.filter(item => `${item.path} ${item.content}`.toLowerCase().includes(query.toLowerCase())).map(item => <button className="library-card" key={item.path} onClick={() => setSelectedDoc(item.path)}><span className="document-icon"><FileText size={22} /></span><ArrowUpRight size={17} className="library-arrow" /><small>{item.path.includes('/') ? item.path.split('/').slice(0, -1).join(' / ').replaceAll('_', ' ') : 'JOURNAL & REFERENCE'}</small><h3>{docName(item.path)}</h3><p>{item.content.replace(/[#*`>]/g, '').trim().slice(0, 125)}…</p></button>)}</div>{!documents.some(item => `${item.path} ${item.content}`.toLowerCase().includes(query.toLowerCase())) && <Empty icon={Search} title="Nothing on this shelf yet">Try a different search.</Empty>}</>}

        <footer><span><Flower2 size={16} /> Made for the long game.</span><span>{overall.done} days learned · Schema v{state.schema_version} · {Object.keys(schema.properties || {}).length} source fields<span className="footer-sync"> · Synced {new Date(payload.loadedAt).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Kolkata' })} IST</span></span></footer>
      </div>
    </main>

    <dialog ref={dialog} className="detail-dialog" onCancel={event => { event.preventDefault(); closeDialog(); }} onClick={event => { if (event.target === dialog.current) closeDialog(); }} aria-labelledby="dialog-title">
      <div className="dialog-top"><span><Flower2 size={19} />{doc ? 'YOUR LEARNING LIBRARY' : 'A CLOSER LOOK'}</span><button className="icon-button" onClick={closeDialog} aria-label="Close details"><X size={21} /></button></div>
      {selected && <div className="day-detail"><div className="eyebrow">MONTH {selected.month} · WEEK {selected.week} · DAY {selected.day_in_week}</div><h2 id="dialog-title">{dayTitle(selected)}</h2><Status status={selected.status} /><div className="detail-grid">{[['Mode', pretty(selected.mode)], ['Confidence', selected.confidence ? pretty(selected.confidence) : 'Not recorded'], ['Study time', selected.time_spent_minutes == null ? 'Not recorded' : `${selected.time_spent_minutes} min`], ['Expected time', selected.expected_minutes == null ? 'Not specified' : `${selected.expected_minutes} min`], ['Difficulty', selected.difficulty ? pretty(selected.difficulty) : 'Not recorded'], ['Reviews', selected.review_history.times_reviewed], ['Last reviewed', selected.review_history.last_reviewed || 'Not yet'], ['Next review', selected.srs.next_review_due || 'Not scheduled']].map(([label, value]) => <div key={label}><small>{label}</small><strong>{value}</strong></div>)}</div>{selected.has_flags && <div className="notice attention">This day has flagged quiz questions.</div>}{selected.tags.length > 0 && <div className="review-tags">{selected.tags.map(tag => <span className="tag" key={tag}>{tag}</span>)}</div>}<h3>Learning materials</h3>{dayDocuments(documents, selected).length ? dayDocuments(documents, selected).map(item => <button key={item.path} className="document-row" onClick={() => setSelectedDoc(item.path)}><FileText size={19} /><span>{docName(item.path)}<small>{item.path.slice(selected.folder.length + 1)}</small></span><ArrowUpRight size={17} /></button>) : <p className="muted">No materials generated for this day yet.</p>}{selected.confusion_notes.length > 0 && <><h3>Confusion notes</h3>{selected.confusion_notes.map((note, index) => <blockquote key={index}>{note.note}<small>{note.date}</small></blockquote>)}</>}<div className="detail-actions"><p>Continue in your curriculum assistant. These buttons copy commands.</p><div><Command value={selected.status === 'current' ? '/i-am-in' : '/progress'} /><Command value="/spaced-review" /></div></div></div>}
    </dialog>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
