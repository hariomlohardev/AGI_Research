# Dashboard architecture

- React + Vite app under web/. Local read-only data comes from server/data.js; source files and notebooks are never executed by the server.
- The adapter reads actual state/schema, local journals, allowlisted Markdown and typed previews from notebooks/, src/ and tests/. It retains the old daily code layout for Month 2. Notebook previews contain input cells only, excluding outputs, execution counts, attachments and widget metadata. Symlinks and dataset/cache/checkpoint folders are not traversed; individual previews are limited to 1 MiB.
- Valid month_N/curriculum.json manifests make prepared plans visible before a month is started. curriculumView derives pending/rest display entries without writing or adding them to the recorded state. Recorded progress takes precedence. Week 1 Day 8 remains a valid distinct day.
- The UI groups each day's files into Notes, Notebooks, Python code and Tests. The library searches/filters those types. DocumentPage renders Markdown, notebook cells or escaped code and links existing notebook/Python counterparts. The existing #document= route supports reload/back navigation for all file types.
- Prepared months are explicitly labeled not started. Their roadmap is readable; starting a month remains a Claude Code action. The current-day card is scoped to the selected month, and lifetime completion totals use recorded state only.
- Vite's local development/preview endpoint reads personal data and binds to loopback. Static public builds call loadExampleCurriculum and contain only the sanitized fixture, with empty documents/files/plans and an example-data label. They never snapshot the learner's live data.
- Refresh retains the last valid payload on error. A partially saved notebook shows a file-level message rather than breaking other materials.
- Existing theme, responsive navigation, reduced-motion styling and accessible day dialog are preserved. New workspace and notebook styles extend the existing surfaces.
- npm test covers calculations, prepared-month isolation, variable week lengths, separate/legacy files, pairing, malformed files and public snapshot privacy. npm run build verifies the production bundle.
