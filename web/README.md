# Bloom — your learning space

A React dashboard with Material 3 Expressive-inspired shapes, tonal colors, motion, light/dark themes, and responsive layouts. All app files are isolated in `web/`; the existing curriculum is read-only.

## Run locally

From this directory:

```powershell
npm install
npm run dev
```

Open http://127.0.0.1:5173. The app reloads curriculum data every 30 seconds, or immediately with the refresh button. No curriculum files are modified. Commands in the interface are copied for use in the curriculum assistant; they do not execute or mark anything complete.

## Views

- Overview: month completion, current day, cross-month study-day streak, recorded hours, week reviews, and earned badges.
- Curriculum: month selection, status filters, topic/tag search, and day details.
- Reviews: due spaced reviews, shaky concepts, flags, and the current month's struggle gate.
- Projects: final-project documents, micro-projects, issues, and submitted PR links.
- Library: searchable learning Markdown and journal files, with an integrated reader.

Unrecorded time, difficulty, confidence, and topics remain explicitly unrecorded. Rest days are visible and excluded from completion totals. A study-day streak follows the schema's sequence rule, rather than claiming a calendar-day streak. Badges are shown only when already recorded as unlocked.

## Build and check

```powershell
npm test
npm run build
npm run preview
```

Preview runs at http://127.0.0.1:4173 and reads current local files. `dist/` can also be served by a static HTTP server: it includes a curriculum snapshot captured at build time. Rebuild to update that snapshot. The built snapshot includes your learning notes, so keep it local unless you intend to share them. No hosting or publishing is configured.

Fonts are loaded from Google Fonts with system fallbacks; the dashboard still works without them. No analytics, account, or external data service is used. Day documents render Markdown without executable HTML. External resources open in a new tab. Theme preference alone is saved in browser storage.

Design reference: [Material 3](https://m3.material.io/get-started).

## Month 3 workspace layout

The local app discovers prepared months from month_N/curriculum.json as well as months already recorded in state.json. A prepared month is labeled not started and does not alter progress. Select Month 3 to read its roadmap and planned days.

Each day's workspace groups Markdown notes, implementation notebooks, Python exports and independent tests. The Library can filter and search these file types. Opening a notebook shows its notes/code cells; opening Python or tests shows read-only source. Notebook/Python pairs link to each other when both exist. Previews do not execute code or load saved notebook outputs.

Month 2's original paths continue to work. The public production bundle remains example-only; local progress, files and coaching are not embedded. Run npm test and npm run build after changing the data contract or views.
