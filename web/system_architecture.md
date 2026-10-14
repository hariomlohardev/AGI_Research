# Dashboard architecture

- Independent React 19 + Vite 6 app; all package installation, build output, and app documentation stay under `web/`.
- `server/data.js` reads `../state.json`, `../state.schema.json`, allowlisted root journals, and Markdown under recorded month folders. It never writes, follows no symlinks, and accepts no user-supplied filesystem path. Code and dependency directories are excluded.
- The Vite plugin provides GET/HEAD `/data/curriculum.json` for development/preview, rejecting mutation requests. A build emits the same payload as a static snapshot. Local servers bind only to loopback.
- `src/model.js` computes non-rest completion, recorded time, due reviews, the unresolved-struggle rule, and the cross-month study-day streak. Nothing derived is persisted to curriculum state.
- `src/main.jsx` contains navigable hash views, filtering, local theme preference, an accessible native modal, and sanitized Markdown rendering. Data refreshes every 30 seconds; refresh failures leave the last valid view with an error banner.
- CSS uses Material 3 Expressive-inspired tonal surfaces, shape variants, emphasized typography, responsive navigation and reduced-motion support.
- Tests cover schema-dependent calculation rules, timezones, month ordering, and the live adapter's read-only behavior. Run `npm test` and `npm run build` after changes.
