# Dashboard implementation

All changes are new files under `web/`. Existing curriculum files, state, schema, and in-progress work are read-only.

- [NEW] package.json, index.html, .gitignore: standalone React and Vite app.
- [NEW] server/data.js, vite.config.js: read existing state/schema and an allowlisted learning-document collection. Serve fresh data locally; emit a snapshot for static builds.
- [NEW] src/model.js: derived progress, streak, review and project selectors without modifying data.
- [NEW] src/main.jsx, src/styles.css: responsive Material 3 Expressive-inspired overview, curriculum, reviews, projects, library, day detail and document reader.
- [NEW] tests/model.test.js: meaningful checks of progress calculations and source integration.
- [NEW] README.md, system_architecture.md, task.md: usage and maintenance notes.

Verify calculations, production build, live endpoints and desktop/mobile browser behavior. Preserve every existing file.
