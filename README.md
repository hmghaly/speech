<img src="assets/images/champolu-logo.png" alt="Champolu" width="96">

# Speech research

A Chinese-first, static frontend for creating sentence projects, recording speakers and collecting listener answers. Uses vanilla HTML/JavaScript and the Tailwind browser CDN.

## Run

Open root `index.html` with VS Code Live Server, or use an existing static server:

```sh
live-server . --no-browser
```

No package installation or build step. Use localhost or HTTPS for microphone access; the Tailwind CDN needs internet access.

## Use

- Import UTF-8 TXT (one sentence per line) or TSV (`text`, optional `reference`).
- Speakers choose a username and record sentences in order.
- Listeners answer 2–4-choice questions with text, audio or both.
- Edit `studies/demo.json` and place audio in `assets/audio/` to create a study. The included English clip is synthetic demo speech.

Data stays in IndexedDB in the same browser and origin; project links do not transfer it across devices. Server integration changes the provider in `assets/js/storage.js`.

Custom CSS/JS lives in `assets/css/` and `assets/js/`; interface translations are keyed by DOM ID in `assets/i18n/messages.json` (Chinese by default; English also included).

## Documentation

- [Storage](docs/storage.md): local persistence, data formats and server adapter integration.
- [Visual design](docs/visual-design.md): branding, layouts, accessibility and internationalization.

Optional checks (Node.js 22.7+): `node tests/core.test.mjs`. Real microphone and target-device verification remain pending.
