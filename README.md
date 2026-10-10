<img src="assets/images/champolu-logo.png" alt="Champolu" width="96">

# Speech research

A Chinese-first, static frontend for creating sentence projects, recording speakers and collecting listener answers. Uses vanilla HTML/JavaScript and the Tailwind browser CDN.

## Run

Open root `index.html` with VS Code Live Server, or use an existing static server:

```sh
live-server . --no-browser
```

If Live Server is unavailable, serve the same files with Python:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open <http://localhost:8000>. No project dependencies or build step. Use localhost or HTTPS for microphone access; the Tailwind CDN needs internet access.

## Use

- Import UTF-8 TXT (one sentence per line) or TSV (`text`, optional `reference`).
- Speakers choose a username and record sentences in order.
- Listeners answer 2–4-choice questions with text, audio or both.
- Edit `studies/demo.json` and place audio in `assets/audio/` to create a study. The included English clip is synthetic demo speech.

Data stays in IndexedDB in the same browser and origin; project links do not transfer it across devices. Server integration changes the provider in `assets/js/storage.js`.

## Project structure

```text
index.html             Project listing
project.html           Project creation and sentence import
record.html            Speaker signup and recording
listener.html          Listener questions and audio playback
assets/
  css/                 Design tokens and shared styles
  js/
    index.js           Project listing controller
    project.js         Project creation controller
    record.js          Speaker session controller
    listener.js        Listener session controller
    ui.js              Shared UI and translation helpers
    data.js            Validation, imports and progress
    recorder.js        Microphone recording
    storage.js         IndexedDB provider and adapter selection
    http-storage.js    Optional server integration example
  i18n/messages.json   Chinese/English text keyed by DOM ID
  images/              Company logo
  audio/               Demo audio
studies/demo.json      Listener study example
tests/core.test.mjs   Focused regression tests
docs/                  Storage and visual design guides
```

## Documentation

- [Storage](docs/storage.md): local persistence, data formats and server adapter integration.
- [Visual design](docs/visual-design.md): branding, layouts, accessibility and internationalization.

Optional checks (Node.js 22.7+): `node tests/core.test.mjs`. Real microphone and target-device verification remain pending.
