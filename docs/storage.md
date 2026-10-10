# Storage handoff

The pages depend on a plain asynchronous object, `store`. Storage details never enter the recording or listener controllers. The default adapter is IndexedDB; there is no application server.

## Connect the server

In `assets/js/storage.js`, add an import and replace the exported factory:

```js
import { createHttpStore } from "./http-storage.js";
export const store = createHttpStore("/api/v1");
```

Those are the only page-facing changes if the server follows the example below. If your existing API differs, edit only `http-storage.js` to map its requests/responses to these shapes. Authentication, server validation, access rules and shared persistence are the server team's responsibility. The example uses same-origin credentials; it does not create authentication or guess participant access policy.

`capabilities.persistence` is `local` or `remote`; `linkScope` is `browser` or `shared`. The UI uses these to describe where a save goes and whether a link can work across devices. `ready()` resolves when the provider is available. Every write promise must resolve **only after durable storage succeeds**.

## Method contract

`operationId` is supplied as the second argument `{ operationId }` for project creation, session creation, recording saves and answer saves. The page retains it and the payload for retries.

| Method                                        | Input                                  | Resolved value                                                                  |
| --------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------- |
| `ready()`                                     | None                                   | No value                                                                        |
| `createProject(project, options)`             | Project JSON                           | `{ project, confirmation }`                                                     |
| `listProjects()`                              | None                                   | Array of projects or summaries with `id`, `title`, `createdAt`, `sentenceCount` |
| `getProject(id)`                              | Project ID                             | Full project                                                                    |
| `findSession(targetId, role)`                 | Project/study ID, `speaker`/`listener` | Latest session progress or `null`                                               |
| `startSession(input, options)`                | Session input below                    | Session progress                                                                |
| `getSession(id)`                              | Session ID                             | Session progress                                                                |
| `saveRecording({ recording, blob }, options)` | Metadata and actual audio Blob         | `{ record, confirmation, sessionProgress }`                                     |
| `getStudy(id)`                                | Study ID                               | Full study JSON                                                                 |
| `resolveAudio(assetId, context)`              | Asset ID; `{ studyId, sessionId }`     | `{ src, mimeType, release() }`                                                  |
| `saveAnswer(answer, options)`                 | Answer metadata                        | `{ record, confirmation, sessionProgress }`                                     |
| `completeSession(id)`                         | Session ID                             | Session progress plus `confirmation`                                            |

`resolveAudio` may return an object URL locally or an authorized playback URL remotely. `release()` releases any temporary resources and may be a no-op for ordinary remote URLs. Audio bytes are never embedded in project/question JSON.

A progress response has this shape:

```json
{
  "session": {
    "schemaVersion": 1,
    "id": "session-id",
    "participantId": "participant-id",
    "role": "speaker",
    "displayName": "reader-name",
    "projectId": "project-id",
    "projectRevision": 1,
    "requiredItemIds": ["sentence-a", "sentence-b"],
    "createdAt": "2026-10-10T10:00:00.000Z",
    "status": "in_progress",
    "completedAt": null
  },
  "requiredItemIds": ["sentence-a", "sentence-b"],
  "savedItemIds": ["sentence-a"],
  "nextItemId": "sentence-b"
}
```

Listener sessions use `studyId`/`studyRevision`, `displayName: null`, and required question IDs. Completed sessions use `status: "completed"` and a timestamp. `nextItemId` is `null` only when all required items are confirmed saved. A speaker item is saved only when **both its metadata and binary audio exist**.

Creation input is `{ id, participantId, role, targetId, createdAt, displayName? }`. IDs and timestamps are generated once by the page. A username is a display label, not an identity or an authentication credential.

`confirmation` is `{ persistence: "local" | "remote", savedAt: ISO timestamp }`. Recording and answer `record` values contain the submitted metadata plus provider fields such as `savedAt`; a stored recording also includes `audioAssetId` and, locally, `digest`.

## Data payloads

A project is:

```json
{
  "schemaVersion": 1,
  "id": "project-id",
  "revision": 1,
  "createdAt": "2026-10-10T10:00:00.000Z",
  "title": "句子朗读研究",
  "stimulusLanguage": "zh-Hans",
  "instructions": "请清楚地朗读。",
  "metadata": { "description": "", "researcherReference": "" },
  "sentences": [
    {
      "id": "sentence-a",
      "order": 0,
      "text": "今天的天气很好。",
      "language": "zh-Hans",
      "sourceLine": 1
    }
  ],
  "importSummary": {
    "format": "txt",
    "encoding": "UTF-8",
    "sourceFilename": "sentences.txt",
    "sentenceCount": 1,
    "blankLines": 0,
    "duplicateLines": []
  }
}
```

TSV sentences may also contain `sourceReference`. Projects are immutable in this starter. Study JSON follows `studies/demo.json`; `getStudy()` reads `studies/<id>.json` locally.

Recording metadata is `{ schemaVersion: 1, id, projectId, projectRevision: 1, participantId, sessionId, sentenceId, createdAt, mimeType, bytes, durationMs }`. `mimeType` comes from the resulting Blob rather than a hardcoded extension. Accept supported WebM, MP4 or Ogg audio, and verify actual bytes on the server. The frontend limits files to 10 MiB and captures to two minutes.

Answer metadata is `{ schemaVersion: 1, id, studyId, studyRevision: 1, participantId, sessionId, questionId, optionId, createdAt }`. Validate that the option belongs to the question and that the question belongs to the session’s study and revision.

## Optional HTTP adapter routes

These routes are an **integration example**, not a claim that an API exists. All successful responses are JSON.

| Request                                                                  | Response                                                     |
| ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| `POST /projects` — project JSON                                          | `{ project, confirmation }`                                  |
| `GET /projects`                                                          | `{ items: [...] }`                                           |
| `GET /projects/:id`                                                      | `{ project }`                                                |
| `GET /sessions/latest?target=:id&role=:role`                             | Progress object or `null`, scoped to the current participant |
| `POST /sessions` — session input                                         | Progress object                                              |
| `GET /sessions/:id`                                                      | Progress object                                              |
| `POST /recordings` — multipart `metadata` JSON part, `audio` binary part | `{ record, confirmation, sessionProgress }`                  |
| `GET /studies/:id`                                                       | `{ study }`                                                  |
| `GET /audio-assets/:id`                                                  | `{ src, mimeType }`, with a URL usable by `<audio>`          |
| `POST /answers` — answer JSON                                            | `{ record, confirmation, sessionProgress }`                  |
| `POST /sessions/:id/complete` — `{}`                                     | Progress object plus confirmation                            |

The adapter sends `Idempotency-Key` for the four mutation types above. On retries, return the same committed record, with current confirmed progress. Reject a key reused with a different payload. Enforce uniqueness for `(sessionId, sentenceId)` and `(sessionId, questionId)`. Completion must check coverage itself and be safe to repeat; the page cannot declare completion on the server’s behalf.

Failures reject with `{ code, params? }`, represented as an Error locally. HTTP failures return `{ "error": { "code": "saveFailed", "params": {} } }` with an appropriate non-2xx status. Use the camelCase codes in `messages.json`, such as `network`, `quota`, `conflict`, `sessionMissing`, `projectMissing`, `invalidStudy`, `incomplete` and `saveFailed`, or normalize your API’s codes in the adapter. Unknown codes receive the generic translated error. Keep server diagnostics out of participant-facing copy.

## Local behavior and limits

The `speech-research` IndexedDB database (version 1) has `projects`, `sessions`, `recordings`, `answers`, `operations` and `audio` stores. One transaction commits audio and recording metadata; completion uses confirmed records. SHA-256 ties a recording retry to the same Blob. Operations store small receipts and payload fingerprints, not another copy of the binary audio.

For simplicity, each operation reads the metadata collection and audio keys. This is suitable for a small local setup, not a large research dataset. There is no offline synchronization, service worker, cleanup dashboard, migration system or cross-device identity. No auto-fallback to memory is used: silently losing research data on refresh would misrepresent a successful save.

If a write fails, the page does not advance. Recording bytes remain in memory for retry/download; an answer retains its original choice and operation ID. Reloading recovers only committed progress. Concurrent tabs cannot overwrite a logical recording or answer silently; a conflicting save requires reloading to see confirmed progress.

## Local content notes

- Sentence import accepts UTF-8 TXT (one sentence per line) or TSV with `text` and optional `reference`. Limits: 1 MiB, 1,000 sentences and 1,000 characters per sentence. Duplicates keep distinct IDs.
- Titles allow 120 characters, instructions/descriptions 2,000, research references 100 and usernames 40.
- Static listener fixtures live in `studies/`; use a new study ID when changing an active study. Keep schema/revision at 1 and stable question/option IDs. The starter has no revision migration. Audio paths start with `./assets/audio/`.
- The included telescope WAV is synthetic English speech, generated with FFmpeg/Flite’s `slt` voice; replace it with researcher-approved stimuli for real studies. Regeneration is optional and is not a runtime dependency:

```sh
ffmpeg -f lavfi -i "flite=text='I saw the boy with the telescope.':voice=slt" -ar 16000 -ac 1 -c:a pcm_s16le assets/audio/telescope.wav
```

Rename/remove the existing WAV before regeneration. Playback is user initiated; this starter permits answering after playback failure. Recording capture is limited to two minutes and 10 MiB. Physical microphone, permission/backgrounding/storage-failure behavior and Safari/Firefox still need target-device verification.
