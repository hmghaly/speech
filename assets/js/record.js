import { store } from "./storage.js";
import { Recorder } from "./recorder.js";
import { AppError, isId, uid, timestamp, validateUsername } from "./data.js";
import { setup, bind, showError, status, focusHeading, time } from "./ui.js";

const get = (id) => document.getElementById(id);
const recorder = new Recorder();
let project;
let progress;
let previous;
let state = "loading";
let pending;
let recordingContext;
let registrationInput;
let alive = true;

// One state owns the controls, preventing repeated clicks from overlapping async work.
function setState(next) {
  state = next;
  const capturing = next === "recording";
  get("record-action").disabled = !["ready", "recording"].includes(next);
  get("record-action").classList.toggle("danger", capturing);
  bind(get("record-action"), capturing ? "record.stop" : "record.record");
  get("record-status").dataset.state = next;
  const statusKey = {
    ready: "record.ready",
    requesting: "record.requesting",
    recording: "record.recording",
    stopping: "record.stopping",
    saving:
      store.capabilities.persistence === "remote"
        ? "record.uploading"
        : "record.saving",
    failed: "record.unsaved",
    checking: "completion.checking",
  }[next];
  if (statusKey) status(get("record-status"), statusKey);
  get("retry-controls").hidden = next !== "failed";
  get("retry-recording").disabled = next !== "failed";
  document.documentElement.style.overscrollBehavior = [
    "recording",
    "failed",
    "saving",
  ].includes(next)
    ? "none"
    : "";
}

// Completion is a provider coverage check, not a count maintained by the page.
async function complete() {
  setState("checking");
  get("page-error").hidden = true;
  try {
    progress = await store.completeSession(progress.session.id);
    if (!alive) return;
    get("recording-panel").hidden = true;
    get("registration").hidden = true;
    get("completion").hidden = false;
    get("completion-detail").hidden =
      store.capabilities.persistence !== "local";
    bind(get("completion-summary"), "record.completeSummary", {
      saved: progress.savedItemIds.length,
      total: progress.requiredItemIds.length,
    });
    setState("completed");
    recorder.cancel();
    focusHeading(get("completion-heading"));
  } catch (failure) {
    showError(get("page-error"), failure);
    get("retry-page").hidden = false;
  }
}

// The next sentence is selected only from adapter-confirmed saved IDs.
async function renderSentence() {
  if (!progress.nextItemId) {
    await complete();
    return;
  }
  const sentence = project.sentences.find(
    (item) => item.id === progress.nextItemId,
  );
  if (!sentence) throw new AppError("conflict");
  get("registration").hidden = true;
  get("recording-panel").hidden = false;
  get("completion").hidden = true;
  bind(get("record-heading"), "record.title");
  bind(get("sentence-position"), "record.position", {
    current: sentence.order + 1,
    total: project.sentences.length,
  });
  const counts = {
    saved: progress.savedItemIds.length,
    total: project.sentences.length,
  };
  bind(get("saved-progress"), "record.progress", counts);
  bind(get("record-progress"), "record.progress", counts, "aria-label");
  get("record-progress").max = project.sentences.length;
  get("record-progress").value = counts.saved;
  for (const id of ["sentence", "recording-panel", "record-action"])
    get(id).dataset.sentenceId = sentence.id;
  get("sentence").textContent = sentence.text;
  get("sentence").dataset.content = "sentence.text";
  get("sentence").lang = sentence.language;
  bind(get("record-timer"), "record.timer", { time: time(0) });
  setState("ready");
  if (!Recorder.supported()) {
    get("record-action").disabled = true;
    showError(
      get("page-error"),
      new AppError(isSecureContext ? "unsupported" : "insecure"),
    );
  } else focusHeading(get("record-heading"));
}

function interrupted() {
  if (!alive || !["recording", "stopping", "requesting"].includes(state))
    return;
  recordingContext = null;
  setState("ready");
  showError(get("page-error"), new AppError("interrupted"));
}

// Keeps the stopped Blob until the provider commits the recording and its associations.
async function saveTake() {
  if (!pending || ["saving", "completed"].includes(state)) return;
  setState("saving");
  get("page-error").hidden = true;
  try {
    const receipt = await store.saveRecording(pending, {
      operationId: pending.recording.id,
    });
    if (!alive) return;
    progress = receipt.sessionProgress;
    pending = null;
    await renderSentence();
  } catch (failure) {
    if (!alive) return;
    setState("failed");
    showError(get("page-error"), failure);
  }
}

// Snapshot metadata stays attached to this take while its final chunks are collected.
async function stopTake() {
  if (state !== "recording") return;
  const context = recordingContext;
  setState("stopping");
  try {
    const take = await recorder.stop();
    if (!alive || recordingContext !== context) return;
    pending = {
      blob: take.blob,
      recording: {
        ...context,
        mimeType: take.mimeType,
        bytes: take.blob.size,
        durationMs: take.durationMs,
      },
    };
    recordingContext = null;
    await saveTake();
  } catch (failure) {
    if (!alive || recordingContext !== context) return;
    recordingContext = null;
    setState("ready");
    showError(get("page-error"), failure);
  }
}

get("record-action").addEventListener("click", async () => {
  if (state === "recording") {
    await stopTake();
    return;
  }
  if (state !== "ready") return;
  get("page-error").hidden = true;
  const session = progress.session;
  recordingContext = {
    schemaVersion: 1,
    id: uid(),
    projectId: project.id,
    projectRevision: project.revision,
    participantId: session.participantId,
    sessionId: session.id,
    sentenceId: progress.nextItemId,
    createdAt: timestamp(),
  };
  const context = recordingContext;
  setState("requesting");
  let warned = false;
  try {
    await recorder.start(
      (seconds) => {
        if (!alive || recordingContext !== context) return;
        bind(get("record-timer"), "record.timer", { time: time(seconds) });
        if (seconds >= 110 && !warned) {
          warned = true;
          status(get("record-status"), "record.limit");
        }
      },
      interrupted,
      stopTake,
    );
    if (alive && recordingContext === context && state === "requesting")
      setState("recording");
  } catch (failure) {
    if (!alive || recordingContext !== context) return;
    recordingContext = null;
    setState("ready");
    showError(get("page-error"), failure);
  }
});

get("username-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state !== "register") return;
  get("page-error").hidden = true;
  try {
    const displayName = validateUsername(get("username").value);
    if (!registrationInput || registrationInput.displayName !== displayName) {
      registrationInput = {
        id: uid(),
        participantId: uid(),
        createdAt: timestamp(),
        role: "speaker",
        targetId: project.id,
        displayName,
      };
    }
    state = "registering";
    get("start-session").disabled =
      get("username").disabled =
      get("resume").disabled =
        true;
    bind(get("start-session"), "record.starting");
    progress = await store.startSession(registrationInput, {
      operationId: registrationInput.id,
    });
    if (alive) await renderSentence();
  } catch (failure) {
    state = "register";
    get("username").setAttribute("aria-invalid", "true");
    showError(get("page-error"), failure);
  } finally {
    get("start-session").disabled =
      get("username").disabled =
      get("resume").disabled =
        false;
    bind(get("start-session"), "record.start");
  }
});

get("username").addEventListener("input", () =>
  get("username").removeAttribute("aria-invalid"),
);
get("resume").addEventListener("click", async () => {
  if (state !== "register" || !previous) return;
  state = "loading";
  get("resume").disabled = true;
  try {
    progress = await store.getSession(previous.session.id);
    await renderSentence();
  } catch (failure) {
    state = "register";
    showError(get("page-error"), failure);
  } finally {
    get("resume").disabled = false;
  }
});
get("retry-recording").addEventListener("click", saveTake);
get("retry-page").addEventListener("click", async () => {
  get("retry-page").hidden = true;
  if (progress) await complete();
  else await loadProject();
});
get("download-recording").addEventListener("click", () => {
  if (!pending) return;
  const url = URL.createObjectURL(pending.blob);
  const link = document.createElement("a");
  link.href = url;
  link.download =
    pending.recording.sentenceId +
    "." +
    { "audio/webm": "webm", "audio/mp4": "mp4", "audio/ogg": "ogg" }[
      pending.blob.type.split(";")[0]
    ];
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// Reopening a link offers the most recent session from this browser/provider.
async function loadProject() {
  try {
    const id = new URLSearchParams(location.search).get("project");
    if (!isId(id)) throw new AppError("projectMissing");
    project = await store.getProject(id);
    get("project-name").textContent = project.title;
    get("project-name").dataset.content = "project.title";
    if (project.instructions) {
      get("project-instructions").textContent = project.instructions;
      get("project-instructions").dataset.content = "project.instructions";
    }
    get("project-instructions").hidden = !project.instructions;
    previous = await store.findSession(project.id, "speaker");
    get("registration").hidden = false;
    get("resume").hidden = !previous;
    state = "register";
    get("page-status").textContent = "";
  } catch (failure) {
    get("registration").hidden = true;
    get("page-status").textContent = "";
    showError(get("page-error"), failure);
    get("retry-page").hidden = false;
  }
}

window.addEventListener("beforeunload", (event) => {
  if (pending || ["recording", "stopping", "saving"].includes(state)) {
    event.preventDefault();
    event.returnValue = "";
  }
});
window.addEventListener("pagehide", () => {
  alive = false;
  recorder.cancel();
});
window.addEventListener("pageshow", (event) => {
  if (event.persisted) location.reload();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && ["recording", "requesting"].includes(state)) {
    recorder.cancel();
    interrupted();
  }
});
try {
  await setup(store);
  await loadProject();
} catch (failure) {
  get("registration").hidden = true;
  showError(get("page-error"), failure);
}
