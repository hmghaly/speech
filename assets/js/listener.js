import { store } from "./storage.js";
import { AppError, uid, timestamp } from "./data.js";
import {
  setup,
  bind,
  content,
  showError,
  status,
  focusHeading,
  time,
} from "./ui.js";

const get = (id) => document.getElementById(id);
const studyId = new URLSearchParams(location.search).get("study") || "demo";
let study;
let progress;
let previous;
let pending;
let sessionInput;
let state = "loading";
let mediaHandle;
let audio = get("question-audio");
let generation = 0;
let alive = true;

// Releasing a question also releases its media URL and silences stale playback callbacks.
function clearAudio() {
  generation++;
  audio.onplay =
    audio.onpause =
    audio.onended =
    audio.onerror =
    audio.ontimeupdate =
    audio.onloadedmetadata =
      null;
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  mediaHandle?.release();
  mediaHandle = null;
}

// A generation token prevents old question audio from updating the current controls.
async function loadAudio(question) {
  clearAudio();
  const token = generation;
  get("play-audio").disabled = true;
  get("retry-audio").hidden = true;
  status(get("audio-status"), "audio.loading");
  try {
    const handle = await store.resolveAudio(question.stimulus.audioAssetId, {
      studyId,
      sessionId: progress.session.id,
    });
    if (!alive || token !== generation) {
      handle.release();
      return;
    }
    mediaHandle = handle;
    audio = document.createElement("audio");
    audio.id = "question-audio";
    audio.preload = "metadata";
    bind(audio, "audio.label", {}, "aria-label");
    get("audio-panel").querySelector("audio").replaceWith(audio);
    const current = () => alive && token === generation;
    const playbackError = () => {
      if (!current()) return;
      status(get("audio-status"), "audio.failed");
      get("retry-audio").hidden = false;
      get("play-audio").disabled = true;
    };
    audio.onerror = playbackError;
    audio.onloadedmetadata = () => {
      if (!current()) return;
      get("audio-status").textContent = "";
      get("play-audio").disabled = false;
      bind(get("audio-time"), "audio.time", {
        elapsed: time(0),
        duration: time(audio.duration),
      });
    };
    audio.ontimeupdate = () => {
      if (current())
        bind(get("audio-time"), "audio.time", {
          elapsed: time(audio.currentTime),
          duration: time(audio.duration),
        });
    };
    audio.onplay = () => {
      if (current()) bind(get("play-audio"), "audio.pause");
    };
    audio.onpause = audio.onended = () => {
      if (current()) bind(get("play-audio"), "audio.play");
    };
    audio.src = handle.src;
    audio.load();
  } catch {
    if (token !== generation || !alive) return;
    status(get("audio-status"), "audio.failed");
    get("retry-audio").hidden = false;
  }
}

// Only provider-confirmed coverage can produce the completion screen.
async function complete() {
  state = "checking";
  clearAudio();
  get("question-panel").hidden = true;
  get("listener-start").hidden = true;
  status(get("page-status"), "completion.checking");
  try {
    progress = await store.completeSession(progress.session.id);
    if (!alive) return;
    state = "completed";
    get("page-status").textContent = "";
    get("completion").hidden = false;
    get("completion-detail").hidden =
      store.capabilities.persistence !== "local";
    bind(get("completion-summary"), "listener.completeSummary", {
      saved: progress.savedItemIds.length,
      total: progress.requiredItemIds.length,
    });
    focusHeading(get("completion-heading"));
  } catch (failure) {
    showError(get("page-error"), failure);
    get("retry-page").hidden = false;
  }
}

// Render only the assigned stimulus mode, including its accessibility-tree representation.
async function renderQuestion() {
  if (!progress.nextItemId) {
    await complete();
    return;
  }
  const question = study.questions.find(
    (item) => item.id === progress.nextItemId,
  );
  if (!question) throw new AppError("conflict");
  clearAudio();
  get("listener-start").hidden = true;
  get("question-panel").hidden = false;
  get("completion").hidden = true;
  get("page-status").textContent = "";
  get("page-error").hidden = true;
  get("answer-status").textContent = "";
  get("answer-group").removeAttribute("aria-invalid");
  bind(get("question-position"), "listener.position", {
    current: question.order + 1,
    total: study.questions.length,
  });
  const counts = {
    saved: progress.savedItemIds.length,
    total: study.questions.length,
  };
  bind(get("answer-progress"), "listener.progress", counts);
  bind(get("listener-progress"), "listener.progress", counts, "aria-label");
  get("listener-progress").max = counts.total;
  get("listener-progress").value = counts.saved;
  if (question.instructions) {
    get("question-instructions").textContent = question.instructions;
    get("question-instructions").dataset.content = "question.instructions";
  } else
    bind(
      get("question-instructions"),
      "listener.instruction." + question.presentationMode,
    );
  const showText = question.presentationMode !== "audio_only";
  get("question-sentence").hidden = !showText;
  get("question-sentence").textContent = showText ? question.stimulus.text : "";
  get("question-sentence").dataset.content = "sentence.text";
  get("question-sentence").lang =
    question.stimulus.language || study.stimulusLanguage;
  get("audio-panel").hidden = question.presentationMode === "text_only";
  get("audio-time").textContent = "";
  bind(get("play-audio"), "audio.play");
  get("question-prompt").textContent = question.prompt;
  get("question-prompt").dataset.content = "question.prompt";
  get("question-prompt").lang = question.language || study.stimulusLanguage;
  const options = get("answer-options");
  options.replaceChildren();
  for (const option of question.options) {
    const prefix = "ui-listener-" + question.id + "-" + option.id;
    const label = document.createElement("label");
    label.className = "option";
    label.htmlFor = prefix + "-radio";
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "answer";
    radio.id = prefix + "-radio";
    radio.value = option.id;
    radio.required = true;
    const caption = content("span", option.text, prefix + "-label");
    caption.lang = question.language || study.stimulusLanguage;
    label.append(radio, caption);
    options.append(label);
  }
  state = "ready";
  pending = null;
  get("submit-answer").disabled = false;
  bind(get("submit-answer"), "listener.submit");
  if (question.presentationMode !== "text_only") void loadAudio(question);
  focusHeading(get("question-prompt"));
}

// Explain the saved session before offering its primary action.
function showSessionChoice() {
  const completed = previous && !previous.nextItemId;
  const variant = !previous ? "start" : completed ? "completed" : "resume";
  bind(get("listener-session-heading"), "listener." + variant + "Title");
  bind(get("listener-session-description"), "listener." + variant + "Help", {
    saved: previous?.savedItemIds.length || 0,
    total: previous?.requiredItemIds.length || study.questions.length,
  });
  bind(get("resume"), completed ? "listener.view" : "listener.resume");
  bind(get("new-session"), previous ? "listener.new" : "listener.start");
  const resume = get("resume");
  const start = get("new-session");
  const continueSession = previous && !completed;
  resume.hidden = !previous;
  resume.classList.toggle("secondary", !continueSession);
  start.classList.toggle("secondary", Boolean(continueSession));
  // Keep the primary action first in both the layout and keyboard order.
  get("listener-start")
    .querySelector(".listener-session-actions")
    .append(...(continueSession ? [resume, start] : [start, resume]));
  get("listener-start").hidden = false;
}

// A failed start retries its original identity; resuming reloads confirmed progress.
async function startSession(resume = false) {
  if (!["start", "loading"].includes(state)) return;
  state = "starting";
  get("new-session").disabled = get("resume").disabled = true;
  try {
    if (resume) progress = await store.getSession(previous.session.id);
    else {
      sessionInput ||= {
        id: uid(),
        participantId: uid(),
        role: "listener",
        targetId: study.id,
        createdAt: timestamp(),
      };
      progress = await store.startSession(sessionInput, {
        operationId: sessionInput.id,
      });
    }
    if (alive) await renderQuestion();
  } catch (failure) {
    state = "start";
    showSessionChoice();
    showError(get("page-error"), failure);
  } finally {
    get("new-session").disabled = get("resume").disabled = false;
  }
}

// After a failed write, retain the same answer and operation ID until a retry is confirmed.
get("question-panel").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!["ready", "failed"].includes(state)) return;
  get("page-error").hidden = true;
  try {
    const optionId =
      get("question-panel").querySelector("input:checked")?.value;
    if (!optionId) throw new AppError("choiceRequired");
    pending ||= {
      schemaVersion: 1,
      id: uid(),
      studyId: study.id,
      studyRevision: study.revision,
      questionId: progress.nextItemId,
      optionId,
      participantId: progress.session.participantId,
      sessionId: progress.session.id,
      createdAt: timestamp(),
    };
    state = "saving";
    get("submit-answer").disabled = true;
    get("answer-group").disabled = true;
    bind(get("submit-answer"), "listener.saving");
    status(get("answer-status"), "listener.saving");
    const receipt = await store.saveAnswer(pending, {
      operationId: pending.id,
    });
    if (!alive) return;
    progress = receipt.sessionProgress;
    pending = null;
    get("answer-group").disabled = false;
    await renderQuestion();
  } catch (failure) {
    if (!alive) return;
    state = pending ? "failed" : "ready";
    get("answer-group").setAttribute("aria-invalid", "true");
    showError(get("page-error"), failure);
    get("submit-answer").disabled = false;
    bind(get("submit-answer"), pending ? "listener.retry" : "listener.submit");
    get("answer-status").textContent = "";
  }
});
get("answer-options").addEventListener("change", () => {
  get("answer-group").removeAttribute("aria-invalid");
  get("page-error").hidden = true;
});
get("play-audio").addEventListener("click", async () => {
  const token = generation;
  try {
    if (audio.paused) await audio.play();
    else audio.pause();
  } catch {
    if (token !== generation || !alive) return;
    status(get("audio-status"), "audio.failed");
    get("retry-audio").hidden = false;
  }
});
get("retry-audio").addEventListener("click", () =>
  loadAudio(study.questions.find((item) => item.id === progress.nextItemId)),
);
get("new-session").addEventListener("click", () => startSession());
get("resume").addEventListener("click", () => startSession(true));
get("retry-page").addEventListener("click", async () => {
  get("retry-page").hidden = true;
  get("page-error").hidden = true;
  if (progress) await complete();
  else await loadStudy();
});

// Study content comes from data, without an authoring dashboard or scoring logic.
async function loadStudy() {
  try {
    study = await store.getStudy(studyId);
    get("demo-notice").hidden = study.id !== "demo";
    get("study-title").textContent = study.title;
    get("study-title").dataset.content = "study.title";
    previous = await store.findSession(study.id, "listener");
    state = "start";
    if (previous) {
      showSessionChoice();
      get("page-status").textContent = "";
    } else await startSession();
  } catch (failure) {
    showError(get("page-error"), failure);
    get("page-status").textContent = "";
    get("retry-page").hidden = false;
  }
}

window.addEventListener("beforeunload", (event) => {
  if (pending) {
    event.preventDefault();
    event.returnValue = "";
  }
});
window.addEventListener("pagehide", () => {
  alive = false;
  clearAudio();
});
window.addEventListener("pageshow", (event) => {
  if (event.persisted) location.reload();
});
try {
  await setup(store);
  await loadStudy();
} catch (failure) {
  showError(get("page-error"), failure);
}
