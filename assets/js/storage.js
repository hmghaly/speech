import {
  AppError,
  isId,
  uid,
  timestamp,
  validateProject,
  validateStudy,
  normalizeSampleStudy,
  validateUsername,
  progressFor,
} from "./data.js";

// Every page depends on this object, not IndexedDB. Swap its factory for remote storage.
export const store = createLocalStore();
const metadataStores = [
  "projects",
  "sessions",
  "recordings",
  "answers",
  "operations",
];
let database;

function openDatabase() {
  if (database) return database;
  database = new Promise((resolve, reject) => {
    if (!globalThis.indexedDB) {
      reject(new AppError("storageUnavailable"));
      return;
    }
    const request = indexedDB.open("speech-research", 1);
    request.onupgradeneeded = () => {
      for (const name of [...metadataStores, "audio"])
        request.result.createObjectStore(name, { keyPath: "id" });
    };
    request.onsuccess = () => {
      request.result.onversionchange = () => {
        request.result.close();
        database = null;
      };
      resolve(request.result);
    };
    request.onerror = () => {
      database = null;
      reject(new AppError("storageUnavailable"));
    };
    request.onblocked = () => reject(new AppError("storageBlocked"));
  });
  return database;
}

// Metadata is small. Read it together, then mutate synchronously inside one transaction.
// Audio is never loaded wholesale; only its keys are read for confirmed progress.
async function transaction(mode, work) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([...metadataStores, "audio"], mode);
    const snapshot = {};
    let pending = metadataStores.length + 1;
    let result;
    let failure;
    const received = () => {
      if (--pending) return;
      try {
        result = work(snapshot, (name, value) =>
          tx.objectStore(name).put(value),
        );
      } catch (error) {
        failure = error;
        tx.abort();
      }
    };
    for (const name of metadataStores) {
      const request = tx.objectStore(name).getAll();
      request.onsuccess = () => {
        snapshot[name] = request.result;
        received();
      };
    }
    const audio = tx.objectStore("audio").getAllKeys();
    audio.onsuccess = () => {
      snapshot.audioIds = audio.result;
      received();
    };
    // Request success is not sufficient: only oncomplete confirms all writes.
    tx.oncomplete = () => resolve(result);
    tx.onabort = tx.onerror = () =>
      reject(
        failure ||
          new AppError(
            tx.error?.name === "QuotaExceededError" ? "quota" : "saveFailed",
          ),
      );
  });
}

function sessionProgress(snapshot, sessionId) {
  const session = snapshot.sessions.find((item) => item.id === sessionId);
  if (!session) throw new AppError("sessionMissing");
  return progressFor(
    session,
    session.role === "speaker" ? snapshot.recordings : snapshot.answers,
    snapshot.audioIds,
  );
}

function confirmation() {
  return { persistence: "local", savedAt: timestamp() };
}

// A mutation key is tied to its payload, so retrying cannot silently change an answer/take.
function replay(snapshot, operationId, fingerprint) {
  const previous = snapshot.operations.find((item) => item.id === operationId);
  if (previous && previous.fingerprint !== fingerprint)
    throw new AppError("conflict");
  return previous?.receipt;
}
function remember(put, operationId, fingerprint, receipt) {
  put("operations", { id: operationId, fingerprint, receipt });
}

// Small domain-oriented adapter: remote implementations can replace each method with fetch.
export function createLocalStore() {
  return {
    capabilities: { persistence: "local", linkScope: "browser" },
    ready: async () => {
      await openDatabase();
    },
    async createProject(project, { operationId = project.id } = {}) {
      validateProject(project);
      const fingerprint = JSON.stringify(project);
      return transaction("readwrite", (snapshot, put) => {
        const old = replay(snapshot, operationId, fingerprint);
        if (old) return old;
        if (snapshot.projects.some((item) => item.id === project.id))
          throw new AppError("conflict");
        put("projects", project);
        const receipt = { project, confirmation: confirmation() };
        remember(put, operationId, fingerprint, receipt);
        return receipt;
      });
    },
    listProjects: () =>
      transaction("readonly", (snapshot) =>
        snapshot.projects.sort((a, b) =>
          b.createdAt.localeCompare(a.createdAt),
        ),
      ),
    getProject: (id) =>
      transaction("readonly", (snapshot) => {
        const project = snapshot.projects.find((item) => item.id === id);
        if (!project) throw new AppError("projectMissing");
        return project;
      }),
    findSession: (targetId, role) =>
      transaction("readonly", (snapshot) => {
        const session = snapshot.sessions
          .filter(
            (item) =>
              item.role === role &&
              (item.projectId === targetId || item.studyId === targetId),
          )
          .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
        return session ? sessionProgress(snapshot, session.id) : null;
      }),
    async startSession(input, { operationId = input.id } = {}) {
      if (
        !["speaker", "listener"].includes(input.role) ||
        !isId(input.id) ||
        (input.role === "speaker" && !isId(input.participantId))
      )
        throw new AppError("conflict");
      const target =
        input.role === "speaker"
          ? await this.getProject(input.targetId)
          : await this.getStudy(input.targetId);
      const participantId =
        input.role === "listener"
          ? validateUsername(input.participantId)
          : input.participantId;
      const displayName =
        input.role === "speaker"
          ? validateUsername(input.displayName)
          : participantId;
      const session = {
        schemaVersion: 1,
        id: input.id,
        participantId,
        role: input.role,
        displayName,
        requiredItemIds: (input.role === "speaker"
          ? target.sentences
          : target.questions
        ).map((item) => item.id),
        ...(input.role === "speaker"
          ? { projectId: target.id, projectRevision: target.revision }
          : { studyId: target.id, studyRevision: target.revision }),
        createdAt: input.createdAt,
        status: "in_progress",
        completedAt: null,
      };
      const fingerprint = JSON.stringify(input);
      return transaction("readwrite", (snapshot, put) => {
        const old = replay(snapshot, operationId, fingerprint);
        if (old) return old;
        if (snapshot.sessions.some((item) => item.id === session.id))
          throw new AppError("conflict");
        put("sessions", session);
        const receipt = progressFor(session, []);
        remember(put, operationId, fingerprint, receipt);
        return receipt;
      });
    },
    getSession: (id) =>
      transaction("readonly", (snapshot) => sessionProgress(snapshot, id)),
    async saveRecording(
      { recording, blob },
      { operationId = recording.id } = {},
    ) {
      if (
        !blob?.size ||
        blob.size > 10 * 1024 * 1024 ||
        recording.bytes !== blob.size ||
        !/^audio\/(webm|mp4|ogg)(;|$)/i.test(recording.mimeType) ||
        recording.mimeType !== blob.type
      )
        throw new AppError("unsupportedMedia");
      const digest = [
        ...new Uint8Array(
          await crypto.subtle.digest("SHA-256", await blob.arrayBuffer()),
        ),
      ]
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
      const fingerprint = JSON.stringify(recording) + digest;
      return transaction("readwrite", (snapshot, put) => {
        const old = replay(snapshot, operationId, fingerprint);
        if (old)
          return {
            ...old,
            sessionProgress: sessionProgress(snapshot, recording.sessionId),
          };
        const progress = sessionProgress(snapshot, recording.sessionId);
        const session = progress.session;
        if (
          session.role !== "speaker" ||
          session.projectId !== recording.projectId ||
          session.projectRevision !== recording.projectRevision ||
          session.participantId !== recording.participantId ||
          !session.requiredItemIds.includes(recording.sentenceId)
        )
          throw new AppError("conflict");
        if (
          !isId(recording.id) ||
          session.status !== "in_progress" ||
          snapshot.recordings.some(
            (item) =>
              item.id === recording.id ||
              (item.sessionId === session.id &&
                item.sentenceId === recording.sentenceId),
          )
        )
          throw new AppError("conflict");
        const record = {
          ...recording,
          digest,
          audioAssetId: uid(),
          savedAt: timestamp(),
        };
        put("audio", { id: record.audioAssetId, blob });
        put("recordings", record);
        snapshot.recordings.push(record);
        snapshot.audioIds.push(record.audioAssetId);
        const receipt = {
          record,
          confirmation: confirmation(),
          sessionProgress: sessionProgress(snapshot, session.id),
        };
        remember(put, operationId, fingerprint, receipt);
        return receipt;
      });
    },
    async getStudy(id) {
      if (!isId(id)) throw new AppError("invalidStudy");
      try {
        const response = await fetch(
          id === "sample"
            ? "./sample/sample.json"
            : "./studies/" + encodeURIComponent(id) + ".json",
        );
        if (!response.ok) throw new Error();
        const data = await response.json();
        const study = id === "sample" ? normalizeSampleStudy(data) : data;

        console.log(study)
        validateStudy(study);
        if (study.id !== id) throw new AppError("invalidStudy");
        return study;
      } catch {
        throw new AppError("invalidStudy");
      }
    },
    async resolveAudio(assetId, { studyId } = {}) {
      const db = await openDatabase();
      const audio = await new Promise((resolve, reject) => {
        const request = db
          .transaction("audio")
          .objectStore("audio")
          .get(assetId);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(new AppError("audioFailed"));
      });
      if (audio) {
        const src = URL.createObjectURL(audio.blob);
        return {
          src,
          mimeType: audio.blob.type,
          release: () => URL.revokeObjectURL(src),
        };
      }
      const study = await this.getStudy(studyId);
      const asset = study.media.find((item) => item.id === assetId);
      if (
        !["./assets/audio/", "./sample/audio/"].some((prefix) =>
          asset?.path?.startsWith(prefix),
        )
      )
        throw new AppError("audioFailed");
      return {
        src: new URL(asset.path, location.href).href,
        mimeType: asset.mimeType,
        release() {},
      };
    },
    async saveAnswer(answer, { operationId = answer.id } = {}) {
      const study = await this.getStudy(answer.studyId);
      const question = study.questions.find(
        (item) => item.id === answer.questionId,
      );
      if (!question?.options.some((option) => option.id === answer.optionId))
        throw new AppError("choiceRequired");
      const fingerprint = JSON.stringify(answer);
      return transaction("readwrite", (snapshot, put) => {
        const old = replay(snapshot, operationId, fingerprint);
        if (old)
          return {
            ...old,
            sessionProgress: sessionProgress(snapshot, answer.sessionId),
          };
        const { session } = sessionProgress(snapshot, answer.sessionId);
        if (
          session.role !== "listener" ||
          session.studyId !== answer.studyId ||
          study.revision !== answer.studyRevision ||
          session.studyRevision !== answer.studyRevision ||
          session.participantId !== answer.participantId ||
          !session.requiredItemIds.includes(answer.questionId)
        )
          throw new AppError("conflict");
        if (
          !isId(answer.id) ||
          session.status !== "in_progress" ||
          snapshot.answers.some(
            (item) =>
              item.id === answer.id ||
              (item.sessionId === session.id &&
                item.questionId === answer.questionId),
          )
        )
          throw new AppError("conflict");
        const record = { ...answer, savedAt: timestamp() };
        put("answers", record);
        snapshot.answers.push(record);
        const receipt = {
          record,
          confirmation: confirmation(),
          sessionProgress: sessionProgress(snapshot, session.id),
        };
        remember(put, operationId, fingerprint, receipt);
        return receipt;
      });
    },
    completeSession: (id) =>
      transaction("readwrite", (snapshot, put) => {
        const progress = sessionProgress(snapshot, id);
        if (progress.nextItemId) throw new AppError("incomplete");
        const session = {
          ...progress.session,
          status: "completed",
          completedAt: progress.session.completedAt || timestamp(),
        };
        put("sessions", session);
        return { ...progress, session, confirmation: confirmation() };
      }),
  };
}
