export const uid = () => crypto.randomUUID();
export const timestamp = () => new Date().toISOString();
export const isId = (value) => /^[a-zA-Z0-9_-]{1,80}$/.test(value || "");

export class AppError extends Error {
  constructor(code, params = {}) {
    super(code);
    this.code = code;
    this.params = params;
  }
}

// Reject non-printing input while allowing tabs and line separators.
// eslint-disable-next-line no-control-regex
const controls = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/u;
const length = (value) => [...value].length;

// Strict import keeps sentence order and intentional duplicate text.
export function parseSentences(bytes, filename, language = "zh-Hans") {
  if (bytes.byteLength > 1024 * 1024) throw new AppError("fileTooLarge");
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new AppError("encodingInvalid");
  }
  const format = filename.toLowerCase().endsWith(".tsv") ? "tsv" : "txt";
  if (!/\.(txt|tsv)$/i.test(filename)) throw new AppError("formatInvalid");
  const lines = text
    .replace(/^\uFEFF/, "")
    .replace(/\r\n?/g, "\n")
    .split("\n");
  // A normal terminal newline is a separator, not an extra blank sentence row.
  if (lines.at(-1) === "") lines.pop();
  let headers = [];
  if (format === "tsv") {
    headers = (lines.shift() || "").split("\t");
    if (
      !headers.includes("text") ||
      new Set(headers).size !== headers.length ||
      headers.some((header) => !["text", "reference"].includes(header))
    ) {
      throw new AppError("headerInvalid");
    }
  }
  const sentences = [];
  const seen = new Map();
  const duplicateLines = [];
  let blankLines = 0;
  lines.forEach((line, index) => {
    const sourceLine = index + (format === "tsv" ? 2 : 1);
    if (!line.trim()) {
      blankLines++;
      return;
    }
    if (controls.test(line))
      throw new AppError("controlCharacters", { line: sourceLine });
    const fields = line.split("\t");
    if (
      (format === "tsv" && fields.length !== headers.length) ||
      (format === "txt" && fields.length !== 1)
    ) {
      throw new AppError("rowMalformed", { line: sourceLine });
    }
    const sentence = (
      format === "tsv" ? fields[headers.indexOf("text")] : line
    ).trim();
    if (!sentence) throw new AppError("rowMalformed", { line: sourceLine });
    if (length(sentence) > 1000)
      throw new AppError("sentenceTooLong", { line: sourceLine });
    if (seen.has(sentence)) duplicateLines.push(sourceLine);
    seen.set(sentence, sourceLine);
    const row = {
      id: uid(),
      order: sentences.length,
      text: sentence,
      language,
      sourceLine,
    };
    if (headers.includes("reference"))
      row.sourceReference = fields[headers.indexOf("reference")].trim();
    sentences.push(row);
  });
  if (!sentences.length) throw new AppError("fileEmpty");
  if (sentences.length > 1000) throw new AppError("tooManySentences");
  return {
    sentences,
    importSummary: {
      format,
      encoding: "UTF-8",
      sourceFilename: filename,
      sentenceCount: sentences.length,
      blankLines,
      duplicateLines,
    },
  };
}

// Validation is shared by the page and local adapter; the server must also validate.
export function validateProject(project) {
  if (
    project.schemaVersion !== 1 ||
    !isId(project.id) ||
    project.revision !== 1
  )
    throw new AppError("schemaUnsupported");
  if (!project.title?.trim()) throw new AppError("titleRequired");
  if (length(project.title) > 120) throw new AppError("titleTooLong");
  if (
    length(project.instructions || "") > 2000 ||
    length(project.metadata?.description || "") > 2000
  )
    throw new AppError("instructionsTooLong");
  if (length(project.metadata?.researcherReference || "") > 100)
    throw new AppError("referenceTooLong");
  if (!["zh-Hans", "zh-Hant", "en"].includes(project.stimulusLanguage))
    throw new AppError("languageInvalid");
  if (
    !Array.isArray(project.sentences) ||
    !project.sentences.length ||
    project.sentences.length > 1000
  )
    throw new AppError("fileEmpty");
  const ids = new Set();
  project.sentences.forEach((sentence, index) => {
    if (
      !isId(sentence.id) ||
      ids.has(sentence.id) ||
      sentence.order !== index ||
      typeof sentence.text !== "string" ||
      !sentence.text.trim() ||
      length(sentence.text) > 1000 ||
      controls.test(sentence.text)
    )
      throw new AppError("rowMalformed", { line: index + 1 });
    ids.add(sentence.id);
  });
}

export function validateUsername(value) {
  const name = typeof value === "string" ? value.trim() : "";
  if (!name) throw new AppError("usernameRequired");
  if (length(name) > 40) throw new AppError("usernameTooLong");
  if (controls.test(name)) throw new AppError("controlCharacters");
  return name;
}

// A refusal is a valid choice; missing consent must never be treated as agreement.
export function consentRecord(consented, recordedAt = timestamp()) {
  if (typeof consented !== "boolean") throw new AppError("consentRequired");
  return {
    consented,
    consentVersion: "2026-10-21",
    consentRecordedAt: recordedAt,
  };
}

// Map the client's sample rows into the listener contract without changing the source JSON.
export function normalizeSampleStudy(rows) {
  if (!Array.isArray(rows) || !rows.length) throw new AppError("invalidStudy");
  const media = new Map();
  const questions = rows.map((row, order) => {
    const audio =
      typeof row?.audio_path === "string"
        ? row.audio_path.match(/^sample\/audio\/([a-zA-Z0-9_-]+)\.wav$/)
        : null;
    if (
      !audio ||
      typeof row.sentence !== "string" ||
      !row.sentence.trim() ||
      typeof row.question !== "string" ||
      !row.question.trim() ||
      !Array.isArray(row.options) ||
      row.options.some((option) => typeof option?.content !== "string")
    )
      throw new AppError("invalidStudy");
    const audioAssetId = audio[1];

    media.set(audioAssetId, {
      id: audioAssetId,
      path: "./" + row.audio_path,
      mimeType: "audio/wav",
    });
    return {
      // Assign identity before shuffling; changing the recording must not rename a question.
      id: audioAssetId + "-" + (order + 1),
      audio_path: row.audio_path,
      alt_audio_paths: row.alt_audio_paths || [],
      original_sentence: row.original_sentence,
      order,
      presentationMode: "both",
      language: "en",
      prompt: row.question,
      stimulus: {
        text: row.sentence,
        language: "zh-Hans",
        audioAssetId,
      },
      // Keep the source option IDs; the answer key is not part of the participant UI.
      options: row.options.map((option) => ({
        id: option.id,
        text: option.content,
      })),
    };
  });
  const study = {
    schemaVersion: 1,
    id: "sample",
    revision: 1,
    title: "Chinese sentence interpretation",
    stimulusLanguage: "zh-Hans",
    questions,
    media: [...media.values()],
  };
  validateStudy(study);
  return study;
}

// Fisher–Yates returns a new array, leaving source rows and audio choices untouched.
function shuffled(items, random) {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

// Draw once when loading a study; the session retains this exact order and audio assignment.
export function randomizeStudy(source, random = Math.random) {
  const study = structuredClone(source);
  const media = new Map((study.media || []).map((asset) => [asset.id, asset]));
  study.questions = shuffled(study.questions, random).map((question, order) => {
    const alternatives = question.alt_audio_paths || [];
    if (
      !Array.isArray(alternatives) ||
      alternatives.some(
        (path) =>
          typeof path !== "string" ||
          !/^sample\/audio\/[a-zA-Z0-9_-]+\.wav$/.test(path),
      )
    )
      throw new AppError("invalidStudy");
    if (alternatives.length) {
      question.audio_path = shuffled(alternatives, random)[0];
      const id = question.audio_path.match(/([a-zA-Z0-9_-]+)\.wav$/)[1];
      question.stimulus.audioAssetId = id;
      media.set(id, {
        id,
        path: "./" + question.audio_path,
        mimeType: "audio/wav",
      });
    }
    question.order = order;
    return question;
  });
  study.media = [...media.values()];
  validateStudy(study);
  return study;
}

// Research fixtures remain data; no automatic translation or experimental logic.
export function validateStudy(study) {
  if (
    study.schemaVersion !== 1 ||
    !isId(study.id) ||
    study.revision !== 1 ||
    !Array.isArray(study.questions) ||
    !study.questions.length
  ) {
    throw new AppError("invalidStudy");
  }
  const ids = new Set();
  study.questions.forEach((question, index) => {
    const mode = question.presentationMode;
    const options = question.options;
    if (
      !isId(question.id) ||
      ids.has(question.id) ||
      question.order !== index ||
      !question.prompt?.trim() ||
      !["audio_only", "text_only", "both"].includes(mode) ||
      !Array.isArray(options) ||
      options.length < 2 ||
      options.length > 4 ||
      options.some((option) => !isId(option.id) || !option.text?.trim()) ||
      new Set(options.map((option) => option.id)).size !== options.length
    ) {
      throw new AppError("invalidStudy");
    }
    ids.add(question.id);
    if (mode !== "audio_only" && !question.stimulus?.text?.trim())
      throw new AppError("invalidStudy");
    if (
      mode !== "text_only" &&
      !study.media?.some(
        (asset) => asset.id === question.stimulus?.audioAssetId,
      )
    )
      throw new AppError("invalidStudy");
  });
}

export function progressFor(session, records, audioIds = []) {
  const savedItemIds = records
    .filter(
      (record) =>
        record.sessionId === session.id &&
        (session.role !== "speaker" || audioIds.includes(record.audioAssetId)),
    )
    .map((record) =>
      session.role === "speaker" ? record.sentenceId : record.questionId,
    );
  return {
    session,
    requiredItemIds: session.requiredItemIds,
    savedItemIds: session.requiredItemIds.filter((id) =>
      savedItemIds.includes(id),
    ),
    nextItemId:
      session.requiredItemIds.find((id) => !savedItemIds.includes(id)) || null,
  };
}
