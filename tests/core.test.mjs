import test from "node:test";
import assert from "node:assert/strict";
import {
  parseSentences,
  validateStudy,
  progressFor,
} from "../assets/js/data.js";
import { Recorder } from "../assets/js/recorder.js";

const bytes = (text) => new TextEncoder().encode(text);

test("UTF-8 import preserves order and duplicate sentences with distinct IDs", () => {
  const result = parseSentences(
    bytes("\uFEFF你好。\r\n\r\n你好。\r\n再见。\r\n"),
    "sentences.txt",
  );
  assert.deepEqual(
    result.sentences.map((row) => row.text),
    ["你好。", "你好。", "再见。"],
  );
  assert.equal(new Set(result.sentences.map((row) => row.id)).size, 3);
  assert.equal(result.importSummary.blankLines, 1);
  assert.deepEqual(result.importSummary.duplicateLines, [3]);
});

test("TSV parses references and rejects malformed, empty, and invalid UTF-8 files", () => {
  const result = parseSentences(
    bytes("text\treference\n你好。\ts1"),
    "sentences.tsv",
  );
  assert.equal(result.sentences[0].sourceReference, "s1");
  assert.throws(() => parseSentences(bytes("text\n"), "empty.tsv"), {
    code: "fileEmpty",
  });
  assert.throws(
    () => parseSentences(bytes("text\treference\n你好。"), "bad.tsv"),
    { code: "rowMalformed" },
  );
  assert.throws(() => parseSentences(new Uint8Array([0xff]), "bad.txt"), {
    code: "encodingInvalid",
  });
});

test("speaker progress requires both the session association and the saved binary", () => {
  const session = {
    id: "session",
    role: "speaker",
    requiredItemIds: ["s1", "s2"],
  };
  const records = [
    { sessionId: "session", sentenceId: "s1", audioAssetId: "audio1" },
    { sessionId: "other", sentenceId: "s2", audioAssetId: "audio2" },
  ];
  assert.equal(progressFor(session, records, []).savedItemIds.length, 0);
  assert.deepEqual(progressFor(session, records, ["audio1"]).savedItemIds, [
    "s1",
  ]);
  assert.equal(progressFor(session, records, ["audio1"]).nextItemId, "s2");
});

test("questions require a valid presentation mode and two to four distinct options", () => {
  const question = {
    id: "q",
    order: 0,
    presentationMode: "text_only",
    prompt: "Who?",
    stimulus: { text: "A sentence." },
    options: [
      { id: "a", text: "One" },
      { id: "b", text: "Two" },
    ],
  };
  const study = {
    schemaVersion: 1,
    id: "study",
    revision: 1,
    questions: [question],
  };
  assert.doesNotThrow(() => validateStudy(study));
  const fourOptions = [
    ...question.options,
    { id: "c", text: "Three" },
    { id: "d", text: "Four" },
  ];
  assert.doesNotThrow(() =>
    validateStudy({
      ...study,
      questions: [{ ...question, options: fourOptions }],
    }),
  );
  assert.throws(
    () =>
      validateStudy({
        ...study,
        questions: [
          { ...question, options: [...fourOptions, { id: "e", text: "Five" }] },
        ],
      }),
    { code: "invalidStudy" },
  );
  assert.throws(
    () =>
      validateStudy({
        ...study,
        questions: [{ ...question, options: [question.options[0]] }],
      }),
    { code: "invalidStudy" },
  );
});

test("recorder includes the final chunk and releases the microphone after stop", async () => {
  const track = new EventTarget();
  let stopped = false;
  track.stop = () => {
    stopped = true;
  };
  const originalMedia = globalThis.MediaRecorder;
  const originalSecure = globalThis.isSecureContext;
  const originalDevices = navigator.mediaDevices;
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: async () => ({ getTracks: () => [track] }) },
  });
  globalThis.isSecureContext = true;
  globalThis.MediaRecorder = class extends EventTarget {
    mimeType = "audio/webm;codecs=opus";
    static isTypeSupported() {
      return true;
    }
    start() {
      this.state = "recording";
    }
    stop() {
      this.state = "inactive";
      queueMicrotask(() => {
        const event = new Event("dataavailable");
        event.data = new Blob(["final audio"]);
        this.dispatchEvent(event);
        this.dispatchEvent(new Event("stop"));
      });
    }
  };
  try {
    const recorder = new Recorder();
    await recorder.start(
      () => {},
      () => assert.fail("Unexpected interruption"),
      () => {},
    );
    const take = await recorder.stop();
    assert.equal(await take.blob.text(), "final audio");
    assert.equal(take.mimeType, "audio/webm;codecs=opus");
    assert.equal(stopped, true);
  } finally {
    globalThis.MediaRecorder = originalMedia;
    globalThis.isSecureContext = originalSecure;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: originalDevices,
    });
  }
});

// Guard the translation contract as pages and changing control states are extended.
test("interface translations use stable DOM IDs with matching English and Chinese entries", async () => {
  const { readFile } = await import("node:fs/promises");
  const dictionary = JSON.parse(
    await readFile(
      new URL("../assets/i18n/messages.json", import.meta.url),
      "utf8",
    ),
  );
  assert.equal(dictionary.defaultLocale, "zh-Hans");
  for (const entry of [
    ...Object.values(dictionary.messages),
    ...Object.values(dictionary.templates),
  ]) {
    assert.ok(entry.en && entry["zh-Hans"]);
    if (typeof entry.en === "object")
      assert.deepEqual(
        Object.keys(entry.en).sort(),
        Object.keys(entry["zh-Hans"]).sort(),
      );
  }
  for (const page of ["index", "project", "record", "listener"]) {
    const html = await readFile(
      new URL("../" + page + ".html", import.meta.url),
      "utf8",
    );
    assert.match(html, /<html lang="zh-Hans" translate="no">/);
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
    assert.equal(new Set(ids).size, ids.length, page + ": duplicate ID");
    for (const [tag] of html.matchAll(/<[^!][^>]*data-i18n[^>]*>/g)) {
      const id = tag.match(/\bid="([^"]+)"/)?.[1];
      assert.ok(id, page + ": missing translation ID");
      const state = tag.match(/data-i18n-state="([^"]+)"/)?.[1];
      for (const [, key] of tag.matchAll(
        /data-i18n(?:-(?:placeholder|aria-label))?="([^"]+)"/g,
      ))
        assert.equal(key, id);
      const entry = dictionary.messages[id];
      for (const language of ["en", "zh-Hans"])
        assert.equal(
          typeof entry?.[language] === "object"
            ? typeof entry[language][state]
            : typeof entry?.[language],
          "string",
          id,
        );
    }
  }
});
