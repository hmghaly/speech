import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import * as data from "../assets/js/data.js";

const rows = JSON.parse(
  await readFile(new URL("../sample/sample.json", import.meta.url), "utf8"),
);

test("random draws keep question identity, source metadata and input data intact", () => {
  const original = structuredClone(rows);
  const source = data.normalizeSampleStudy(rows);
  const first = data.randomizeStudy(source, () => 0);
  const second = data.randomizeStudy(source, () => 0.999999);
  assert.notDeepEqual(
    first.questions.map((q) => q.id),
    second.questions.map((q) => q.id),
  );
  assert.deepEqual(
    first.questions.map((q) => q.id).sort(),
    source.questions.map((q) => q.id).sort(),
  );
  for (const [order, question] of first.questions.entries()) {
    const index = source.questions.findIndex((q) => q.id === question.id);
    assert.equal(question.order, order);
    assert.equal(question.original_sentence, rows[index].original_sentence);
    assert.equal(question.audio_path, rows[index].alt_audio_paths[1]);
    assert.equal(
      first.media.find((a) => a.id === question.stimulus.audioAssetId).path,
      "./" + question.audio_path,
    );
    assert.equal(
      second.questions.find((q) => q.id === question.id).audio_path,
      rows[index].alt_audio_paths[0],
    );
  }
  assert.deepEqual(rows, original);
  assert.deepEqual(data.normalizeSampleStudy(rows), source);
  data.validateStudy(first);
  data.validateStudy(second);
});

test("missing or empty alternate audio keeps the original; invalid paths are rejected", () => {
  for (const alternatives of [undefined, []]) {
    const row = { ...rows[0], alt_audio_paths: alternatives };
    const study = data.randomizeStudy(
      data.normalizeSampleStudy([row]),
      () => 0,
    );
    assert.equal(study.questions[0].audio_path, row.audio_path);
  }
  assert.throws(
    () =>
      data.randomizeStudy(
        data.normalizeSampleStudy([
          { ...rows[0], alt_audio_paths: ["../outside.wav"] },
        ]),
      ),
    { code: "invalidStudy" },
  );
});
