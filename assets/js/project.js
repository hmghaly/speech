import { store } from "./storage.js";
import {
  AppError,
  uid,
  timestamp,
  parseSentences,
  validateProject,
} from "./data.js";
import { setup, bind, textNode, content, showError } from "./ui.js";

const form = document.querySelector("#project-form");
const fileInput = document.querySelector("#sentence-file");
const language = document.querySelector("#project-language");
const error = document.querySelector("#form-error");
const save = document.querySelector("#save-project");
let parsed;
let parseFailure;
let fileGeneration = 0;
let pendingProject;
let busy = false;
let errorField;

// Keep one error beside its field; storage/bootstrap failures belong beside Save.
function showProjectError(failure) {
  const fields = {
    titleRequired: form.elements.title,
    titleTooLong: form.elements.title,
    referenceTooLong: form.elements.reference,
    languageInvalid: language,
    instructionsTooLong:
      [...form.elements.instructions.value].length > 2000
        ? form.elements.instructions
        : form.elements.description,
  };
  errorField =
    fields[failure.code] ||
    (/^(file|format|encoding|header|row|sentence|tooMany|control)/.test(
      failure.code || "",
    )
      ? fileInput
      : null);
  error.classList.toggle("field-error", Boolean(errorField));
  if (errorField) {
    const disclosure = errorField.closest("details");
    if (disclosure) disclosure.open = true;
    errorField.insertAdjacentElement("afterend", error);
    errorField.setAttribute("aria-invalid", "true");
    const descriptions = new Set(
      (errorField.getAttribute("aria-describedby") || "")
        .split(" ")
        .filter(Boolean),
    );
    descriptions.add(error.id);
    errorField.setAttribute("aria-describedby", [...descriptions].join(" "));
  } else form.querySelector(".form-footer").prepend(error);
  showError(error, failure, { focus: false });
  error.focus({ preventScroll: true });
  const bounds = error.getBoundingClientRect();
  if (bounds.top < 24 || bounds.bottom > innerHeight - 24) {
    error.scrollIntoView({ block: "center", behavior: "instant" });
  }
}

// Editing the affected field removes its stale message without clearing unrelated copy.
function clearError() {
  error.hidden = true;
  error.textContent = "";
  if (errorField) {
    errorField.removeAttribute("aria-invalid");
    const descriptions = (errorField.getAttribute("aria-describedby") || "")
      .split(" ")
      .filter((id) => id && id !== error.id);
    if (descriptions.length)
      errorField.setAttribute("aria-describedby", descriptions.join(" "));
    else errorField.removeAttribute("aria-describedby");
  }
  errorField = null;
}

// Discards outdated file reads if a second file is selected before the first finishes.
async function preview() {
  const generation = ++fileGeneration;
  parsed = pendingProject = parseFailure = null;
  clearError();
  document.querySelector("#file-preview").hidden = true;
  const file = fileInput.files[0];
  if (!file) return;
  save.disabled = true;
  try {
    if (file.size > 1024 * 1024) throw new AppError("fileTooLarge");
    const bytes = await file.arrayBuffer();
    if (generation !== fileGeneration) return;
    parsed = parseSentences(bytes, file.name, language.value);
    bind(document.querySelector("#preview-summary"), "create.previewCount", {
      count: parsed.sentences.length,
      blanks: parsed.importSummary.blankLines,
      duplicates: parsed.importSummary.duplicateLines.length,
    });
    document.querySelector("#duplicate-note").hidden =
      !parsed.importSummary.duplicateLines.length;
    const rows = document.querySelector("#preview-sentences");
    rows.replaceChildren();
    parsed.sentences.slice(0, 5).forEach((sentence) => {
      const row = document.createElement("li");
      row.append(
        textNode(
          "span",
          "create.sourceLine",
          { line: sentence.sourceLine },
          "ui-preview-" + sentence.id + "-line",
          "ui-preview-{id}-line",
        ),
        content("p", sentence.text, "ui-preview-" + sentence.id + "-text"),
      );
      rows.append(row);
    });
    document.querySelector("#file-preview").hidden = false;
  } catch (failure) {
    if (generation === fileGeneration) {
      parseFailure = failure;
      showProjectError(failure);
    }
  } finally {
    if (generation === fileGeneration) save.disabled = false;
  }
}

fileInput.addEventListener("change", preview);
language.addEventListener("change", preview);
form.addEventListener("input", (event) => {
  if (event.target !== fileInput) pendingProject = null;
  if (!errorField || event.target === errorField) clearError();
});

// Freeze one payload per save attempt; retries use the same project/operation identity.
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  clearError();
  try {
    if (!parsed) throw parseFailure || new AppError("fileRequired");
    pendingProject ||= {
      schemaVersion: 1,
      id: uid(),
      revision: 1,
      createdAt: timestamp(),
      title: form.elements.title.value.trim(),
      stimulusLanguage: language.value,
      instructions: form.elements.instructions.value.trim(),
      metadata: {
        description: form.elements.description.value.trim(),
        researcherReference: form.elements.reference.value.trim(),
      },
      ...parsed,
    };
    validateProject(pendingProject);
    busy = true;
    for (const field of form.elements) field.disabled = true;
    bind(save, "create.saving");
    const receipt = await store.createProject(pendingProject, {
      operationId: pendingProject.id,
    });
    location.href =
      "./record.html?project=" + encodeURIComponent(receipt.project.id);
  } catch (failure) {
    showProjectError(failure);
  } finally {
    busy = false;
    for (const field of form.elements) field.disabled = false;
    bind(save, "create.save");
  }
});

try {
  await setup(store);
} catch (failure) {
  showProjectError(failure);
  save.disabled = true;
}
