import { AppError } from "./data.js";

// Owns microphone resources; the page decides which sentence is being recorded.
export class Recorder {
  stream = null;
  recorder = null;
  timer = null;
  generation = 0;
  resolveStop = null;
  rejectStop = null;

  static supported() {
    return (
      globalThis.isSecureContext &&
      navigator.mediaDevices?.getUserMedia &&
      globalThis.MediaRecorder
    );
  }

  async start(onTime, onInterrupted, onLimit) {
    if (!Recorder.supported())
      throw new AppError(
        globalThis.isSecureContext ? "unsupported" : "insecure",
      );
    if (this.stream || this.recorder) throw new AppError("deviceBusy");
    const generation = ++this.generation;
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      throw new AppError(
        {
          NotAllowedError: "permissionDenied",
          NotFoundError: "deviceMissing",
          NotReadableError: "deviceBusy",
        }[error.name] || "deviceBusy",
      );
    }
    if (generation !== this.generation) {
      stream.getTracks().forEach((track) => track.stop());
      throw new AppError("interrupted");
    }
    this.stream = stream;
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/mp4",
      "audio/ogg;codecs=opus",
      "",
    ];
    for (const mimeType of candidates) {
      try {
        if (mimeType && !MediaRecorder.isTypeSupported(mimeType)) continue;
        this.recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        break;
      } catch {
        this.recorder = null;
      }
    }
    if (
      !this.recorder ||
      !/^audio\/(webm|mp4|ogg)(;|$)/i.test(this.recorder.mimeType)
    ) {
      this.cancel();
      throw new AppError("unsupportedMedia");
    }
    const recorder = this.recorder;
    const chunks = [];
    const started = performance.now();
    let intentionalStop = false;
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener(
      "stop",
      () => {
        if (generation !== this.generation) return;
        const durationMs = Math.round(performance.now() - started);
        const blob = new Blob(chunks, { type: recorder.mimeType });
        const resolve = this.resolveStop;
        const reject = this.rejectStop;
        this.release();
        if (!intentionalStop) {
          reject?.(new AppError("interrupted"));
          onInterrupted();
        } else if (!blob.size) reject?.(new AppError("emptyTake"));
        else resolve?.({ blob, durationMs, mimeType: blob.type });
      },
      { once: true },
    );
    recorder.addEventListener(
      "error",
      () => {
        if (generation !== this.generation) return;
        this.rejectStop?.(new AppError("interrupted"));
        this.cancel();
        onInterrupted();
      },
      { once: true },
    );
    for (const track of stream.getTracks()) {
      track.addEventListener(
        "ended",
        () => {
          if (generation === this.generation && !intentionalStop) {
            this.cancel();
            onInterrupted();
          }
        },
        { once: true },
      );
    }
    this.stopRecording = () => {
      intentionalStop = true;
      recorder.stop();
    };
    try {
      recorder.start();
    } catch {
      this.cancel();
      throw new AppError("unsupportedMedia");
    }
    this.timer = setInterval(() => {
      const elapsed = Math.floor((performance.now() - started) / 1000);
      onTime(elapsed);
      if (elapsed >= 120) {
        clearInterval(this.timer);
        onLimit();
      }
    }, 250);
    onTime(0);
  }

  // The stop event follows the final dataavailable event; never build the Blob earlier.
  stop() {
    if (this.recorder?.state !== "recording")
      return Promise.reject(new AppError("interrupted"));
    return new Promise((resolve, reject) => {
      this.resolveStop = resolve;
      this.rejectStop = reject;
      this.stopRecording();
    });
  }

  // Stop tracks after finalization; a subsequent sentence acquires a fresh stream.
  release() {
    clearInterval(this.timer);
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = this.recorder = this.timer = null;
    this.resolveStop = this.rejectStop = null;
  }

  // Invalidates pending permission requests as well as active captures.
  cancel() {
    this.generation++;
    this.rejectStop?.(new AppError("interrupted"));
    if (this.recorder?.state === "recording") this.recorder.stop();
    this.release();
  }
}
