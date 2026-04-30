// pcm-recorder-worklet.js — captures Float32 audio frames and posts them
// to the main thread for WAV-encoding when recording stops.
// Loaded via audioContext.audioWorklet.addModule('/pcm-recorder-worklet.js').
class PCMRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    // input[0] is mono channel Float32Array; copy + post to main
    const ch0 = input[0];
    if (ch0 && ch0.length > 0) {
      this.port.postMessage(ch0.slice());
    }
    return true;
  }
}
registerProcessor("pcm-recorder", PCMRecorderProcessor);
