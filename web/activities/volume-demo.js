export class VolumeDemo {
  constructor({ slider, value, status, toggleButton }) {
    this.slider = slider;
    this.value = value;
    this.status = status;
    this.toggleButton = toggleButton;
    this.audioContext = null;
    this.audioOscillator = null;
    this.audioGain = null;

    slider.addEventListener("input", () => this.setVolume(slider.value));
    toggleButton.addEventListener("click", () => this.toggle());
  }

  setVolume(value) {
    const volume = Math.max(0, Math.min(100, Math.round(value)));
    this.slider.value = String(volume);
    this.value.textContent = String(volume);
    if (this.audioGain && this.audioContext) {
      this.audioGain.gain.setTargetAtTime(
        volume / 100 * 0.18,
        this.audioContext.currentTime,
        0.04
      );
    }
  }

  setStatus(message) {
    this.status.textContent = message;
  }

  stop() {
    if (this.audioOscillator) {
      this.audioOscillator.stop();
      this.audioOscillator.disconnect();
      this.audioOscillator = null;
    }
    this.audioGain?.disconnect();
    this.audioGain = null;

    if (this.audioContext) {
      void this.audioContext.close();
      this.audioContext = null;
    }
    this.toggleButton.textContent = "Enable demo sound";
    this.setStatus("Sound is off. Enable it to hear the gesture-controlled tone.");
  }

  async toggle() {
    if (this.audioContext) {
      this.stop();
      return;
    }

    try {
      const AudioContextConstructor = window.AudioContext ?? window.webkitAudioContext;
      if (!AudioContextConstructor) {
        throw new Error("This browser does not support Web Audio.");
      }

      const context = new AudioContextConstructor();
      this.audioContext = context;
      await context.resume();
      if (this.audioContext !== context) {
        return;
      }

      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = 440;
      gain.gain.value = Number(this.slider.value) / 100 * 0.18;
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start();
      this.audioOscillator = oscillator;
      this.audioGain = gain;
      this.toggleButton.textContent = "Stop demo sound";
      this.setStatus("Tone is playing. Use your hand or the slider to change its volume.");
    } catch (error) {
      this.stop();
      this.setStatus(error instanceof Error ? error.message : "The demo sound could not be started.");
    }
  }
}
