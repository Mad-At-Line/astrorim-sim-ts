/// <reference lib="webworker" />
/** Runs the simulator off the main thread so the sliders stay responsive. */

import { handle } from "./engine";

self.onmessage = (ev: MessageEvent) => {
  const { message, transfer } = handle(ev.data);
  (self as unknown as Worker).postMessage(message, transfer);
};
