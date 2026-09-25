import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// `npm run build`        -> dist/ for GitHub Pages
// `npm run build:single` -> dist-single/index.html, one self-contained file
export default defineConfig(({ mode }) => ({
  root: "web",
  base: "./",
  plugins: mode === "single" ? [viteSingleFile()] : [],
  worker: { format: "es" },
  build: {
    outDir: mode === "single" ? "../dist-single" : "../dist",
    emptyOutDir: true,
  },
}));
