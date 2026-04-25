import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: { port: 5178, host: "127.0.0.1" },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: { target: "es2022", sourcemap: true },
});
