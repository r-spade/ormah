import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ui": "http://localhost:8787",
      "/agent": "http://localhost:8787",
      "/admin": "http://localhost:8787",
    },
  },
  build: {
    outDir: "../src/ormah/ui_dist",
    emptyOutDir: true,
  },
});
