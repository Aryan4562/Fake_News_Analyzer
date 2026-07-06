import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { inspectAttr } from "kimi-plugin-inspect-react"

export default defineConfig({
  base: "./",
  plugins: [inspectAttr(), react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: true,        // accept requests from any host (needed for Cloudflare)
    port: 5173,
    strictPort: true,
    allowedHosts: "all" // allow any external host (dev only)
  }
})
