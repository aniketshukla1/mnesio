// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkGfm from "remark-gfm";

// NOTE: set `site` to your deployed origin. For a GitHub Pages *project* site
// (https://<user>.github.io/mneme) also uncomment `base: "/mneme"`.
export default defineConfig({
  site: "https://mneme.dev",
  // base: "/mneme",
  // GFM (tables, strikethrough, task lists) for both .md and .mdx. The MDX
  // integration inherits this markdown config by default.
  markdown: {
    remarkPlugins: [remarkGfm],
  },
  integrations: [
    starlight({
      title: "mneme",
      tagline: "A memory that gets verifiably better.",
      description:
        "A self-improving long-term memory layer for AI agents — append-only, bi-temporal, erasable, and verifiably better over time.",
      logo: { src: "./src/assets/logo.svg", alt: "mneme" },
      customCss: ["./src/styles/custom.css"],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/aniketshukla1/mneme",
        },
      ],
      editLink: {
        baseUrl: "https://github.com/aniketshukla1/mneme/edit/main/website/",
      },
      lastUpdated: true,
      sidebar: [
        {
          label: "Start",
          items: [
            { label: "Why mneme", slug: "start/why" },
            { label: "Getting started", slug: "start/getting-started" },
            { label: "Quickstart (MCP)", slug: "start/quickstart" },
          ],
        },
        {
          label: "Concepts",
          items: [
            { label: "Architecture", slug: "concepts/architecture" },
            { label: "The seven hard rules", slug: "concepts/hard-rules" },
            { label: "The procedural wedge", slug: "concepts/wedge" },
          ],
        },
        {
          label: "Guides",
          items: [
            { label: "Agent integration (MCP)", slug: "guides/integration" },
            { label: "OpenClaw & Hermes", slug: "guides/openclaw-hermes" },
            { label: "KV cartridges (GPU)", slug: "guides/kv-cartridges" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "Benchmarks", slug: "reference/benchmarks" },
            { label: "Roadmap", slug: "reference/roadmap" },
          ],
        },
      ],
    }),
  ],
});
