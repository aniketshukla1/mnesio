// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkGfm from "remark-gfm";

// Deploy target. The site serves at the **root** (`/`). That requires a
// root-serving Pages target — either a custom domain, or an org/user site repo
// named `mnesio.github.io` — rather than the `github.com/mnesio/mnesio` *project*
// path, which GitHub would otherwise serve under `/mnesio/`.
//
// To serve under a project sub-path instead, set `BASE = "/mnesio"`; the
// `remarkBaseLinks` plugin below then prefixes every internal link so nothing
// 404s. For a custom domain, also set `site` to that origin and add the domain
// in Settings → Pages → Custom domain (which writes a CNAME file).
const BASE = "";

/**
 * Prefix root-absolute markdown links with the deploy `base`.
 *
 * Astro rewrites the links *it* generates (assets, the sidebar), but not
 * hand-written `[text](/concepts/foo/)` links in content — those would 404 on a
 * project-path deploy. Rewriting them here keeps the content portable: prose
 * stays written against the site root, and switching between a root and a
 * sub-path deploy is a one-line change to `BASE` rather than an edit to every
 * link. A no-op when `BASE` is empty (root deploy).
 *
 * Walks the mdast directly so this needs no extra dependency.
 */
function remarkBaseLinks() {
  const prefix = BASE.replace(/\/$/, "");
  return (tree) => {
    if (!prefix) return;
    const visit = (node) => {
      if (node.type === "link" && typeof node.url === "string") {
        const u = node.url;
        // Internal, root-absolute, not already prefixed, not protocol-relative.
        if (u.startsWith("/") && !u.startsWith("//") && !u.startsWith(`${prefix}/`)) {
          node.url = prefix + u;
        }
      }
      for (const child of node.children ?? []) visit(child);
    };
    visit(tree);
  };
}

export default defineConfig({
  site: "https://mnesio.github.io",
  base: BASE || "/",
  // GFM (tables, strikethrough, task lists) for both .md and .mdx. The MDX
  // integration inherits this markdown config by default.
  markdown: {
    remarkPlugins: [remarkGfm, remarkBaseLinks],
  },
  integrations: [
    starlight({
      title: "mnesio",
      tagline: "A memory that gets verifiably better.",
      description:
        "A self-improving long-term memory layer for AI agents — append-only, bi-temporal, erasable, and verifiably better over time.",
      logo: { src: "./src/assets/logo.svg", alt: "mnesio" },
      customCss: ["./src/styles/custom.css"],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/mnesio/mnesio",
        },
      ],
      editLink: {
        baseUrl: "https://github.com/mnesio/mnesio/edit/main/website/",
      },
      lastUpdated: true,
      sidebar: [
        {
          label: "Start",
          items: [
            { label: "Why mnesio", slug: "start/why" },
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
            { label: "How mnesio differs", slug: "reference/comparison" },
            { label: "Benchmarks", slug: "reference/benchmarks" },
          ],
        },
      ],
    }),
  ],
});
