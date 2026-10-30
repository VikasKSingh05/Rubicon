// Stage the committed E2E artifacts into the paths the services actually read.
//
// The ai-engine container mounts ./ai-engine/data and loads its model from
// /app/data/models, but ai-engine/data is gitignored (it holds the full Houston
// corpus and real training weights). Copying the small fixtures in here means CI
// and a fresh clone can boot the real inference path without training anything.
import { cpSync, existsSync, mkdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..");

const MODEL_FILES = ["model.pt", "pca.pkl", "metrics.json"];
const SCAN_FILES = ["hsi.tiff", "lidar.las"];

function copyInto(srcDir, destDir, files) {
  const copied = [];
  for (const file of files) {
    const src = path.join(srcDir, file);
    if (!existsSync(src)) {
      throw new Error(`missing committed fixture: ${path.relative(repoRoot, src)}`);
    }
    const dest = path.join(destDir, file);
    const current = existsSync(dest) ? statSync(dest).size : null;
    if (current === statSync(src).size) continue;
    mkdirSync(destDir, { recursive: true });
    cpSync(src, dest);
    copied.push(path.relative(repoRoot, dest));
  }
  return copied;
}

const modelCopied = copyInto(
  path.join(here, "fixtures", "model"),
  path.join(repoRoot, "ai-engine", "data", "models"),
  MODEL_FILES,
);

// The scan pair is only needed for engine-only runs; the E2E harness uploads it
// straight from e2e/fixtures/scan. Stage it too so a local `predict` against
// ai-engine/data works after a clone.
const scanCopied = copyInto(
  path.join(here, "fixtures", "scan"),
  path.join(repoRoot, "ai-engine", "data", "houston2013"),
  SCAN_FILES,
);

const staged = [...modelCopied, ...scanCopied];
console.log(staged.length ? `staged:\n  ${staged.join("\n  ")}` : "fixtures already staged");