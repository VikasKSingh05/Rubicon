// Persist uploaded evidence under STORAGE_DIR/<assessmentId>/<kind><ext>.
// Returns the relative paths stored on the Assessment document so the
// download endpoints and the async pipeline can resolve them later.
import path from "node:path";
import { mkdir, writeFile } from "node:fs/promises";
import { config } from "../config.js";

export async function persistFiles(assessmentId, files) {
  const dir = path.join(config.storageDir, assessmentId);
  await mkdir(dir, { recursive: true });
  const stored = {};
  for (const [kind, file] of Object.entries(files)) {
    const ext = path.extname(file.originalname).toLowerCase();
    const rel = path.join(assessmentId, `${kind}${ext}`);
    await writeFile(path.join(config.storageDir, rel), file.buffer);
    stored[kind] = rel;
  }
  return stored;
}

export function resolveStoredPath(relativePath) {
  return path.join(config.storageDir, relativePath);
}
