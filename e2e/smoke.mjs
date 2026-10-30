/**
 * Rubicon end-to-end smoke test.
 *
 * Drives the real, running stack (backend + ai-engine + agent-service + frontend)
 * through every user-facing flow and asserts the responses against the contracts
 * in docs/api-contracts.md. Zero dependencies: Node's global fetch/FormData only.
 *
 * Usage:
 *   node e2e/smoke.mjs                 # against http://localhost:*
 *   E2E_BACKEND=https://api.example.com node e2e/smoke.mjs
 *
 * Exits 0 when every check passes, 1 otherwise (CI gate).
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(__dirname, "fixtures", "scan");
const HSI_FIXTURE = join(FIXTURES, "hsi.tiff");
const LIDAR_FIXTURE = join(FIXTURES, "lidar.las");

const BACKEND = process.env.E2E_BACKEND || "http://localhost:4000";
const ENGINE = process.env.E2E_ENGINE || "http://localhost:8000";
const AGENT = process.env.E2E_AGENT || "http://localhost:8001";
const FRONTEND = process.env.E2E_FRONTEND || "http://localhost:3000";

const jar = new Map();

let passed = 0;
const failures = [];

function cookieHeader() {
  return [...jar.entries()].map(([k, v]) => `${k}=${v}`).join("; ");
}

function captureCookies(res) {
  const raw = res.headers.getSetCookie?.() ?? [];
  for (const entry of raw) {
    const [pair] = entry.split(";");
    const idx = pair.indexOf("=");
    const name = pair.slice(0, idx).trim();
    const value = pair.slice(idx + 1).trim();
    if (value === "") jar.delete(name);
    else jar.set(name, value);
  }
}

async function call(url, { method = "GET", body, token, headers = {}, raw = false } = {}) {
  const h = { ...headers };
  if (jar.size) h.cookie = cookieHeader();
  if (token) h.authorization = `Bearer ${token}`;
  const res = await fetch(url, { method, body, headers: h, redirect: "manual" });
  captureCookies(res);
  if (raw) return { res, bytes: Buffer.from(await res.arrayBuffer()) };
  const text = await res.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = null;
  }
  return { res, json, text };
}

function check(name, ok, detail) {
  if (ok) {
    passed += 1;
    console.log(`  ok   ${name}`);
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ""}`);
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

const section = (title) => console.log(`\n[${title}]`);

function scanForm() {
  const hsi = new Blob([readFileSync(HSI_FIXTURE)], { type: "image/tiff" });
  const lidar = new Blob([readFileSync(LIDAR_FIXTURE)], {
    type: "application/octet-stream",
  });
  const form = new FormData();
  form.append("hsi", hsi, "houston_sample.tif");
  form.append("lidar", lidar, "houston_sample.las");
  return form;
}

const uniqueEmail = `e2e-${Date.now()}-${Math.floor(Math.random() * 1e6)}@rubicon.test`;
const PASSWORD = "e2e-initial-pass";

async function run() {
  console.log(`Rubicon e2e smoke\n  backend=${BACKEND} engine=${ENGINE} agent=${AGENT} frontend=${FRONTEND}`);

  section("1. health / liveness");
  const engineHealth = await call(`${ENGINE}/health`);
  check(
    "ai-engine /health reports the served artifact",
    engineHealth.res.ok && engineHealth.json?.service === "ai-engine" && engineHealth.json?.model_loaded === true,
    `status=${engineHealth.res.status} body=${engineHealth.text.slice(0, 160)}`,
  );

  const backendHealth = await call(`${BACKEND}/health`);
  check(
    "backend /health reports mongo connectivity",
    backendHealth.res.ok && backendHealth.json?.service === "backend" && backendHealth.json?.mongo === "connected",
    `status=${backendHealth.res.status} body=${backendHealth.text.slice(0, 160)}`,
  );

  const agentHealth = await call(`${AGENT}/health`);
  check("agent-service /health", agentHealth.res.ok && agentHealth.json?.service === "agent-service",
    `status=${agentHealth.res.status}`);

  const frontendHealth = await call(`${FRONTEND}/health`);
  check("frontend /health serves the built SPA", frontendHealth.res.ok,
    `status=${frontendHealth.res.status}`);

  const frontendIndex = await call(`${FRONTEND}/`, { raw: true });
  check(
    "frontend index.html is served",
    frontendIndex.res.ok && frontendIndex.bytes.toString("utf8").includes("<div id=\"root\">"),
    `status=${frontendIndex.res.status}`,
  );

  section("2. ai-engine /predict (real model path)");
  const enginePredict = await call(`${ENGINE}/predict`, { method: "POST", body: scanForm() });
  check(
    "engine /predict returns contract fields",
    enginePredict.res.ok &&
      typeof enginePredict.json?.prediction === "string" &&
      typeof enginePredict.json?.confidence === "number" &&
      enginePredict.json?.confidence > 0 &&
      enginePredict.json?.confidence <= 1 &&
      enginePredict.json?.geojson_polygon?.type === "Polygon" &&
      typeof enginePredict.json?.model_version === "string",
    `status=${enginePredict.res.status} body=${enginePredict.text.slice(0, 200)}`,
  );
  check(
    "engine served the trained model (not the pre-model stub)",
    enginePredict.json?.model_version && enginePredict.json.model_version !== "stub-v0",
    `model_version=${enginePredict.json?.model_version}`,
  );

  const badMagic = new FormData();
  badMagic.append("hsi", new Blob([readFileSync(LIDAR_FIXTURE)]), "not-a-tiff.tif");
  badMagic.append("lidar", new Blob([readFileSync(LIDAR_FIXTURE)]), "points.las");
  const engineBad = await call(`${ENGINE}/predict`, { method: "POST", body: badMagic });
  check("engine rejects a non-TIFF hsi with 400", engineBad.res.status === 400,
    `status=${engineBad.res.status}`);

  section("3. auth: register / login / refresh / change-password / logout");
  const noAuth = await call(`${BACKEND}/assessments`);
  check("protected route rejects anonymous access (401)", noAuth.res.status === 401,
    `status=${noAuth.res.status}`);

  const register = await call(`${BACKEND}/auth/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email: uniqueEmail, password: PASSWORD }),
  });
  check("register issues a session (201)", register.res.status === 201 && !!register.json?.token,
    `status=${register.res.status} body=${register.text.slice(0, 160)}`);

  const duplicate = await call(`${BACKEND}/auth/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email: uniqueEmail, password: PASSWORD }),
  });
  check("duplicate register is rejected (409)", duplicate.res.status === 409,
    `status=${duplicate.res.status}`);

  const login = await call(`${BACKEND}/auth/login`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email: uniqueEmail, password: "wrong-password" }),
  });
  check("login with a wrong password is rejected (401)", login.res.status === 401,
    `status=${login.res.status}`);

  const loginOk = await call(`${BACKEND}/auth/login`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email: uniqueEmail, password: PASSWORD }),
  });
  check("login issues access token + refresh cookie",
    loginOk.res.ok && !!loginOk.json?.token && jar.has("rubicon_refresh"),
    `status=${loginOk.res.status} cookies=${[...jar.keys()].join(",")}`);
  let token = loginOk.json?.token;

  const firstRefresh = jar.get("rubicon_refresh");
  const refreshed = await call(`${BACKEND}/auth/refresh`, { method: "POST" });
  check("refresh rotates the cookie and mints a new access token",
    refreshed.res.ok && !!refreshed.json?.token && jar.has("rubicon_refresh") && jar.get("rubicon_refresh") !== firstRefresh,
    `status=${refreshed.res.status}`);
  token = refreshed.json?.token;

  const replayOld = await fetch(`${BACKEND}/auth/refresh`, {
    method: "POST",
    headers: { cookie: `rubicon_refresh=${firstRefresh}` },
  });
  check("a reused (rotated) refresh cookie is rejected (401)", replayOld.status === 401,
    `status=${replayOld.status}`);

  const changed = await call(`${BACKEND}/auth/change-password`, {
    method: "POST",
    token,
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ currentPassword: PASSWORD, newPassword: "e2e-rotated-pass" }),
  });
  check("change-password succeeds (bearer, { ok: true })",
    changed.res.ok && changed.json?.ok === true,
    `status=${changed.res.status} body=${changed.text.slice(0, 160)}`);

  // The session that changed the password must survive; only other devices die.
  const ownSession = await call(`${BACKEND}/auth/refresh`, { method: "POST" });
  check("the caller's own refresh session survives a password change",
    ownSession.res.ok && !!ownSession.json?.token,
    `status=${ownSession.res.status}`);
  token = ownSession.json?.token;

  const oldLogin = await call(`${BACKEND}/auth/login`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email: uniqueEmail, password: PASSWORD }),
  });
  check("the previous password no longer works (401)", oldLogin.res.status === 401,
    `status=${oldLogin.res.status}`);

  const newLogin = await call(`${BACKEND}/auth/login`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email: uniqueEmail, password: "e2e-rotated-pass" }),
  });
  check("the new password works", newLogin.res.ok && !!newLogin.json?.token,
    `status=${newLogin.res.status}`);
  token = newLogin.json?.token;

  section("4. upload → inference → proof pipeline");
  const rejectedUpload = await call(`${BACKEND}/upload`, {
    method: "POST",
    token,
    body: (() => {
      const form = new FormData();
      form.append("hsi", new Blob([readFileSync(LIDAR_FIXTURE)]), "wrong.tif");
      form.append("lidar", new Blob([readFileSync(LIDAR_FIXTURE)]), "points.las");
      return form;
    })(),
  });
  check("upload rejects a non-TIFF hsi by magic bytes (400)", rejectedUpload.res.status === 400,
    `status=${rejectedUpload.res.status} body=${rejectedUpload.text.slice(0, 160)}`);

  const upload = await call(`${BACKEND}/upload`, { method: "POST", token, body: scanForm() });
  const created = upload.json ?? {};
  check("upload returns 201 with the created assessment",
    upload.res.status === 201 &&
      typeof created.assessmentId === "string" &&
      typeof created.prediction === "string" &&
      typeof created.confidence === "number" &&
      created.confidence > 0 && created.confidence <= 1 &&
      created.geojson_polygon?.type === "Polygon" &&
      !!created.links?.detail,
    `status=${upload.res.status} body=${upload.text.slice(0, 240)}`);
  check("upload ran the trained engine, not the backend stub",
    created.model_version && created.model_version !== "stub-v0",
    `model_version=${created.model_version}`);
  check("upload state is analyzed (simulated chain) or chain_logged (real chain)",
    created.state === "analyzed" || created.state === "chain_logged",
    `state=${created.state}`);

  const assessmentId = created.assessmentId;

  section("5. assessments list + detail");
  const list = await call(`${BACKEND}/assessments`, { token });
  const listed = (list.json?.assessments ?? []).find((a) => a.id === assessmentId);
  check("GET /assessments lists the new assessment with a map-marker envelope",
    list.res.ok && !!listed &&
      typeof listed.center?.lng === "number" && typeof listed.center?.lat === "number" &&
      ["none", "moderate", "severe"].includes(listed.severity) &&
      typeof listed.chainVerified === "boolean" &&
      typeof list.json?.total === "number",
    `status=${list.res.status}`);

  const detail = await call(`${BACKEND}/assessments/${assessmentId}`, { token });
  const d = detail.json ?? {};
  check("GET /assessments/:id returns the stored record + proof fields",
    detail.res.ok &&
      d.id === assessmentId &&
      typeof d.hsiCid === "string" && d.hsiCid.length > 20 &&
      typeof d.lidarCid === "string" &&
      d.filename?.hsi && d.filename?.lidar &&
      !!d.storage?.hsi && !!d.storage?.lidar &&
      typeof d.timestamps?.uploaded === "string",
    `status=${detail.res.status} body=${detail.text.slice(0, 200)}`);

  const foreign = await call(`${BACKEND}/assessments/${assessmentId}`);
  check("detail is not readable without a token (401)", foreign.res.status === 401,
    `status=${foreign.res.status}`);

  section("6. original-file downloads (owner-scoped)");
  const hsiBytes = readFileSync(HSI_FIXTURE);
  const download = await call(`${BACKEND}/files/${assessmentId}/hsi`, { token, raw: true });
  check("GET /files/:id/hsi returns the original HSI bytes",
    download.res.ok && download.bytes.equals(hsiBytes),
    `status=${download.res.status} got=${download.bytes.length} want=${hsiBytes.length}`);

  const lidarBytes = readFileSync(LIDAR_FIXTURE);
  const downloadLidar = await call(`${BACKEND}/files/${assessmentId}/lidar`, { token, raw: true });
  check("GET /files/:id/lidar returns the original LiDAR bytes",
    downloadLidar.res.ok && downloadLidar.bytes.equals(lidarBytes),
    `status=${downloadLidar.res.status} got=${downloadLidar.bytes.length} want=${lidarBytes.length}`);

  const badKind = await call(`${BACKEND}/files/${assessmentId}/secrets`, { token });
  check("unknown file kind is rejected (400)", badKind.res.status === 400,
    `status=${badKind.res.status}`);

  // Regression: a malformed id used to reject inside mongoose and crash the
  // backend process (unhandled rejection), taking the API down entirely.
  const malformed = await call(`${BACKEND}/files/zzz/hsi`, { token });
  check("malformed file id is a 404, not a crash", malformed.res.status === 404,
    `status=${malformed.res.status}`);

  const malformedDetail = await call(`${BACKEND}/assessments/zzz`, { token });
  check("malformed assessment id is a 404", malformedDetail.res.status === 404,
    `status=${malformedDetail.res.status}`);

  const stillAlive = await call(`${BACKEND}/health`);
  check("backend is still serving after malformed ids", stillAlive.res.ok,
    `status=${stillAlive.res.status}`);

  const traversal = await call(`${BACKEND}/files/${assessmentId}..%2F..%2F.env`, { token });
  check("path traversal in the file route is rejected (400)", traversal.res.status === 400,
    `status=${traversal.res.status}`);

  const otherEmail = `e2e-other-${Date.now()}@rubicon.test`;
  await call(`${BACKEND}/auth/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email: otherEmail, password: "other-user-pass" }),
  });
  const otherLogin = await call(`${BACKEND}/auth/login`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email: otherEmail, password: "other-user-pass" }),
  });
  const crossRead = await call(`${BACKEND}/assessments/${assessmentId}`, {
    token: otherLogin.json?.token,
  });
  check("another user cannot read the assessment (404)", crossRead.res.status === 404,
    `status=${crossRead.res.status}`);

  const crossDownload = await call(`${BACKEND}/files/${assessmentId}/hsi`, {
    token: otherLogin.json?.token,
  });
  check("another user cannot download the original (404)", crossDownload.res.status === 404,
    `status=${crossDownload.res.status}`);

  section("7. chain verification");
  const verify = await call(`${BACKEND}/chain/verify/${encodeURIComponent(d.hsiCid)}`, { token });
  check("GET /chain/verify/:cid answers with the proof shape",
    verify.res.ok && verify.json?.cid === d.hsiCid && typeof verify.json?.verified === "boolean",
    `status=${verify.res.status} body=${verify.text.slice(0, 200)}`);
  check("proof state matches the stored record",
    verify.json?.verified === d.chainVerified && (verify.json?.txHash ?? null) === (d.txHash ?? null),
    `verify=${JSON.stringify(verify.json)} record txHash=${d.txHash} chainVerified=${d.chainVerified}`);

  section("8. tool-grounded agent");
  const tools = await call(`${AGENT}/tools`);
  const toolNames = (tools.json?.tools ?? []).map((t) => t.name ?? t);
  check("GET /tools lists the grounded toolset",
    tools.res.ok && toolNames.includes("get_recent_assessments") && toolNames.includes("get_chain_status"),
    `status=${tools.res.status} tools=${JSON.stringify(toolNames)}`);

  const agentAnswer = await call(`${AGENT}/agent/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query: "How many assessments are logged and which are on-chain?", token }),
  });
  check("POST /agent/query answers from tools",
    agentAnswer.res.ok &&
      typeof agentAnswer.json?.answer === "string" && agentAnswer.json.answer.length > 0 &&
      Array.isArray(agentAnswer.json?.tool_calls),
    `status=${agentAnswer.res.status} body=${agentAnswer.text.slice(0, 200)}`);

  section("9. observability");
  const metrics = await call(`${BACKEND}/metrics`, { raw: true });
  const metricsText = metrics.bytes.toString("utf8");
  check("GET /metrics exposes the Prometheus counters",
    metrics.res.ok && metricsText.includes("rubicon_uploads_total") && metricsText.includes("rubicon_errors_total"),
    `status=${metrics.res.status}`);

  section("10. logout revokes the session");
  await call(`${BACKEND}/auth/logout`, { method: "POST" });
  const afterLogout = await call(`${BACKEND}/assessments`, { token });
  check("the access token is dead after logout (401)", afterLogout.res.status === 401,
    `status=${afterLogout.res.status}`);

  const afterLogoutRefresh = await fetch(`${BACKEND}/auth/refresh`, {
    method: "POST",
    headers: { cookie: cookieHeader() },
  });
  check("the refresh cookie is dead after logout (401)", afterLogoutRefresh.status === 401,
    `status=${afterLogoutRefresh.status}`);

  console.log(`\n${passed} checks passed, ${failures.length} failed`);
  if (failures.length) {
    console.log("\nFailures:");
    for (const f of failures) console.log(`  - ${f}`);
    process.exit(1);
  }
  process.exit(0);
}

run().catch((err) => {
  console.error("\nsmoke run crashed:", err?.message || err);
  if (err?.cause) console.error("cause:", err.cause.message);
  process.exit(1);
});