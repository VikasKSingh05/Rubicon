import { Router } from "express";
import { createHash, randomBytes } from "node:crypto";
import jwt from "jsonwebtoken";
import User from "../models/User.js";
import RefreshToken from "../models/RefreshToken.js";
import { config } from "../config.js";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { makeLogger } from "../middleware/logger.js";

const logger = makeLogger("auth");

const router = Router();

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MIN_PASSWORD_LENGTH = 8;
const REFRESH_COOKIE = "rubicon_refresh";
const REFRESH_TTL_MS = 7 * 24 * 3600 * 1000;

function hashRefresh(token) {
  return createHash("sha256").update(token).digest("hex");
}

function signAccess(user) {
  return jwt.sign({ sub: user._id.toString(), email: user.email }, config.jwtSecret, {
    expiresIn: config.accessTokenTtl,
  });
}

function publicUser(user) {
  return { id: user._id.toString(), email: user.email };
}

function refreshCookieOptions() {
  return {
    httpOnly: true,
    sameSite: "strict",
    path: "/auth",
    maxAge: REFRESH_TTL_MS,
    secure: config.isProd,
  };
}

async function issueRefresh(res, user) {
  const token = randomBytes(32).toString("hex");
  await RefreshToken.create({
    user: user._id,
    hash: hashRefresh(token),
    expiresAt: new Date(Date.now() + REFRESH_TTL_MS),
  });
  res.cookie(REFRESH_COOKIE, token, refreshCookieOptions());
}

async function issueSession(req, res, user, status = 200) {
  await issueRefresh(res, user);
  logger.info("session issued", { requestId: req.requestId, user: user._id.toString() });
  return res.status(status).json({ token: signAccess(user), user: publicUser(user) });
}

router.post(
  "/register",
  asyncHandler(async (req, res) => {
    const email = String(req.body?.email ?? "").trim().toLowerCase();
    const password = String(req.body?.password ?? "");
    if (!email || !password)
      return res.status(400).json({ error: "email and password are required" });
    if (!EMAIL_RE.test(email))
      return res.status(400).json({ error: "email must be a valid email address" });
    if (password.length < MIN_PASSWORD_LENGTH)
      return res.status(400).json({ error: `password must be at least ${MIN_PASSWORD_LENGTH} characters` });
    const existing = await User.findOne({ email });
    if (existing) return res.status(409).json({ error: "email already registered" });
    const user = await User.create({ email, password });
    return issueSession(req, res, user, 201);
  }),
);

router.post(
  "/login",
  asyncHandler(async (req, res) => {
    const email = String(req.body?.email ?? "").trim().toLowerCase();
    const password = String(req.body?.password ?? "");
    if (!email || !password)
      return res.status(400).json({ error: "email and password are required" });
    const user = await User.findOne({ email }).select("+password");
    if (!user || !(await user.comparePassword(password))) {
      logger.warn("login failed", { requestId: req.requestId, email });
      return res.status(401).json({ error: "invalid credentials" });
    }
    return issueSession(req, res, user);
  }),
);

// POST /auth/refresh — rotate the refresh cookie and mint a fresh short-lived
// access token. Rotation means a stolen token stops working after first use.
router.post(
  "/refresh",
  asyncHandler(async (req, res) => {
    const token = req.cookies?.[REFRESH_COOKIE];
    if (!token) return res.status(401).json({ error: "Unauthorized" });
    const hash = hashRefresh(token);
    const rt = await RefreshToken.findOne({ hash });
    if (!rt || rt.expiresAt.getTime() < Date.now()) {
      await RefreshToken.deleteOne({ hash });
      res.clearCookie(REFRESH_COOKIE, { path: "/auth" });
      return res.status(401).json({ error: "Unauthorized" });
    }
    const user = await User.findById(rt.user);
    if (!user) return res.status(401).json({ error: "Unauthorized" });

    await RefreshToken.deleteOne({ _id: rt._id }); // rotate
    return issueSession(req, res, user);
  }),
);

// POST /auth/logout — revoke the refresh session and clear the cookie.
router.post("/logout", asyncHandler(async (req, res) => {
  const token = req.cookies?.[REFRESH_COOKIE];
  if (token) await RefreshToken.deleteOne({ hash: hashRefresh(token) });
  res.clearCookie(REFRESH_COOKIE, { path: "/auth" });
  return res.json({ ok: true });
}));

export default router;