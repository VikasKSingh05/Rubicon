import { Router } from "express";
import jwt from "jsonwebtoken";
import User from "../models/User.js";
import { config } from "../config.js";
import { asyncHandler } from "../middleware/asyncHandler.js";

const router = Router();

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function sign(user) {
  return jwt.sign({ sub: user._id.toString(), email: user.email }, config.jwtSecret, {
    expiresIn: "7d",
  });
}

function publicUser(user) {
  return { id: user._id.toString(), email: user.email };
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
    if (password.length < 6)
      return res.status(400).json({ error: "password must be at least 6 characters" });
    const existing = await User.findOne({ email });
    if (existing) return res.status(409).json({ error: "email already registered" });
    const user = await User.create({ email, password });
    return res.status(201).json({ token: sign(user), user: publicUser(user) });
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
      return res.status(401).json({ error: "invalid credentials" });
    }
    return res.json({ token: sign(user), user: publicUser(user) });
  }),
);

export default router;