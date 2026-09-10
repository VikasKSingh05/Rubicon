// Wraps async route handlers so rejections reach the global error middleware
// instead of hanging the request (Express 4 does not catch async throws).
export function asyncHandler(fn) {
  return (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
}