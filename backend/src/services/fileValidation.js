export const ALLOWED_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".tiff", ".tif"]);

/**
 * Magic-byte validation. Returns an error string, or null when the buffer
 * matches a supported satellite image.
 */
export function validateMagic(buf, kind) {
  if (!buf || buf.length < 4) return "file payload too small or empty";
  if (kind === "pre" || kind === "post") {
    const isPng = buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47;
    const isJpg = buf[0] === 0xff && buf[1] === 0xd8;
    const isTiffLittle = buf[0] === 0x49 && buf[1] === 0x49 && buf[2] === 0x2a && buf[3] === 0x00;
    const isTiffBig = buf[0] === 0x4d && buf[1] === 0x4d && buf[2] === 0x00 && buf[3] === 0x2a;
    return isPng || isJpg || isTiffLittle || isTiffBig
      ? null
      : `${kind} image must be a valid PNG, JPEG, or TIFF`;
  }
  throw new TypeError(`unsupported satellite image kind: ${kind}`);
}
