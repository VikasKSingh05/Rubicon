// CI-friendly compile check: exits non-zero if PramaanLedger.sol stops compiling.
import { compileContract } from "./compile.js";

try {
  const { bytecode, abi } = compileContract();
  console.log(
    `[compile-check] PramaanLedger compiles (bytecode ${(bytecode.length / 2).toLocaleString()} bytes, ABI ${abi.length} entries)`,
  );
} catch (err) {
  console.error(`[compile-check] FAILED\n${err.message}`);
  process.exit(1);
}