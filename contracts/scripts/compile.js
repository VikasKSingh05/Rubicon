// Shared solc compilation for PramaanLedger.sol. Used by both the deploy
// script (which needs the bytecode) and the CI compile-check (which only
// verifies that the contract still compiles).

import { readFileSync } from "node:fs";
import solc from "solc";

export function compileContract() {
  const source = readFileSync(new URL("../contracts/PramaanLedger.sol", import.meta.url), "utf8");

  const input = {
    language: "Solidity",
    sources: { "PramaanLedger.sol": { content: source } },
    settings: {
      optimizer: { enabled: true, runs: 200 },
      outputSelection: { "*": { "*": ["abi", "evm.bytecode"] } },
    },
  };

  const output = JSON.parse(solc.compile(JSON.stringify(input)));
  const errors = output.errors?.filter((e) => e.severity === "error") ?? [];
  if (errors.length > 0) {
    throw new Error(errors.map((e) => e.formattedMessage).join("\n"));
  }
  const compiled = output.contracts["PramaanLedger.sol"].PramaanLedger;
  return { abi: compiled.abi, bytecode: compiled.evm.bytecode.object };
}