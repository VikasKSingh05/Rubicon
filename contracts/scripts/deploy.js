// Deploy PramaanLedger.sol to Polygon Amoy (chain ID 80002).
//
// Manual, one-time step:
//   cd contracts
//   npm install
//   npx hardhat --version   # not needed; this script compiles with solc + deploys with ethers
//
// Requires env (or a root .env):
//   AMOY_RPC_URL=https://rpc-amoy.polygon.technology
//   DEPLOYER_PRIVATE_KEY=<0x-prefixed testnet key with a little MATIC for gas>
//   CHAIN_ID=80002
// After deployment print the address and, if RUBICON_CONTRACT_OUT env is set,
// write CONTRACT_ADDRESS=<addr> into the given .env file.

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { ethers } from "ethers";
import solc from "solc";

const rpcUrl = process.env.AMOY_RPC_URL || "https://rpc-amoy.polygon.technology";
const privateKey = process.env.DEPLOYER_PRIVATE_KEY;
const chainId = Number(process.env.CHAIN_ID || 80002);
const outEnv = process.env.RUBICON_CONTRACT_OUT;

if (!privateKey) {
  console.error("DEPLOYER_PRIVATE_KEY is required (0x-prefixed Amoy testnet key funded with MATIC).");
  process.exit(1);
}

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
  console.error("Compilation failed:");
  errors.forEach((e) => console.error(`  ${e.formattedMessage}`));
  process.exit(1);
}
const compiled = output.contracts["PramaanLedger.sol"].PramaanLedger;

console.log(`[deploy] provider=${rpcUrl} chainId=${chainId}`);
console.log(`[deploy] deploying from ${new ethers.Wallet(privateKey).address} ...`);

const provider = new ethers.JsonRpcProvider(rpcUrl);
const network = await provider.getNetwork();
if (Number(network.chainId) !== chainId) {
  console.error(`wrong chain: provider says ${network.chainId}, expected ${chainId}`);
  process.exit(1);
}
const wallet = new ethers.Wallet(privateKey, provider);
const factory = new ethers.ContractFactory(compiled.abi, compiled.evm.bytecode.object, wallet);
const contract = await factory.deploy();
await contract.waitForDeployment();
const address = await contract.getAddress();
const txHash = contract.deploymentTransaction().hash;

console.log(`[deploy] PramaanLedger deployed at ${address}`);
console.log(`[deploy] tx ${txHash}`);

if (outEnv && existsSync(outEnv)) {
  let text = readFileSync(outEnv, "utf8");
  text = text.replace(/^CONTRACT_ADDRESS=.*$/m, `CONTRACT_ADDRESS=${address}`);
  writeFileSync(outEnv, text, "utf8");
  console.log(`[deploy] wrote CONTRACT_ADDRESS=${address} into ${outEnv}`);
}

if (!outEnv) {
  console.log("[deploy] set CONTRACT_ADDRESS in your .env to enable real on-chain logging.");
}