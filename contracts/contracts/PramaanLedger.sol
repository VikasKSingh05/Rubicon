// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

// Rubicon — Pramaan Ledger
// Placeholder copied in during Phase 0 (contracts-first). Wired and deployed to the
// Polygon Amoy testnet (chain ID 80002) in Phase 5.
//
// Customization notes (see Appendix C):
// - Add onlyOwner/role-based access if only the backend's wallet should be able to write.
//   Discuss the centralization-vs-openness tradeoff in the report.

contract PramaanLedger {
    struct LogEntry {
        string hsiCid;
        string lidarCid;
        string prediction;
        uint256 timestamp;
        address logger;
    }

    LogEntry[] public entries;

    event AssessmentLogged(uint256 indexed id, string hsiCid, string prediction, uint256 timestamp);

    function logAssessment(
        string memory hsiCid,
        string memory lidarCid,
        string memory prediction
    ) external {
        entries.push(LogEntry(hsiCid, lidarCid, prediction, block.timestamp, msg.sender));
        emit AssessmentLogged(entries.length - 1, hsiCid, prediction, block.timestamp);
    }

    function getEntry(uint256 id) external view returns (LogEntry memory) {
        return entries[id];
    }

    function totalEntries() external view returns (uint256) {
        return entries.length;
    }
}
