# Security Policy

## Supported version

Security fixes currently target the latest `0.1.x` release candidate.

## Report privately

Please use GitHub's private vulnerability reporting feature for this repository. Include the affected commit, reproduction steps, impact, and whether hidden lifetime data, evaluator tokens, evidence-chain integrity, or safety-decision authority may be involved.

Do not include real battery datasets, credentials, access tokens, or proprietary cell identifiers in the report. If private reporting is unavailable, contact the repository owner through the email address shown on their GitHub profile and request a secure channel before sharing details.

## Security boundary

BatteryGuard is an offline research prototype and is not a certified control or safety system. The repository intentionally has no hardware interface. A report that demonstrates a path from this software to physical actuation should be treated as a design-boundary violation as well as a security issue.

The local Evidence Ledger is tamper-evident, not an external trust anchor. An attacker who can coherently replace both the ledger and its checkpoint remains outside the current threat model.
