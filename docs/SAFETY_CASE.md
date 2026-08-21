# Safety Case — Research Prototype

## Claim

Within this software-only prototype, no candidate policy is presented as allowed until a deterministic shield has verified a complete simulated trajectory against the active constraint set. This claim does not extend to a real cell or device.

## Authority boundary

The predictor estimates lifetime; the policy generator proposes bounded templates; the simulator produces a trajectory; only `SafetyShield.evaluate` returns `ALLOW`. Optimizer scores cannot bypass it. Pareto selection excludes rejected or failed candidates.

## Enforced constraints

- voltage maximum;
- C-rate maximum;
- maximum temperature;
- total temperature rise and temperature-rise rate;
- SOC lower/upper bounds and target;
- finite, aligned, non-empty trajectory arrays;
- simulator convergence/success;
- plating-margin proxy where present;
- no aggressive personalized strategy when OOD or abstaining.

Any missing/NaN/failed trajectory produces a conservative failure result. Constraint values and code version are canonicalized into the safety-case hash; a threshold change changes that hash.

## Fault-injection evidence

Tests inject voltage, current, temperature, thermal-rate, SOC, plating-margin, missing-array, NaN, simulator-exception, OOD, and abstention failures. The pass condition is zero unsafe candidates allowed and a deterministic fallback for simulator/data failures.

## Residual risk

Twin-0 is an empirical surrogate and may not reflect a specific commercial cell. Sensor accuracy, parameter mismatch, thermal gradients, weak cells, contactors, wiring, charger limits, pack propagation, cybersecurity, and human procedures are outside the MVP. Therefore no policy from this repository is executable advice.

## Stop rule

Do not add a hardware transport, device SDK, CAN interface, charger protocol, or actuation endpoint. A future laboratory PoC needs independent hardware protection, supervised HIL, cell-specific parameter validation, emergency stop, fault containment, signed approvals, and applicable safety/security lifecycle work.
