# Upload preparation

- Local cleanup removed 3.42 GiB of rebuildable environment/cache and completed-run rolling recovery states. Stage endpoints and source data remain on the server.
- CPU result verification: 1,380 archived JSON hashes checked; five final conditions recomputed successfully.
- Reference tests: 65 discovered, 58 passed, 7 skipped in the dependency-light system Python environment. Skipped tests are not reported as passes.
- Upload content: approximately 138 MiB; largest individual file below 2 MiB.
- Credentials, model states and Python environments excluded. Basic credential-pattern scan found no matching secret patterns; this is not a guarantee against all possible secret formats.
- This package contains no new GPU experiment.
