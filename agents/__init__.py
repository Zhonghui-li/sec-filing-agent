"""Package-level defaults shared by the agent, its observability and the eval harness.

The generation model's fallback used to be written out at six separate `os.environ.get`
call sites. Commit 138a3b9 changed four of them and missed .github/workflows/eval.yml,
so every manual L2 run after it evaluated a model the repo had already replaced. The
value itself is not the point — the point is that a run's recorded model and the model
that actually answered cannot drift apart if they read the same name.

Changed from o4-mini to gpt-5.4-mini on 2026-10-12, ahead of o4-mini's 2026-10-23 shutdown,
by Phase 1.7 (eval/_migration_17_round3/RESULTS.md). Changing it here changes the Python side
in one edit — the six call sites and eval/runconfig's snapshot default all read this name. The
two non-Python sites (.github/workflows/eval.yml, service/deploy.sh) set the env var explicitly
and must be changed with it.
"""
DEFAULT_GEN_MODEL = "gpt-5.4-mini"
