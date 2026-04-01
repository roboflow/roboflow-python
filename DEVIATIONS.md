# CLI Modernization: Plan Deviations

This document records deviations from the original plan made during implementation, per the orchestration guidelines.

## Deviations

### 1. Graceful handler error handling in auto-discovery
**Plan**: Auto-discovery loads all handlers without error handling.
**Change**: Added try/except around each handler's `register()` call so a broken handler doesn't crash the entire CLI.
**Reason**: During Wave 1, engineer-5's in-progress deployment handler had a bug that crashed every CLI command. This was a QA blocker.
**Assessment**: Good permanent change. A broken handler should never take down the CLI.

### 2. SDK stdout suppression via context manager
**Plan**: Not explicitly planned.
**Change**: Added `suppress_sdk_output(args)` context manager in `_output.py` that redirects stdout when `--json` or `--quiet` is active. Used by search and model handlers.
**Reason**: The SDK's `Roboflow()` and `rf.workspace()` print "loading Roboflow workspace..." to stdout, which corrupts `--json` output for piping. QA flagged this as a bug.
**Assessment**: Correct fix. The SDK's chatty output is a design debt that should eventually be addressed at the SDK level, but suppressing at the CLI layer is the right short-term approach.

### 3. Error message extraction from JSON-encoded exceptions
**Plan**: Not explicitly planned.
**Change**: Added `_extract_error_message()` helper in model.py and train.py that parses JSON error strings from `RoboflowError` exceptions into clean messages.
**Reason**: QA found that API errors were double-encoded in `--json` output (JSON string inside JSON). The API returns error bodies as exception message strings.
**Assessment**: Good fix. Should eventually be centralized into `_output.py` rather than duplicated.

### 4. Legacy aliases show ==SUPPRESS== in help
**Plan**: Legacy aliases would be completely hidden from help.
**Change**: Used `argparse.SUPPRESS` for help text, which hides the description but still shows the command name in the choices list with `==SUPPRESS==` text.
**Known limitation**: argparse doesn't support fully hiding subparser choices. Would need a custom HelpFormatter to fix completely.
**Assessment**: Cosmetic issue. The commands work correctly. Can be addressed in a follow-up.

### 5. No separate worktree branches to merge
**Plan**: Engineers work in isolated worktrees, lead merges branches.
**Actual**: Engineers' worktrees shared the filesystem with the main branch (worktree isolation cleaned up but files persisted). Changes were committed directly to the working directory.
**Assessment**: Worked fine in practice — no merge conflicts since each engineer owned distinct files. The worktree isolation still prevented engineers from interfering with each other's running processes.
