# Red-flag checklist for reading diffs

Use in Step 5. Ask the learner which of these appear in the diff before revealing yours. Each row says what to look for and why it matters, so you can explain it.

## Silent behavior changes
- **Changed operators or constants**: `>` to `>=`, `and` to `or`, `0.9` to `0.8`, an index of `0` to `1`. The code still runs, but means something different. Ask: "was this in the request?"
- **Reordered logic**: moved checks, changed early returns, reordered operations that have side effects.
- **Default values changed**: function parameters, config defaults, timeouts.

## Removed safety
- **Deleted lines starting with `-`**: validation, auth checks, null checks, logging, cleanup (closing files or connections). Ask what each deleted line was protecting.
- **Weakened conditions**: a strict check relaxed to make something work.

## Hidden failures
- **Swallowed errors**: bare `except:`, `catch {}`, `except: pass`, errors logged but execution continues as if fine.
- **Fallbacks that mask problems**: `or default_value`, returning `None` or an empty result on failure with no signal.
- **Missing cleanup or timeouts**: network calls with no timeout, files never closed.

## Secrets and portability
- **Hardcoded secrets**: API keys, tokens, passwords, connection strings. Also check `.env` files or credentials in the file list.
- **Hardcoded paths or hosts**: `C:\Users\...`, `/home/...`, `localhost` in non-test code.
- **Magic numbers** that should be named config.

## Dependencies
- **New imports or entries in requirements, package.json, or lockfiles**: is it needed, real, maintained, pinned? AI can invent package names that do not exist, or that an attacker has registered.
- **Version bumps** nobody asked for.

## Tests
- **Edited assertions or expected values**: was the code wrong, or was the test bent to pass?
- **Tests that cannot fail**: asserting something always true, or mirroring the implementation.
- **Skipped or deleted tests** (`skip`, `xfail`, commented out).
- **No tests** for new behavior, or none for the edge cases.

## Scope and honesty
- **Scope creep**: unrelated refactors, renames, reformatting, or new files outside the stated intent. These hide real changes in noise.
- **Unused or duplicate code**: functions nothing calls, copy-pasted blocks.
- **Placeholders**: `TODO`, `pass`, `NotImplemented`, stub returns, presented as finished.
- **Docs or comments that overclaim**: a README or comment describing behavior the code does not have, or reporting results that were not actually run.

## Security quick checks
- User input reaching a shell, SQL, file path, `eval`, or HTML without validation or escaping.
- Authorization: can one user reach another's data?
- Overly broad permissions or disabled verification (`verify=False`, `shell=True`).

## Edge cases to ask about for any hunk
Empty, null or missing, zero, negative, very large, duplicates, wrong type, unicode, concurrent calls, and the failure path (network down, file missing).
