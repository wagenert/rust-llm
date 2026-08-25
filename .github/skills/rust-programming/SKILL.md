---
name: rust-programming
description: 'Develop, debug, review, and test Rust projects with Cargo. Use for Rust implementation tasks, compiler errors, ownership and borrowing issues, trait and lifetime design, workspace packages, async code, performance, Burn/tensor code, formatting, Clippy, and focused test validation.'
argument-hint: '[Rust task, error, or target package]'
user-invocable: true
disable-model-invocation: false
---

# Rust Programming

## Purpose

Use this skill to make small, verifiable changes in Rust codebases while preserving the project's existing APIs, crate structure, feature flags, and style.

## Workflow

1. **Find the controlling code path.** Start from the named file, symbol, compiler error, failing test, or command. Read the nearest implementation, its call sites, and one neighboring test or example. State one falsifiable hypothesis about the behavior and one cheap check that could disconfirm it.
2. **Identify the Cargo scope.** Read the nearest `Cargo.toml` and the workspace manifest. Determine the affected package, binary or library target, feature flags, and relevant build script. Prefer `cargo check -p <package>` and `cargo test -p <package>` over workspace-wide commands when the change is local.
3. **Preserve local design.** Reuse existing types, traits, error conventions, modules, and dependency versions. Avoid introducing a new abstraction, clone, `unwrap`, `expect`, unsafe block, or broad refactor unless the behavior requires it. Keep public API changes explicit.
4. **Make the smallest coherent edit.** Fix the ownership, type, lifetime, trait-bound, control-flow, or data-shape cause at the point where it is controlled. Keep unrelated user changes intact. Add a focused regression test when the behavior is testable and no suitable test exists.
5. **Validate immediately.** After the first substantive edit, run the narrowest useful executable check before more exploration or edits:
   - syntax/type issue: `cargo check -p <package> [--features <features>]`
   - behavior issue: `cargo test -p <package> <test_filter> -- --exact` or the narrowest relevant test
   - formatting issue: `cargo fmt --all -- --check`
   - lint or API quality issue: `cargo clippy -p <package> --all-targets --all-features -- -D warnings`
   - binary behavior: `cargo run -p <package> -- <args>`
6. **Interpret failures locally.** If the check supports the hypothesis, repair the same slice and rerun the same command. If it falsifies the hypothesis, take one nearby hop toward the code that actually computes or mutates the behavior. Do not widen the search until the nearby path is exhausted.
7. **Run proportional final checks.** For a library or shared helper, run its focused tests plus `cargo check -p <package>`. For a workspace contract or dependency change, run `cargo test --workspace` and, when practical, `cargo clippy --workspace --all-targets --all-features -- -D warnings`. Run `cargo fmt --all -- --check` before finishing.
8. **Report precisely.** Summarize the root cause, changed files, commands run, and any remaining warnings, untested paths, feature-specific limitations, or environment constraints. Do not claim checks that were not run.

## Rust-Specific Decisions

- Prefer borrowing over cloning when ownership permits; when a value must outlive its source, make the ownership transfer explicit rather than hiding it behind repeated clones.
- Use `Result` and `?` for recoverable failures. Preserve context with the project's existing error type or error-reporting style.
- Let the compiler guide lifetime and trait-bound fixes. Avoid adding `'static`, `Send`, or `Sync` bounds unless the caller or runtime genuinely requires them.
- Keep generic bounds near the item that needs them. Use associated types and existing trait abstractions when they clarify the contract.
- For iterators, prefer clear iterator composition when it remains readable; use a loop when mutation, early exit, or error handling is clearer.
- Treat `unsafe` as an interface requiring justification, a narrowly documented invariant, and focused tests. Do not add it to bypass borrow-checker errors.
- Keep conversions explicit at boundaries. Validate dimensions, indices, encodings, and device/backend assumptions before tensor or numerical operations.
- For Burn or other tensor code, confirm tensor rank, shape, dtype, device, backend, and autodiff requirements at the call site. Test both values and shapes where appropriate; avoid assuming a backend-specific behavior is universal.
- In workspace repositories, do not assume the current directory identifies the intended package. Use package names from `Cargo.toml` and pass them explicitly to Cargo commands.

## Debugging Checklist

- Reproduce the failure with the smallest command that still fails.
- Read the complete compiler diagnostic, including the primary span and notes.
- Check whether the issue is a moved value, temporary lifetime, mismatched generic type, missing trait import, feature-gated item, wrong package, or backend/shape mismatch.
- Compare the failing code with the nearest working implementation or test before inventing a new pattern.
- After a fix, exercise the original failure path and one boundary case such as empty input, invalid input, a repeated call, or a different backend when relevant.

## Completion Criteria

A Rust task is complete when:

- the requested behavior is implemented at its controlling code path;
- the smallest relevant check passes, or a concrete blocker is reported;
- formatting and applicable lint/test checks have been run;
- no unrelated files or user changes were altered; and
- the final report names residual risk instead of silently omitting it.
