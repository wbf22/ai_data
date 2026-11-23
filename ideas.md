A few real gaps, roughly in order of how much they'd matter to a buyer:
1. The actual code changes are missing from the readable transcript. render_part explicitly skips snapshot/patch parts (aidata.py:610), so the .md transcript shows the agent's words and tool calls but not the diff of what it changed. The patches are in the export JSON, but a buyer reading the MD can't see the outcome. Worse, there's no post-session state: src.zip is captured at start, before any work, so nothing proves what the repo looked like when the agent finished.
2. Outcome evidence isn't captured, even though your rubric asks for it. The rubric asks you to type "proof (file, command, test name)", but the pipeline doesn't record it. Auto-capturing a final git diff and commit SHA + dirty flag at end (plus optional test/build run) would turn "trust me, it worked" into verifiable evidence — exactly the "reproducible, not just my opinion" item in your notes.
3. Sampling parameters are ignored. The plugin has a chat.params hook but only uses it to snapshot tool defs (aidata-capture.ts). Temperature/top_p/max_tokens are never captured. Since these sessions are training examples, the parameters that produced the outputs are part of "reproducible," and right now they're lost.
4. There's no dataset-level catalog. Everything is per-session folders; nothing ties them together. A top-level manifest (one JSON/README listing each session's category, outcome, model, models used, token counts, duration, difficulty, repo size, language, file hashes) is what makes it purchasable as a dataset rather than a pile of folders.
5. Metadata you listed but don't capture: task statement (the first user message as an explicit field), difficulty, task language, and repo size — the latter two are auto-derivable.
6. Small stuff that's in the JSON but not surfaced: per-message timestamps and token/cost usage. Cheap to add to the MD/metadata.
If I had to pick, I'd do 2 + 1 first (git diff/commit SHA at end, and stop dropping patches from the MD), then 4 (catalog), then 3 (plugin params). Want me to start with the git-diff/commit capture?




1. nice catch, yeah I think we would want the patches in the md transcript. Maybe we capture the session state after then too

2. I think capturing the final repo state would work for this too right?

3. yeah we could capture those too if it's available

4. yeah that might be nice to make a script that produces a markdown file for the whole dataset would be nice. I think it'd be good if it could be rerun whenever to generate the file instead of trying to keep it up to date manually or something

5. maybe we have the user rate the diffulty 1-10. But capturing programming language and repo size would be great

