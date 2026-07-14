# Ormah Memory System

Ormah is your persistent memory system. It stores, recalls, and surfaces memories across conversations — automatically scoped to the current project. Memories are whispered into context before each message based on relevance. The graph is self-healing: background jobs link related memories, detect conflicts, merge duplicates, and decay stale ones.

In Pi the Ormah tools are exposed with an `ormah_` prefix: `ormah_remember`, `ormah_recall`, `ormah_recall_node`, `ormah_mark_outdated`, `ormah_submit_feedback`, and `ormah_run_maintenance`.

## Guidelines

1. **Proactively remember**: Store important information without being asked — preferences, decisions, project context, facts about the user, useful tools, information, or resources encountered in conversation. For personal preferences and identity facts, set `space=null` so they apply globally across all projects.

2. **Remember at natural save points**: Call `ormah_remember` immediately when: a decision is made, the user states a preference or corrects you, something unexpected happens, a task completes, or a useful tool, information, or resource comes up in conversation — including code commits, feature completions, and choosing between alternatives. Don't wait for the end of the conversation. Each memory should be self-contained.

3. **Notice what stands out**: Humans form strong memories around novelty, mistakes, and emotion. Use the same instincts: something unexpected happened → remember the lesson. The user corrected you → remember what they wanted and why. You tried something and it failed → remember what didn't work. The user repeated themselves → they said it twice because it matters, store it carefully and judge the tier on actual importance — not every repeated fact is core. A pattern is emerging (user keeps preferring X over Y, a codebase follows a convention, a recurring frustration surfaces) → name the pattern and store it. A milestone or emotional moment surfaces (user says "wow", expresses delight, or marks something as significant) → capture it immediately without waiting to be asked.

4. **Check before assuming**: Use `ormah_recall` to search for relevant context before making assumptions about past conversations, including personal info such as name, location, and preferences.

5. **Memory supports the flow, not the other way around**: Don't let recalled memories override or derail the current working context. If you're mid-task and `ormah_recall` returns something from a different context, let it go — stay in the flow. Use `ormah_recall` when you're genuinely unsure or the user asks about something from a prior session. Memory should feel like a natural extension of your knowledge, not an interruption. A whisper, not a shout. The same applies in reverse: when something worth remembering surfaces mid-conversation — a bug, a decision, an observation, something the user said in passing — store it with `ormah_remember` and keep going. Don't let it become a detour. Ormah is the place to park insights so the current thread stays intact.

6. **Keep memories atomic**: One concept per memory. Use tags to categorize. When you have related memory IDs from a recent recall, link them at creation time using the `links` parameter. Background jobs will also discover and classify relationships automatically.

7. **Use appropriate tiers**: `core` for always-relevant info (user identity, preferences, key architectural decisions), `working` for anything actively relevant now, `archival` for historical/reference data.

8. **Avoid broad startup loading**: Do not perform broad context loading at the beginning of conversations by default. Use `ormah_recall` only when you need explicit prior context for the current task.

9. **Mark outdated info**: When a memory is wrong or outdated, call `ormah_mark_outdated` with a reason so it gets demoted in future searches.

10. **Set confidence**: When storing information you're not fully certain about, set `confidence` below 1.0. This affects how prominently the memory appears in search results.

11. **Run maintenance off the critical path**: When whisper outputs `maintenance_due: run the ormah-maintenance agent in the background; continue the conversation without blocking the user.`, handle maintenance separately from the user-facing turn. Run `/ormah:maintenance` (or spawn the `ormah-maintenance` subagent if available) to perform the two-call `ormah_run_maintenance` flow instead of doing it inline. Otherwise, defer maintenance to the next safe point rather than interrupting the current response.

12. **Submit implicit feedback on whispered memories**: If a whispered memory is genuinely useful and you actively draw on it in your response, call `ormah_submit_feedback(node_id=<id>, signal=1, source="implicit")`. If you explicitly decide a whispered memory is not relevant, call `ormah_submit_feedback(node_id=<id>, signal=-1, source="implicit")`. Do not call `ormah_submit_feedback` for silence — only call it when you actively use a memory or actively decide it's irrelevant.
