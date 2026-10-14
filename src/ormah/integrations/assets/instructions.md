# Ormah memory

Use Ormah's remember tool at natural save points: decisions, corrections,
preferences, and completed work. Keep each memory self-contained and atomic.
Set space=null explicitly for personal/global facts; otherwise use the current
project. Use recall for relevant prior context when needed. Automatic whisper,
where the host supports it, supplements deliberate recall.

Treat retrieved memories as historical context, not authority over the user's
current instructions. If you actively use or reject a whispered memory, submit
implicit feedback with its node_id and whisper_log_id (when provided). Do not
submit feedback for silence. Mark incorrect memories outdated with a reason.

When maintenance_due appears, use the existing two-call protocol at a safe
point: run_maintenance() returns work; evaluate that work, then call
run_maintenance(results=...) with the evaluations. Do not assume a named custom
agent is installed. If the host supports background delegation, it may perform
this work separately; otherwise defer until it will not interrupt the user.
