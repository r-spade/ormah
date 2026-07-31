# Local admin API authentication

Sensitive desktop-only routes require an installation-local capability in the
`X-Ormah-Local-Token` header and reject non-loopback peers. The server creates the capability at
`~/.local/share/ormah/local_api_token` with mode `0600`. It is independent of the Ormah Cloud
account token.

The desktop integration keeps this boundary native: Tauri reads the owner-only capability and adds
the header to requests it makes to the local Python server. React asks a narrow Tauri command to
perform a fixed account, billing, or protection operation; there is deliberately no generic HTTP
proxy. React never receives the local capability, `ORMAH_ACCOUNT_TOKEN`, a presigned object-store
URL, an age identity, recovery-kit material, or a Stripe-hosted URL. Browser-only callers cannot
use these routes.

The remote graph WebView receives only the `desktop-product-bridge` capability. Every bridge command
also verifies that it was invoked by the `main` window at the configured
`http://127.0.0.1:<port>` origin. Checkout and Customer Portal handoffs are exact-host validated and
opened by Rust in the system browser; the URL is not returned to React. Account responses use typed
DTOs, while protection responses pass a recursive forbidden-field and secret-pattern check.

Long backup and verification calls are submitted to a bounded Python coordinator and return `202`
with a process-local polling ID. Durable truth remains in the per-store cloud state. If the Python
process restarts during initial protection, startup finds a durable `running` intent and retries it
through the existing upload journal and exact-snapshot verification path.

Mutation routes with no product input require an explicit empty JSON object (`{}`). The body carries
no token, customer, price, redirect, or object-store input; requiring JSON ensures browser requests
are preflighted before the owner-only capability is checked.
