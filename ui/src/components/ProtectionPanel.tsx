import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronRight,
  Circle,
  CloudOff,
  CreditCard,
  ExternalLink,
  LoaderCircle,
  LockKeyhole,
  LogIn,
  LogOut,
  RefreshCw,
  ShieldCheck,
  X,
} from "lucide-react";
import {
  isDesktopApp,
  effectiveProtectionState,
  operationPhaseIsActive,
  productBridge,
  protectionPresentation,
  protectionRepairAction,
  recoveryKitSectionVisible,
  type AccountStatus,
  type BillingOffer,
  type OperationPhase,
  type ProtectionOperation,
  type ProtectionStatus,
} from "../productBridge";
import RecoveryKitSection from "./RecoveryKitSection";

interface Props {
  open: boolean;
  onClose: () => void;
  onToast: (message: string, type?: "success" | "error" | "info") => void;
  onStatusChange?: (status: ProtectionStatus) => void;
}

type View = "summary" | "email" | "code" | "checkout";
type LoginPurpose = "account" | "protect";

function operationIsActive(operation: ProtectionOperation | null): boolean {
  return Boolean(
    operation && (
      operation.status === "queued"
      || operation.status === "running"
    )
  );
}

function formatDate(value: string | null): string {
  if (!value) return "not yet";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "unavailable";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
}

function formatPrice(offer: BillingOffer | null): string | null {
  if (!offer) return null;
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: offer.currency.toUpperCase(),
    }).format(offer.unit_amount / 100);
  } catch {
    return null;
  }
}

function operationLabel(
  operation: ProtectionOperation | null,
  status: ProtectionStatus | null,
): string | null {
  if (!operation) return null;
  if (operation.status === "queued") return "Queued";
  if (operation.status !== "running") return null;
  const phase = status?.last_operation_phase || operation.phase;
  switch (phase) {
    case "pending": return "Queued";
    case "running":
    case "preparing": return "Creating a local recovery point";
    case "encrypting": return "Encrypting on this device";
    case "uploading": return "Uploading encrypted data";
    case "finalizing": return "Securing the cloud recovery point";
    case "downloading": return "Downloading a temporary recovery test";
    case "verifying": return "Decrypting and checking every file";
    case "rebuilding": return "Rebuilding memory and testing search";
    default: return "Finishing protection check";
  }
}

const PROTECTION_STAGES = [
  { phase: "preparing", label: "Create local recovery point" },
  { phase: "encrypting", label: "Encrypt on this device" },
  { phase: "uploading", label: "Upload encrypted data" },
  { phase: "finalizing", label: "Secure cloud recovery point" },
  { phase: "downloading", label: "Download a temporary test copy" },
  { phase: "verifying", label: "Decrypt and check every file" },
  { phase: "rebuilding", label: "Rebuild memory and test search" },
] as const;

function phaseIndex(phase: OperationPhase | null | undefined): number {
  if (phase === "pending" || phase === "running") return 0;
  return PROTECTION_STAGES.findIndex((stage) => stage.phase === phase);
}

function operationSuccessMessage(operation: ProtectionOperation): string {
  switch (operation.kind) {
    case "enable": return "Memory protected and verified.";
    case "backup": return "Encrypted recovery point uploaded and restore-tested.";
    case "verify": return "Recovery point verified.";
    case "disable": return "Future cloud backups stopped.";
    default: return "Operation completed.";
  }
}

function errorMessage(value: unknown, fallback: string): string {
  if (value instanceof Error && value.message.trim()) return value.message;
  if (typeof value === "string" && value.trim()) return value;
  return fallback;
}

export default function ProtectionPanel({ open, onClose, onToast, onStatusChange }: Props) {
  const [account, setAccount] = useState<AccountStatus | null>(null);
  const [status, setStatus] = useState<ProtectionStatus | null>(null);
  const [offer, setOffer] = useState<BillingOffer | null>(null);
  const [operation, setOperation] = useState<ProtectionOperation | null>(null);
  const [view, setView] = useState<View>("summary");
  const [loginPurpose, setLoginPurpose] = useState<LoginPurpose>("account");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checkoutPolling, setCheckoutPolling] = useState(false);
  const [confirmStop, setConfirmStop] = useState(false);
  const [recoveryBusy, setRecoveryBusy] = useState<"save" | "print" | null>(null);
  const [recoveryError, setRecoveryError] = useState<string | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const checkoutCheckInFlight = useRef(false);
  const offerRequested = useRef(false);
  const desktop = isDesktopApp();

  const refresh = useCallback(async () => {
    if (!desktop) return;
    setLoading(true);
    try {
      try {
        const bridge = await productBridge.info();
        if (bridge.version < 1) throw new Error("unsupported bridge");
      } catch {
        throw new Error("Update Ormah Desktop to use cloud protection.");
      }
      // Account status refreshes a stale entitlement cache. Read protection
      // after it so an active subscriber is never painted as paused from an
      // expired cache entry on first open.
      const nextAccount = await productBridge.accountStatus();
      const nextStatus = await productBridge.status();
      setAccount(nextAccount);
      setStatus(nextStatus);
      onStatusChange?.(nextStatus);
      setError(null);
      if (nextAccount.signed_in && !offerRequested.current) {
        offerRequested.current = true;
        productBridge.offer().then(setOffer).catch(() => {
          offerRequested.current = false;
        });
      }
    } catch (err) {
      setError(errorMessage(err, "Protection status is unavailable."));
    } finally {
      setLoading(false);
    }
  }, [desktop, onStatusChange]);

  useEffect(() => {
    if (!open) return;
    void refresh();
    requestAnimationFrame(() => headingRef.current?.focus());
  }, [open, refresh]);

  useEffect(() => {
    if (!open || !operationIsActive(operation)) return;
    const operationId = operation?.operation_id;
    if (!operationId) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await productBridge.operation(operationId);
        setOperation(next);
        const nextStatus = await productBridge.status();
        setStatus(nextStatus);
        onStatusChange?.(nextStatus);
        if (!operationIsActive(next)) {
          window.clearInterval(timer);
          await refresh();
          setOperation(null);
          if (next.phase === "completed") onToast(operationSuccessMessage(next), "success");
          else if (next.message) setError(next.message);
        }
      } catch (err) {
        // Polling IDs are process-local. After a Python restart, discard the
        // stale ID and continue from the durable per-store status.
        setOperation(null);
        try {
          const nextStatus = await productBridge.status();
          setStatus(nextStatus);
          onStatusChange?.(nextStatus);
          setError(null);
        } catch {
          setError(errorMessage(err, "Could not read operation status."));
        }
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [open, operation?.operation_id, operation?.phase, operation?.status, onStatusChange, onToast, refresh]);

  useEffect(() => {
    if (
      !open
      || operationIsActive(operation)
      || status?.protection_intent_status !== "running"
      || !operationPhaseIsActive(status?.last_operation_phase)
    ) return;
    const timer = window.setInterval(async () => {
      try {
        const nextStatus = await productBridge.status();
        setStatus(nextStatus);
        onStatusChange?.(nextStatus);
      } catch (err) {
        setError(errorMessage(err, "Could not refresh protection progress."));
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [
    open,
    operation,
    onStatusChange,
    status?.last_operation_phase,
    status?.protection_intent_status,
  ]);

  const bindAndContinue = useCallback(async (intentId: string) => {
    const bound = await productBridge.bindIntent(intentId);
    setOperation(bound);
    if (bound.protection_state === "subscription_required" || bound.reason_code === "subscription_required") {
      setView("checkout");
      const nextOffer = await productBridge.offer();
      setOffer(nextOffer);
      return;
    }
    if (bound.reason_code) {
      throw new Error(bound.message || "Protection could not continue.");
    }
    const started = await productBridge.enable(intentId);
    setOperation(started);
    setView("summary");
  }, []);

  const beginProtection = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      setLoginPurpose("protect");
      const intent = await productBridge.createIntent();
      if (intent.reason_code && intent.reason_code !== "sign_in_required") {
        throw new Error(intent.message || "Protection could not start.");
      }
      const intentId = intent.protection_intent_id;
      if (!intentId) throw new Error("Protection could not create a durable request.");
      setOperation(intent);
      if (!account?.signed_in || intent.protection_state === "sign_in_required") {
        setView("email");
      } else {
        await bindAndContinue(intentId);
      }
    } catch (err) {
      setError(errorMessage(err, "Protection could not start."));
    } finally {
      setBusy(false);
    }
  }, [account?.signed_in, bindAndContinue]);

  const requestCode = useCallback(async () => {
    if (!email.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await productBridge.requestCode(email.trim().toLowerCase());
      setView("code");
    } catch (err) {
      setError(errorMessage(err, "Could not request a sign-in code."));
    } finally {
      setBusy(false);
    }
  }, [email]);

  const verifyCode = useCallback(async () => {
    if (!/^\d{6}$/.test(code)) return;
    setBusy(true);
    setError(null);
    try {
      const nextAccount = await productBridge.verifyCode(email.trim().toLowerCase(), code);
      setAccount(nextAccount);
      if (loginPurpose === "account") {
        setOperation(null);
        setView("summary");
        await refresh();
        onToast("Signed in to Ormah Cloud.", "success");
        return;
      }
      const intentId = operation?.protection_intent_id || status?.protection_intent_id;
      if (!intentId) {
        setOperation(null);
        setView("summary");
        throw new Error("The protection request expired. Start again.");
      }
      await bindAndContinue(intentId);
      await refresh();
    } catch (err) {
      setError(errorMessage(err, "That code could not be verified."));
    } finally {
      setBusy(false);
    }
  }, [bindAndContinue, code, email, loginPurpose, onToast, operation?.protection_intent_id, refresh, status?.protection_intent_id]);

  const openCheckout = useCallback(async () => {
    const intentId = operation?.protection_intent_id || status?.protection_intent_id;
    if (!intentId) return;
    setBusy(true);
    setError(null);
    try {
      const handoff = await productBridge.openCheckout(intentId);
      if (handoff.status === "already_subscribed") {
        await bindAndContinue(intentId);
      } else {
        if (!handoff.opened) {
          setError("Your subscription is still being processed. Check again shortly or manage it in Stripe.");
        }
        setCheckoutPolling(true);
      }
    } catch (err) {
      setError(errorMessage(err, "Checkout could not open."));
    } finally {
      setBusy(false);
    }
  }, [bindAndContinue, operation?.protection_intent_id, status?.protection_intent_id]);

  const checkPayment = useCallback(async (silent = false) => {
    if (checkoutCheckInFlight.current) return;
    const intentId = operation?.protection_intent_id || status?.protection_intent_id;
    if (!intentId) return;
    checkoutCheckInFlight.current = true;
    setBusy(true);
    setError(null);
    try {
      const bound = await productBridge.bindIntent(intentId);
      setOperation(bound);
      if (bound.protection_state === "subscription_required" || bound.reason_code === "subscription_required") {
        if (!silent) setError("Payment is not active yet. Stripe may still be processing it.");
        return;
      }
      setCheckoutPolling(false);
      await bindAndContinue(intentId);
    } catch (err) {
      setError(errorMessage(err, "Payment status is unavailable."));
    } finally {
      checkoutCheckInFlight.current = false;
      setBusy(false);
    }
  }, [bindAndContinue, operation?.protection_intent_id, status?.protection_intent_id]);

  useEffect(() => {
    if (!checkoutPolling) return;
    let checks = 0;
    const poll = window.setInterval(() => {
      checks += 1;
      if (checks >= 20) {
        window.clearInterval(poll);
        setCheckoutPolling(false);
        return;
      }
      void checkPayment(true);
    }, 3000);
    const onFocus = () => void checkPayment(false);
    window.addEventListener("focus", onFocus);
    return () => {
      window.clearInterval(poll);
      window.removeEventListener("focus", onFocus);
    };
  }, [checkoutPolling, checkPayment]);

  const runOperation = useCallback(async (action: "backup" | "verify" | "disable") => {
    setBusy(true);
    setError(null);
    try {
      const next = action === "backup"
        ? await productBridge.backupNow()
        : action === "verify"
          ? await productBridge.verifyNow()
          : await productBridge.disable();
      setOperation(next);
      await refresh();
    } catch (err) {
      setError(errorMessage(err, "The operation could not start."));
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  const saveRecoveryKit = useCallback(async () => {
    setRecoveryBusy("save");
    setRecoveryError(null);
    try {
      const result = await productBridge.saveRecoveryKit();
      if (result.status === "canceled") return;
      if (!result.recovery_kit_verified_at) {
        throw new Error("The saved recovery kit could not be verified.");
      }
      await refresh();
      onToast(
        result.device_loss_recovery_ready
          ? "Recovery kit saved and verified."
          : "Recovery kit saved and verified; recovery readiness is waiting for current protection verification.",
        "success",
      );
    } catch (err) {
      setRecoveryError(errorMessage(err, "The recovery kit could not be saved."));
    } finally {
      setRecoveryBusy(null);
    }
  }, [onToast, refresh]);

  const openPrintableRecoveryKit = useCallback(async () => {
    setRecoveryBusy("print");
    setRecoveryError(null);
    try {
      const result = await productBridge.openPrintableRecoveryKit();
      if (!result.opened) throw new Error("The printable copy could not be opened.");
      onToast("Recovery kit sent to your system viewer.", "info");
    } catch (err) {
      setRecoveryError(errorMessage(err, "The printable copy could not be opened."));
    } finally {
      setRecoveryBusy(null);
    }
  }, [onToast]);

  const activeOperation = operationIsActive(operation);
  const presentation = useMemo(
    () => protectionPresentation(effectiveProtectionState(operation, status)),
    [operation, status],
  );
  const price = formatPrice(offer);
  const activeLabel = operationLabel(operation, status);
  const activePhase = activeOperation
    ? (status?.last_operation_phase || operation?.phase)
    : null;
  const activeStageIndex = phaseIndex(activePhase);
  const summaryTone = activeOperation ? "working" : presentation.tone;
  const repairAction = status ? protectionRepairAction(status) : "none";

  const runRepairAction = useCallback(async () => {
    if (!status) return;
    if (repairAction === "signin") {
      await beginProtection();
      return;
    }
    if (repairAction === "subscribe") {
      setView("checkout");
      if (!offer) productBridge.offer().then(setOffer).catch(() => undefined);
      return;
    }
    if (repairAction === "resume") {
      const intentId = status.protection_intent_id;
      if (!intentId) return;
      setBusy(true);
      setError(null);
      try {
        setOperation(await productBridge.enable(intentId));
      } catch (err) {
        setError(errorMessage(err, "Protection could not resume."));
      } finally {
        setBusy(false);
      }
      return;
    }
    if (repairAction === "backup" || repairAction === "verify") {
      await runOperation(repairAction);
      return;
    }
    await refresh();
  }, [beginProtection, offer, refresh, repairAction, runOperation, status]);

  return (
    <aside className={`side-panel protection-panel ${open ? "open" : ""}`} aria-hidden={!open}>
      <div className="side-panel-header">
        <div>
          <div className="protection-eyebrow">Ormah Cloud</div>
          <h2 className="review-title" ref={headingRef} tabIndex={-1}>Protection</h2>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close protection">
          <X size={16} />
        </button>
      </div>

      {!desktop ? (
        <div className="protection-empty">
          <ShieldCheck size={24} />
          <strong>Open Ormah Desktop to protect this memory</strong>
          <p>The desktop app keeps account credentials and encryption material out of this page.</p>
        </div>
      ) : loading && !status ? (
        <div className="protection-empty"><LoaderCircle className="spin" size={22} />Loading protection status</div>
      ) : (
        <>
          <section className={`protection-summary tone-${summaryTone}`}>
            <div className="protection-status-icon" aria-hidden="true">
              {summaryTone === "success" ? <ShieldCheck size={22} />
                : summaryTone === "danger" ? <AlertTriangle size={22} />
                  : summaryTone === "warning" ? <CloudOff size={22} />
                    : summaryTone === "working" ? <LoaderCircle className="spin" size={22} />
                      : <LockKeyhole size={22} />}
            </div>
            <div>
              <h3>{activeLabel || presentation.title}</h3>
              <p>{activeOperation
                ? "Ormah is creating and restore-testing an encrypted recovery point."
                : presentation.detail}</p>
            </div>
          </section>

          {activeOperation && activeStageIndex >= 0 && (
            <section className="protection-progress" aria-label="Protection progress">
              <div className="protection-progress-heading">
                <span>Recovery check</span>
                <strong>In progress</strong>
              </div>
              <ol>
                {PROTECTION_STAGES.map((stage, index) => {
                  const stageState = index < activeStageIndex
                    ? "complete"
                    : index === activeStageIndex
                      ? "current"
                      : "upcoming";
                  return (
                    <li className={stageState} key={stage.phase}>
                      <span aria-hidden="true">
                        {stageState === "complete" ? <Check size={13} />
                          : stageState === "current" ? <LoaderCircle className="spin" size={13} />
                            : <Circle size={10} />}
                      </span>
                      <span>{stage.label}</span>
                    </li>
                  );
                })}
              </ol>
              <p>Verification uses a temporary copy. Your live memory is never replaced.</p>
            </section>
          )}

          {error && (
            <div className="protection-error" role="alert">
              <AlertTriangle size={15} />
              <span>{error}</span>
            </div>
          )}

          <div className="sr-status" aria-live="polite">{activeLabel || presentation.title}</div>

          {view === "email" && (
            <section className="protection-step">
              <button className="step-back" onClick={() => setView("summary")}>Back</button>
              <h3>{loginPurpose === "protect" ? "Sign in to continue" : "Sign in to Ormah Cloud"}</h3>
              <p>Enter your email. Ormah will send a one-time code; there is no password.</p>
              <label className="protection-field">
                <span>Email</span>
                <input
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  onKeyDown={(event) => { if (event.key === "Enter") void requestCode(); }}
                  autoFocus
                />
              </label>
              <button className="protection-primary" disabled={busy || !email.trim()} onClick={() => void requestCode()}>
                {busy ? <LoaderCircle className="spin" size={15} /> : <ChevronRight size={15} />}
                Send code
              </button>
            </section>
          )}

          {view === "code" && (
            <section className="protection-step">
              <button className="step-back" onClick={() => setView("email")}>Change email</button>
              <h3>Check your email</h3>
              <p>Enter the six-digit code sent to <strong>{email}</strong>.</p>
              <label className="protection-field">
                <span>One-time code</span>
                <input
                  className="otp-input"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  maxLength={6}
                  value={code}
                  onChange={(event) => setCode(event.target.value.replace(/\D/g, "").slice(0, 6))}
                  onKeyDown={(event) => { if (event.key === "Enter") void verifyCode(); }}
                  autoFocus
                />
              </label>
              <button className="protection-primary" disabled={busy || code.length !== 6} onClick={() => void verifyCode()}>
                {busy ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}
                {loginPurpose === "protect" ? "Verify and continue" : "Sign in"}
              </button>
              <button className="protection-secondary" disabled={busy} onClick={() => void requestCode()}>
                Send another code
              </button>
            </section>
          )}

          {view === "checkout" && (
            <section className="protection-step">
              <h3>Start cloud protection</h3>
              <p>Card details are handled by Stripe in your browser. Ormah never receives them.</p>
              {offer && (
                <div className="protection-offer">
                  <span>{offer.name}</span>
                  <strong>{price || `${offer.unit_amount} ${offer.currency}`} / {offer.interval}</strong>
                </div>
              )}
              <button className="protection-primary" disabled={busy} onClick={() => void openCheckout()}>
                {busy ? <LoaderCircle className="spin" size={15} /> : <ExternalLink size={15} />}
                Subscribe with Stripe
              </button>
              {checkoutPolling && (
                <div className="checkout-waiting">
                  <LoaderCircle className="spin" size={15} /> Waiting for Stripe to confirm payment
                </div>
              )}
                <button className="protection-secondary" disabled={busy} onClick={() => void checkPayment(false)}>
                <RefreshCw size={14} /> I've paid, check again
              </button>
            </section>
          )}

          {view === "summary" && !activeLabel && (
            <section className="protection-actions">
              {!account?.signed_in && (
                <button className="protection-primary" disabled={busy} onClick={() => {
                  setLoginPurpose("account");
                  setError(null);
                  setView("email");
                }}>
                  <LogIn size={16} /> Sign in to Ormah Cloud
                </button>
              )}
              {account?.signed_in && ["local_only", "sign_in_required", "stopped"].includes(status?.protection_state || "local_only") && (
                <button className="protection-primary" disabled={busy} onClick={() => void beginProtection()}>
                  <ShieldCheck size={16} /> Protect this memory
                </button>
              )}
              {account?.signed_in && ["protected", "changes_pending"].includes(status?.protection_state || "") && (
                <button className="protection-primary" disabled={busy} onClick={() => void runOperation("backup")}>
                  <RefreshCw size={15} /> Back up now
                </button>
              )}
              {account?.signed_in && status?.protection_state === "verification_pending" && (
                <button className="protection-primary" disabled={busy} onClick={() => void runOperation("verify")}>
                  <ShieldCheck size={15} /> Verify this recovery point
                </button>
              )}
              {["attention_required", "offline"].includes(status?.protection_state || "") && repairAction !== "none" && (
                <button className="protection-primary" disabled={busy} onClick={() => void runRepairAction()}>
                  <RefreshCw size={15} />
                  {repairAction === "verify" ? "Retry recovery check"
                    : repairAction === "signin" ? "Sign in again"
                      : repairAction === "subscribe" ? "Reactivate subscription"
                        : repairAction === "refresh" ? "Check connection"
                          : repairAction === "resume" ? "Resume protection"
                            : "Retry encrypted backup"}
                </button>
              )}
              {["subscription_required", "paused"].includes(status?.protection_state || "") && (
                <button className="protection-primary" disabled={busy} onClick={() => {
                  setView("checkout");
                  if (!offer) productBridge.offer().then(setOffer).catch(() => undefined);
                }}>
                  <CreditCard size={15} /> Reactivate protection
                </button>
              )}
            </section>
          )}

          {(status?.last_successful_verify_at || status?.last_successful_upload_at) && (
            <section className="protection-facts" aria-label="Protection details">
              <div><span>Last verified restorable</span><strong>{formatDate(status.last_successful_verify_at)}</strong></div>
              <div><span>Last encrypted upload</span><strong>{formatDate(status.last_successful_upload_at)}</strong></div>
              <div><span>Cloud can read</span><strong>metadata only</strong></div>
            </section>
          )}

          {recoveryKitSectionVisible(status) && view === "summary" && (
            <RecoveryKitSection
              ready={Boolean(status?.device_loss_recovery_ready)}
              verifiedAt={formatDate(status?.recovery_kit_verified_at ?? null)}
              busy={recoveryBusy}
              error={recoveryError}
              onSave={() => void saveRecoveryKit()}
              onOpenPrintable={() => void openPrintableRecoveryKit()}
            />
          )}

          {status?.warnings?.length ? (
            <section className="protection-warnings">
              {status.warnings.map((warning) => <p key={warning}>{warning}</p>)}
            </section>
          ) : null}

          {account?.signed_in && view === "summary" && (
            <section className="protection-account">
              <div>
                <span>Account</span>
                <strong>{account.email}</strong>
              </div>
              <div className="protection-account-actions">
                <button className="icon-text-button" onClick={async () => {
                  try {
                    await productBridge.openPortal();
                  } catch (err) {
                    setError(errorMessage(err, "Subscription management could not open."));
                  }
                }}>
                  <CreditCard size={14} /> Manage subscription
                </button>
                <button className="icon-text-button" onClick={async () => {
                  setBusy(true);
                  try {
                    const result = await productBridge.logout();
                    if (result.warning) onToast(result.warning, "info");
                    setOperation(null);
                    setView("summary");
                    await refresh();
                  } catch (err) {
                    setError(errorMessage(err, "Could not sign out locally."));
                  } finally {
                    setBusy(false);
                  }
                }}>
                  <LogOut size={14} /> Sign out
                </button>
              </div>
            </section>
          )}

          {status?.enabled && view === "summary" && (
            confirmStop ? (
              <section className="protection-stop-confirm">
                <strong>Stop cloud protection?</strong>
                <p>Future uploads will stop. Existing recovery points remain available, and your subscription is managed separately.</p>
                <div>
                  <button className="protection-secondary" onClick={() => setConfirmStop(false)}>Keep protection on</button>
                  <button className="protection-danger" disabled={busy} onClick={async () => {
                    await runOperation("disable");
                    setConfirmStop(false);
                  }}>Stop protection</button>
                </div>
              </section>
            ) : (
              <button className="protection-stop" disabled={busy} onClick={() => setConfirmStop(true)}>
                <CloudOff size={15} /> Stop future cloud backups
              </button>
            )
          )}
        </>
      )}
    </aside>
  );
}
