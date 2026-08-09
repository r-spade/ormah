import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronRight,
  Circle,
  CloudOff,
  CloudDownload,
  CreditCard,
  ExternalLink,
  LoaderCircle,
  LockKeyhole,
  KeyRound,
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
  protectionCompletionSummary,
  protectionPresentation,
  protectionReconnectDelay,
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
  onRestoreComplete?: () => Promise<void> | void;
}

type View =
  | "summary"
  | "email"
  | "code"
  | "checkout"
  | "restore_key"
  | "restore_ready"
  | "restore_complete";
type LoginPurpose = "account" | "protect" | "restore";

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
  if (operation.kind === "restore") {
    switch (phase) {
      case "pending": return "Queued";
      case "running":
      case "discovering": return "Finding your latest recovery point";
      case "downloading": return "Downloading encrypted memory";
      case "decrypting": return "Decrypting on this device";
      case "checking": return "Checking every memory file";
      case "safety_backup": return "Protecting your current memory first";
      case "restoring": return "Restoring verified memory";
      case "rebuilding": return "Rebuilding local search";
      case "reloading": return "Refreshing your memory graph";
      default: return "Finishing recovery";
    }
  }
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

const RESTORE_PREPARE_STAGES = [
  { phase: "discovering", label: "Find latest recovery point" },
  { phase: "downloading", label: "Download encrypted memory" },
  { phase: "decrypting", label: "Decrypt on this device" },
  { phase: "checking", label: "Check files, identity, and search" },
] as const;

const RESTORE_APPLY_STAGES = [
  { phase: "safety_backup", label: "Save current memory locally" },
  { phase: "restoring", label: "Replace with verified recovery point" },
  { phase: "rebuilding", label: "Rebuild local search" },
  { phase: "reloading", label: "Refresh the memory graph" },
] as const;

function phaseIndex(phase: OperationPhase | null | undefined): number {
  if (phase === "pending" || phase === "running") return 0;
  return PROTECTION_STAGES.findIndex((stage) => stage.phase === phase);
}

function restorePhaseIndex(
  stages: ReadonlyArray<{ phase: string }>,
  phase: OperationPhase | null | undefined,
): number {
  if (phase === "pending" || phase === "running") return 0;
  return stages.findIndex((stage) => stage.phase === phase);
}

function operationSuccessMessage(operation: ProtectionOperation): string {
  switch (operation.kind) {
    case "enable": return "Memory protected and verified.";
    case "backup": return "Encrypted recovery point uploaded and restore-tested.";
    case "verify": return "Recovery point verified.";
    case "disable": return "Future cloud backups stopped.";
    case "restore": return "Latest verified memory restored.";
    default: return "Operation completed.";
  }
}

function errorMessage(value: unknown, fallback: string): string {
  if (value instanceof Error && value.message.trim()) return value.message;
  if (typeof value === "string" && value.trim()) return value;
  return fallback;
}

export default function ProtectionPanel({
  open,
  onClose,
  onToast,
  onStatusChange,
  onRestoreComplete,
}: Props) {
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
  const [refreshFailed, setRefreshFailed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checkoutPolling, setCheckoutPolling] = useState(false);
  const [confirmStop, setConfirmStop] = useState(false);
  const [recoveryBusy, setRecoveryBusy] = useState<"save" | "print" | null>(null);
  const [recoveryError, setRecoveryError] = useState<string | null>(null);
  const [bridgeVersion, setBridgeVersion] = useState(0);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const checkoutCheckInFlight = useRef(false);
  const offerRequested = useRef(false);
  const preparedRestoreRef = useRef<ProtectionOperation | null>(null);
  const reconnectAttempt = useRef(0);
  const desktop = isDesktopApp();

  const refresh = useCallback(async () => {
    if (!desktop) return;
    setLoading(true);
    try {
      try {
        const bridge = await productBridge.info();
        if (bridge.version < 1) throw new Error("unsupported bridge");
        setBridgeVersion(bridge.version);
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
      setRefreshFailed(false);
      onStatusChange?.(nextStatus);
      setError(null);
      if (nextAccount.signed_in && !offerRequested.current) {
        offerRequested.current = true;
        productBridge.offer().then(setOffer).catch(() => {
          offerRequested.current = false;
        });
      }
    } catch (err) {
      setRefreshFailed(true);
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
    if (!open || (!refreshFailed && status?.protection_state !== "offline")) {
      reconnectAttempt.current = 0;
      return;
    }
    if (loading || operationIsActive(operation)) return;
    const delay = protectionReconnectDelay(reconnectAttempt.current);
    const timer = window.setTimeout(() => {
      reconnectAttempt.current += 1;
      void refresh();
    }, delay);
    return () => window.clearTimeout(timer);
  }, [loading, open, operation?.status, refresh, refreshFailed, status?.protection_state]);

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
          if (next.kind === "restore" && next.phase === "ready") {
            preparedRestoreRef.current = next;
            setView("restore_ready");
            setError(null);
          } else if (next.kind === "restore" && next.reason_code === "key_missing") {
            setView("restore_key");
            setError(null);
          } else if (next.phase === "completed") {
            await refresh();
            if (next.kind === "restore") {
              preparedRestoreRef.current = null;
              await onRestoreComplete?.();
              setView("restore_complete");
            }
            onToast(operationSuccessMessage(next), "success");
          } else if (
            next.kind === "restore"
            && next.reason_code === "store_busy"
            && preparedRestoreRef.current
          ) {
            setOperation(preparedRestoreRef.current);
            setView("restore_ready");
            setError(next.message || "Memory is busy. Close active work and try restore again.");
          } else {
            if (next.kind === "restore") preparedRestoreRef.current = null;
            setOperation(null);
            if (next.message) setError(next.message);
          }
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
  }, [
    open,
    operation?.operation_id,
    operation?.phase,
    operation?.status,
    onRestoreComplete,
    onStatusChange,
    onToast,
    refresh,
  ]);

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

  const startRestorePreparation = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      setOperation(await productBridge.prepareRestore());
      setView("summary");
    } catch (err) {
      setError(errorMessage(err, "Recovery could not start."));
    } finally {
      setBusy(false);
    }
  }, []);

  const beginRestore = useCallback(async () => {
    setLoginPurpose("restore");
    setError(null);
    if (bridgeVersion < 2) {
      setError("Update Ormah Desktop to restore a cloud recovery point in the app.");
      return;
    }
    if (!account?.signed_in) {
      setView("email");
      return;
    }
    await startRestorePreparation();
  }, [account?.signed_in, bridgeVersion, startRestorePreparation]);

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
      if (loginPurpose === "restore") {
        await refresh();
        await startRestorePreparation();
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
  }, [
    bindAndContinue,
    code,
    email,
    loginPurpose,
    onToast,
    operation?.protection_intent_id,
    refresh,
    startRestorePreparation,
    status?.protection_intent_id,
  ]);

  const importRecoveryKit = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await productBridge.importRecoveryKit();
      if (result.status === "canceled") return;
      onToast("Recovery kit imported on this device.", "success");
      await startRestorePreparation();
    } catch (err) {
      setError(errorMessage(err, "The recovery kit could not be imported."));
    } finally {
      setBusy(false);
    }
  }, [onToast, startRestorePreparation]);

  const confirmRestore = useCallback(async () => {
    if (!operation?.operation_id || operation.phase !== "ready") return;
    setBusy(true);
    setError(null);
    try {
      setOperation(await productBridge.confirmRestore(operation.operation_id));
      setView("summary");
    } catch (err) {
      setError(errorMessage(err, "Memory could not be restored."));
    } finally {
      setBusy(false);
    }
  }, [operation?.operation_id, operation?.phase]);

  const cancelPreparedRestore = useCallback(async (closeAfter = false) => {
    const prepared = preparedRestoreRef.current;
    if (!prepared?.operation_id) {
      if (closeAfter) onClose();
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await productBridge.cancelRestore(prepared.operation_id);
      preparedRestoreRef.current = null;
      setOperation(null);
      setView("summary");
      if (closeAfter) onClose();
    } catch (err) {
      setError(errorMessage(err, "The temporary recovery copy could not be discarded."));
    } finally {
      setBusy(false);
    }
  }, [onClose]);

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
  const restoreApplying = operation?.kind === "restore"
    && ["safety_backup", "restoring", "rebuilding", "reloading"].includes(activePhase || "");
  const activeStages = operation?.kind === "restore"
    ? restoreApplying ? RESTORE_APPLY_STAGES : RESTORE_PREPARE_STAGES
    : PROTECTION_STAGES;
  const activeStageIndex = operation?.kind === "restore"
    ? restorePhaseIndex(activeStages, activePhase)
    : phaseIndex(activePhase);
  const completionSummary = protectionCompletionSummary(operation);
  const completedStages = operation?.kind === "verify"
    ? PROTECTION_STAGES.slice(4)
    : PROTECTION_STAGES;
  const summaryTone = activeOperation
    ? "working"
    : view === "restore_ready" || view === "restore_complete"
      ? "success"
      : view === "restore_key"
        ? "warning"
        : presentation.tone;
  const restoreFlow = operation?.kind === "restore"
    || view === "restore_key"
    || view === "restore_ready"
    || view === "restore_complete";
  const summaryTitle = activeLabel
    || (view === "restore_ready" ? "Recovery point ready"
      : view === "restore_complete" ? "Memory restored"
        : view === "restore_key" ? "Recovery kit needed"
          : presentation.title);
  const summaryDetail = activeOperation && operation?.kind === "restore"
    ? restoreApplying
      ? "Your current memory is saved locally before the verified copy replaces it."
      : "Ormah is checking a temporary copy. Your current memory is unchanged."
    : view === "restore_ready"
      ? "This recovery point passed file, identity, index, and search checks on this device."
      : view === "restore_complete"
        ? "The graph and local search now use the recovered memory."
        : view === "restore_key"
          ? "Choose the recovery kit saved when cloud protection was created."
          : activeOperation
            ? "Ormah is creating and restore-testing an encrypted recovery point."
            : presentation.detail;
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
    reconnectAttempt.current = 0;
    await refresh();
  }, [beginProtection, offer, refresh, repairAction, runOperation, status]);

  return (
    <aside className={`side-panel protection-panel ${open ? "open" : ""}`} aria-hidden={!open}>
      <div className="side-panel-header">
        <div>
          <div className="protection-eyebrow">Ormah Cloud</div>
          <h2 className="review-title" ref={headingRef} tabIndex={-1}>Protection</h2>
        </div>
        <button
          className="icon-button"
          onClick={() => {
            if (view === "restore_ready") void cancelPreparedRestore(true);
            else onClose();
          }}
          aria-label="Close protection"
        >
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
              <h3>{summaryTitle}</h3>
              <p>{summaryDetail}</p>
            </div>
          </section>

          {activeOperation && activeStageIndex >= 0 && (
            <section className="protection-progress" aria-label={restoreFlow ? "Recovery progress" : "Protection progress"}>
              <div className="protection-progress-heading">
                <span>{operation?.kind === "restore" ? "Restore memory" : "Recovery check"}</span>
                <strong>In progress</strong>
              </div>
              <ol>
                {activeStages.map((stage, index) => {
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
              <p>{operation?.kind === "restore" && restoreApplying
                ? "Ormah creates a local safety backup before replacing anything."
                : "Verification uses a temporary copy. Your live memory is never replaced."}</p>
            </section>
          )}

          {completionSummary && (
            <section className="protection-progress protection-complete" aria-label="Protection completed">
              <div className="protection-progress-heading">
                <span>Recovery check</span>
                <strong>Complete</strong>
              </div>
              <ol>
                {completedStages.map((stage) => (
                  <li className="complete" key={stage.phase}>
                    <span aria-hidden="true"><Check size={13} /></span>
                    <span>{stage.label}</span>
                  </li>
                ))}
              </ol>
              <p>{completionSummary}</p>
            </section>
          )}

          {error && (
            <div className="protection-error" role="alert">
              <AlertTriangle size={15} />
              <span>{error}</span>
            </div>
          )}

          <div className="sr-status" aria-live="polite">{summaryTitle}</div>

          {view === "email" && (
            <section className="protection-step">
              <button className="step-back" onClick={() => setView("summary")}>Back</button>
              <h3>{loginPurpose === "protect"
                ? "Sign in to continue"
                : loginPurpose === "restore"
                  ? "Sign in to recover memory"
                  : "Sign in to Ormah Cloud"}</h3>
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
                {loginPurpose === "account" ? "Sign in" : "Verify and continue"}
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

          {view === "restore_key" && (
            <section className="protection-step restore-step">
              <button className="step-back" onClick={() => {
                setOperation(null);
                setView("summary");
              }}>Back</button>
              <h3>Unlock your encrypted recovery point</h3>
              <p>
                The recovery kit contains the encryption identity needed on this device.
                It stays local and is never sent to Ormah Cloud.
              </p>
              <button className="protection-primary" disabled={busy} onClick={() => void importRecoveryKit()}>
                {busy ? <LoaderCircle className="spin" size={15} /> : <KeyRound size={15} />}
                Choose recovery kit
              </button>
              <button className="protection-secondary" disabled={busy} onClick={() => {
                setOperation(null);
                setView("summary");
              }}>Cancel</button>
            </section>
          )}

          {view === "restore_ready" && operation && (
            <section className="protection-step restore-step restore-confirm">
              <button
                className="step-back"
                disabled={busy}
                onClick={() => void cancelPreparedRestore()}
              >Cancel</button>
              <div className="restore-proof">
                <ShieldCheck size={18} />
                <div>
                  <strong>{new Intl.NumberFormat().format(operation.verified_node_count || 0)} memories checked</strong>
                  <span>Recovered from {formatDate(operation.snapshot_created_at || null)}</span>
                </div>
              </div>
              {Boolean(operation.skipped_newer_snapshots) && (
                <div className="restore-fallback" role="status">
                  <AlertTriangle size={14} />
                  The newest recovery point did not pass local checks, so Ormah selected the next safe one.
                </div>
              )}
              <h3>Replace this device's memory?</h3>
              <p>
                Ormah first saves the current graph as a local safety backup, then restores this verified copy.
              </p>
              <button className="protection-primary" disabled={busy} onClick={() => void confirmRestore()}>
                {busy ? <LoaderCircle className="spin" size={15} /> : <CloudDownload size={15} />}
                Restore {new Intl.NumberFormat().format(operation.verified_node_count || 0)} memories
              </button>
            </section>
          )}

          {view === "restore_complete" && operation && (
            <section className="protection-step restore-step restore-complete">
              <div className="restore-proof">
                <Check size={18} />
                <div>
                  <strong>{new Intl.NumberFormat().format(operation.verified_node_count || 0)} memories restored</strong>
                  <span>Your graph and search index are ready.</span>
                </div>
              </div>
              {operation.safety_backup_name && (
                <p>Your previous memory was saved locally as <strong>{operation.safety_backup_name}</strong>.</p>
              )}
              <button className="protection-primary" onClick={() => {
                setOperation(null);
                setView("summary");
                onClose();
              }}>
                <Check size={15} /> View restored memory
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

          {view === "summary" && !activeLabel && (
            <section className="restore-entry">
              <div>
                <CloudDownload size={17} />
                <div>
                  <strong>Recover this memory</strong>
                  <span>Restore the latest cloud recovery point that passes local checks.</span>
                </div>
              </div>
              <button className="protection-secondary" disabled={busy} onClick={() => void beginRestore()}>
                Restore latest verified backup
              </button>
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
