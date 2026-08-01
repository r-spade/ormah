import {
  AlertTriangle,
  Check,
  Download,
  LoaderCircle,
  Printer,
} from "lucide-react";

interface Props {
  ready: boolean;
  verifiedAt: string;
  busy: "save" | "print" | null;
  error: string | null;
  onSave: () => void;
  onOpenPrintable: () => void;
}

export default function RecoveryKitSection({
  ready,
  verifiedAt,
  busy,
  error,
  onSave,
  onOpenPrintable,
}: Props) {
  return (
    <section
      className={`recovery-kit-section ${ready ? "is-ready" : "action-required"}`}
      aria-label="Device-loss recovery"
      aria-busy={busy !== null}
    >
      <div className="recovery-kit-heading">
        <span aria-hidden="true">
          {ready ? <Check size={16} /> : <AlertTriangle size={16} />}
        </span>
        <div>
          <h3>Device-loss recovery</h3>
          <strong>{ready ? "Ready" : "Action required"}</strong>
        </div>
      </div>
      {ready ? (
        <p>Your saved recovery kit was reopened and verified {verifiedAt}.</p>
      ) : (
        <p>
          This file is the only way to restore if all trusted devices are lost.
          Save a separate copy now.
        </p>
      )}
      <div className="recovery-kit-warning">
        Anyone with this file can read your backups. Ormah cannot recover it for you.
      </div>
      {error && (
        <div className="recovery-kit-error" role="alert">
          <AlertTriangle size={14} />
          <span>{error}</span>
        </div>
      )}
      <button
        className="protection-primary"
        disabled={busy !== null}
        onClick={onSave}
      >
        {busy === "save" ? <LoaderCircle className="spin" size={15} /> : <Download size={15} />}
        {busy === "save" ? "Saving and checking…" : ready ? "Save another copy" : "Save recovery kit"}
      </button>
      <button
        className="protection-secondary"
        disabled={busy !== null}
        onClick={onOpenPrintable}
      >
        {busy === "print" ? <LoaderCircle className="spin" size={14} /> : <Printer size={14} />}
        {busy === "print" ? "Opening…" : "Open printable copy"}
      </button>
      <p className="recovery-print-note">
        Opens in your system viewer so you can print it. Opening it does not verify a saved copy.
      </p>
      <div className="sr-status" aria-live="polite">
        {busy === "save" ? "Saving and checking recovery kit"
          : busy === "print" ? "Opening printable recovery kit"
            : ready ? "Device-loss recovery ready"
              : "Device-loss recovery action required"}
      </div>
    </section>
  );
}
