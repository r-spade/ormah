export interface ToastData {
  id: number;
  message: string;
  type: "success" | "error" | "info";
}

interface Props {
  toasts: ToastData[];
}

export default function ToastContainer({ toasts }: Props) {
  if (toasts.length === 0) return null;

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.type}`}>
          {t.message}
        </div>
      ))}
    </div>
  );
}
