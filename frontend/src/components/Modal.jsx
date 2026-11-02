import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

export default function Modal({
  children,
  onClose,
  labelledBy,
  panelClassName,
  closeOnBackdrop = false,
  closeDisabled = false,
}) {
  const dialogRef = useRef(null);
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;

    document.body.style.overflow = "hidden";
    dialogRef.current?.focus();

    function handleKeyDown(event) {
      if (event.key === "Escape" && !closeDisabled) {
        onCloseRef.current();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) {
        previousFocus.focus();
      }
    };
  }, [closeDisabled]);

  return createPortal(
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby={labelledBy}
      tabIndex={-1}
      className="fixed inset-0 z-[2000] flex items-center justify-center overflow-y-auto bg-black/50 p-4 backdrop-blur-sm outline-none"
      onClick={(event) => {
        if (
          closeOnBackdrop &&
          !closeDisabled &&
          event.target === event.currentTarget
        ) {
          onClose();
        }
      }}
    >
      <div
        className={`relative w-full max-h-[calc(100dvh-2rem)] overflow-y-auto ${panelClassName}`}
      >
        {children}
      </div>
    </div>,
    document.body,
  );
}
