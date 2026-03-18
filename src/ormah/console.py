"""Shared output formatting for CLI and setup — matches install.sh visual style."""

from __future__ import annotations

import os
import random
import shutil
import sys
import threading
import time

# ── Color detection ──────────────────────────────────────────────────────────

_use_color: bool | None = None


def _should_use_color() -> bool:
    global _use_color
    if _use_color is not None:
        return _use_color
    if os.environ.get("NO_COLOR"):
        _use_color = False
    elif os.environ.get("FORCE_COLOR"):
        _use_color = True
    else:
        _use_color = sys.stdout.isatty()
    return _use_color


def _reset_color_cache() -> None:
    """Reset color detection cache (for testing)."""
    global _use_color
    _use_color = None


_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_TEAL = "\033[38;5;37m"


# ── Output functions (mirror install.sh lines 22-26) ────────────────────────


def step(msg: str) -> None:
    """Section header: \\n==> msg (bold)."""
    if _should_use_color():
        print(f"\n{_BOLD}==>{_RESET} {msg}")
    else:
        print(f"\n==> {msg}")


def info(msg: str) -> None:
    """In-progress: [..] msg (bold brackets)."""
    if _should_use_color():
        print(f"{_BOLD}[..]{_RESET} {msg}")
    else:
        print(f"[..] {msg}")


def ok(msg: str) -> None:
    """Success: [ok] msg (green)."""
    if _should_use_color():
        print(f"{_GREEN}[ok]{_RESET} {msg}")
    else:
        print(f"[ok] {msg}")


def warn(msg: str) -> None:
    """Warning: [!!] msg (yellow)."""
    if _should_use_color():
        print(f"{_YELLOW}[!!]{_RESET} {msg}")
    else:
        print(f"[!!] {msg}")


def fail(msg: str) -> None:
    """Error: [xx] msg (bold, to stderr)."""
    if _should_use_color():
        print(f"{_BOLD}[xx]{_RESET} {msg}", file=sys.stderr)
    else:
        print(f"[xx] {msg}", file=sys.stderr)


# ── Spinner ──────────────────────────────────────────────────────────────────

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Spinner:
    """Animated braille spinner with elapsed time display.

    Usage::

        with Spinner("Loading...") as sp:
            do_work()
            sp.update("Still going...")
        # prints [ok] Done  or [!!] Failed  on exit
    """

    def __init__(self, message: str = "") -> None:
        self._message = message
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._is_tty = sys.stdout.isatty()
        self._final_printed = False
        self._cols = shutil.get_terminal_size().columns

    def __enter__(self) -> Spinner:
        self._start_time = time.monotonic()
        if self._is_tty:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            # Non-TTY: print initial message as [..] line
            if self._message:
                print(f"[..] {self._message}", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._is_tty and not self._final_printed:
            # Clear the spinner line
            sys.stdout.write("\r" + " " * self._cols + "\r")
            sys.stdout.flush()
        return None

    def update(self, message: str) -> None:
        """Thread-safe message change."""
        with self._lock:
            old = self._message
            self._message = message
        # Non-TTY: print each phase change as a new line
        if not self._is_tty and message != old:
            print(f"[..] {message}", flush=True)

    def succeed(self, message: str) -> None:
        """Stop spinner and print [ok] final line."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        elapsed = time.monotonic() - self._start_time
        self._final_printed = True
        if self._is_tty:
            sys.stdout.write("\r" + " " * self._cols + "\r")
            sys.stdout.flush()
        elapsed_str = f"  {elapsed:.0f}s" if elapsed >= 3.0 else ""
        if _should_use_color():
            print(f"{_GREEN}[ok]{_RESET} {message}{elapsed_str}")
        else:
            print(f"[ok] {message}{elapsed_str}")

    def fail(self, message: str) -> None:
        """Stop spinner and print [!!] final line."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._final_printed = True
        if self._is_tty:
            sys.stdout.write("\r" + " " * self._cols + "\r")
            sys.stdout.flush()
        if _should_use_color():
            print(f"{_YELLOW}[!!]{_RESET} {message}")
        else:
            print(f"[!!] {message}")

    def _spin(self) -> None:
        """Background thread: render braille animation."""
        idx = 0
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    msg = self._message
                elapsed = time.monotonic() - self._start_time
                elapsed_str = f"  {elapsed:.0f}s" if elapsed >= 3.0 else ""

                char = _BRAILLE[idx % len(_BRAILLE)]
                # Truncate to terminal width: "  X msg  NNs"
                max_msg = self._cols - len(elapsed_str) - 4  # "  X " prefix
                if len(msg) > max_msg > 0:
                    msg = msg[: max_msg - 1] + "\u2026"
                line = f"  {char} {msg}{elapsed_str}"

                sys.stdout.write("\r" + " " * self._cols + "\r" + line)
                sys.stdout.flush()
                idx += 1
                self._stop_event.wait(0.1)
        except Exception:
            pass


# ── Finale animation ────────────────────────────────────────────────────────


def play_finale() -> None:
    """Play a ~2.5s terminal animation: 'ormah' dissolves into a sphere.

    TTY only — skips silently when piped.
    """
    if not sys.stdout.isatty():
        return

    cols = shutil.get_terminal_size().columns
    rows = 7  # vertical space used

    # Center of the display area
    center_col = cols // 2
    # We'll use rows 0..6 relative to current cursor position

    # Hide cursor
    sys.stdout.write("\033[?25l")
    # Reserve vertical space
    sys.stdout.write("\n" * rows)
    # Move cursor up to the start of our animation region
    sys.stdout.write(f"\033[{rows}A")
    sys.stdout.flush()

    # Save the base row (we'll use relative positioning from here)
    use_color = _should_use_color()
    teal = _TEAL if use_color else ""
    bold = _BOLD if use_color else ""
    reset = _RESET if use_color else ""

    word = "ormah"
    word_row = 3  # middle of our 7-row space
    word_start_col = center_col - len(word) // 2

    # Target for sphere: right-center area
    sphere_col = center_col
    sphere_row = 3

    # Sphere pattern (small)
    sphere = [
        "  ##  ",
        " #### ",
        " #### ",
        "  ##  ",
    ]
    sphere_h = len(sphere)
    sphere_w = max(len(r) for r in sphere)
    sphere_top = sphere_row - sphere_h // 2
    sphere_left = sphere_col - sphere_w // 2

    def _goto(row: int, col: int) -> str:
        """Move cursor to row,col relative to animation region top."""
        # row is 0-based from the top of our reserved space
        return f"\033[{row + 1};{max(1, col + 1)}H"

    def _clear_region() -> None:
        """Clear the animation region."""
        # Save cursor, clear each row, restore
        for r in range(rows):
            sys.stdout.write(f"\033[{r + 1};1H" + " " * cols)
        sys.stdout.flush()

    # Save absolute position so we can get back
    # We're currently at the top of the animation region
    # Use absolute positioning by getting current position
    # Actually, we'll use save/restore cursor
    sys.stdout.write("\033[s")  # save cursor position (top of animation region)

    # --- Phase 1: Display the word (frames 1-4, ~400ms) ---
    _clear_region()
    for i in range(4):
        sys.stdout.write("\033[u")  # restore to top
        sys.stdout.write(f"\033[{word_row + 1};{word_start_col + 1}H")
        sys.stdout.write(f"{bold}{teal}{word}{reset}")
        sys.stdout.flush()
        time.sleep(0.1)

    # --- Phase 2: Dissolve (frames 5-12, ~800ms) ---
    # Create particles from each letter
    particles: list[dict] = []
    for ci, ch in enumerate(word):
        col_pos = word_start_col + ci
        # Each letter becomes 2-3 particles
        for _ in range(random.randint(2, 3)):
            particles.append({
                "row": float(word_row),
                "col": float(col_pos),
                "char": random.choice(["·", "•", "∘"]),
                "vx": random.uniform(0.3, 1.2),
                "vy": random.uniform(-0.6, 0.6),
                "dissolve_frame": 5 + ci,  # stagger left-to-right
                "alive": False,
            })

    visible_word = list(word)
    for frame in range(5, 13):
        _clear_region()
        sys.stdout.write("\033[u")

        # Draw remaining letters
        for ci, ch in enumerate(visible_word):
            if ch != " ":
                sys.stdout.write(
                    f"\033[{word_row + 1};{word_start_col + ci + 1}H"
                    f"{bold}{teal}{ch}{reset}"
                )

        # Activate and move particles
        for p in particles:
            if frame >= p["dissolve_frame"]:
                if not p["alive"]:
                    p["alive"] = True
                    # Remove corresponding letter
                    letter_idx = int(p["col"] - word_start_col)
                    if 0 <= letter_idx < len(visible_word):
                        visible_word[letter_idx] = " "

                # Update position
                p["col"] += p["vx"]
                p["row"] += p["vy"]
                p["vy"] += random.uniform(-0.1, 0.1)  # slight vertical wander

                # Draw particle
                pr = int(round(p["row"]))
                pc = int(round(p["col"]))
                if 0 <= pr < rows and 0 < pc < cols:
                    sys.stdout.write(
                        f"\033[{pr + 1};{pc + 1}H{teal}{p['char']}{reset}"
                    )

        sys.stdout.flush()
        time.sleep(0.1)

    # --- Phase 3: Converge (frames 13-20, ~800ms) ---
    for frame in range(13, 21):
        _clear_region()
        sys.stdout.write("\033[u")

        progress = (frame - 13) / 7.0  # 0.0 -> 1.0
        for p in particles:
            if not p["alive"]:
                continue
            # Lerp toward sphere center
            target_row = float(sphere_row)
            target_col = float(sphere_col)
            p["row"] += (target_row - p["row"]) * (0.2 + 0.3 * progress)
            p["col"] += (target_col - p["col"]) * (0.2 + 0.3 * progress)
            # Add diminishing randomness
            scatter = 0.3 * (1.0 - progress)
            p["row"] += random.uniform(-scatter, scatter)
            p["col"] += random.uniform(-scatter, scatter)

            pr = int(round(p["row"]))
            pc = int(round(p["col"]))
            if 0 <= pr < rows and 0 < pc < cols:
                sys.stdout.write(
                    f"\033[{pr + 1};{pc + 1}H{teal}{p['char']}{reset}"
                )

        sys.stdout.flush()
        time.sleep(0.1)

    # --- Phase 4: Form sphere (frames 21-25, ~500ms) ---
    for frame in range(21, 26):
        _clear_region()
        sys.stdout.write("\033[u")

        # Alternate between filled and ringed for pulse effect
        use_ring = frame in (22, 24)
        dot = "\u25c9" if use_ring else "\u25cf"  # ◉ vs ●

        for ri, row_str in enumerate(sphere):
            for ci_s, ch in enumerate(row_str):
                if ch == "#":
                    r = sphere_top + ri
                    c = sphere_left + ci_s
                    if 0 <= r < rows and 0 < c < cols:
                        sys.stdout.write(
                            f"\033[{r + 1};{c + 1}H{teal}{dot}{reset}"
                        )

        sys.stdout.flush()
        time.sleep(0.1)

    # Hold the final frame briefly
    time.sleep(0.3)

    # Clean up: move cursor below the animation region and show cursor
    sys.stdout.write(f"\033[{rows + 1};1H")
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()
