"""
Desktop GUI for Language Model Testing and Inference.
Built with Python Tkinter and TTK.
Features:
- Checkpoint loader with model parameter inspector.
- Real-time token-by-token streaming generation.
- Interactive parameter controls: Temperature, Top-K, Top-P, Repetition Penalty, Max Tokens.
- Thread-safe asynchronous generation with Stop/Cancel control.
- Preset prompt selector, copy-to-clipboard, and live speed metrics.
"""

import os
import sys
import time
import queue
import threading
from typing import Optional, List
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Ensure utf-8 console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure scripts dir in sys.path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from config import MODELS_DIR, DEVICE
from inference import ModelInferenceEngine

# -----------------------------------------------------------------------------
# Color Palette & Styles (Modern Clean Slate Theme)
# -----------------------------------------------------------------------------
BG_COLOR = "#181825"        # Main Background
PANEL_BG = "#1E1E2E"        # Cards / Panels Background
BORDER_COLOR = "#313244"    # Subtle borders
TEXT_COLOR = "#CDD6F4"      # Primary Text
SUBTEXT_COLOR = "#A6ADC8"   # Secondary Text
ACCENT_COLOR = "#89B4FA"    # Blue Accent
ACCENT_HOVER = "#B4BEFE"
SUCCESS_COLOR = "#A6E3A1"   # Green
DANGER_COLOR = "#F38BA8"    # Red (Stop)
INPUT_BG = "#11111B"        # Text Boxes

class LanguageModelGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Antigravity LM Studio - Desktop Testing Interface")
        self.root.geometry("1100x780")
        self.root.minsize(900, 650)
        self.root.configure(bg=BG_COLOR)

        # State
        self.engine: Optional[ModelInferenceEngine] = None
        self.is_generating = False
        self.stop_requested = threading.Event()
        self.token_queue = queue.Queue()

        # Setup modern TTK styles
        self._setup_styles()

        # Build UI layout
        self._build_layout()

        # Auto-detect checkpoints and load default
        self._populate_checkpoints()

        # Start queue polling for thread-safe UI updates
        self.root.after(20, self._process_token_queue)

    def _setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG_COLOR)
        style.configure("Panel.TFrame", background=PANEL_BG, relief="flat")

        # Labels
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=TEXT_COLOR, background=PANEL_BG)
        style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "bold"), foreground=ACCENT_COLOR, background=PANEL_BG)
        style.configure("Panel.TLabel", background=PANEL_BG, foreground=TEXT_COLOR)
        style.configure("Subtext.TLabel", background=PANEL_BG, foreground=SUBTEXT_COLOR, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=INPUT_BG, foreground=SUBTEXT_COLOR, font=("Segoe UI", 9))

        # Combobox
        style.configure("TCombobox", fieldbackground=INPUT_BG, background=BORDER_COLOR, foreground=TEXT_COLOR)
        style.map("TCombobox", fieldbackground=[("readonly", INPUT_BG)], foreground=[("readonly", TEXT_COLOR)])

        # Scales (Sliders)
        style.configure("Horizontal.TScale", background=PANEL_BG, troughcolor=INPUT_BG)

    def _build_layout(self):
        # Main split container
        main_container = tk.Frame(self.root, bg=BG_COLOR)
        main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left Sidebar (Parameters & Model selection)
        left_sidebar = tk.Frame(main_container, bg=PANEL_BG, width=340, padx=14, pady=14, highlightthickness=1, highlightbackground=BORDER_COLOR)
        left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_sidebar.pack_propagate(False)

        # Right Main Panel (Prompts & Real-time generation)
        right_panel = tk.Frame(main_container, bg=PANEL_BG, padx=16, pady=14, highlightthickness=1, highlightbackground=BORDER_COLOR)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_sidebar(left_sidebar)
        self._build_main_panel(right_panel)

    def _build_sidebar(self, parent: tk.Frame):
        # 1. Header
        header = ttk.Label(parent, text="⚙ Settings & Model", style="Header.TLabel")
        header.pack(anchor="w", pady=(0, 12))

        # 2. Checkpoint Selection Card
        ckpt_frame = tk.Frame(parent, bg=PANEL_BG)
        ckpt_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(ckpt_frame, text="Model Checkpoint:", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 4))
        
        self.ckpt_combo = ttk.Combobox(ckpt_frame, state="readonly", font=("Segoe UI", 9))
        self.ckpt_combo.pack(fill=tk.X, pady=(0, 6))
        self.ckpt_combo.bind("<<ComboboxSelected>>", self._on_checkpoint_selected)

        btn_row = tk.Frame(ckpt_frame, bg=PANEL_BG)
        btn_row.pack(fill=tk.X)
        browse_btn = tk.Button(btn_row, text="📁 Browse...", bg=BORDER_COLOR, fg=TEXT_COLOR, activebackground=ACCENT_COLOR,
                               activeforeground="#000", bd=0, padx=8, pady=3, font=("Segoe UI", 8), cursor="hand2", command=self._browse_checkpoint)
        browse_btn.pack(side=tk.LEFT, padx=(0, 6))

        reload_btn = tk.Button(btn_row, text="🔄 Reload", bg=BORDER_COLOR, fg=TEXT_COLOR, activebackground=ACCENT_COLOR,
                                activeforeground="#000", bd=0, padx=8, pady=3, font=("Segoe UI", 8), cursor="hand2", command=self._reload_model)
        reload_btn.pack(side=tk.LEFT)

        # Model Info Label
        self.model_info_lbl = ttk.Label(ckpt_frame, text="Model: Not loaded", style="Subtext.TLabel")
        self.model_info_lbl.pack(anchor="w", pady=(6, 0))

        # Divider
        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill=tk.X, pady=10)

        # 3. Sampling Parameters
        ttk.Label(parent, text="Generation Parameters", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 8))

        # Temperature
        self.temp_var = tk.DoubleVar(value=0.4)
        self.temp_lbl = ttk.Label(parent, text="Temperature: 0.40", style="Panel.TLabel")
        self.temp_lbl.pack(anchor="w")
        temp_scale = ttk.Scale(parent, from_=0.0, to=2.0, variable=self.temp_var, command=lambda v: self.temp_lbl.config(text=f"Temperature: {float(v):.2f}"))
        temp_scale.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(parent, text="0.0 = Deterministic (Greedy), 1.0 = Creative", style="Subtext.TLabel").pack(anchor="w", pady=(0, 8))

        # Top-K
        self.top_k_var = tk.IntVar(value=40)
        self.top_k_lbl = ttk.Label(parent, text="Top-K: 40", style="Panel.TLabel")
        self.top_k_lbl.pack(anchor="w")
        top_k_scale = ttk.Scale(parent, from_=0, to=100, variable=self.top_k_var, command=lambda v: self.top_k_lbl.config(text=f"Top-K: {int(float(v))}"))
        top_k_scale.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(parent, text="Restricts sampling to K highest probability tokens (0 = off)", style="Subtext.TLabel").pack(anchor="w", pady=(0, 8))

        # Top-P (Nucleus)
        self.top_p_var = tk.DoubleVar(value=0.90)
        self.top_p_lbl = ttk.Label(parent, text="Top-P (Nucleus): 0.90", style="Panel.TLabel")
        self.top_p_lbl.pack(anchor="w")
        top_p_scale = ttk.Scale(parent, from_=0.1, to=1.0, variable=self.top_p_var, command=lambda v: self.top_p_lbl.config(text=f"Top-P (Nucleus): {float(v):.2f}"))
        top_p_scale.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(parent, text="Cumulative probability cutoff threshold", style="Subtext.TLabel").pack(anchor="w", pady=(0, 8))

        # Max New Tokens
        self.max_tokens_var = tk.IntVar(value=80)
        self.max_tokens_lbl = ttk.Label(parent, text="Max New Tokens: 80", style="Panel.TLabel")
        self.max_tokens_lbl.pack(anchor="w")
        max_tokens_scale = ttk.Scale(parent, from_=10, to=300, variable=self.max_tokens_var, command=lambda v: self.max_tokens_lbl.config(text=f"Max New Tokens: {int(float(v))}"))
        max_tokens_scale.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(parent, text="Length of generated token sequence", style="Subtext.TLabel").pack(anchor="w", pady=(0, 8))

        # Repetition Penalty
        self.rep_pen_var = tk.DoubleVar(value=1.10)
        self.rep_pen_lbl = ttk.Label(parent, text="Repetition Penalty: 1.10", style="Panel.TLabel")
        self.rep_pen_lbl.pack(anchor="w")
        rep_scale = ttk.Scale(parent, from_=1.0, to=2.0, variable=self.rep_pen_var, command=lambda v: self.rep_pen_lbl.config(text=f"Repetition Penalty: {float(v):.2f}"))
        rep_scale.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(parent, text="Discourages repeating recent words", style="Subtext.TLabel").pack(anchor="w", pady=(0, 8))

        # Preset Prompts Dropdown
        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill=tk.X, pady=8)
        ttk.Label(parent, text="Preset Prompts:", style="Panel.TLabel").pack(anchor="w", pady=(0, 4))
        
        presets = [
            "User: Hello!\nAssistant:",
            "User: What is artificial intelligence?\nAssistant:",
            "User: Who made you?\nAssistant:",
            "User: Can you tell me a joke?\nAssistant:",
            "User: What is Python?\nAssistant:",
            "Wikipedia is an online",
            "The history of"
        ]
        self.preset_combo = ttk.Combobox(parent, values=presets, state="readonly", font=("Segoe UI", 9))
        self.preset_combo.pack(fill=tk.X)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

    def _build_main_panel(self, parent: tk.Frame):
        # 1. Header & Title
        title_row = tk.Frame(parent, bg=PANEL_BG)
        title_row.pack(fill=tk.X, pady=(0, 8))

        title = ttk.Label(title_row, text="Transformer Language Model Playground", style="Header.TLabel")
        title.pack(side=tk.LEFT)

        self.device_badge = tk.Label(title_row, text=f"⚡ Device: {DEVICE}", bg="#313244", fg=SUCCESS_COLOR, font=("Segoe UI", 9, "bold"), padx=8, pady=2)
        self.device_badge.pack(side=tk.RIGHT)

        # 2. Input Prompt Card
        ttk.Label(parent, text="Input Prompt (Chatbot / Text):", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 4))
        
        self.prompt_text = tk.Text(parent, height=4, bg=INPUT_BG, fg=TEXT_COLOR, insertbackground=ACCENT_COLOR,
                                   relief="flat", highlightthickness=1, highlightbackground=BORDER_COLOR,
                                   font=("Consolas", 10), padx=8, pady=8)
        self.prompt_text.pack(fill=tk.X, pady=(0, 8))
        self.prompt_text.insert(tk.END, "User: Hello!\nAssistant:")

        # 3. Action Buttons Row
        action_row = tk.Frame(parent, bg=PANEL_BG)
        action_row.pack(fill=tk.X, pady=(0, 10))

        self.generate_btn = tk.Button(
            action_row,
            text="▶  Generate Text",
            bg=ACCENT_COLOR,
            fg="#11111B",
            activebackground=ACCENT_HOVER,
            activeforeground="#000",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            padx=18,
            pady=7,
            cursor="hand2",
            command=self._toggle_generation
        )
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 8))

        clear_btn = tk.Button(
            action_row,
            text="Clear",
            bg=BORDER_COLOR,
            fg=TEXT_COLOR,
            activebackground=PANEL_BG,
            font=("Segoe UI", 10),
            bd=0,
            padx=14,
            pady=7,
            cursor="hand2",
            command=self._clear_all
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 8))

        copy_btn = tk.Button(
            action_row,
            text="📋 Copy Output",
            bg=BORDER_COLOR,
            fg=TEXT_COLOR,
            activebackground=PANEL_BG,
            font=("Segoe UI", 10),
            bd=0,
            padx=14,
            pady=7,
            cursor="hand2",
            command=self._copy_output
        )
        copy_btn.pack(side=tk.LEFT)

        # 4. Output Display (Streaming)
        ttk.Label(parent, text="Generated Text (Real-time Token-by-Token Stream):", style="SubHeader.TLabel").pack(anchor="w", pady=(6, 4))

        output_frame = tk.Frame(parent, bg=INPUT_BG, highlightthickness=1, highlightbackground=BORDER_COLOR)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.output_text = tk.Text(
            output_frame,
            bg=INPUT_BG,
            fg=TEXT_COLOR,
            insertbackground=ACCENT_COLOR,
            relief="flat",
            wrap=tk.WORD,
            font=("Consolas", 11),
            padx=10,
            pady=10
        )
        scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tag for prompt highlight vs generated text
        self.output_text.tag_configure("prompt_tag", foreground="#89DCEB", font=("Consolas", 11, "bold"))
        self.output_text.tag_configure("token_tag", foreground=TEXT_COLOR)

        # 5. Status / Telemetry Bar
        self.status_bar = tk.Frame(parent, bg=INPUT_BG, padx=8, pady=4, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.status_bar.pack(fill=tk.X)

        self.status_lbl = ttk.Label(self.status_bar, text="Status: Ready", style="Status.TLabel")
        self.status_lbl.pack(side=tk.LEFT)

        self.speed_lbl = ttk.Label(self.status_bar, text="Tokens: 0 | Speed: 0.0 tok/s | Time: 0.0s", style="Status.TLabel")
        self.speed_lbl.pack(side=tk.RIGHT)

    def _populate_checkpoints(self):
        ckpt_files = []
        if os.path.exists(MODELS_DIR):
            for f in os.listdir(MODELS_DIR):
                if f.endswith(".pt"):
                    ckpt_files.append(f)
        
        # Sort so chatbot_model.pt or best_model.pt is first
        priority = ["chatbot_model.pt", "best_model.pt", "latest_checkpoint.pt", "checkpoint_step_60.pt", "checkpoint_step_30.pt", "model_checkpoint.pt"]
        sorted_files = []
        for p in priority:
            if p in ckpt_files and p not in sorted_files:
                sorted_files.append(p)
        for f in sorted(ckpt_files):
            if f not in sorted_files:
                sorted_files.append(f)

        self.ckpt_combo["values"] = sorted_files
        if sorted_files:
            self.ckpt_combo.current(0)
            self._load_selected_checkpoint()

    def _on_checkpoint_selected(self, event=None):
        self._load_selected_checkpoint()

    def _browse_checkpoint(self):
        filename = filedialog.askopenfilename(
            initialdir=MODELS_DIR,
            title="Select PyTorch Checkpoint",
            filetypes=[("PyTorch Checkpoints", "*.pt"), ("All Files", "*.*")]
        )
        if filename:
            current_vals = list(self.ckpt_combo["values"])
            basename = os.path.basename(filename)
            if basename not in current_vals:
                current_vals.append(basename)
                self.ckpt_combo["values"] = current_vals
            self.ckpt_combo.set(basename)
            self._load_selected_checkpoint(full_path=filename)

    def _reload_model(self):
        self._load_selected_checkpoint()

    def _load_selected_checkpoint(self, full_path: Optional[str] = None):
        selected_name = self.ckpt_combo.get()
        if not selected_name and not full_path:
            return

        ckpt_path = full_path or os.path.join(MODELS_DIR, selected_name)
        if not os.path.exists(ckpt_path):
            self.model_info_lbl.config(text=f"Error: {selected_name} not found")
            return

        self.status_lbl.config(text=f"Loading {selected_name}...")
        self.root.update_idletasks()

        try:
            self.engine = ModelInferenceEngine(checkpoint_path=ckpt_path)
            param_count = sum(p.numel() for p in self.engine.model.parameters())
            cfg = self.engine.config
            info_str = f"Params: {param_count:,} | {cfg.n_layer}L, {cfg.n_embd}D, {cfg.n_head}H"
            self.model_info_lbl.config(text=info_str)
            self.status_lbl.config(text=f"Loaded: {selected_name}")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load checkpoint:\n{e}")
            self.model_info_lbl.config(text="Error loading model")
            self.status_lbl.config(text="Model error")

    def _on_preset_selected(self, event=None):
        selected = self.preset_combo.get()
        if selected:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, selected)

    def _toggle_generation(self):
        if self.is_generating:
            # Request Stop
            self.stop_requested.set()
            self.status_lbl.config(text="Stopping generation...")
        else:
            # Start Generation
            self._start_generation()

    def _start_generation(self):
        if self.engine is None or self.engine.model is None:
            messagebox.showwarning("Model Warning", "Please select and load a checkpoint first.")
            return

        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Prompt Empty", "Please type an input prompt.")
            return

        # Read Parameters
        temp = float(self.temp_var.get())
        top_k = int(self.top_k_var.get())
        top_p = float(self.top_p_var.get())
        max_tokens = int(self.max_tokens_var.get())
        rep_penalty = float(self.rep_pen_var.get())

        # Reset UI
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, prompt + " ", "prompt_tag")
        self.output_text.see(tk.END)

        self.is_generating = True
        self.stop_requested.clear()

        # Update button to Stop state
        self.generate_btn.config(text="⏹  Stop Generation", bg=DANGER_COLOR, activebackground="#EBA0AC")
        self.status_lbl.config(text="Status: Generating tokens...")

        # Launch background thread
        thread = threading.Thread(
            target=self._worker_generate,
            args=(prompt, max_tokens, temp, top_k, top_p, rep_penalty),
            daemon=True
        )
        thread.start()

    def _worker_generate(self, prompt: str, max_tokens: int, temp: float, top_k: int, top_p: float, rep_pen: float):
        try:
            token_count = 0
            for chunk, token_id, elapsed, speed in self.engine.stream_generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_pen,
                stop_on_eos=True
            ):
                token_count += 1
                self.token_queue.put(("token", chunk, token_count, max_tokens, speed, elapsed))

                if self.stop_requested.is_set():
                    break

            self.token_queue.put(("done", "", token_count, max_tokens, speed, elapsed))
        except Exception as e:
            self.token_queue.put(("error", str(e), 0, 0, 0, 0))

    def _process_token_queue(self):
        try:
            while not self.token_queue.empty():
                msg_type, content, tok_count, max_toks, speed, elapsed = self.token_queue.get_nowait()

                if msg_type == "token":
                    display_chunk = content.replace("<|endoftext|>", "")
                    if display_chunk:
                        self.output_text.insert(tk.END, display_chunk, "token_tag")
                        self.output_text.see(tk.END)
                    self.speed_lbl.config(
                        text=f"Tokens: {tok_count}/{max_toks} | Speed: {speed:.1f} tok/s | Time: {elapsed:.2f}s"
                    )

                elif msg_type == "done":
                    self._finish_generation(tok_count, speed, elapsed)

                elif msg_type == "error":
                    self._finish_generation(0, 0, 0)
                    messagebox.showerror("Generation Error", f"An error occurred during inference:\n{content}")

        finally:
            self.root.after(20, self._process_token_queue)

    def _finish_generation(self, tok_count: int, speed: float, elapsed: float):
        self.is_generating = False
        self.generate_btn.config(text="▶  Generate Text", bg=ACCENT_COLOR, activebackground=ACCENT_HOVER)
        self.status_lbl.config(text="Status: Completed")
        if tok_count > 0:
            self.speed_lbl.config(
                text=f"Completed {tok_count} tokens | Avg Speed: {speed:.1f} tok/s | Total Time: {elapsed:.2f}s"
            )

    def _clear_all(self):
        self.prompt_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.speed_lbl.config(text="Tokens: 0 | Speed: 0.0 tok/s | Time: 0.0s")
        self.status_lbl.config(text="Status: Cleared")

    def _copy_output(self):
        text = self.output_text.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_lbl.config(text="Output copied to clipboard!")
        else:
            self.status_lbl.config(text="Nothing to copy.")

def main():
    root = tk.Tk()
    app = LanguageModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
