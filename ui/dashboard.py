import os
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, ProgressBar, Log, Button

# Adjust imports to match your local parser/alignment module filenames if needed
from parser.fasta_parser import encode_sequence, parse_fasta_chunks, generate_mock_fasta
from alignment.cpu_baseline import run_cpu_benchmark
from alignment.gpu_aligner import run_gpu_alignment, is_cuda_available

FASTA_PATH = "data/mock_genome.fasta"
GRNA_TEST = "GGTGAGTGAGTACACCAAAC"  # 20bp CRISPR gRNA target

class GeneWeaverDashboard(App):
    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }
    .panel {
        border: solid $primary;
        padding: 1 2;
        margin: 1;
        height: auto;
    }
    #top-row {
        height: 14;
    }
    #controls-box {
        width: 1fr;
    }
    #metrics-box {
        width: 1fr;
    }
    #log_window {
        height: 1fr;
        border: solid $secondary;
        margin: 1;
        padding: 1;
    }
    Button {
        margin-top: 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Horizontal(
                Vertical(
                    Static("[b]Genomic Chunking & Scan Progress[/b]"),
                    ProgressBar(id="chunk_progress", total=100, show_eta=True),
                    Button("Run CPU vs. CUDA Benchmark", id="start_btn", variant="success"),
                    id="controls-box",
                    classes="panel"
                ),
                Vertical(
                    Static("[b]Hardware & Speedup Telemetry[/b]"),
                    Static("Hardware Status: Checking...", id="hw_status"),
                    Static("CPU Baseline Time: -- s", id="cpu_metric"),
                    Static("GPU CUDA Time:     -- s", id="gpu_metric"),
                    Static("Speedup Factor:    -- x", id="speedup_metric"),
                    id="metrics-box",
                    classes="panel"
                ),
                id="top-row"
            ),
            Log(id="log_window")
        )
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#log_window", Log)
        hw_label = self.query_one("#hw_status", Static)

        if is_cuda_available():
            hw_label.update("Hardware Status: [green]NVIDIA CUDA GPU Ready[/green]")
            log.write_line("[SYSTEM] NVIDIA GPU detected via Numba CUDA.")
        else:
            hw_label.update("Hardware Status: [yellow]CPU Only (No Local CUDA GPU)[/yellow]")
            log.write_line("[WARNING] No local NVIDIA GPU detected. CUDA kernel will skip on this machine.")

        if not os.path.exists(FASTA_PATH):
            log.write_line(f"[PARSER] Generating mock genome at {FASTA_PATH}...")
            generate_mock_fasta(FASTA_PATH, length_bp=2_000_000)
            log.write_line("[PARSER] Mock genome ready.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start_btn":
            event.button.disabled = True
            self.run_worker(self.run_pipeline, thread=True)

    def run_pipeline(self) -> None:
        log = self.query_one("#log_window", Log)
        progress = self.query_one("#chunk_progress", ProgressBar)
        progress.update(progress=0)

        log.write_line("[PIPELINE] Encoding target gRNA sequence...")
        grna_encoded = encode_sequence(GRNA_TEST)

        log.write_line("[PIPELINE] Streaming genomic chunks via BioPython...")
        chunks = list(parse_fasta_chunks(FASTA_PATH, chunk_size=500_000))
        total_chunks = len(chunks)
        log.write_line(f"[PIPELINE] Loaded {total_chunks} chunks (500,000 BP each).")

        # Benchmark on Chunk 0
        _, target_chunk = chunks[0]

        # 1. CPU Baseline
        log.write_line("[CPU] Running sequential baseline alignment on 500,000 BP...")
        cpu_hits, cpu_time = run_cpu_benchmark(target_chunk, grna_encoded, max_mismatches=3)
        self.call_from_thread(
            self.query_one("#cpu_metric", Static).update,
            f"CPU Baseline Time: {cpu_time:.4f} s"
        )
        log.write_line(f"[CPU] Finished in {cpu_time:.4f}s | Hits: {len(cpu_hits)}")
        self.call_from_thread(progress.advance, 50)

        # 2. GPU CUDA Kernel
        log.write_line("[GPU] Attempting Host -> Device VRAM transfer & @cuda.jit launch...")
        try:
            gpu_hits, gpu_time, _ = run_gpu_alignment(target_chunk, grna_encoded, max_mismatches=3)
            self.call_from_thread(
                self.query_one("#gpu_metric", Static).update,
                f"GPU CUDA Time:     {gpu_time:.4f} s"
            )
            log.write_line(f"[GPU] Finished in {gpu_time:.4f}s | Hits: {len(gpu_hits)}")

            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                self.call_from_thread(
                    self.query_one("#speedup_metric", Static).update,
                    f"Speedup Factor:    [bold green]{speedup:.2f}x[/bold green]"
                )
                log.write_line(f"[BENCHMARK] Achieved {speedup:.2f}x speedup over CPU baseline!")
        except RuntimeError as e:
            self.call_from_thread(
                self.query_one("#gpu_metric", Static).update,
                "GPU CUDA Time:     [yellow]Skipped (Run on Colab)[/yellow]"
            )
            log.write_line(f"[GPU SKIPPED] {str(e)}")

        self.call_from_thread(progress.advance, 50)
        self.call_from_thread(setattr, self.query_one("#start_btn", Button), "disabled", False)
        log.write_line("[PIPELINE] Scan complete.")

if __name__ == "__main__":
    GeneWeaverDashboard().run()