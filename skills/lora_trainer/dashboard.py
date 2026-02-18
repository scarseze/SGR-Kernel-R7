"""
LoRA Experiment Dashboard — CLI & Markdown report generator.

Generates:
- Trial comparison table
- Loss progression (ASCII)
- Best config summary
- Budget usage chart
"""
from typing import Optional
from skills.lora_trainer.state import ExperimentState, TrialRecord
from skills.lora_trainer.experiment_spec import TrialStatus


class ExperimentDashboard:
    """
    Renders experiment state as human-readable reports.
    
    Usage:
        dashboard = ExperimentDashboard(experiment_state)
        print(dashboard.render_text())
        # or
        markdown = dashboard.render_markdown()
    """

    def __init__(self, state: ExperimentState):
        self.state = state

    def render_text(self) -> str:
        """Render full CLI text report."""
        lines = []
        lines.append(self._header())
        lines.append(self._trial_table())
        lines.append(self._loss_chart())
        lines.append(self._budget_bar())
        lines.append(self._best_summary())
        return "\n".join(lines)

    def render_markdown(self) -> str:
        """Render Markdown report suitable for README/artifact."""
        lines = []
        lines.append(f"# Experiment Report")
        lines.append(f"**Goal**: {self.state.experiment_goal}")
        lines.append(f"**Metric**: {self.state.best_metric_name}")
        lines.append("")

        # Trial table
        lines.append("## Trials")
        lines.append("")
        lines.append("| # | Status | Rank | LR | Loss | PPL | Δ Base | GPU h |")
        lines.append("|---|--------|------|----|------|-----|--------|-------|")
        for t in self.state.trial_history:
            is_best = "🏆" if self.state.best_trial and t.trial_id == self.state.best_trial.trial_id else ""
            lines.append(
                f"| {t.trial_id}{is_best} | {t.status.value if hasattr(t.status, 'value') else t.status} | "
                f"{t.config.lora_rank} | {t.config.learning_rate:.0e} | "
                f"{t.eval_loss if t.eval_loss is not None else '-'} | "
                f"{f'{t.eval_perplexity:.1f}' if t.eval_perplexity is not None else '-'} | "
                f"{f'{t.delta_vs_base:+.3f}' if t.delta_vs_base is not None else '-'} | "
                f"{t.gpu_hours:.2f} |"
            )
        lines.append("")

        # Budget
        lines.append("## Budget")
        used = self.state.budget_used_gpu_hours
        total = self.state.budget_total_gpu_hours
        pct = (used / total * 100) if total > 0 else 0
        lines.append(f"- GPU Hours: **{used:.2f}** / {total:.1f} ({pct:.0f}%)")
        lines.append(f"- Trials: **{len(self.state.trial_history)}** / {self.state.budget_max_trials}")
        lines.append("")

        # Best trial
        if self.state.best_trial:
            bt = self.state.best_trial
            lines.append("## Best Trial")
            lines.append(f"- **Trial {bt.trial_id}** | loss={bt.eval_loss}")
            lines.append(f"- rank={bt.config.lora_rank}, alpha={bt.config.lora_alpha}, lr={bt.config.learning_rate}")
            lines.append(f"- modules={bt.config.target_modules}")
            if bt.adapter_path:
                lines.append(f"- Adapter: `{bt.adapter_path}`")
            lines.append("")

        # Stop flags
        if self.state.stop_flags:
            lines.append("## Stop Reason")
            for flag in self.state.stop_flags:
                lines.append(f"- {flag}")

        return "\n".join(lines)

    def _header(self) -> str:
        total = len(self.state.trial_history)
        done = sum(1 for t in self.state.trial_history if t.status == TrialStatus.DONE)
        failed = sum(1 for t in self.state.trial_history if t.status == TrialStatus.FAILED)
        return (
            f"╔══════════════════════════════════════════╗\n"
            f"║  LoRA Experiment Dashboard               ║\n"
            f"╠══════════════════════════════════════════╣\n"
            f"║  Goal: {self.state.experiment_goal[:33]:<33} ║\n"
            f"║  Trials: {done}✅ {failed}❌ / {total} total{' ' * (18 - len(str(total)))} ║\n"
            f"╚══════════════════════════════════════════╝"
        )

    def _trial_table(self) -> str:
        if not self.state.trial_history:
            return "\n  No trials yet.\n"

        lines = ["\n  Trial │ Status │ Rank │ LR       │ Loss   │ GPU h"]
        lines.append("  ──────┼────────┼──────┼──────────┼────────┼──────")

        for t in self.state.trial_history:
            is_best = "★" if self.state.best_trial and t.trial_id == self.state.best_trial.trial_id else " "
            status_icon = {"done": "✅", "failed": "❌", "running": "🔄", "pruned": "✂️"}.get(
                t.status.value if hasattr(t.status, "value") else str(t.status), "?"
            )
            loss_str = f"{t.eval_loss:.4f}" if t.eval_loss is not None else "  -   "
            lines.append(
                f"  {is_best}{t.trial_id:4d} │ {status_icon:6s} │ {t.config.lora_rank:4d} │ "
                f"{t.config.learning_rate:.2e} │ {loss_str} │ {t.gpu_hours:.2f}"
            )
        return "\n".join(lines) + "\n"

    def _loss_chart(self) -> str:
        """ASCII loss progression chart."""
        completed = [t for t in self.state.trial_history if t.eval_loss is not None]
        if not completed:
            return ""

        losses = [t.eval_loss for t in completed]
        min_loss = min(losses)
        max_loss = max(losses)
        span = max_loss - min_loss if max_loss > min_loss else 1.0

        width = 30
        lines = ["\n  Loss Progression:"]
        for t in completed:
            bar_len = int((t.eval_loss - min_loss) / span * width)
            bar = "█" * max(1, bar_len) + "░" * (width - max(1, bar_len))
            marker = " ◄ best" if t.eval_loss == min_loss else ""
            lines.append(f"  T{t.trial_id:02d} │{bar}│ {t.eval_loss:.4f}{marker}")
        return "\n".join(lines) + "\n"

    def _budget_bar(self) -> str:
        """ASCII budget usage bar."""
        used = self.state.budget_used_gpu_hours
        total = self.state.budget_total_gpu_hours
        if total <= 0:
            return ""

        pct = min(1.0, used / total)
        width = 30
        filled = int(pct * width)
        bar = "█" * filled + "░" * (width - filled)
        emoji = "🟢" if pct < 0.7 else ("🟡" if pct < 0.9 else "🔴")

        return (
            f"\n  Budget: {emoji}\n"
            f"  GPU │{bar}│ {used:.2f}/{total:.1f}h ({pct*100:.0f}%)\n"
        )

    def _best_summary(self) -> str:
        """Best trial summary."""
        if not self.state.best_trial:
            return "\n  No best trial yet.\n"

        bt = self.state.best_trial
        return (
            f"\n  🏆 Best: Trial {bt.trial_id}\n"
            f"     Loss:    {bt.eval_loss}\n"
            f"     Rank:    {bt.config.lora_rank}\n"
            f"     Alpha:   {bt.config.lora_alpha}\n"
            f"     LR:      {bt.config.learning_rate}\n"
            f"     Modules: {bt.config.target_modules}\n"
            f"     Adapter: {bt.adapter_path or 'N/A'}\n"
        )


def generate_report(state: ExperimentState, format: str = "text") -> str:
    """Convenience function to generate experiment report."""
    dashboard = ExperimentDashboard(state)
    if format == "markdown":
        return dashboard.render_markdown()
    return dashboard.render_text()
