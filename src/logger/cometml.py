from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


class CometMLWriter:
    def __init__(
        self,
        project_name: str,
        project_config: dict | None = None,
        workspace: str | None = None,
        run_name: str | None = None,
        api_key: str | None = None,
        mode: str = "online",
        **kwargs,
    ) -> None:
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()
        self.disabled = mode in {"disabled", "off", "none"}
        self.exp = None

        if self.disabled:
            return

        try:
            import comet_ml
        except ImportError as exc:
            raise RuntimeError("Install comet_ml or set use_comet=false") from exc

        exp_class = comet_ml.OfflineExperiment if mode == "offline" else comet_ml.Experiment
        exp_kwargs = {
            "project_name": project_name,
            "workspace": workspace,
            "auto_metric_logging": kwargs.get("auto_metric_logging", False),
            "auto_param_logging": kwargs.get("auto_param_logging", False),
        }
        if api_key:
            exp_kwargs["api_key"] = api_key

        self.exp = exp_class(**exp_kwargs)
        if run_name:
            self.exp.set_name(run_name)
        if project_config:
            self.exp.log_parameters(project_config)

    def _name(self, name: str) -> str:
        return f"{name}_{self.mode}" if self.mode else name

    def set_step(self, step: int, mode: str = "train") -> None:
        previous_step = self.step
        self.step = step
        self.mode = mode

        if step == 0:
            self.timer = datetime.now()
            return

        duration = (datetime.now() - self.timer).total_seconds()
        if duration > 0 and step != previous_step:
            self.add_scalar("steps_per_sec", (step - previous_step) / duration)
        self.timer = datetime.now()

    def log_parameters(self, params: dict) -> None:
        if self.exp is not None:
            self.exp.log_parameters(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.exp is not None:
            self.exp.log_metrics(metrics, step=self.step if step is None else step)

    def log_audio(
        self,
        audio_data,
        sample_rate: int,
        file_name: str,
        step: int | None = None,
    ) -> None:
        if self.exp is not None:
            self.exp.log_audio(
                audio_data=audio_data,
                sample_rate=sample_rate,
                file_name=file_name,
                step=self.step if step is None else step,
            )

    def log_model(self, name: str, path: str | Path) -> None:
        if self.exp is not None:
            self.exp.log_model(name=name, file_or_folder=str(path), overwrite=True)

    def log_other(self, key: str, value: Any) -> None:
        if self.exp is not None:
            self.exp.log_other(key, value)

    def add_tag(self, tag: str) -> None:
        if self.exp is not None:
            self.exp.add_tag(tag)

    def end(self) -> None:
        if self.exp is not None:
            self.exp.end()

    def add_checkpoint(self, checkpoint_path, save_dir=None) -> None:
        self.log_model("checkpoints", checkpoint_path)

    def add_scalar(self, scalar_name: str, scalar) -> None:
        self.log_metrics({self._name(scalar_name): scalar})

    def add_scalars(self, scalars: dict) -> None:
        self.log_metrics({self._name(name): value for name, value in scalars.items()})

    def add_image(self, image_name: str, image) -> None:
        if self.exp is not None:
            self.exp.log_image(image_data=image, name=self._name(image_name), step=self.step)

    def add_images(self, image_names, images) -> None:
        for name, image in zip(image_names, images):
            self.add_image(name, image)

    def add_audio(self, audio_name: str, audio, sample_rate: int | None = None) -> None:
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        self.log_audio(audio, sample_rate=sample_rate, file_name=self._name(audio_name))

    def add_text(self, text_name: str, text: str) -> None:
        if self.exp is not None:
            self.exp.log_text(text=text, step=self.step, metadata={"name": self._name(text_name)})

    def add_histogram(self, hist_name: str, values_for_hist, bins=None) -> None:
        if self.exp is not None:
            if hasattr(values_for_hist, "detach"):
                values_for_hist = values_for_hist.detach().cpu().numpy()
            self.exp.log_histogram_3d(values=values_for_hist, name=self._name(hist_name), step=self.step)

    def add_table(self, table_name: str, table) -> None:
        if self.exp is not None:
            self.exp.log_table(filename=self._name(table_name) + ".csv", tabular_data=table, headers=True)

    def add_pr_curve(self, curve_name: str, curve) -> None:
        if self.exp is not None:
            self.exp.log_other(self._name(curve_name), curve)

    def add_embedding(self, embedding_name: str, embedding) -> None:
        if self.exp is not None:
            self.exp.log_other(self._name(embedding_name), str(embedding))
