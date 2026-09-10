#!/usr/bin/env python3
"""Expand reproducible training sweeps into local commands or batch scripts."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import shlex
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import fire

def _coerce_list(value: Any, name: str, cast: type | None = None) -> list[Any]:
    """Accept list-like inputs from Fire and coerce to a list of cast values."""

    if value is None: # user did not specify any specific value for this tunable parameter
        out: list[Any] = []
    elif isinstance(value, (list, tuple)): # iterables are casted to list
        out = list(value)
    elif isinstance(value, str): # parsing strings and wrapping them into lists
        text = value.strip()
        if not text:
            out = []
        elif text.startswith("["):
            out = json.loads(text)
        elif "," in text:
            out = [piece.strip() for piece in text.split(",") if piece.strip()]
        else:
            out = [piece.strip() for piece in text.split() if piece.strip()]
    else:
        out = [value]

    if cast is None:
        return out
    try:
        return [cast(item) for item in out]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be castable to {cast.__name__}") from exc


def _coerce_dict(value: Any, name: str) -> dict[str, Any]:
    """Accept dict-like inputs from Fire and coerce to a dictionary."""

    if value in (None, "", {}):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
        
    raise ValueError(f"{name} must be a dictionary or JSON dictionary string")


def _coerce_train_kwargs_variants(value: Any) -> list[tuple[str, dict[str, Any]]]:
    """Parse independently scheduled training-argument variants.

    Each mapping may contain an optional ``_name`` used only in batch job names.
    The remaining keys are merged over ``train_kwargs`` for that job.
    """

    if value in (None, "", []):
        return [("", {})]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("train_kwargs_variants must be a list of dictionaries")

    variants: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            item = json.loads(item)
        if not isinstance(item, dict):
            raise ValueError("train_kwargs_variants entries must be dictionaries")
        name = _sanitize_name(str(item.get("_name", f"variant{index}")))
        variants.append((name, {key: val for key, val in item.items() if key != "_name"}))
    return variants


def _format_cli_value(value: Any) -> str:
    """Serialize values so Fire can reconstruct structured arguments."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    
    return str(value)


def _coerce_optional_int_list(value: Any, name: str) -> list[int | None]:
    """Accept list-like inputs and coerce entries to int or None.
    That's because window is an integer unless 'full' is specificed, which we handle
    differently (it may have extra arguments when set to full, like min_length
    and max_length)
    """

    raw = _coerce_list(value, name)
    out: list[int | None] = []
    for item in raw:
        if item is None:
            out.append(None)
            continue
        if isinstance(item, str) and item.strip().lower() in {"", "none", "null"}:
            out.append(None)
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} entries must be int or null/None") from exc
    return out


def _coerce_window_list(value: Any, name: str = "windows") -> list[int | str]:
    """Accept list-like inputs and coerce each window to int or string token."""
    raw = _coerce_list(value, name)
    out: list[int | str] = []
    for item in raw:
        if isinstance(item, int):
            out.append(item)
            continue
        if isinstance(item, float) and float(item).is_integer():
            out.append(int(item))
            continue
        text = str(item).strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            out.append(int(text))
        else:
            out.append(text)
    return out


def _sanitize_name(text: str) -> str:
    """Return a scheduler-safe job tag while preserving uniqueness after truncation."""

    max_length = 100
    hash_length = 8

    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")

    if len(text) <= max_length:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:hash_length]
    prefix_length = max_length - hash_length - 1
    return f"{text[:prefix_length].rstrip('-')}-{digest}"


def _load_config_file(config_path: str) -> dict[str, Any]:
    # Config files are the way of setting an experiment. This loads these files with error handling

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            ) from exc
        config = yaml.safe_load(path.read_text())
    else:
        config = json.loads(path.read_text())

    if not isinstance(config, dict):
        raise ValueError("Config file must parse to a dictionary/object.")
    
    return config


class SubmitJobs:
    """Generate local commands or portable Slurm batch scripts for a sweep."""

    def run(
        self,
        models: list[str] | tuple[str, ...] = (),
        seeds_file: str = "seeds.txt",
        ntrials: int = 10,
        missingnesses: list[float] | tuple[float, ...] = (0.3,),
        horizons: list[int] | tuple[int, ...] = (0,),
        windows: list[int | str] | tuple[int | str, ...] = (60,),
        chunksizes: list[int] | tuple[int, ...] = (900,),
        min_lengths: list[int | None] | tuple[int | None, ...] = (None,),
        max_lengths: list[int | None] | tuple[int | None, ...] = (None,),
        sample_rates: list[float] | tuple[float, ...] = (0.25,),
        lab_order_delays: list[int] | tuple[int, ...] = (30,),
        default_batch_size: int = 128,
        batch_sizes: list[int] | str | None = None,
        default_fit_kwargs: dict[str, Any] | None = None,
        data_dir: str = "./data/",
        savedir: str = "results/",
        gpu: int = 0,
        splits_file: str = "data/train_val_test_splits.csv",
        model_batch_size_map: dict[str, int] | None = None,
        model_chunksize_map: dict[str, Any] | None = None,
        model_fit_kwargs_map: dict[str, Any] | None = None,
        account: str | None = None,
        partition: str | None = None,
        gpus: int | str = 1,
        time_limit: str = "01:00:00",
        ntasks: int = 1,
        max_parallel: int | None = None,
        cpus_per_task: int = 1,
        mem: str = "128G",
        qos: str | None = None,
        exclude_nodes: str | None = None,
        exclude_nodes_file: str | None = None,
        python_exec: str = "uv run --active python",
        train_script: str = "train_time_to_event_models.py",
        trace_file: str | None = None,
        label_trainfile: str | None = None,
        label_testfile: str | None = None,
        train_kwargs: dict[str, Any] | None = None,
        train_kwargs_variants: list[dict[str, Any]] | None = None,
        model_train_kwargs_variants_map: dict[str, Any] | None = None,
        jobs_dir: str = "jobs",
        logs_dir: str = "logs",
        submit: bool = False,
        print_only: bool = False,
        local: bool = False,
        override: bool = False,
        confirm_before_run: bool = False,
    ):
        """Expand a sweep configuration into job scripts or local submissions.

        Structured inputs accept JSON, comma-separated strings, repeated Fire list
        inputs, or native Python values.

        Returns
        -------
        dict[str, Any]
            Summary information about generated scripts, printed commands, and
            submitted job IDs.
        """

        models_list = _coerce_list(models, "models", str)
        if not models_list:
            raise ValueError(
                "At least one model is required. Pass --models as a JSON list, "
                "comma-separated string, or repeated Fire list input."
            )

        seeds = self._load_seeds(seeds_file, ntrials) # Using fixed seeds!!!

        # Making sure everything is a list. Parsing them to a fixed datatype so we can identify runs with results
        missingness_list = _coerce_list(missingnesses, "missingnesses", float)
        horizon_list = _coerce_list(horizons, "horizons", int)
        window_list = _coerce_window_list(windows, "windows")
        chunksize_list = _coerce_list(chunksizes, "chunksizes", int)
        min_length_list = _coerce_optional_int_list(min_lengths, "min_lengths")
        max_length_list = _coerce_optional_int_list(max_lengths, "max_lengths")
        sample_rate_list = _coerce_list(sample_rates, "sample_rates", float)
        lab_delay_list = _coerce_list(lab_order_delays, "lab_order_delays", int)

        batch_overrides = _coerce_dict(model_batch_size_map, "model_batch_size_map")
        batch_size_list = _coerce_list(
            [default_batch_size] if batch_sizes is None else batch_sizes,
            "batch_sizes", int,
        )
        model_batch_sizes = {
            model: _coerce_list(
                batch_overrides.get(model, batch_size_list),
                f"model_batch_size_map[{model}]", int,
            )
            for model in models_list
        }
        chunksize_overrides = _coerce_dict(model_chunksize_map, "model_chunksize_map")
        unknown_chunksize_models = set(chunksize_overrides).difference(models_list)
        if unknown_chunksize_models:
            raise ValueError(
                "model_chunksize_map contains models absent from models: "
                f"{sorted(unknown_chunksize_models)}"
            )
        model_chunksizes = {
            model: _coerce_list(
                chunksize_overrides.get(model, chunksize_list),
                f"model_chunksize_map[{model}]",
                int,
            )
            for model in models_list
        }

        # This one is specific to the model, and not the training procedure. It is a dict instead of values directly
        fit_kwargs_overrides = _coerce_dict(
            model_fit_kwargs_map,
            "model_fit_kwargs_map",
        )
        global_fit_kwargs = _coerce_dict(default_fit_kwargs, "default_fit_kwargs")
        extra_train_kwargs = _coerce_dict(train_kwargs, "train_kwargs")
        train_variants = _coerce_train_kwargs_variants(train_kwargs_variants)
        model_train_variant_overrides = _coerce_dict(
            model_train_kwargs_variants_map,
            "model_train_kwargs_variants_map",
        )
        unknown_variant_models = set(model_train_variant_overrides).difference(models_list)
        if unknown_variant_models:
            raise ValueError(
                "model_train_kwargs_variants_map contains models absent from models: "
                f"{sorted(unknown_variant_models)}"
            )
        model_train_variants = {
            model: _coerce_train_kwargs_variants(
                model_train_variant_overrides.get(model, train_kwargs_variants)
            )
            for model in models_list
        }
        model_chunk_variant_pairs = [
            (model, chunksize, batch_size, variant)
            for model in models_list
            for chunksize in model_chunksizes[model]
            for batch_size in model_batch_sizes[model]
            for variant in model_train_variants[model]
        ]

        excluded_nodes = self._resolve_excluded_nodes(exclude_nodes, exclude_nodes_file)
        jobs_path = Path(jobs_dir)
        logs_path = Path(logs_dir)
        jobs_path.mkdir(parents=True, exist_ok=True)
        logs_path.mkdir(parents=True, exist_ok=True)

        # Find previous results
        run_records: list[dict[str, Any]] = []
        if not override:
            savedir_path = Path(savedir)
            if savedir_path.exists(): # Making sure we dont write on top of previous results.
                for path in savedir_path.glob("run_*.json"):
                    try:
                        data = json.loads(path.read_text()) # We need to load the data - job skipping relies on comparing the settings of previous results
                    except (OSError, json.JSONDecodeError) as exc:
                        print(f"skipping unreadable run file {path}: {exc}")
                        continue
                    data["_run_path"] = str(path)
                    run_records.append(data)

        created_scripts: list[Path] = []
        submitted_job_ids: list[str] = []
        local_commands: list[list[str]] = []
        pending_submissions: list[Path] = []
        skipped_existing = 0
        printed_jobs = 0

        combos = itertools.product(
            seeds, # Seeds first so we run all configurations first before moving to the second repetition 

            # Each experimental setting
            missingness_list,
            horizon_list,
            window_list,
            min_length_list,
            max_length_list,
            sample_rate_list,
            lab_delay_list,
            # Each model with its independently scheduled training configurations.
            model_chunk_variant_pairs,
        )

        # All possible configurations of tunable settings
        for (seed, missingness, horizon, window, min_length, max_length,
             sample_rate, lab_delay, (model, chunksize, batch_size, (train_variant_name, train_variant))) in combos:
            
            # Let's load everything and use default values if the user have not specified it.
            # By doing this, we ensure we can find repeated results when one or many parameters
            # are not set by the user.
            effective_train_kwargs = {**extra_train_kwargs, **train_variant}

            fit_kwargs_variants = self._resolve_fit_kwargs_variants(
                model=model,
                fit_kwargs_overrides=fit_kwargs_overrides,
                global_fit_kwargs=global_fit_kwargs,
            )

            for variant_name, fit_kwargs in fit_kwargs_variants:
                fit_kwargs_json = json.dumps(fit_kwargs, separators=(",", ":"))

                # Stuff for survival models
                duration_mode = effective_train_kwargs.get("duration_mode", "time_since_start")
                optimizer_name = effective_train_kwargs.get("optimizer", "adafactor")

                variant_suffix = f"_{variant_name}" if variant_name else ""
                train_variant_suffix = f"_{train_variant_name}" if train_variant_name else ""
                evaluation_suffix = ""
                if effective_train_kwargs.get("eval_only"):
                    checkpoint_path = effective_train_kwargs.get("checkpoint_path", "auto")
                    evaluation_name = effective_train_kwargs.get("evaluation_name", "evaluation")
                    checkpoint_label = (
                        "auto" if str(checkpoint_path).strip().lower() == "auto"
                        else Path(str(checkpoint_path)).stem[-12:]
                    )
                    evaluation_suffix = (
                        f"_eval-{evaluation_name}_"
                        f"{checkpoint_label}"
                    )

                run_tag = _sanitize_name(
                    (
                        f"{model}{variant_suffix}{train_variant_suffix}_s{seed}_{optimizer_name}"
                        f"{evaluation_suffix}"
                        f"_m{int(round(missingness * 100))}"
                        f"_bs{batch_size}_cs{chunksize}"
                        f"_lod{lab_delay}"
                        f"_dm-{duration_mode}"
                    )
                )

                # We also need default values so this line below works and always
                # use the same command to call the jobs
                if Path(train_script).name == "train_time_to_event_models.py":
                    if not trace_file or not label_trainfile:
                        raise ValueError(
                            "Time-to-event configs require trace_file and label_trainfile."
                        )
                    command = [
                        *shlex.split(str(python_exec)), train_script, "run",
                        "--trace_file", trace_file,
                        "--label_trainfile", label_trainfile,
                        "--ml", model, "--gpu", str(gpu), "--savedir", savedir,
                        "--batch_size", str(batch_size), "--random_state", str(seed),
                        "--lab_order_delay", str(lab_delay),
                        "--chunk_window_size", str(chunksize), "--fit_kwargs", fit_kwargs_json,
                        "--splits_file", splits_file,
                    ]
                    if label_testfile:
                        command.extend(["--label_testfile", label_testfile])
                else:
                    command = [
                        *shlex.split(str(python_exec)),
                        train_script, "run", "--data_dir", data_dir, "--horizon", str(horizon),
                        "--ml", model, "--window", str(window), "--gpu", str(gpu),
                        "--savedir", savedir, "--missingness", str(missingness),
                        "--batch_size", str(batch_size), "--random_state", str(seed),
                        "--sample_rate", str(sample_rate), "--lab_order_delay", str(lab_delay),
                        "--chunksize", str(chunksize), "--fit_kwargs", fit_kwargs_json,
                        "--splits_file", splits_file,
                    ]

                # full tracing settings
                if Path(train_script).name != "train_time_to_event_models.py" and min_length is not None:
                    command.extend(["--min_length", str(min_length)])
                if Path(train_script).name != "train_time_to_event_models.py" and max_length is not None:
                    command.extend(["--max_length", str(max_length)])

                for key, value in effective_train_kwargs.items():
                    if value is None:
                        continue
                    command.extend([f"--{key}", _format_cli_value(value)])
                command_str = " ".join(shlex.quote(piece) for piece in command)

                signature = {
                    "data_dir": data_dir,
                    "horizon": horizon,
                    "ml": model,
                    "window": window,
                    "min_length": min_length,
                    "max_length": max_length,
                    "gpu": gpu,
                    "savedir": savedir,
                    "missingness": missingness,
                    "batch_size": batch_size,
                    "random_state": seed,
                    "sample_rate": sample_rate,
                    "lab_order_delay": lab_delay,
                    "chunk_window_size": chunksize,
                    "fit_kwargs": fit_kwargs,
                    "splits_file": splits_file,
                }
                signature.update(effective_train_kwargs)

                if not override and run_records: # Checking prev results
                    matched = None
                    for record in run_records:
                        ok = True
                        for key, value in signature.items(): # searching for prev results
                            if key == "fit_kwargs":
                                try:
                                    record_value = _coerce_dict(record.get(key), "fit_kwargs")
                                except ValueError: # The setting was not defined in the experiment.yml
                                    ok = False
                                    break
                                if record_value != value: # The setting has no available results
                                    ok = False
                                    break
                            elif key == "duration_mode":
                                record_value = record.get(key, "time_since_start")
                                if record_value != value:
                                    ok = False
                                    break
                            elif key == "chunk_window_size":
                                record_value = record.get(key, record.get("chunksize"))
                                if record_value != value:
                                    ok = False
                                    break
                            elif record.get(key) != value:
                                ok = False
                                break
                        if ok:
                            matched = record
                            break

                    if matched:
                        # Run id is also in the filename, but matching it there would be annoying. lets just recover it from the structured result json
                        run_id = matched.get("run_id")
                        if not run_id and matched.get("_run_path"):
                            run_id = Path(matched["_run_path"]).stem.replace("run_", "")
                        model_name = matched.get("model_name")
                        if not model_name:
                            if matched.get("initial_model") is None:
                                model_name = f"{matched.get('ml')}_run_{run_id}"
                            else:
                                model_name = f"run_{run_id}"

                        # Checking all relavant files exists - the ones required to run the model
                        expected = [
                            Path(savedir) / "models" / f"{model_name}.pt",
                            Path(savedir) / f"{model_name}.history",
                            Path(savedir) / f"run_{run_id}.csv",
                        ]
                        if all(path.exists() for path in expected):
                            skipped_existing += 1
                            print(
                                f"skipping existing results for {run_tag} "
                                f"(run_id={run_id})"
                            )
                            continue

                if print_only:
                    print(f"- [{run_tag}] {command_str}")
                    printed_jobs += 1
                    continue

                if local:
                    local_commands.append(command)
                    continue

                # Batch-script version.
                script_path = jobs_path / f"{run_tag}.sbatch"
                script_text = self._build_script(
                    job_name=run_tag,
                    log_dir=logs_path,
                    account=account,
                    partition=partition,
                    gpus=str(gpus),
                    time_limit=time_limit,
                    ntasks=ntasks,
                    cpus_per_task=cpus_per_task,
                    mem=mem,
                    qos=qos,
                    exclude_nodes=excluded_nodes,
                    command_str=command_str,
                )
                script_path.write_text(script_text)
                created_scripts.append(script_path)

                if submit and not local:
                    pending_submissions.append(script_path)
                else:
                    print(f"generated {script_path}")

        if print_only:
            planned_jobs = printed_jobs
            print(f"printed_jobs={printed_jobs}")
        else:
            planned_jobs = len(local_commands) + len(pending_submissions)
            if not submit and not local:
                planned_jobs = len(created_scripts)
            print(f"total_jobs={planned_jobs}")

        print(f"skipped_existing={skipped_existing}")

        if confirm_before_run and (local_commands or pending_submissions):
            input("Press Enter to continue with execution...")

        if local and local_commands:
            parallel_limit = max(1, int(ntasks if max_parallel is None else max_parallel))
            print(
                f"running locally: {len(local_commands)} jobs "
                f"(max_parallel={parallel_limit})"
            )
            self._run_local_commands(local_commands, max_parallel=parallel_limit)

        # Submit generated batch scripts only when explicitly requested.
        if pending_submissions:
            print(
                f"submitting {len(pending_submissions)} batch jobs in configured order "
                "(all configurations for each seed before the next seed)"
            )
            for script_path in pending_submissions:
                sbatch_cmd = ["sbatch", str(script_path)]
                try:
                    proc = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as exc:
                    stdout = (exc.stdout or "").strip()
                    stderr = (exc.stderr or "").strip()
                    raise RuntimeError(
                        "sbatch failed for "
                        f"{script_path}\n"
                        f"command: {' '.join(sbatch_cmd)}\n"
                        f"stdout: {stdout or '<empty>'}\n"
                        f"stderr: {stderr or '<empty>'}"
                    ) from exc
                output = proc.stdout.strip()
                job_id = output.split()[-1] if output else ""
                submitted_job_ids.append(job_id)
                print(f"submitted {script_path} -> {output}")

        if submit and submitted_job_ids:
            print(f"submitted_jobs={len(submitted_job_ids)}")
            print(f"job_ids={','.join(submitted_job_ids)}")

        return {
            "total_jobs": len(created_scripts),
            "jobs_dir": str(jobs_path),
            "logs_dir": str(logs_path),
            "submitted_job_ids": submitted_job_ids,
            "skipped_existing": skipped_existing,
            "printed_jobs": printed_jobs,
        }

    def run_config(
        self,
        config: str,
        submit: bool | None = None,
        print_only: bool | None = None,
        local: bool | None = None,
        max_parallel: int | None = None,
    ):
        """Load a JSON/YAML config and run the same sweep pipeline.

        Command-line execution controls override values in the configuration.
        In particular, ``local=True`` runs the expanded training commands as
            child processes of this shell and never creates or submits batch jobs,
        even when ``submit=True`` is also supplied.
        """

        config_path = Path(config)
        params = _load_config_file(config)
        savedir = Path(params["savedir"])
        savedir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, savedir / config_path.name)

        if submit is not None:
            params["submit"] = submit
        if print_only is not None:
            params["print_only"] = print_only
        if local is not None:
            params["local"] = local
        if max_parallel is not None:
            params["max_parallel"] = max_parallel

        return self.run(**params)

    @staticmethod
    def _resolve_excluded_nodes(
        exclude_nodes: str | None,
        exclude_nodes_file: str | None,
    ) -> str | None:
        """Return a de-duplicated scheduler --exclude hostlist from inline/file input."""
        if exclude_nodes is not None and exclude_nodes_file is not None:
            raise ValueError("Specify at most one of exclude_nodes and exclude_nodes_file")
        if exclude_nodes_file is not None:
            path = Path(exclude_nodes_file)
            if not path.is_file():
                raise FileNotFoundError(f"exclude_nodes_file not found: {path}")
            raw = "\n".join(
                line.split("#", 1)[0] for line in path.read_text().splitlines()
            )
        else:
            raw = exclude_nodes or ""
        nodes = [node for node in re.split(r"[\s,]+", raw.strip()) if node]
        if not nodes:
            return None
        return ",".join(dict.fromkeys(nodes))

    @staticmethod
    def _load_seeds(seeds_file: str, ntrials: int) -> list[int]:
        seeds_path = Path(seeds_file)

        # we MUST have a seeds file - we want reproducible results
        if not seeds_path.exists():
            raise FileNotFoundError(f"seeds file not found: {seeds_file}")

        seeds: list[int] = []
        for line in seeds_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            seeds.append(int(stripped))
            if len(seeds) >= ntrials:
                break

        if len(seeds) < ntrials:
            raise ValueError(
                f"Requested ntrials={ntrials}, but only found "
                f"{len(seeds)} seeds in {seeds_file}"
            )
        return seeds

    @staticmethod
    def _resolve_fit_kwargs_variants(
        *,
        model: str,
        fit_kwargs_overrides: dict[str, Any],
        global_fit_kwargs: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        # several attempts to load what we expect to be a json object

        raw = fit_kwargs_overrides.get(model, global_fit_kwargs)
        if isinstance(raw, str):
            raw = json.loads(raw)

        if isinstance(raw, dict):
            return [("", raw)]

        if isinstance(raw, list):
            variants: list[tuple[str, dict[str, Any]]] = []
            for idx, item in enumerate(raw, start=1):
                if isinstance(item, str):
                    item = json.loads(item)
                if not isinstance(item, dict):
                    raise ValueError(
                        f"model_fit_kwargs_map[{model}] list entries must be dicts"
                    )
                name = item.get("_name", f"cfg{idx}")
                cfg = {k: v for k, v in item.items() if k != "_name"}
                variants.append((_sanitize_name(str(name)), cfg))
            return variants

        raise ValueError(
            f"model_fit_kwargs_map[{model}] must be a dict or list of dicts"
        )

    @staticmethod
    def _build_script(
        *,
        job_name: str,
        log_dir: Path,
        account: str | None,
        partition: str | None,
        gpus: str,
        time_limit: str,
        ntasks: int,
        cpus_per_task: int,
        mem: str,
        qos: str | None,
        exclude_nodes: str | None,
        command_str: str,
    ) -> str:
        now = datetime.now().isoformat(timespec="seconds")
        lines = [
            "#!/bin/bash",
            f"# generated {now}",
            f"#SBATCH --gres=gpu:{gpus}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={log_dir}/%j_{job_name}.txt",
            f"#SBATCH --ntasks={ntasks}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={mem}",
        ]
        if account:
            lines.append(f"#SBATCH --account={account}")
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if qos:
            lines.append(f"#SBATCH --qos={qos}")
        if exclude_nodes:
            lines.append(f"#SBATCH --exclude={exclude_nodes}")
        lines.append("set -euo pipefail")
        lines.extend(
            [
                "hostname",
                command_str,
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _run_local_commands(
        commands: list[list[str]],
        *,
        max_parallel: int,
    ) -> None:
        """Execute local training commands with bounded concurrency."""

        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(subprocess.run, command, check=True): command
                for command in commands
            }
            for future in as_completed(futures):
                command = futures[future]
                try:
                    future.result()
                except subprocess.CalledProcessError as exc:
                    cmd_str = " ".join(shlex.quote(piece) for piece in command)
                    raise RuntimeError(
                        "local command failed\n"
                        f"command: {cmd_str}\n"
                        f"returncode: {exc.returncode}"
                    ) from exc


if __name__ == "__main__":
    fire.Fire(SubmitJobs)
