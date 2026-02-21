import asyncio
import argparse
import atexit
import json
import logging
import os
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse
from tqdm import tqdm

import httpx
import evaluate
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from translation_bench.data.afridoct import AfriDocMTDataset, ModelType
from translation_bench.data.data_class import MiniBatch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


DATASET_REGISTRY = {
    "AfriDocMTDataset": AfriDocMTDataset,
}


@dataclass
class DatasetConfig:
    name: str
    kwargs: dict
    output_file: Optional[str] = None


@dataclass
class PipelineConfig:
    model_name: str
    model_type: str
    datasets: List[DatasetConfig]
    server: Optional[str]
    api_key: Optional[str]
    port: int
    tensor_parallel_size: int
    data_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    max_tokens: int
    temperature: float
    batch_size: int
    max_retries: int
    vllm_extra_args: List[str]
    metrics: List[str]
    output_dir: str
    save_translations: bool

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg["model"]
        inf_cfg = cfg.get("inference", {})
        out_cfg = cfg.get("output", {})

        raw_datasets = cfg.get("datasets", [])
        if not raw_datasets and "dataset" in cfg:
            raw_datasets = [cfg["dataset"]]

        dataset_configs = []
        for ds_cfg in raw_datasets:
            ds_kwargs = {k: v for k, v in ds_cfg.items() if k not in ("name", "output_file")}
            dataset_configs.append(
                DatasetConfig(
                    name=ds_cfg["name"],
                    kwargs=ds_kwargs,
                    output_file=ds_cfg.get("output_file"),
                )
            )

        return cls(
            model_name=model_cfg["name"],
            model_type=model_cfg.get("model_type", "gemma"),
            datasets=dataset_configs,
            server=inf_cfg.get("server"),
            api_key=inf_cfg.get("api_key"),
            port=inf_cfg.get("port", 30024),
            tensor_parallel_size=inf_cfg.get("tensor_parallel_size", 1),
            data_parallel_size=inf_cfg.get("data_parallel_size", 1),
            gpu_memory_utilization=inf_cfg.get("gpu_memory_utilization", 0.90),
            max_model_len=inf_cfg.get("max_model_len", 8192),
            max_tokens=inf_cfg.get("max_tokens", 4096),
            temperature=inf_cfg.get("temperature", 0.0),
            batch_size=inf_cfg.get("batch_size", 16),
            max_retries=inf_cfg.get("max_retries", 3),
            vllm_extra_args=inf_cfg.get("vllm_extra_args", []),
            metrics=cfg.get("metrics", ["bleu", "chrf", "ter"]),
            output_dir=out_cfg.get("dir", "results"),
            save_translations=out_cfg.get("save_translations", True),
        )


def build_dataset(ds_config: DatasetConfig, model_name: str, model_type_str: str) -> AfriDocMTDataset:
    if ds_config.name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{ds_config.name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds_cls = DATASET_REGISTRY[ds_config.name]
    model_type = ModelType(model_type_str)

    return ds_cls(
        tokenizer=tokenizer,
        model_type=model_type,
        **ds_config.kwargs,
    )


def collate_fn(batch: list) -> MiniBatch:
    return MiniBatch(
        input_prompts=[item["input_prompt"] for item in batch],
        expected_outputs=[item["expected_output"] for item in batch],
        inputs=[item["input"] for item in batch],
    )


# ── vLLM server management ──────────────────────────────────────────────────


async def start_vllm_server(config: PipelineConfig) -> asyncio.subprocess.Process:
    cmd = [
        "vllm",
        "serve", config.model_name,
        "--host", "0.0.0.0",
        "--port", str(config.port),
        "--disable-log-requests",
        "--served-model-name", config.model_name,
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--data-parallel-size", str(config.data_parallel_size),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-model-len", str(config.max_model_len),
    ]
    if config.vllm_extra_args:
        cmd.extend(config.vllm_extra_args)

    logger.info(f"Starting vLLM: {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )

    async def _drain_stdout():
        while proc.stdout and not proc.stdout.at_eof():
            line = await proc.stdout.readline()
            if line:
                logger.info(f"[vllm] {line.decode().rstrip()}")

    asyncio.create_task(_drain_stdout())

    def _kill():
        try:
            proc.terminate()
        except Exception:
            pass

    atexit.register(_kill)
    return proc


async def wait_for_server(
    base_url: str,
    api_key: Optional[str] = None,
    max_attempts: int = 300,
    proc: Optional[asyncio.subprocess.Process] = None,
):
    url = f"{base_url.rstrip('/')}/models"
    for attempt in range(1, max_attempts + 1):
        if proc is not None and proc.returncode is not None:
            raise RuntimeError(
                f"vLLM process exited with code {proc.returncode} before becoming ready. "
                "Check the [vllm] log lines above for details."
            )
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    logger.info("vLLM server is ready")
                    return
        except Exception:
            if attempt % 10 == 0:
                logger.info(f"Waiting for vLLM server... (attempt {attempt}/{max_attempts})")
        await asyncio.sleep(1)
    raise RuntimeError("vLLM server did not become ready")


# ── Inference ────────────────────────────────────────────────────────────────


async def apost(url: str, json_data: dict, api_key: Optional[str] = None):
    parsed = urlparse(url)
    host = parsed.hostname
    use_ssl = parsed.scheme == "https"
    port = parsed.port or (443 if use_ssl else 80)
    path = parsed.path or "/"

    writer = None
    try:
        if use_ssl:
            ctx = ssl.create_default_context()
            reader, writer = await asyncio.open_connection(host, port, ssl=ctx)
        else:
            reader, writer = await asyncio.open_connection(host, port)

        payload = json.dumps(json_data)
        headers = [
            f"POST {path} HTTP/1.1",
            f"Host: {host}",
            "Content-Type: application/json",
            f"Content-Length: {len(payload)}",
        ]
        if api_key:
            headers.append(f"Authorization: Bearer {api_key}")
        headers.append("Connection: close")

        request = "\r\n".join(headers) + "\r\n\r\n" + payload
        writer.write(request.encode())
        await writer.drain()

        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response")
        status_code = int(status_line.decode().strip().split(" ", 2)[1])

        resp_headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode().partition(":")
            resp_headers[k.strip().lower()] = v.strip()

        if "content-length" in resp_headers:
            body = await reader.readexactly(int(resp_headers["content-length"]))
        elif resp_headers.get("transfer-encoding") == "chunked":
            chunks = []
            while True:
                size_line = await reader.readline()
                chunk_size = int(size_line.strip(), 16)
                if chunk_size == 0:
                    await reader.readline()
                    break
                chunks.append(await reader.readexactly(chunk_size))
                await reader.readline()
            body = b"".join(chunks)
        else:
            body = await reader.read()

        return status_code, body
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


async def run_inference_single(
    messages: list,
    base_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    api_key: Optional[str],
    max_retries: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        try:
            status, body = await apost(url, payload, api_key=api_key)
            if status == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            if status != 200:
                logger.warning(f"HTTP {status}: {body[:200]}")
                await asyncio.sleep(1)
                continue
            data = json.loads(body)
            return data["choices"][0]["message"]["content"].strip()
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"Connection error attempt {attempt}: {e}")
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"Error attempt {attempt}: {e}")
            await asyncio.sleep(1)

    logger.error(f"Failed after {max_retries} retries")
    return ""


async def run_inference_batch(
    batch: MiniBatch,
    base_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    api_key: Optional[str],
    max_retries: int,
    semaphore: asyncio.Semaphore,
) -> List[str]:
    async def _infer(messages):
        async with semaphore:
            return await run_inference_single(
                messages, base_url, model_name,
                max_tokens, temperature, api_key, max_retries,
            )

    tasks = [_infer(msgs) for msgs in batch.input_prompts]
    return await asyncio.gather(*tasks)


# ── Metrics ──────────────────────────────────────────────────────────────────


METRIC_FNS = {
    "bleu": lambda hyps, refs: evaluate.load("sacrebleu").compute(predictions=hyps, references=[[r] for r in refs])["score"],
    "chrf": lambda hyps, refs: evaluate.load("chrf").compute(
        predictions=hyps, references=[[r] for r in refs])["score"],
    "ter": lambda hyps, refs: evaluate.load("ter").compute(predictions=hyps, references=[[r] for r in refs])["score"],
    "chrf++": lambda hyps, refs: evaluate.load("chrf").compute(
        predictions=hyps, references=[[r] for r in refs], word_order=2)["score"]
}


def compute_metrics(
    hypotheses: List[str], references: List[str], metric_names: List[str]
) -> dict:
    results = {}
    for name in metric_names:
        if name not in METRIC_FNS:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(METRIC_FNS.keys())}")
        results[name] = METRIC_FNS[name](hypotheses, references)
    return results


# ── Main pipeline ────────────────────────────────────────────────────────────


async def run_single_dataset(
    ds_config: DatasetConfig,
    config: PipelineConfig,
    base_url: str,
) -> dict:
    logger.info(f"Loading dataset '{ds_config.name}'...")
    dataset = build_dataset(ds_config, config.model_name, config.model_type)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_hypotheses = []
    all_references = []
    all_sources = []
    concurrency = asyncio.Semaphore(config.batch_size * 2)

    logger.info(f"Running inference on {len(dataset)} samples for '{ds_config.name}'...")
    start = time.time()

    for batch in tqdm(loader, desc=f"Inference [{ds_config.name}]"):
        translations = await run_inference_batch(
            batch, base_url, config.model_name,
            config.max_tokens, config.temperature,
            config.api_key, config.max_retries, concurrency,
        )
        all_hypotheses.extend(translations)
        all_references.extend(batch.expected_outputs)
        all_sources.extend(batch.inputs)

    elapsed = time.time() - start
    logger.info(f"Inference for '{ds_config.name}' completed in {elapsed:.1f}s")

    logger.info("Computing metrics...")
    scores = compute_metrics(all_hypotheses, all_references, config.metrics)

    for name, val in scores.items():
        logger.info(f"  {name}: {val:.4f}")

    results = {
        "model": config.model_name,
        "dataset": ds_config.name,
        "dataset_kwargs": ds_config.kwargs,
        "metrics": scores,
        "num_samples": len(all_hypotheses),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }

    if config.save_translations:
        results["translations"] = [
            {"source": s, "reference": r, "hypothesis": h}
            for s, r, h in zip(all_sources, all_references, all_hypotheses)
        ]

    os.makedirs(config.output_dir, exist_ok=True)
    if ds_config.output_file:
        out_path = os.path.join(config.output_dir, ds_config.output_file)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = config.model_name.replace("/", "_")
        out_path = os.path.join(
            config.output_dir,
            f"{safe_model}_{ds_config.name}_{ts}.json",
        )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {out_path}")

    return results


async def run_pipeline(config: PipelineConfig):
    use_internal = config.server is None

    vllm_proc = None
    if use_internal:
        base_url = f"http://localhost:{config.port}/v1"
        vllm_proc = await start_vllm_server(config)
    else:
        base_url = config.server

    try:
        await wait_for_server(base_url, config.api_key, proc=vllm_proc)

        all_results = []
        for i, ds_config in enumerate(config.datasets):
            logger.info(f"\n{'='*60}")
            logger.info(f"Dataset {i+1}/{len(config.datasets)}: {ds_config.name}")
            logger.info(f"{'='*60}")

            if ds_config.output_file:
                out_path = os.path.join(config.output_dir, ds_config.output_file)
                if os.path.exists(out_path):
                    logger.info(f"Skipping '{ds_config.name}' — output file already exists: {out_path}")
                    continue

            result = await run_single_dataset(ds_config, config, base_url)
            all_results.append(result)

        return all_results

    finally:
        if vllm_proc is not None:
            vllm_proc.terminate()
            try:
                await asyncio.wait_for(vllm_proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                vllm_proc.kill()


def main():
    parser = argparse.ArgumentParser(description="Translation benchmark pipeline")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--server", default=None, help="Override vLLM server URL")
    parser.add_argument("--api-key", default=None, help="Override API key")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)

    if args.server is not None:
        config.server = args.server
    if args.api_key is not None:
        config.api_key = args.api_key
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    all_results = asyncio.run(run_pipeline(config))

    for results in all_results:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Model:   {results['model']}")
        print(f"Dataset: {results['dataset']}")
        print(f"Samples: {results['num_samples']}")
        print(f"Time:    {results['elapsed_seconds']}s")
        print("-" * 60)
        for name, val in results["metrics"].items():
            print(f"  {name:>10}: {val:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
