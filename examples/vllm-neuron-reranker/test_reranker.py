"""
vLLM-Neuron Reranker Benchmark - Generic Implementation

This test is designed to work with various Reranker models by using configuration
from config.yaml. Customize the config file for your specific model.

Run with:
    pytest test_reranker.py --benchmark-only --benchmark-json=results.json -v
"""

import csv
import gc
import logging
from pathlib import Path

import pytest
import vllm
from vllm import SamplingParams
from benchmark_capture import profile

# Configure logging for real-time progress updates
logger = logging.getLogger(__name__)


@pytest.mark.benchmark(group="reranker")
@pytest.mark.vllm
@pytest.mark.neuron
@profile()  # Auto-detect Neuron hardware
def test_vllm_neuron_reranker(
    benchmark,
    model_path,
    vllm_config,
    reranker_config,
    benchmark_config,
    reranker_prompts,
    token_ids,
):
    """Generic vLLM-Neuron reranker benchmark."""

    # Load CSV data
    csv_file = Path(__file__).parent / reranker_config['input_file']
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    num_queries = min(len(rows), benchmark_config['num_test_queries'])
    search_num = reranker_config['search_num']
    batch_size = reranker_config['batch_size']
    max_length = reranker_config['max_length']

    logger.info(f"Loaded {len(rows)} queries from {csv_file}")
    logger.info(f"Testing with first {num_queries} queries")

    def setup():
        """Initialize vLLM with reranker model"""
        logger.info("Initializing vLLM-Neuron reranker...")
        logger.info(f"Model: {model_path}")
        logger.info(f"Config: block_size={vllm_config['block_size']}, "
                   f"max_num_seqs={vllm_config['max_num_seqs']}, "
                   f"tensor_parallel_size={vllm_config['tensor_parallel_size']}")

        llm = vllm.LLM(model=model_path, **vllm_config)

        # Get tokenizer and token IDs
        tokenizer = llm.get_tokenizer()
        token_false_id = tokenizer.convert_tokens_to_ids(token_ids['false'])
        token_true_id = tokenizer.convert_tokens_to_ids(token_ids['true'])

        logger.info(f"Token IDs: {token_ids['true']}={token_true_id}, "
                   f"{token_ids['false']}={token_false_id}")

        # Encode prompt templates
        prefix_tokens = tokenizer.encode(
            reranker_prompts['prefix'], add_special_tokens=False
        )
        suffix_tokens = tokenizer.encode(
            reranker_prompts['suffix'], add_special_tokens=False
        )

        logger.info(f"Prefix tokens: {len(prefix_tokens)}, Suffix tokens: {len(suffix_tokens)}")

        return llm, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens

    def format_instruction(query: str, doc: str) -> str:
        """Format instruction for reranker"""
        instruction = reranker_prompts['instruction']
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        # Truncate if too long
        if len(output) >= 2000:
            output = output[:2000]
        return output

    def build_prompts_for_vllm(pairs, tokenizer, prefix_tokens, suffix_tokens):
        """Build prompts with proper tokenization"""
        prompts = []
        budget = max_length - len(prefix_tokens) - len(suffix_tokens)

        # Tokenize pairs
        enc = tokenizer(
            list(pairs),
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            add_special_tokens=False,
            max_length=max(8, budget),
        )

        # Build final prompts: prefix + content + suffix
        for ids in enc["input_ids"]:
            final_ids = prefix_tokens + ids + suffix_tokens
            text = tokenizer.decode(final_ids, skip_special_tokens=False)
            prompts.append(text)

        return prompts

    def run_reranker(llm, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens):
        """Run reranker on queries"""

        # Create SamplingParams
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=20,
            detokenize=True,
            allowed_token_ids=[token_true_id, token_false_id]
        )

        logger.info(f"SamplingParams configured: max_tokens=1, "
                    f"allowed_tokens=[{token_ids['true']}, {token_ids['false']}]")

        # Process each query
        total_processed = 0
        for query_idx, row in enumerate(rows[:num_queries]):
            query = row["query"]

            # Get candidates
            candidates = [
                row[f"answer_{i}"]
                for i in range(search_num)
                if f"answer_{i}" in row
            ]

            # Format query-document pairs
            pairs = [format_instruction(query, doc) for doc in candidates[:search_num]]

            # Build prompts with tokenization
            prompts = build_prompts_for_vllm(pairs, tokenizer, prefix_tokens, suffix_tokens)

            # Process in batches
            query_outputs = []
            for s in range(0, len(prompts), batch_size):
                batch_prompts = prompts[s:s + batch_size]
                outputs = llm.generate(batch_prompts, sampling_params)
                query_outputs.extend(outputs)

            total_processed += len(query_outputs)

            if query_idx == 0:
                # Show first result for verification
                logger.info(f"Query 1: {query[:80]}...")
                logger.info(f"Generated {len(query_outputs)} scores for "
                           f"{len(candidates[:search_num])} candidates")
                if query_outputs:
                    first_output = query_outputs[0]
                    logger.info(f"First output: {first_output.outputs[0].text} "
                               f"(token_ids={first_output.outputs[0].token_ids})")

        logger.info(f"Benchmark completed: processed {total_processed} reranker pairs")
        return total_processed

    def teardown(llm):
        """Cleanup"""
        del llm
        gc.collect()

    # Setup
    llm, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens = setup()

    try:
        # Benchmark
        benchmark.pedantic(
            run_reranker,
            args=(llm, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens),
            rounds=benchmark_config['rounds'],
            warmup_rounds=benchmark_config['warmup_rounds'],
        )

        # Calculate per-query metrics
        total_time_ms = benchmark.stats['mean'] * 1000
        latency_per_query_ms = total_time_ms / num_queries
        qps = num_queries / benchmark.stats['mean']

        # Add custom metrics to benchmark
        benchmark.extra_info['num_queries'] = num_queries
        benchmark.extra_info['total_pairs_processed'] = num_queries * search_num
        benchmark.extra_info['latency_per_query_ms'] = latency_per_query_ms
        benchmark.extra_info['throughput_qps'] = qps
        benchmark.extra_info['block_size'] = vllm_config['block_size']
        benchmark.extra_info['batch_size'] = batch_size
        benchmark.extra_info['tensor_parallel_size'] = vllm_config['tensor_parallel_size']

        # Display results with more decimal places
        print(f"\n{'='*80}")
        print(f"✅ Benchmark Results")
        print(f"{'='*80}")
        print(f"\n📊 Overall Performance:")
        print(f"   Total time (mean): {total_time_ms:.3f} ms")
        print(f"   Min: {benchmark.stats['min']*1000:.3f} ms")
        print(f"   Max: {benchmark.stats['max']*1000:.3f} ms")
        print(f"   Median: {benchmark.stats['median']*1000:.3f} ms")
        print(f"   StdDev: {benchmark.stats['stddev']*1000:.3f} ms")
        print(f"\n📈 Per-Query Metrics:")
        print(f"   Latency per query: {latency_per_query_ms:.3f} ms ({latency_per_query_ms/1000:.4f} s)")
        print(f"   Throughput (QPS): {qps:.4f} queries/second")
        print(f"\n🔢 Configuration:")
        print(f"   Total queries: {num_queries}")
        print(f"   Candidates per query: {search_num}")
        print(f"   Total pairs: {num_queries * search_num}")
        print(f"   Batch size: {batch_size}")
        print(f"   Block size: {vllm_config['block_size']}")
        print(f"   Tensor parallel size: {vllm_config['tensor_parallel_size']}")
        print(f"{'='*80}\n")

    finally:
        teardown(llm)
