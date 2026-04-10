from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.config import LLM_MODEL, MODELS_DIR


def get_llm() -> HuggingFacePipeline:
    """Loads Gemma-2b-it locally. Downloads on first run, then uses cache thereafter.

    Returns:
        HuggingFacePipeline: The HuggingFacePipeline class to use HF API.
    """
    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, cache_dir=MODELS_DIR)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        cache_dir=MODELS_DIR,
        dtype="auto",
        device_map="auto",
    )

    # Transformers Pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)
