"""
Generator module for OmniRAG Studio.
Constructs strictly grounded prompts with source attribution, supports multiple LLM providers
(Anthropic Claude, OpenAI GPT, and offline heuristic engine), and tracks token economics & faithfulness.
"""

import time
import re
from typing import List, Dict, Any, Optional
import os


class PromptBuilder:
    """Formats retrieved context and enforces negative constraint prompt rules."""
    
    @staticmethod
    def build_prompt(persona: str, retrieved_sources: List[Dict[str, Any]], query: str) -> str:
        """Constructs grounded prompt with [Source X] attribution delimiters."""
        if retrieved_sources:
            context_blocks = []
            for i, r in enumerate(retrieved_sources):
                src_num = i + 1
                title = r.get("source_title", f"Doc {r.get('doc_id', '')}")
                context_blocks.append(f"[Source {src_num}] ({title}):\n{r['text']}")
            context = "\n\n".join(context_blocks)
        else:
            context = "NO RELEVANT CONTEXT FOUND."

        prompt = (
            f"You are {persona}.\n\n"
            f"Instructions:\n"
            f"1. Answer using ONLY the factual context provided below.\n"
            f"2. If the answer is not directly contained in the context, or if the context is insufficient, "
            f"explicitly state: 'I don't have that information.'\n"
            f"3. Always cite sources where relevant using the format [Source X].\n"
            f"4. Never hallucinate, extrapolate, or introduce assumptions outside the documented text.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return prompt


class GroundedGenerator:
    """Manages LLM invocation and observability metric computation."""
    
    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 400,
        temperature: float = 0.0
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Resolve defaults based on provider
        if self.provider == "anthropic":
            self.model = self.model or "claude-3-5-sonnet-20241022"
            self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "openai":
            self.model = self.model or "gpt-4o"
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        else:
            self.provider = "offline"

    def generate(
        self,
        persona: str,
        retrieved_sources: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Executes generation and returns answer payload with token economics and quality metrics.
        """
        prompt = PromptBuilder.build_prompt(persona, retrieved_sources, query)
        start_time = time.time()
        
        # Check if we should use Anthropic, OpenAI, or Offline
        if self.provider == "anthropic" and self.api_key:
            return self._generate_anthropic(prompt, retrieved_sources, start_time)
        elif self.provider == "openai" and self.api_key:
            return self._generate_openai(prompt, retrieved_sources, start_time)
        else:
            return self._generate_offline(prompt, retrieved_sources, query, start_time)

    def _generate_anthropic(
        self,
        prompt: str,
        retrieved_sources: List[Dict[str, Any]],
        start_time: float
    ) -> Dict[str, Any]:
        """Inference via Anthropic Claude API."""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        model_name: str = self.model or "claude-3-5-sonnet-20241022"
        response = client.messages.create(
            model=model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency_ms = (time.time() - start_time) * 1000
        answer = response.content[0].text
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens
        
        # Estimate pricing (Claude 3.5 Sonnet: ~$3/M in, $15/M out)
        cost_usd = (in_tokens * 0.000003) + (out_tokens * 0.000015)
        
        return self._format_payload(answer, prompt, retrieved_sources, in_tokens, out_tokens, cost_usd, latency_ms)

    def _generate_openai(
        self,
        prompt: str,
        retrieved_sources: List[Dict[str, Any]],
        start_time: float
    ) -> Dict[str, Any]:
        """Inference via OpenAI GPT API."""
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        
        model_name: str = self.model or "gpt-4o"
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency_ms = (time.time() - start_time) * 1000
        answer = response.choices[0].message.content or ""
        in_tokens = response.usage.prompt_tokens if response.usage is not None else int(len(prompt.split()) * 1.3)
        out_tokens = response.usage.completion_tokens if response.usage is not None else int(len(answer.split()) * 1.3)
        
        # Estimate pricing (GPT-4o: ~$2.50/M in, $10/M out)
        cost_usd = (in_tokens * 0.0000025) + (out_tokens * 0.00001)
        
        return self._format_payload(answer, prompt, retrieved_sources, in_tokens, out_tokens, cost_usd, latency_ms)

    def _generate_offline(
        self,
        prompt: str,
        retrieved_sources: List[Dict[str, Any]],
        query: str,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Deterministic, offline grounded generator.
        Allows testing the full pipeline and verifying PRD failure modes without active API keys.
        """
        # Estimated token counts (roughly 1 token per 0.75 words)
        in_tokens = int(len(prompt.split()) * 1.3)
        
        # If no chunks were retrieved or scores are near zero
        if not retrieved_sources or all(s.get("score", 0.0) < 0.05 for s in retrieved_sources):
            answer = "I don't have that information."
            time.sleep(0.05)  # simulate brief inference
        else:
            top_source = retrieved_sources[0]
            # Simple extractive summary from top source
            top_sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', top_source["text"]) if s.strip()]
            summary = " ".join(top_sentences[:2]) if top_sentences else top_source["text"]
            answer = f"Based on the provided documentation, {summary.lower().capitalize()} [Source 1]"
            if len(retrieved_sources) > 1:
                answer += f". Additional details can be verified in [Source 2]."
            time.sleep(0.08)

        latency_ms = (time.time() - start_time) * 1000
        out_tokens = int(len(answer.split()) * 1.3)
        cost_usd = 0.0  # Offline runs have zero API cost
        
        return self._format_payload(answer, prompt, retrieved_sources, in_tokens, out_tokens, cost_usd, latency_ms)

    def _format_payload(
        self,
        answer: str,
        prompt: str,
        retrieved_sources: List[Dict[str, Any]],
        in_tokens: int,
        out_tokens: int,
        cost_usd: float,
        latency_ms: float
    ) -> Dict[str, Any]:
        """Assembles rich observability payload fulfilling PRD Section 5.2."""
        # Check source citations
        citations_found = re.findall(r'\[Source\s*(\d+)\]', answer, re.IGNORECASE)
        has_citations = len(citations_found) > 0
        
        # Check refusal / fallback
        is_refusal = (
            "don't have that information" in answer.lower() or
            "do not have that information" in answer.lower() or
            "not mentioned in the provided" in answer.lower()
        )
        
        # Context token estimation
        context_words = sum(s.get("word_count", len(s.get("text", "").split())) for s in retrieved_sources)
        context_tokens = int(context_words * 1.3)
        
        return {
            "answer": answer,
            "prompt": prompt,
            "token_metrics": {
                "prompt_tokens": in_tokens,
                "context_tokens": context_tokens,
                "completion_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
                "estimated_cost_usd": round(cost_usd, 6)
            },
            "quality_metrics": {
                "citations_detected": citations_found,
                "has_citations": has_citations,
                "fallback_detected": is_refusal,
                "grounded_faithfulness_score": 100.0 if (has_citations or is_refusal) else 50.0
            },
            "latency_ms": round(latency_ms, 2)
        }
