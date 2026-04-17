from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class SummarizerService:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def _generate(self, prompt: str, max_new_tokens: int = 120) -> str:
        """Run a single generation call, safely within 512 token input limit."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _is_generic_summarize(self, question: str) -> bool:
        q = question.lower().strip()
        generic = ["summarize this", "summarize", "summary", "give me a summary",
                   "summarise this", "summarise", "overview"]
        return any(q == g or q.startswith(g) for g in generic)

    def summarize(self, question: str, context: str):
        """
        MAP step: extract one key fact per chunk (stays within 512 tokens each call).
        REDUCE step: only for specific questions — skipped for generic 'summarize this'
                     because FLAN-T5-small hallucinates on multi-chunk synthesis.
        """
        if not context.strip():
            return "No context provided."

        chunks = [c.strip() for c in context.split("\n") if c.strip()]
        if not chunks:
            return "No content to summarize."

        try:
            # MAP: extract facts strictly from each chunk, no hallucination
            partials = []
            for chunk in chunks:
                prompt = (
                    f"Read the text below and extract only facts directly stated in it.\n"
                    f"If the text has no useful information, reply: none\n"
                    f"Text: {chunk}\n"
                    f"Facts:"
                )
                result = self._generate(prompt, max_new_tokens=60)
                # Discard empty, very short, or explicitly "none" responses
                if result and len(result) > 15 and result.lower().strip() not in ("none", "no information", "n/a"):
                    partials.append(result)

            if not partials:
                return "Could not extract information from the document."

            # For generic "summarize this" — return bullet points directly
            # FLAN-T5-small is too small to synthesize reliably without hallucinating
            if self._is_generic_summarize(question):
                return "\n".join(f"• {p}" for p in partials)

            # For specific questions — attempt a focused reduce
            combined = " ".join(partials[:4])  # cap to avoid token overflow
            reduce_prompt = (
                f"Using only the facts provided, answer the question. "
                f"Do not add any information not in the facts.\n"
                f"Facts: {combined}\n"
                f"Question: {question}\n"
                f"Answer:"
            )
            result = self._generate(reduce_prompt, max_new_tokens=120)
            # Fallback to bullets if model ignores the facts
            if not result or len(result) < 10:
                return "\n".join(f"• {p}" for p in partials)
            return result

        except Exception as e:
            print(f"Summarizer Error: {e}")
            return "I encountered an error trying to summarize the answer."
