from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class QAService:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)

    def _answer_single(self, question: str, context: str):
        """Run QA on a single context chunk. Returns (answer, score)."""
        inputs = self.tokenizer(
            question, context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        seq_len = start_logits.shape[0]

        best_score = float("-inf")
        best_start, best_end = 0, 0
        for s in range(seq_len):
            for e in range(s, min(s + 30, seq_len)):
                score = start_logits[s] + end_logits[e]
                if score > best_score:
                    best_score = score
                    best_start, best_end = s, e

        tokens = inputs["input_ids"][0][best_start: best_end + 1]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return answer, best_score

    def answer(self, question: str, context: str):
        """Answers a question by running QA on each chunk and picking the best answer."""
        if not context.strip():
            return "No context provided to answer the question."

        # context may be multiple chunks joined by newline — score each separately
        chunks = [c.strip() for c in context.split("\n") if c.strip()]
        if not chunks:
            chunks = [context]

        try:
            best_answer, best_score = "", float("-inf")
            for chunk in chunks:
                ans, score = self._answer_single(question, chunk)
                if score > best_score and ans:
                    best_score = score
                    best_answer = ans

            return best_answer if best_answer else "I could not find a specific answer in the documents."
        except Exception as e:
            print(f"QA Error: {e}")
            return "I could not find a specific answer in the documents."
