import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math


class CausalNaturalnessScorer:
    """
    基于自回归模型的代码自然度/隐蔽性评分器。
    通过计算代码片段的困惑度 (Perplexity) 来量化其自然度。
    困惑度越低，代码越自然（隐蔽性越高）。
    """

    def __init__(self, model_name_or_path: str = "deepseek-ai/deepseek-coder-1.3b-base", device: str = None):
        """
        初始化自回归模型和分词器。
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Loading autoregressive model {model_name_or_path} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        # 加载模型并设置为评估模式
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def calculate_naturalness_score(self, code_snippet: str) -> dict:
        """
        计算单段代码的自然度得分。

        Args:
            code_snippet (str): 待评估的代码片段（建议输入包含目标标识符的完整函数或代码块）。

        Returns:
            dict: 包含 Loss (负对数似然) 和 PPL (困惑度) 的字典。
        """
        if not code_snippet.strip():
            return {"loss": float('inf'), "perplexity": float('inf')}

        # Tokenize 输入代码
        inputs = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(self.device)

        # 自回归模型计算 Loss 时，将 labels 设置为 input_ids，
        # 内部会自动进行 shift (预测下一个 token) 并计算 Cross-Entropy Loss
        outputs = self.model(input_ids=input_ids, labels=input_ids)

        loss = outputs.loss.item()

        # 处理可能出现的溢出情况
        try:
            ppl = math.exp(loss)
        except OverflowError:
            ppl = float('inf')

        return {
            "loss": loss,  # 交叉熵损失 (通常可以直接作为负向得分)
            "perplexity": ppl  # 困惑度 (直观的自然度指标，越低越好)
        }

    def evaluate_candidate(self, original_code: str, adv_code: str, ppl_tolerance_ratio: float = 1.1) -> bool:
        """
        评估对抗样本是否满足隐蔽性阈值。

        Args:
            original_code (str): 原始代码片段。
            adv_code (str): 替换标识符后的对抗代码片段。
            ppl_tolerance_ratio (float): PPL 容忍度倍数。1.1 表示对抗样本的 PPL 不能超过原始 PPL 的 10%。

        Returns:
            bool: 是否通过自然度检验。
        """
        orig_metrics = self.calculate_naturalness_score(original_code)
        adv_metrics = self.calculate_naturalness_score(adv_code)

        orig_ppl = orig_metrics["perplexity"]
        adv_ppl = adv_metrics["perplexity"]

        # 严格过滤：如果对抗代码的困惑度大幅超过原始代码，则视为不自然，触发丢弃
        is_stealthy = adv_ppl <= (orig_ppl * ppl_tolerance_ratio)

        return is_stealthy, orig_ppl, adv_ppl