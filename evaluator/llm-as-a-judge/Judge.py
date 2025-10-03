from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class Judge:
    def __init__(self, model_name, cache_dir, device='cpu', quantization=False):
        
        self.device = device
        self.cache_dir = cache_dir

        # Quantization configuration
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            cache_dir=cache_dir,
            quantization_config=self.quantization_config if quantization else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

    def evaluate(self, prompt, temperature=0.01, max_new_tokens=512):
        try:
            # Format the prompt into messages
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Prepare model inputs
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # Apply attention mask
            attention_mask = model_inputs.attention_mask

            # Generate response
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Extract the newly generated tokens
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # Decode the response
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response

        except Exception as e:
            print(f"Error in evaluate function: {e}")
            return None


class GPTJudge(Judge):

    def __init__(self, cache_dir, device='cpu', quantization=True):
        super().__init__("openai/gpt-oss-20b", cache_dir, device, quantization)


class SeleneJudge(Judge):

    def __init__(self, cache_dir, device='cpu', quantization=False):
        super().__init__("AtlaAI/Selene-1-Mini-Llama-3.1-8B", cache_dir, device, quantization)

    def parse_atla_response(self, response):
        """
        Parse ATLA model response to extract reasoning and score.

        Args:
            response (str): Raw response from ATLA model

        Returns:
            tuple: (critique, score) where critique is a string and score is an integer
        """
        try:
            # Split into lines and clean up
            lines = [line.strip() for line in response.split('\n') if line.strip()]

            # Extract critique (everything between **Reasoning:** and **Result:**)
            critique = None
            score = None

            for i, line in enumerate(lines):
                if line.startswith("**Reasoning:**"):
                    critique = lines[i].replace("**Reasoning:**", "").strip()
                elif line.startswith("**Result:**"):
                    score = lines[i].replace("**Result:**", "").strip()

            # Remove style tag if present
            if critique and "<userStyle>" in critique:
                critique = critique.split("<userStyle>")[0].strip()

            return critique, score

        except Exception as e:
            print(f"Error parsing ATLA response: {e}")
            return None, None