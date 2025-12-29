"""
LLM-based linguistic feature extraction with LoRA fine-tuning.

Based on the paper's methodology:
1. Fine-tune a pre-trained LLM (Mistral-7B) using LoRA
2. Extract features from the last hidden layer (4096-dim for 7B models)
3. Use these features for downstream classification

The prompt format follows the paper's Section III-B.
"""

import os
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from tqdm import tqdm


# Default picture description (Cookie Theft from Boston Diagnostic Aphasia Examination)
DEFAULT_PICTURE_DESCRIPTION = """a kitchen scene where a woman is washing dishes at a sink. The water is overflowing from the sink onto the floor. Behind her, two children (a boy and a girl) are attempting to steal cookies from a cookie jar on a high shelf. The boy is standing on a stool that is tipping over. The girl is reaching up to receive a cookie from the boy. Through the window, you can see the backyard with a path."""


def create_prompt(
    transcript: str,
    picture_description: str = DEFAULT_PICTURE_DESCRIPTION
) -> str:
    """
    Create the prompt for AD detection.
    
    From paper Section III-B:
    "Carefully analyze the following interview with a person describing 
    the Cookie Theft picture from the Boston Diagnostic Aphasia Examination, 
    in which [PICTURE DESCRIPTION]. Based on your analysis, reply with 'YES' 
    if you believe the person has signs of cognitive impairment, otherwise 
    reply with 'NO'": [INTERVIEW]."
    """
    prompt = f"""Carefully analyze the following interview with a person describing the Cookie Theft picture from the Boston Diagnostic Aphasia Examination, in which {picture_description}. Based on your analysis, reply with 'YES' if you believe the person has signs of cognitive impairment, otherwise reply with 'NO'.

Interview transcript:
{transcript}

Analysis:"""
    
    return prompt


class ADDetectionDataset(Dataset):
    """Dataset for AD detection fine-tuning."""
    
    def __init__(
        self,
        transcripts: List[str],
        labels: List[int],  # 0=HC, 1=AD
        tokenizer,
        max_length: int = 2048,
        picture_description: str = DEFAULT_PICTURE_DESCRIPTION
    ):
        self.transcripts = transcripts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.picture_description = picture_description
    
    def __len__(self):
        return len(self.transcripts)
    
    def __getitem__(self, idx):
        transcript = self.transcripts[idx]
        label = self.labels[idx]
        
        # Create prompt
        prompt = create_prompt(transcript, self.picture_description)
        
        # Add target response
        target = " YES" if label == 1 else " NO"
        full_text = prompt + target
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels (mask prompt, only predict response)
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        labels = encoding['input_ids'].clone()
        labels[0, :prompt_length] = -100  # Mask prompt tokens
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'label': label
        }


class LLMFeatureExtractor:
    """
    LLM-based feature extractor with LoRA fine-tuning.
    
    Extracts 4096-dimensional features from the last hidden layer
    of a fine-tuned LLM (Mistral-7B by default).
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        max_length: int = 2048,
        lora_config: Optional[dict] = None
    ):
        """
        Initialize LLM feature extractor.
        
        Args:
            model_name: HuggingFace model name.
            device: Device to use.
            load_in_8bit: Use 8-bit quantization.
            load_in_4bit: Use 4-bit quantization.
            max_length: Maximum sequence length.
            lora_config: LoRA configuration dictionary.
        """
        self.model_name = model_name
        self.max_length = max_length
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Quantization config
        self.quantization_config = None
        if load_in_4bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif load_in_8bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Default LoRA config (from paper Table II for Mistral)
        self.lora_config = lora_config or {
            'r': 20,
            'alpha': 40,
            'dropout': 0.01,
            'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"],
            'bias': "none",
            'task_type': "CAUSAL_LM"
        }
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.is_fine_tuned = False
    
    def load_model(self):
        """Load the base model and tokenizer."""
        print(f"Loading {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16  # renamed from torch_dtype
        )
        
        # Prepare for k-bit training if quantized
        if self.quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)
        
        print("Model loaded successfully")
        return self
    
    def setup_lora(self):
        """Setup LoRA adapters for fine-tuning."""
        if self.model is None:
            self.load_model()
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['alpha'],
            lora_dropout=self.lora_config['dropout'],
            target_modules=self.lora_config['target_modules'],
            bias=self.lora_config['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
        
        return self
    
    def fine_tune(
        self,
        train_transcripts: List[str],
        train_labels: List[int],
        val_transcripts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 5,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 1e-5,
        output_dir: str = "./checkpoints/llm",
        **kwargs
    ):
        """
        Fine-tune the LLM with LoRA.
        
        Args:
            train_transcripts: Training transcripts.
            train_labels: Training labels (0=HC, 1=AD).
            val_transcripts: Validation transcripts.
            val_labels: Validation labels.
            epochs: Number of training epochs.
            batch_size: Per-device batch size.
            gradient_accumulation_steps: Gradient accumulation steps.
            learning_rate: Learning rate.
            output_dir: Output directory for checkpoints.
        """
        if self.peft_model is None:
            self.setup_lora()
        
        # Create datasets
        train_dataset = ADDetectionDataset(
            train_transcripts,
            train_labels,
            self.tokenizer,
            max_length=self.max_length
        )
        
        val_dataset = None
        if val_transcripts is not None:
            val_dataset = ADDetectionDataset(
                val_transcripts,
                val_labels,
                self.tokenizer,
                max_length=self.max_length
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=kwargs.get('weight_decay', 0.01),
            warmup_ratio=kwargs.get('warmup_ratio', 0.1),
            logging_steps=10,
            eval_strategy="epoch" if val_dataset else "no",  # renamed from evaluation_strategy
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=val_dataset is not None,
            fp16=True,
            optim=kwargs.get('optimizer', 'paged_adamw_8bit'),
            gradient_checkpointing=kwargs.get('gradient_checkpointing', False),
            report_to="tensorboard"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # Train
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save final model
        self.peft_model.save_pretrained(os.path.join(output_dir, "final"))
        
        self.is_fine_tuned = True
        print("Fine-tuning complete")
        
        return self
    
    def extract_features(
        self,
        transcripts: List[str],
        batch_size: int = 4,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract features from transcripts.
        
        Extracts the last hidden state from the LLM for each transcript.
        
        Args:
            transcripts: List of transcripts.
            batch_size: Batch size for processing.
            show_progress: Show progress bar.
            
        Returns:
            Feature array of shape (n_samples, hidden_dim).
            Hidden dim is 4096 for 7B models.
        """
        model = self.peft_model if self.peft_model is not None else self.model
        if model is None:
            self.load_model()
            model = self.model
        
        model.eval()
        features = []
        
        # Process in batches
        n_batches = (len(transcripts) + batch_size - 1) // batch_size
        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting LLM features")
        
        with torch.no_grad():
            for batch_idx in iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(transcripts))
                batch_transcripts = transcripts[start_idx:end_idx]
                
                # Create prompts
                prompts = [create_prompt(t) for t in batch_transcripts]
                
                # Tokenize
                inputs = self.tokenizer(
                    prompts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get last hidden state
                last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
                
                # Pool: take mean of non-padded tokens
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = last_hidden * attention_mask
                summed = masked_hidden.sum(dim=1)
                counts = attention_mask.sum(dim=1)
                pooled = summed / counts  # (batch, hidden_dim)
                
                features.append(pooled.cpu().numpy())
        
        return np.vstack(features)
    
    def save(self, path: str):
        """Save the fine-tuned model."""
        if self.peft_model is not None:
            self.peft_model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        else:
            print("No fine-tuned model to save")
    
    def load(self, path: str):
        """Load a fine-tuned model."""
        from peft import PeftModel
        
        if self.model is None:
            self.load_model()
        
        # Load LoRA weights
        self.peft_model = PeftModel.from_pretrained(self.model, path)
        self.is_fine_tuned = True
        print(f"Loaded fine-tuned model from {path}")
        
        return self


class LLMFeatureExtractorSmallGPU(LLMFeatureExtractor):
    """
    LLM feature extractor optimized for small GPUs (4GB VRAM).
    
    Uses:
    - Smaller model (TinyLlama-1.1B instead of Mistral-7B)
    - 4-bit quantization
    - Gradient checkpointing
    - Minimal LoRA config
    """
    
    # Small models suitable for 4GB VRAM
    SMALL_MODELS = {
        'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # 1.1B params, ~1.5GB in 4-bit
        'phi2': 'microsoft/phi-2',  # 2.7B params, ~2GB in 4-bit
        'opt-1.3b': 'facebook/opt-1.3b',  # 1.3B params
        'pythia-1b': 'EleutherAI/pythia-1b',  # 1B params
        'distilgpt2': 'distilgpt2',  # 82M params, very small
    }
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_length: int = 512,
        lora_config: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize small GPU extractor.
        
        Args:
            model_name: Small model name (default: TinyLlama-1.1B).
            max_length: Max sequence length (shorter for memory).
            lora_config: LoRA configuration.
        """
        # Force 4-bit quantization
        super().__init__(
            model_name=model_name,
            load_in_8bit=False,
            load_in_4bit=True,
            max_length=max_length,
            **kwargs
        )
        
        # Smaller LoRA config for memory efficiency
        self.lora_config = lora_config or {
            'r': 8,
            'alpha': 16,
            'dropout': 0.05,
            'target_modules': ["q_proj", "v_proj"],  # Only attention
            'bias': "none",
            'task_type': "CAUSAL_LM"
        }
        
        print(f"Small GPU mode: Using {model_name}")
        print(f"  Max length: {max_length}")
        print(f"  LoRA r={self.lora_config['r']}, alpha={self.lora_config['alpha']}")
    
    def fine_tune(
        self,
        train_transcripts: List[str],
        train_labels: List[int],
        val_transcripts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 5,
        batch_size: int = 1,  # Minimal batch size
        gradient_accumulation_steps: int = 16,  # Compensate with accumulation
        learning_rate: float = 2e-5,  # Slightly higher for smaller model
        output_dir: str = "./checkpoints/llm",
        gradient_checkpointing: bool = True,  # Always True for small GPU
        **kwargs
    ):
        """Fine-tune with memory-optimized settings for 4GB VRAM."""
        # Remove gradient_checkpointing from kwargs if present to avoid duplicate
        kwargs.pop('gradient_checkpointing', None)
        
        return super().fine_tune(
            train_transcripts=train_transcripts,
            train_labels=train_labels,
            val_transcripts=val_transcripts,
            val_labels=val_labels,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            output_dir=output_dir,
            gradient_checkpointing=True,  # Essential for 4GB, always force True
            **kwargs
        )


def create_llm_extractor(config: dict, small_gpu: bool = False) -> LLMFeatureExtractor:
    """Create LLM feature extractor from config."""
    llm_config = config.get('llm', {})
    
    if small_gpu:
        small_gpu_config = config.get('small_gpu', {})
        # Use smaller model for 4GB VRAM
        model_name = small_gpu_config.get(
            'model_name', 
            'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        )
        lora_config = small_gpu_config.get('lora', None)
        
        return LLMFeatureExtractorSmallGPU(
            model_name=model_name,
            max_length=small_gpu_config.get('max_length', 512),
            lora_config=lora_config
        )
    else:
        return LLMFeatureExtractor(
            model_name=llm_config.get('model_name', 'mistralai/Mistral-7B-v0.1'),
            load_in_8bit=llm_config.get('load_in_8bit', False),
            load_in_4bit=llm_config.get('load_in_4bit', True),
            max_length=llm_config.get('max_length', 2048),
            lora_config=llm_config.get('lora', None)
        )


if __name__ == "__main__":
    # Test LLM feature extraction
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create extractor
    extractor = create_llm_extractor(config, small_gpu=False)
    extractor.load_model()
    
    # Test feature extraction
    test_transcripts = [
        "well there's a boy getting cookies from the cookie jar and he's falling off the stool",
        "the woman is washing dishes and the water is overflowing"
    ]
    
    features = extractor.extract_features(test_transcripts)
    print(f"Feature shape: {features.shape}")

