"""Export and quantize trained model for RUBIK Pi deployment."""
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import os

class ModelExporter:
    def __init__(self, base_model="google/paligemma-3b-pt-224", 
                 lora_path="./math_lora_checkpoints/lora_epoch_final"):
        print("Loading base model...")
        self.base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        print("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(base_model)
    
    def merge_and_save(self, output_dir="./merged_model"):
        """Merge LoRA weights with base model."""
        print("Merging LoRA weights...")
        merged_model = self.model.merge_and_unload()
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving merged model to {output_dir}...")
        merged_model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        print("✓ Model merged and saved!")
        return merged_model
    
    def quantize_int8(self, model, output_dir="./quantized_model"):
        """Quantize model to INT8 for edge deployment."""
        print("Quantizing to INT8...")
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving quantized model to {output_dir}...")
        
        torch.save(quantized_model.state_dict(), 
                  os.path.join(output_dir, "model_int8.pt"))
        self.processor.save_pretrained(output_dir)
        
        print("✓ Model quantized and saved!")
        
        # Print size comparison
        original_size = sum(p.numel() * p.element_size() 
                          for p in model.parameters()) / 1024**3
        quantized_size = os.path.getsize(
            os.path.join(output_dir, "model_int8.pt")) / 1024**3
        
        print(f"\nModel size comparison:")
        print(f"  Original: {original_size:.2f} GB")
        print(f"  Quantized: {quantized_size:.2f} GB")
        print(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        return quantized_model
    
    def test_inference(self, stroke_text):
        """Test inference with sample input."""
        print("\nTesting inference...")
        
        inputs = self.processor(
            text=stroke_text,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=4
            )
        
        predicted = self.processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {stroke_text[:100]}...")
        print(f"Predicted: {predicted}")
        
        return predicted


if __name__ == "__main__":
    exporter = ModelExporter(
        lora_path="./math_lora_checkpoints/lora_epoch_final"
    )
    
    # Merge LoRA weights
    merged_model = exporter.merge_and_save()
    
    # Quantize for edge deployment
    exporter.quantize_int8(merged_model)
    
    # Test with sample
    sample_stroke = "<stroke> 112 99 115 102 118 105"
    exporter.test_inference(sample_stroke)

