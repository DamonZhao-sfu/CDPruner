import time
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from cdencoder import CLIPVisionTower  # Assuming first file is saved as clipEncoder.py
import pandas as pd # Import pandas for CSV saving
from vllm import LLM, SamplingParams # Import LLM and SamplingParams for direct vLLM calls


# Global variable for vLLM model
vllm_model = None

def initialize_vllm_model(model_path):
    """Initialize vLLM model once"""
    global vllm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if vllm_model is None:
        print(f"Initializing vLLM model: {model_path}")
        vllm_model = LLM(model=model_path, gpu_memory_utilization=0.8, tensor_parallel_size=2)
        print("vLLM model initialized successfully")
    return vllm_model

def call_vllm_generate_with_embeds(image_embedding, question="What's in this image?", model_path="llava-hf/llava-1.5-7b-hf"):
    """
    Call vLLM generate API with image embeddings
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings
        question: Question to ask about the image
        model_path: Model path for vLLM
        
    Returns:
        Generated text response
    """
    global vllm_model
    
    try:
        # Initialize vLLM model if not already done
        if vllm_model is None:
            vllm_model = initialize_vllm_model(model_path)
        
        # Format prompt according to LLaVA format
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            stop_token_ids=None
        )
        
        print(image_embedding.shape)
        # Prepare input for vLLM
        # Ensure image_embedding is on CPU and converted to numpy if it's not already
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_embedding.detach().cpu()},
        }
        
        print(f"Generating response with vLLM...")
        
        # Generate response
        outputs = vllm_model.generate(
            inputs, 
            sampling_params=sampling_params
        )

        
        
        # Extract generated text
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            return {
                "choices": [
                    {
                        "message": {
                            "content": generated_text
                        }
                    }
                ]
            }
        else:
            print("No output generated from vLLM")
            return None
            
    except Exception as e:
        print(f"Error calling vLLM generate: {e}")
        import traceback
        traceback.print_exc()
        return None

vision_tower_name = "/data/models/clip-vit-p14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"  # Default CLIP model
class MockArgs:
    def __init__(self):
        self.mm_vision_select_layer = -2
        self.mm_vision_select_feature = 'patch'

mock_args = MockArgs()
vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
vision_tower = vision_tower.to("cpu")


def getPrunedVisualToken(model, processor,image_path, texts):

    image = Image.open(image_path)        
    inputs = processor.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    image_stream = torch.cuda.Stream()
    text_stream = torch.cuda.Stream()
    
    with torch.cuda.stream(image_stream):
        image_forward_outs = vision_tower.vision_tower(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(images.dtype)
    
    if texts is not None:
        with torch.cuda.stream(text_stream):
            text_inputs = vision_tower.text_tokenizer(text=texts, return_tensors="pt")
            text_segment = (text_inputs.input_ids.shape[1] - 1) // vision_tower.max_position_embeddings + 1
            text_padding = vision_tower.max_position_embeddings * text_segment - text_inputs.input_ids.shape[1]
            text_inputs = {
                k: torch.cat([v, v.new_zeros((v.shape[0], text_padding))], 
                                dim=1).reshape(-1, vision_tower.max_position_embeddings).to(device=vision_tower.device)
                for k, v in text_inputs.items()
            }
            text_embeds = vision_tower.text_tower(**text_inputs).text_embeds
    
    torch.cuda.synchronize()

    if texts is not None:
        image_embeds = vision_tower.vision_tower.vision_model.post_layernorm(image_outputs)
        image_embeds = vision_tower.vision_tower.visual_projection(image_embeds.float())
        #image_features = (image_features, image_embeds, text_embeds)

    B, N, C = image_features.shape
    device = image_features.device
    index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
    
    image_features = model.multi_modal_projector(image_features.to(torch.float16))
    
    # [CDPruner] Calculate cosine similarity
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
    image_normalized = image_normalized.float() # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)

    # [CDPruner] Calculate query relevance
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) # (B, N, C)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # (M, C)
    relevance = torch.matmul(image_embeds, text_embeds.t()) # (B, N, M)
    relevance = (-relevance).mean(dim=-1) # (B, N)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)

    # [CDPruner] Construct kernel matrix
    # You can use an additional hyperparameter theta to control the influence of the relevance score.
    # theta = 0.5
    # alpha = theta / (2 * (1 - theta))
    # relevance = torch.exp(alpha * relevance) # (B, N)
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1) # (B, N, N)

    # [CDPruner] Fast MAP inference of conditional DPP
    cis = torch.zeros((144, B, N), device=device) # (T, B, N)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
    select_idx = torch.empty((144, B), dtype=torch.long, device=device) # (T, B)
    for i in range(144):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')
    
    select_idx = torch.sort(select_idx.t()).values # (B, T)
    index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    index_masks.scatter_(1, select_idx, True)
    image_features = image_features[index_masks].unsqueeze(0)
    print(image_features.shape)
    return image_features



def main():
    global vllm_model # Declare global to ensure initialization is shared
    MODEL_PATH = "/data/models/llava-1.5-7b-hf"
    IMAGE_PATH = "/home/haikai/AI_UDF/sparkai/examples/python/35b31d9b4f723f806fd32662ef29edf7.jpg"
    API_URL = "http://localhost:8005" # This URL is no longer directly used for calls in main()


    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto",
        attn_implementation="eager"
    )
    
    processor = LlavaProcessor.from_pretrained(MODEL_PATH, patch_size=14)
    
    
    print("Comparing different token pruning methods...")
    
    # Initialize results storage
    results_data = []
    

    # Get sample images for testing
    spark = SparkSession.builder.appName("AudioVisualQAProcessor") \
        .master("local[*]") \
        .config("spark.driver.memory", "16g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    # Load CSV data with questions and hints
    df = spark.read.parquet("/home/haikai/MMbench/dev-00000-of-00001.parquet")
    df.show()

    print("Extracting sample data for testing...")
    sample_data = df.select(
        col("index"),
        col("question"), 
        col("hint"), 
        col("answer"),
        col("A"),
        col("B"), 
        col("C"),
        col("D"),
    ).limit(1000).collect()
    
    print(f"Testing {len(sample_data)} questions...")
    print("-" * 60)

    # Run tests with CSV logging
    for i, row in enumerate(sample_data, 1):
        question = row['question']   if row['question'] else ""
        hint = row['hint'] if row['hint'] else ""
        correct_answer = row['answer']   if row['answer']   else ""
        option_a = row['A'] if row['A'] else ""
        option_b = row['B'] if row['B'] else ""
        option_c = row['C'] if row['C'] else ""
        option_d = row['D'] if row['D'] else ""
        image_path = "/home/haikai/MMbench/extracted_images/" + str(i-1) + ".jpg" if row['index'] else ""
        
        # Format the complete question with options
        formatted_question = f"Question: {question}\n\nHint: {hint}\n\nOptions:\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nPlease analyze the image and answer the question."

        print(f"Test {i}/{len(sample_data)}: {image_path}")
        print(f"Question: {question}")
        print(f"Correct Answer: {correct_answer}")
        
        # Initialize result record for this iteration
        result_record = {
            'test_number': i,
            'total_tests': len(sample_data),
            'sample_image_path': image_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'embed_time': 0,
            'api_call_time': 0, # Renamed from 'api_call_time' to 'generate_time' for clarity
            'total_time': 0,
            'api_success': False, # Renamed from 'api_success' to 'generation_success'
            'generated_text': '',
            'predicted_answer': '',
            'is_correct': False,
            'error_message': '',
            'full_response': '',
            'model_path': MODEL_PATH,
            'original_token': 0,
            'pruned_token': 0,
            'api_url': API_URL, # Still include for context, though not used in direct call
            'method': 'embeddings_direct_vllm' # Updated method name
        }
        
        try:
            # Embedding method - use existing pruning logic
            prune_time_begin = time.time()
            reduced_tokens = getPrunedVisualToken(model, processor, image_path, formatted_question)
            prune_time_end = time.time()

            embed_time = prune_time_end - prune_time_begin
            result_record['original_token'] = 576 # Placeholder, update if actual token count is available
            result_record['pruned_token'] = 0    # Placeholder, update if actual token count is available
            result_record['preprocess_time'] = 0 # Placeholder, update if actual time is available
            result_record['encode_time'] = 0     # Placeholder, update if actual time is available
            result_record['project_time'] = 0    # Placeholder, update if actual time is available
            result_record['prune_time'] = 0      # Placeholder, update if actual time is available

            generate_time_begin = time.time()
            response = call_vllm_generate_with_embeds(
                image_embedding=reduced_tokens.to(torch.float16),
                question=formatted_question,
                model_path=MODEL_PATH # Use model_path here
            )
            generate_time_end = time.time()
            
            generate_time = generate_time_end - generate_time_begin
            result_record['embed_time'] = embed_time
            result_record['api_call_time'] = generate_time # Update key name
            result_record['total_time'] = embed_time + generate_time
            
            if response:
                result_record['api_success'] = True # Update key name
                print(f"embed time: {embed_time:.2f} seconds")
                print(f"generate time: {generate_time:.2f} seconds")
                print("=" * 60)
                print("GENERATION RESPONSE:")
                print("=" * 60)
                
                if 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content']
                    result_record['generated_text'] = content
                    
                    # Extract predicted answer (A, B, C, or D)
                    predicted_answer = ""
                    content_upper = content.upper().strip()
                    if content_upper in ['A', 'B', 'C', 'D']:
                        predicted_answer = content_upper
                    elif 'A)' in content_upper or content_upper.startswith('A'):
                        predicted_answer = 'A'
                    elif 'B)' in content_upper or content_upper.startswith('B'):
                        predicted_answer = 'B'
                    elif 'C)' in content_upper or content_upper.startswith('C'):
                        predicted_answer = 'C'
                    elif 'D)' in content_upper or content_upper.startswith('D'):
                        predicted_answer = 'D'
                    
                    result_record['predicted_answer'] = predicted_answer
                    result_record['is_correct'] = (predicted_answer == correct_answer.upper())
                    
                    print(f"Generated text: {content}")
                    print(f"Predicted answer: {predicted_answer}")
                    print(f"Correct answer: {correct_answer}")
                    print(f"Is correct: {result_record['is_correct']}")
                else:
                    result_record['full_response'] = json.dumps(response, indent=2) # json is not imported, keep as string or import it
                    print(f"Full response: {response}") # Print raw response if json not imported
            else:
                result_record['api_success'] = False
                result_record['error_message'] = "Failed to get response from vLLM generate"
                print("Failed to get response from vLLM generate")
                
        except Exception as e:
            result_record['api_success'] = False
            result_record['error_message'] = str(e)
            print(f"Error processing test {i}: {e}")
            import traceback
            traceback.print_exc()
        
        # Add the result to our data list
        results_data.append(result_record)
        print()

    # Calculate accuracy
    successful_tests = [r for r in results_data if r['api_success'] and r['predicted_answer']]
    if successful_tests:
        accuracy = sum(1 for r in successful_tests if r['is_correct']) / len(successful_tests)
        print(f"\nOverall Accuracy: {accuracy:.2%} ({sum(1 for r in successful_tests if r['is_correct'])}/{len(successful_tests)})")

    print(f"\nToken pruning with vLLM generate completed successfully using embedding method!")

    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    method_suffix = "cdprune_direct_vllm_generate"
    results_csv_path = f"llava_eval_results_{method_suffix}.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved detailed results to {results_csv_path}")

    # Calculate summary stats
    successful = results_df[results_df['api_success'] & results_df['predicted_answer'].astype(bool)]

    summary = {}

    if not successful.empty:
        summary['accuracy'] = successful['is_correct'].mean()
        summary['accuracy_count'] = f"{successful['is_correct'].sum()}/{len(successful)}"
        summary['method'] = 'embeddings_direct_vllm' # Updated method name

        # Only calculate these stats for embedding method
        summary['project_time_avg'] = successful['project_time'].mean()
        summary['project_time_min'] = successful['project_time'].min()
        summary['project_time_max'] = successful['project_time'].max()

        summary['preprocess_time_avg'] = successful['preprocess_time'].mean()
        summary['preprocess_time_min'] = successful['preprocess_time'].min()
        summary['preprocess_time_max'] = successful['preprocess_time'].max()

        summary['encode_time_avg'] = successful['encode_time'].mean()
        summary['encode_time_min'] = successful['encode_time'].min()
        summary['encode_time_max'] = successful['encode_time'].max()

        summary['prune_time_avg'] = successful['prune_time'].mean()
        summary['prune_time_min'] = successful['prune_time'].min()
        summary['prune_time_max'] = successful['prune_time'].max()

        summary['token_original_avg'] = successful['original_token'].mean()
        summary['token_pruned_avg'] = successful['pruned_token'].mean()

        summary['generate_time_avg'] = successful['api_call_time'].mean() # Updated key name
        summary['generate_time_min'] = successful['api_call_time'].min() # Updated key name
        summary['generate_time_max'] = successful['api_call_time'].max() # Updated key name

        summary['total_time_avg'] = successful['total_time'].mean()
        summary['total_time_min'] = successful['total_time'].min()
        summary['total_time_max'] = successful['total_time'].max()
        summary['total_time_sum'] = successful['total_time'].sum()

        print(f"\n=== Summary Statistics ({summary['method']}) ===")
        print(f"Accuracy:        {summary['accuracy']:.2%} ({summary['accuracy_count']})")
        
        print(f"PreProcess Time: avg={summary['preprocess_time_avg']:.2f}s, min={summary['preprocess_time_min']:.2f}s, max={summary['preprocess_time_max']:.2f}s")
        print(f"Encode Time:     avg={summary['encode_time_avg']:.2f}s, min={summary['encode_time_min']:.2f}s, max={summary['encode_time_max']:.2f}s")
        print(f"Project Time:    avg={summary['project_time_avg']:.2f}s, min={summary['project_time_min']:.2f}s, max={summary['project_time_max']:.2f}s")
        print(f"Prune Time:      avg={summary['prune_time_avg']:.2f}s, min={summary['prune_time_min']:.2f}s, max={summary['prune_time_max']:.2f}s")
        print(f"Tokens:          avg original={summary['token_original_avg']:.1f}, avg pruned={summary['token_pruned_avg']:.1f}")
        
        print(f"Generate Time:   avg={summary['generate_time_avg']:.2f}s, min={summary['generate_time_min']:.2f}s, max={summary['generate_time_max']:.2f}s")
        print(f"Total Time:      avg={summary['total_time_avg']:.2f}s, min={summary['total_time_min']:.2f}s, max={summary['total_time_max']:.2f}s, sum={summary['total_time_sum']:.2f}s")

        # Save summary stats
        summary_df = pd.DataFrame([summary])
        summary_csv_path = f"llava_summary_stats_{method_suffix}.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved summary statistics to {summary_csv_path}")

    else:
        print("No successful generation responses to compute summary statistics.")


if __name__ == '__main__':
    main()