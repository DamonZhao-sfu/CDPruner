import time
import torch
import numpy as np
from PIL import Image
import requests
import json
import io
import base64
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, CLIPImageProcessor
from cdencoder import CLIPVisionTower  # Assuming first file is saved as clipEncoder.py
vision_tower_name = "/data/models/clip-vit-p14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"  # Default CLIP model
class MockArgs:
    def __init__(self):
        self.mm_vision_select_layer = -2
        self.mm_vision_select_feature = 'patch'

mock_args = MockArgs()
vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
vision_tower = vision_tower.to("cuda")

# def getPrunedVisualToken(model,image_path, texts):

#     image = Image.open(image_path)        
#     inputs = vision_tower.image_processor(image, return_tensors="pt")
#     images = inputs["pixel_values"]
#     image_stream = torch.cuda.Stream()
#     text_stream = torch.cuda.Stream()
    
#     with torch.cuda.stream(image_stream):
#         image_forward_outs = vision_tower.vision_tower(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
#         image_outputs = vision_tower.feature_select(image_forward_outs)
#         image_features = image_outputs.to(images.dtype)
    
#     if texts is not None:
#         with torch.cuda.stream(text_stream):
#             text_inputs = vision_tower.text_tokenizer(text=texts, return_tensors="pt")
#             text_segment = (text_inputs.input_ids.shape[1] - 1) // vision_tower.max_position_embeddings + 1
#             text_padding = vision_tower.max_position_embeddings * text_segment - text_inputs.input_ids.shape[1]
#             text_inputs = {
#                 k: torch.cat([v, v.new_zeros((v.shape[0], text_padding))], 
#                                 dim=1).reshape(-1, vision_tower.max_position_embeddings).to(device=vision_tower.device)
#                 for k, v in text_inputs.items()
#             }
#             text_embeds = vision_tower.text_tower(**text_inputs).text_embeds.cpu()
    
#     torch.cuda.synchronize()

#     if texts is not None:
#         image_embeds = vision_tower.vision_tower.vision_model.post_layernorm(image_outputs)
#         image_embeds = vision_tower.vision_tower.visual_projection(image_embeds.float())
#         #image_features = (image_features, image_embeds, text_embeds)

#     B, N, C = image_features.shape
#     device = image_features.device
#     index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
    
#     image_features = model.multi_modal_projector(image_features.to(torch.float16)).cpu()
    
#     # [CDPruner] Calculate cosine similarity
#     image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
#     image_normalized = image_normalized.float() # (B, N, D)
#     similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)

#     # [CDPruner] Calculate query relevance
#     image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) # (B, N, C)
#     text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # (M, C)
#     relevance = torch.matmul(image_embeds, text_embeds.t()) # (B, N, M)
#     relevance = (-relevance).mean(dim=-1) # (B, N)
#     relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)

#     # [CDPruner] Construct kernel matrix
#     # You can use an additional hyperparameter theta to control the influence of the relevance score.
#     # theta = 0.5
#     # alpha = theta / (2 * (1 - theta))
#     # relevance = torch.exp(alpha * relevance) # (B, N)
#     kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1) # (B, N, N)

#     # [CDPruner] Fast MAP inference of conditional DPP
#     cis = torch.zeros((144, B, N), device=device) # (T, B, N)
#     di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
#     select_idx = torch.empty((144, B), dtype=torch.long, device=device) # (T, B)
#     for i in range(144):
#         j = torch.argmax(di2s, dim=-1)
#         select_idx[i] = j

#         eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
#             / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
#         cis[i, :, :] = eis
#         di2s -= torch.square(eis)
#         di2s[torch.arange(B), j] = -float('inf')
    
#     select_idx = torch.sort(select_idx.t()).values # (B, T)
#     index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
#     index_masks.scatter_(1, select_idx, True)
#     image_features = image_features[index_masks].unsqueeze(0)
#     print(image_features.shape)
#     return image_features.detach().cpu()


def getPrunedVisualToken(model, image_path, texts):
    image = Image.open(image_path)        
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    image_stream = torch.cuda.Stream()
    text_stream = torch.cuda.Stream()
    
    model_device = vision_tower.device
    # print(f"Model device: {model_device}")
    # print(f"visionTower device: {vision_tower.device}")
    
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
            # Keep text_embeds on GPU
            text_embeds = vision_tower.text_tower(**text_inputs).text_embeds
    
    torch.cuda.synchronize()

    if texts is not None:
        image_embeds = vision_tower.vision_tower.vision_model.post_layernorm(image_outputs)
        image_embeds = vision_tower.vision_tower.visual_projection(image_embeds.float())

    B, N, C = image_features.shape
    
    # Move image_features to the same device as the model BEFORE applying projection
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    
    # Ensure all tensors are on the same device as the model
    image_embeds = image_embeds.to(device=model_device)
    text_embeds = text_embeds.to(device=model_device)
    
    # print(f"Before projection - image_features device: {image_features.device}")
    # print(f"Model multi_modal_projector device: {model_device}")
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)

    # Apply projection - now both tensors are on the same device
    image_features = model.multi_modal_projector(image_features)
    
    # print(f"After projection - image_features device: {image_features.device}")
    # print(f"image_embeds device: {image_embeds.device}")
    # print(f"text_embeds device: {text_embeds.device}")
    
    # [CDPruner] Calculate cosine similarity - ALL ON SAME DEVICE
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # (B, N, D)
    image_normalized = image_normalized.float()  # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # (B, N, N)

    # [CDPruner] Calculate query relevance - ALL ON SAME DEVICE
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # (B, N, C)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # (M, C)
    relevance = torch.matmul(image_embeds, text_embeds.t())  # (B, N, M)
    relevance = (-relevance).mean(dim=-1)  # (B, N)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min())  # (B, N)

    # [CDPruner] Construct kernel matrix - ALL ON SAME DEVICE
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # (B, N, N)
    
    # Verify all tensors are on the same device
    # print(f"similarity device: {similarity.device}")
    # print(f"relevance device: {relevance.device}")
    # print(f"kernel device: {kernel.device}")
    
    # [CDPruner] Fast MAP inference of conditional DPP - ALL ON SAME DEVICE
    cis = torch.zeros((144, B, N), device=model_device)  # (T, B, N)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()  # (B, N)
    select_idx = torch.empty((144, B), dtype=torch.long, device=model_device)  # (T, B)
    
    for i in range(144):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')
    
    select_idx = torch.sort(select_idx.t()).values  # (B, T)
    index_masks = torch.zeros(B, N, dtype=torch.bool, device=model_device)
    index_masks.scatter_(1, select_idx, True)
    
    # Apply mask and move final result to CPU only at the very end
    image_features_selected = image_features[index_masks].unsqueeze(0).detach().cpu()
    print(f"Final output shape: {image_features_selected.shape}")
    
    return image_features_selected

def encode_image_embedding_to_base64(image_embedding):
    """
    Encode image embedding tensor to base64 string
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings
        
    Returns:
        base64 encoded string of the tensor
    """
    buffer = io.BytesIO()
    torch.save(image_embedding, buffer)
    buffer.seek(0)
    binary_data = buffer.read()
    base64_image_embedding = base64.b64encode(binary_data).decode('utf-8')
    return base64_image_embedding


def call_vllm_api_with_embeds(image_embedding, question="What's in this image?", model="llava-hf/llava-1.5-7b-hf", api_url="http://localhost:8005"):
    """
    Call vLLM API with encoded image embedding
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings
        question: Question to ask about the image
        model: Model name to use
        api_url: API endpoint URL
        
    Returns:
        API response
    """
    # Encode image embedding
    base64_image_embedding = encode_image_embedding_to_base64(image_embedding)
    
    # Prepare the request payload
    embeds = {
        "type": "image_embeds",
        "image_embeds": base64_image_embedding
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    embeds,
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "guided_choice": ["A", "B", "C", "D"]  # Add this line
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request to {api_url}/v1/chat/completions...")
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return None



def call_vllm_api_with_image_base64(image_path, question="What's in this image?", model="llava-hf/llava-1.5-7b-hf", api_url="http://localhost:8005"):
    """
    Call vLLM API with direct image path using base64 encoding
    
    Args:
        image_path: Path to the image file (local path or URL)
        question: Question to ask about the image
        model: Model name to use
        api_url: API endpoint URL
        
    Returns:
        API response
    """
    try:
        # Read and encode image to base64
        if image_path.startswith(('http://', 'https://')):
            # Download image from URL
            response = requests.get(image_path)
            response.raise_for_status()
            image_data = response.content
        else:
            # Read local image file
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Detect image format from file extension or content
        if image_path.lower().endswith(('.png', '.PNG')):
            image_format = 'png'
        elif image_path.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
            image_format = 'jpeg'
        elif image_path.lower().endswith(('.gif', '.GIF')):
            image_format = 'gif'
        elif image_path.lower().endswith(('.webp', '.WEBP')):
            image_format = 'webp'
        else:
            # Default to jpeg if format cannot be determined
            image_format = 'jpeg'
        
        # Create data URL with base64 encoded image
        image_url = f"data:image/{image_format};base64,{base64_image}"
        
    except Exception as e:
        print(f"Error reading/encoding image: {e}")
        return None
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "guided_choice": ["A", "B", "C", "D"]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request to {api_url}/v1/chat/completions with base64 encoded image")
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return None

def call_vllm_api_with_image_path(image_path, question="What's in this image?", model="llava-hf/llava-1.5-7b-hf", api_url="http://localhost:8005"):
    """
    Call vLLM API with direct image path/URL
    
    Args:
        image_path: Path to the image file (local path or URL)
        question: Question to ask about the image
        model: Model name to use
        api_url: API endpoint URL
        
    Returns:
        API response
    """
    # Convert local path to file URL if it's a local path
    if os.path.exists(image_path) and not image_path.startswith(('http://', 'https://')):
        # Convert local path to file URL
        image_url = f"file://{os.path.abspath(image_path)}"
    else:
        # Assume it's already a URL
        image_url = image_path
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "guided_choice": ["A", "B", "C", "D"]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request to {api_url}/v1/chat/completions with image URL: {image_url}")
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return None


# Wrapper function to choose between embed and image path methods
def call_vllm_api(image_embedding=None, image_path=None, question="What's in this image?", model="llava-hf/llava-1.5-7b-hf", api_url="http://localhost:8005", use_image_path=False):
    """
    Call vLLM API with either image embedding or direct image path
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings (for embed method)
        image_path: Path to image file (for direct path method)
        question: Question to ask about the image
        model: Model name to use
        api_url: API endpoint URL
        use_image_path: If True, use direct image path method; if False, use embedding method
        
    Returns:
        API response
    """
    if use_image_path:
        if image_path is None:
            raise ValueError("image_path must be provided when use_image_path=True")
        return call_vllm_api_with_image_path(image_path, question, model, api_url)
    else:
        if image_embedding is None:
            raise ValueError("image_embedding must be provided when use_image_path=False")
        return call_vllm_api_with_embeds(image_embedding, question, model, api_url)

        

def main():
    MODEL_PATH = "/data/models/llava-1.5-7b-hf"
    IMAGE_PATH = "/home/haikai/AI_UDF/sparkai/examples/python/35b31d9b4f723f806fd32662ef29edf7.jpg"
    API_URL = "http://localhost:8005"

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="cuda",
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
        question = row['question']  if row['question'] else ""
        hint = row['hint'] if row['hint'] else ""
        correct_answer = row['answer']  if row['answer']  else ""
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
            'api_call_time': 0,
            'total_time': 0,
            'api_success': False,
            'generated_text': '',
            'predicted_answer': '',
            'is_correct': False,
            'error_message': '',
            'full_response': '',
            'model_path': MODEL_PATH,
            'original_token': 0,
            'pruned_token': 0,
            'api_url': API_URL,
            'method': 'embeddings'
        }
        
        try:
            if True:
                # Embedding method - use existing pruning logic
                prune_time_begin = time.time()
                reduced_tokens = getPrunedVisualToken(model, image_path, formatted_question)
                #reduced_tokens, originaTokenNum, preprocess_time, encode_time, project_time, pruneTime = divprune(model, processor, image_path, device="cuda", prune=False)
                prune_time_end = time.time()

                embed_time = prune_time_end - prune_time_begin
                result_record['original_token'] = 576
                result_record['pruned_token'] = 0
                result_record['preprocess_time'] = 0
                result_record['encode_time'] = 0
                result_record['project_time'] = 0
                result_record['prune_time'] = 0

                api_time_begin = time.time()
                response = call_vllm_api(
                    image_embedding=reduced_tokens.to(torch.float16),
                    question=formatted_question,
                    model="/data/models/llava-1.5-7b-hf",
                    api_url=API_URL,
                    use_image_path=False
                )
                api_time_end = time.time()
            
            api_call_time = api_time_end - api_time_begin
            result_record['embed_time'] = embed_time
            result_record['api_call_time'] = api_call_time
            result_record['total_time'] = embed_time + api_call_time
            
            if response:
                result_record['api_success'] = True
                print(f"embed time: {embed_time:.2f} seconds")
                print(f"api call time: {api_call_time:.2f} seconds")
                print("=" * 60)
                print("API RESPONSE:")
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
                    result_record['full_response'] = json.dumps(response, indent=2)
                    print(f"Full response: {json.dumps(response, indent=2)}")
            else:
                result_record['api_success'] = False
                result_record['error_message'] = "Failed to get response from vLLM API"
                print("Failed to get response from vLLM API")
                
        except Exception as e:
            result_record['api_success'] = False
            result_record['error_message'] = str(e)
            print(f"Error processing test {i}: {e}")
            import traceback
            traceback.print_exc()
        
        # Add the result to our data list
        results_data.append(result_record)
        print()

    # # Calculate accuracy
    # successful_tests = [r for r in results_data if r['api_success'] and r['predicted_answer']]
    # if successful_tests:
    #     accuracy = sum(1 for r in successful_tests if r['is_correct']) / len(successful_tests)
    #     print(f"\nOverall Accuracy: {accuracy:.2%} ({sum(1 for r in successful_tests if r['is_correct'])}/{len(successful_tests)})")

    # print(f"\nToken pruning with vLLM API completed successfully using {'direct image path' if USE_IMAGE_PATH else 'embedding'} method!")

    # # Save results to CSV
    # #results_df = pd.DataFrame(results_data)
    # method_suffix = "cdprune"
    # results_csv_path = f"llava_eval_results_{method_suffix}.csv"
    # results_df.to_csv(results_csv_path, index=False)
    # print(f"\nSaved detailed results to {results_csv_path}")

    # # Calculate summary stats
    # #successful = results_df[results_df['api_success'] & results_df['predicted_answer'].astype(bool)]

    # summary = {}

    # if not successful.empty:
    #     summary['accuracy'] = successful['is_correct'].mean()
    #     summary['accuracy_count'] = f"{successful['is_correct'].sum()}/{len(successful)}"
    #     summary['method'] = 'direct_image_path' if USE_IMAGE_PATH else 'embeddings'

    #     if not USE_IMAGE_PATH:
    #         # Only calculate these stats for embedding method
    #         summary['project_time_avg'] = successful['project_time'].mean()
    #         summary['project_time_min'] = successful['project_time'].min()
    #         summary['project_time_max'] = successful['project_time'].max()

    #         summary['preprocess_time_avg'] = successful['preprocess_time'].mean()
    #         summary['preprocess_time_min'] = successful['preprocess_time'].min()
    #         summary['preprocess_time_max'] = successful['preprocess_time'].max()

    #         summary['encode_time_avg'] = successful['encode_time'].mean()
    #         summary['encode_time_min'] = successful['encode_time'].min()
    #         summary['encode_time_max'] = successful['encode_time'].max()

    #         summary['prune_time_avg'] = successful['prune_time'].mean()
    #         summary['prune_time_min'] = successful['prune_time'].min()
    #         summary['prune_time_max'] = successful['prune_time'].max()

    #         summary['token_original_avg'] = successful['original_token'].mean()
    #         summary['token_pruned_avg'] = successful['pruned_token'].mean()

    #     summary['api_call_time_avg'] = successful['api_call_time'].mean()
    #     summary['api_call_time_min'] = successful['api_call_time'].min()
    #     summary['api_call_time_max'] = successful['api_call_time'].max()

    #     summary['total_time_avg'] = successful['total_time'].mean()
    #     summary['total_time_min'] = successful['total_time'].min()
    #     summary['total_time_max'] = successful['total_time'].max()
    #     summary['total_time_sum'] = successful['total_time'].sum()

    #     print(f"\n=== Summary Statistics ({summary['method']}) ===")
    #     print(f"Accuracy:       {summary['accuracy']:.2%} ({summary['accuracy_count']})")
        
    #     if not USE_IMAGE_PATH:
    #         print(f"PreProcess Time:     avg={summary['preprocess_time_avg']:.2f}s, min={summary['preprocess_time_min']:.2f}s, max={summary['preprocess_time_max']:.2f}s")
    #         print(f"Encode Time:     avg={summary['encode_time_avg']:.2f}s, min={summary['encode_time_min']:.2f}s, max={summary['encode_time_max']:.2f}s")
    #         print(f"Project Time:     avg={summary['project_time_avg']:.2f}s, min={summary['project_time_min']:.2f}s, max={summary['project_time_max']:.2f}s")
    #         print(f"Prune Time:     avg={summary['prune_time_avg']:.2f}s, min={summary['prune_time_min']:.2f}s, max={summary['prune_time_max']:.2f}s")
    #         print(f"Tokens:         avg original={summary['token_original_avg']:.1f}, avg pruned={summary['token_pruned_avg']:.1f}")
        
    #     print(f"API Call Time:  avg={summary['api_call_time_avg']:.2f}s, min={summary['api_call_time_min']:.2f}s, max={summary['api_call_time_max']:.2f}s")
    #     print(f"Total Time:     avg={summary['total_time_avg']:.2f}s, min={summary['total_time_min']:.2f}s, max={summary['total_time_max']:.2f}s, sum={summary['total_time_sum']:.2f}s")

    #     # Save summary stats
    #     summary_df = pd.DataFrame([summary])
    #     summary_csv_path = f"llava_summary_stats_{method_suffix}.csv"
    #     summary_df.to_csv(summary_csv_path, index=False)
    #     print(f"Saved summary statistics to {summary_csv_path}")

    # else:
    #     print("No successful API responses to compute summary statistics.")


if __name__ == '__main__':
    main()