import json
import os
import time
from google import genai
from google.genai import types
from PIL import Image

IMAGE_DIR = "test"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg')
OUTPUT_JSON_FILE = "underwater_detections.json"

MAX_RETRIES = 5
INITIAL_DELAY = 2

BASE_CATEGORIES = [
    'Fish', 'Reefs', 'Aquatic plants',
    'Wrecks/ruins', 'Human divers', 'Robots', 'Sea-floor'
]

CATEGORY_DEFINITIONS = """
- Fish: Underwater vertebrates, e.g., fish, turtles.
- Reefs: Underwater invertebrates and coral reefs.
- Aquatic plants: Aquatic plants and flora.
- Wrecks/ruins: Wrecks, ruins, and damaged artifacts.
- Human divers: Human divers and their equipment.
- Robots: Underwater robots like AUV, ROV.
- Sea-floor: Rocks and bottom substrate (distinct from living coral).
"""


def generate_content_with_retry(client, img, prompt, config, max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[img, prompt],
                config=config
            )
            return response

        except Exception as e:
            error_message = str(e)

            is_retriable_error = (
                    "RESOURCE_EXHAUSTED" in error_message or
                    "429" in error_message or
                    "InternalServerError" in error_message or
                    "500" in error_message or
                    "ServiceUnavailable" in error_message or
                    "503" in error_message
            )

            if is_retriable_error and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(
                    f"  [Retry {attempt + 1}/{max_retries}]: API error (Potential Quota/Transient: {e})."
                    f" Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                if is_retriable_error:
                    print(f"  [Retry Failed]: API request failed after {max_retries} attempts.")
                else:
                    print(f"  [Fatal API Error]: Non-retryable error: {e}")

                raise e

    return None


def get_bounding_boxes(image_file_path):
    try:
        client = genai.Client()

        if not os.path.exists(image_file_path):
            print(f"Error: Image file not found at {image_file_path}.")
            return None

        img = Image.open(image_file_path)
        file_name_without_ext = os.path.splitext(os.path.basename(image_file_path))[0]

        bounding_box_schema = types.Schema(
            type=types.Type.ARRAY,
            description="List of detected objects and their normalized bounding boxes.",
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "label": types.Schema(type=types.Type.STRING, description="The primary category label."),
                    "detailed_label": types.Schema(type=types.Type.STRING,
                                                   description="A specific description of the item."),
                    "box_2d": types.Schema(type=types.Type.ARRAY,
                                           description="Normalized bounding box: [ymin, xmin, ymax, xmax] (0-1000).",
                                           items=types.Schema(type=types.Type.NUMBER))
                },
                required=["label", "detailed_label", "box_2d"]
            )
        )

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="application/json",
            response_schema=bounding_box_schema
        )

        prompt = (
            "You are an expert underwater image analyst. Your task is to perform **saliency detection** on this image. "
            "Identify the most prominent, attention-grabbing underwater items present.\n\n"

            "For each salient item detected, provide:\n"
            "1. `box_2d`: The bounding box coordinates normalized to 0-1000 in the format [ymin, xmin, ymax, xmax].\n"
            "2. `label`: The strict category classification.\n"
            "3. `detailed_label`: A specific, descriptive name (e.g., 'Green Sea Turtle', 'Rusting Shipwreck', "
            "'Brain Coral').\n\n"

            "**CATEGORY DEFINITIONS (Strictly adhere to these):**\n"
            f"{CATEGORY_DEFINITIONS}\n"

            "**CONSTRAINTS:**\n"
            f"- The `label` field must ONLY be one of these exact strings: {BASE_CATEGORIES}.\n"
            "- If an object is a living invertebrate or coral, label it 'Reefs'. If it is geological rock/sand, "
            "label it 'Sea-floor'.\n"
            "- Output the result as a valid JSON list of objects."
        )

        response = generate_content_with_retry(client, img, prompt, config)

        if response and response.usage_metadata:
            total_tokens = response.usage_metadata.total_token_count
            print(f"  -> Total Tokens for this Request: {total_tokens}")

        coordinates_json = response.text.strip()
        detections_list = json.loads(coordinates_json)

        formatted_result = {
            "file_name": file_name_without_ext,
            "detections": detections_list
        }

        return formatted_result

    except Exception as e:
        print(f"  [ERROR] Skipping {image_file_path} due to fatal error: {e}")
        return None


def process_multiple_images(image_paths, output_json_file):
    all_results = []

    print(f"🌊 Starting batch processing for {len(image_paths)} images...")

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i + 1}/{len(image_paths)}] Processing: **{image_path}**")

        file_data = get_bounding_boxes(image_path)

        if file_data and file_data.get("detections") is not None:
            all_results.append(file_data)
            print(f"  -> Successfully detected {len(file_data['detections'])} objects.")
        else:
            print(f"  -> Failed to get valid data. Skipping file.")

    try:
        with open(output_json_file, 'w') as f:
            json.dump(all_results, f, indent=4)

        total_detections = sum(len(d['detections']) for d in all_results)

        print(f"\n--- Batch Processing Complete ---")
        print(f"Total images processed successfully: **{len(all_results)}**")
        print(f"Total detections saved: **{total_detections}**")
        print(f"All results written to: **{output_json_file}**")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


if __name__ == "__main__":

    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Directory '{IMAGE_DIR}' not found. Please create it and place your images inside.")
    else:
        image_files_to_process = [
            os.path.join(IMAGE_DIR, f)
            for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if not image_files_to_process:
            print(f"Warning: No JPG/JPEG files found in the '{IMAGE_DIR}' directory.")
        else:
            process_multiple_images(
                image_files_to_process,
                OUTPUT_JSON_FILE
            )
