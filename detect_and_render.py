import json
import os
from google import genai
from google.genai import types
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

matplotlib.use('TkAgg')
IMAGE_FILE_PATH = "tests/val_01368.jpg"


def get_bounding_boxes(image_file_path):
    try:
        client = genai.Client()
        print(f"Loading image: {image_file_path}...")

        if not os.path.exists(image_file_path):
            print(f"Error: Image file not found at {image_file_path}.")
            return None, None

        img = Image.open(image_file_path)
        width, height = img.size

        bounding_box_schema = types.Schema(
            type=types.Type.ARRAY,
            description="List of detected objects and their normalized bounding boxes.",
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "label": types.Schema(
                        type=types.Type.STRING,
                        description="Description of the detected item."
                    ),
                    "detailed_label": types.Schema(
                        type=types.Type.STRING,
                        description="A specific description of the item (e.g., 'golden fish', 'blue coral reef')."
                    ),
                    "box_2d": types.Schema(
                        type=types.Type.ARRAY,
                        description="Normalized bounding box: [ymin, xmin, ymax, xmax] (0-1000).",
                        items=types.Schema(type=types.Type.NUMBER)
                    )
                },
                required=["label", "box_2d"]
            )
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=bounding_box_schema
        )

        base_categories = [
            'Fish', 'Reefs', 'Aquatic plants',
            'Wrecks/ruins', 'Human divers', 'Robots', 'Sea-floor'
        ]

        prompt = (
            "Detect all prominent items in the image. "
            "For each item, provide a category label, a detailed description, and its box_2d normalized to 0-1000. "
            "The box_2d format must be [ymin, xmin, ymax, xmax]. "
            "**The 'label' field must ONLY be one of the following categories:** "
            f"{', '.join(base_categories)}. "
            "**The 'detailed_label' field must be a specific, descriptive name for the item found in the image** "
            "(e.g., if the category is 'Fish', a detailed label could be 'Lionfish' or 'Turtle')."
        )

        print("Sending request to Gemini model with JSON schema...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, prompt],
            config=config
        )

        coordinates_json = response.text.strip()
        print("\n--- Model Response JSON ---")
        print(coordinates_json)
        print(f"Successfully received {len(coordinates_json)} bytes of JSON data.")
        print("---------------------------\n")

        data = json.loads(coordinates_json)
        absolute_boxes = []

        for item in data:
            ymin_norm, xmin_norm, ymax_norm, xmax_norm = item['box_2d']
            label = item['label']
            detailed_label = item['detailed_label']

            abs_xmin = int(xmin_norm * width / 1000)
            abs_ymin = int(ymin_norm * height / 1000)
            abs_xmax = int(xmax_norm * width / 1000)
            abs_ymax = int(ymax_norm * height / 1000)

            absolute_boxes.append({
                "label": label,
                "detailed_label": detailed_label,
                "box": [abs_xmin, abs_ymin, abs_xmax, abs_ymax]
            })

            print(f"Detected: **{label}** ({detailed_label}) at ({abs_xmin}, {abs_ymin}) to ({abs_xmax}, {abs_ymax})")

        return img, absolute_boxes

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


def visualize_bounding_boxes(img, absolute_boxes):
    if img is None or not absolute_boxes:
        print("No image or bounding boxes to display.")
        return

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for item in absolute_boxes:
        xmin, ymin, xmax, ymax = item['box']
        label = item['label']

        width = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            ymin - 5,
            label,
            color='white',
            fontsize=8,
            bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=2)
        )

    ax.axis('off')

    plt.title("Image with Bounding Boxes (Extracted by Gemini)")
    plt.show()
    print("\nBounding box visualization displayed successfully.")


if __name__ == "__main__":
    image, boxes = get_bounding_boxes(IMAGE_FILE_PATH)
    visualize_bounding_boxes(image, boxes)
