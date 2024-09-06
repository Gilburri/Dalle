import os
from openai import OpenAI
import re
from datetime import datetime


huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

class HuggingFaceInferenceNode:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api-inference.huggingface.co/v1/",
            api_key=huggingface_token,
        )
        self.prompts_dir = "./prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    def save_prompt(self, prompt):
        filename_text = "hf_" + prompt.split(',')[0].strip()
        filename_text = re.sub(r'[^\w\-_\. ]', '_', filename_text)
        filename_text = filename_text[:30]  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)
        
        with open(filename, "w") as file:
            file.write(prompt)
        
        print(f"Prompt saved to {filename}")

    def generate(self, input_text, happy_talk, compress, compression_level, poster, prompt_type, custom_base_prompt=""):
        try:
            default_happy_prompt = """Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like "create an image").Remove any mention of digital artwork or artwork style. Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Always frame the scene, including details about the film grain, color grading, and any artifacts or characteristics specific."""

            default_simple_prompt = """Create a brief, straightforward caption for this description, suitable for a text-to-image AI system. Focus on the main elements, key characters, and overall scene without elaborate details. Provide a clear and concise description in one or two sentences."""

            poster_prompt = """Analyze the provided description and extract key information to create a movie poster style description. Format the output as follows:
Title: A catchy, intriguing title that captures the essence of the scene, place the title in "".
Main character: Give a description of the main character.
Background: Describe the background in detail.
Supporting characters: Describe the supporting characters
Branding type: Describe the branding type
Tagline: Include a tagline that captures the essence of the movie.
Visual style: Ensure that the visual style fits the branding type and tagline.
You are allowed to make up film and branding names, and do them like 80's, 90's or modern movie posters."""

            only_objects_prompt = """Create a highly detailed and visually rich description focusing solely on inanimate objects, without including any human or animal figures. Describe the objects' shapes, sizes, colors, textures, and materials in great detail. Pay attention to their arrangement, positioning, and how they interact with light and shadow. Include information about the setting or environment these objects are in, such as indoor/outdoor, time of day, weather conditions, and any atmospheric effects. Mention any unique features, patterns, or imperfections on the objects. Describe the overall composition, perspective, and any artistic techniques that might be employed to render these objects (e.g., photorealism, impressionistic style, etc.). Your description should paint a vivid picture that allows someone to imagine the scene without seeing it, focusing on the beauty, complexity, or significance of everyday objects."""

            no_figure_prompt = """Generate a comprehensive and visually evocative description of a scene or landscape without including any human or animal figures. Focus on the environment, natural elements, and man-made structures if present. Describe the topography, vegetation, weather conditions, and time of day in great detail. Pay attention to colors, textures, and how light interacts with different elements of the scene. If there are buildings or other structures, describe their architecture, condition, and how they fit into the landscape. Include sensory details beyond just visual elements - mention sounds, smells, and the overall atmosphere or mood of the scene. Describe any notable features like bodies of water, geological formations, or sky phenomena. Consider the perspective from which the scene is viewed and how this affects the composition. Your description should transport the reader to this location, allowing them to vividly imagine the scene without any living subjects present."""

            landscape_prompt = """Create an immersive and detailed description of a landscape, focusing on its natural beauty and geographical features. Begin with the overall topography - is it mountainous, coastal, forested, desert, or a combination? Describe the horizon and how land meets sky. Detail the vegetation, noting types of trees, flowers, or grass, and how they're distributed across the landscape. Include information about any water features - rivers, lakes, oceans - and how they interact with the land. Describe the sky, including cloud formations, color gradients, and any celestial bodies visible. Pay attention to the quality of light, time of day, and season, explaining how these factors affect the colors and shadows in the scene. Include details about weather conditions and how they impact the landscape. Mention any geological features like rock formations, cliffs, or unique land patterns. If there are any distant man-made elements, describe how they integrate with the natural setting. Your description should capture the grandeur and mood of the landscape, allowing the reader to feel as if they're standing within this awe-inspiring natural scene."""

            fantasy_prompt = """Craft an extraordinarily detailed and imaginative description of a fantasy scene, blending elements of magic, otherworldly creatures, and fantastical environments. Begin by setting the overall tone - is this a dark and foreboding realm, a whimsical fairytale setting, or an epic high-fantasy world? Describe the landscape, including any impossible or magical geographical features like floating islands, crystal forests, or rivers of starlight. Detail the flora and fauna, focusing on fantastical plants and creatures that don't exist in our world. Include descriptions of any structures or ruins, emphasizing their otherworldly architecture and magical properties. Describe the sky and any celestial bodies, considering how they might differ from our reality. Include details about the presence of magic - how it manifests visually, its effects on the environment, and any magical phenomena occurring in the scene. If there are characters present, describe their appearance, focusing on non-human features, magical auras, or fantastical clothing and accessories. Pay attention to colors, textures, and light sources, especially those that couldn't exist in the real world. Your description should transport the reader to a realm of pure imagination, where the laws of physics and nature as we know them don't apply."""

            prompt_types = {
                "happy": default_happy_prompt,
                "simple": default_simple_prompt,
                "poster": poster_prompt,
                "only_objects": only_objects_prompt,
                "no_figure": no_figure_prompt,
                "landscape": landscape_prompt,
                "fantasy": fantasy_prompt
            }

            print(f"Received prompt_type: '{prompt_type}'")  # Debug print

            if prompt_type and prompt_type.strip() in prompt_types:
                base_prompt = prompt_types[prompt_type.strip()]
                print(f"Using {prompt_type} prompt")
            elif custom_base_prompt.strip():
                base_prompt = custom_base_prompt
                print("Using custom base prompt")
            else:
                base_prompt = default_happy_prompt
                print(f"Warning: Unknown or empty prompt type '{prompt_type}'. Using default happy prompt.")

            if compress and not poster:
                compression_chars = {
                    "soft": 600 if happy_talk else 300,
                    "medium": 400 if happy_talk else 200,
                    "hard": 200 if happy_talk else 100
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

            system_message = "You are a helpful assistant. Try your best to give the best response possible to the user."
            user_message = f"{base_prompt}\nDescription: {input_text}"

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                messages=messages,
            )

            output = response.choices[0].message.content.strip()

            # Clean up the output
            if ": " in output:
                output = output.split(": ", 1)[1].strip()
            elif output.lower().startswith("here"):
                sentences = output.split(". ")
                if len(sentences) > 1:
                    output = ". ".join(sentences[1:]).strip()
            
            return output

        except Exception as e:
            print(f"An error occurred: {e}")
            return f"Error occurred while processing the request: {str(e)}"