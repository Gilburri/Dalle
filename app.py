import gradio as gr
import random
import json
import os
import re
from openai import OpenAI
from datetime import datetime

# Load JSON files
def load_json_file(file_name):
    file_path = os.path.join("data", file_name)
    with open(file_path, "r") as file:
        return json.load(file)

ARTFORM = load_json_file("artform.json")
PHOTO_TYPE = load_json_file("photo_type.json")
BODY_TYPES = load_json_file("body_types.json")
DEFAULT_TAGS = load_json_file("default_tags.json")
ROLES = load_json_file("roles.json")
HAIRSTYLES = load_json_file("hairstyles.json")
ADDITIONAL_DETAILS = load_json_file("additional_details.json")
PHOTOGRAPHY_STYLES = load_json_file("photography_styles.json")
DEVICE = load_json_file("device.json")
PHOTOGRAPHER = load_json_file("photographer.json")
ARTIST = load_json_file("artist.json")
DIGITAL_ARTFORM = load_json_file("digital_artform.json")
PLACE = load_json_file("place.json")
LIGHTING = load_json_file("lighting.json")
CLOTHING = load_json_file("clothing.json")
COMPOSITION = load_json_file("composition.json")
POSE = load_json_file("pose.json")
BACKGROUND = load_json_file("background.json")

class PromptGenerator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def split_and_choose(self, input_str):
        choices = [choice.strip() for choice in input_str.split(",")]
        return self.rng.choices(choices, k=1)[0]

    def get_choice(self, input_str, default_choices):
        if input_str.lower() == "disabled":
            return ""
        elif "," in input_str:
            return self.split_and_choose(input_str)
        elif input_str.lower() == "random":
            return self.rng.choices(default_choices, k=1)[0]
        else:
            return input_str

    def clean_consecutive_commas(self, input_string):
        cleaned_string = re.sub(r',\s*,', ',', input_string)
        return cleaned_string

    def process_string(self, replaced, seed):
        replaced = re.sub(r'\s*,\s*', ',', replaced)
        replaced = re.sub(r',+', ',', replaced)
        original = replaced
        
        first_break_clipl_index = replaced.find("BREAK_CLIPL")
        second_break_clipl_index = replaced.find("BREAK_CLIPL", first_break_clipl_index + len("BREAK_CLIPL"))
        
        if first_break_clipl_index != -1 and second_break_clipl_index != -1:
            clip_content_l = replaced[first_break_clipl_index + len("BREAK_CLIPL"):second_break_clipl_index]
            replaced = replaced[:first_break_clipl_index].strip(", ") + replaced[second_break_clipl_index + len("BREAK_CLIPL"):].strip(", ")
            clip_l = clip_content_l
        else:
            clip_l = ""
        
        first_break_clipg_index = replaced.find("BREAK_CLIPG")
        second_break_clipg_index = replaced.find("BREAK_CLIPG", first_break_clipg_index + len("BREAK_CLIPG"))
        
        if first_break_clipg_index != -1 and second_break_clipg_index != -1:
            clip_content_g = replaced[first_break_clipg_index + len("BREAK_CLIPG"):second_break_clipg_index]
            replaced = replaced[:first_break_clipg_index].strip(", ") + replaced[second_break_clipg_index + len("BREAK_CLIPG"):].strip(", ")
            clip_g = clip_content_g
        else:
            clip_g = ""
        
        t5xxl = replaced
        
        original = original.replace("BREAK_CLIPL", "").replace("BREAK_CLIPG", "")
        original = re.sub(r'\s*,\s*', ',', original)
        original = re.sub(r',+', ',', original)
        clip_l = re.sub(r'\s*,\s*', ',', clip_l)
        clip_l = re.sub(r',+', ',', clip_l)
        clip_g = re.sub(r'\s*,\s*', ',', clip_g)
        clip_g = re.sub(r',+', ',', clip_g)
        if clip_l.startswith(","):
            clip_l = clip_l[1:]
        if clip_g.startswith(","):
            clip_g = clip_g[1:]
        if original.startswith(","):
            original = original[1:]
        if t5xxl.startswith(","):
            t5xxl = t5xxl[1:]

        return original, seed, t5xxl, clip_l, clip_g

    def generate_prompt(self, **kwargs):
        seed = kwargs.get("seed", 0)
        if seed is not None:
            self.rng = random.Random(seed)
        components = []
        custom = kwargs.get("custom", "")
        if custom:
            components.append(custom)
        is_photographer = kwargs.get("artform", "").lower() == "photography" or (
            kwargs.get("artform", "").lower() == "random"
            and self.rng.choice([True, False])
        )

        subject = kwargs.get("subject", "")

        if is_photographer:
            selected_photo_style = self.get_choice(kwargs.get("photography_styles", ""), PHOTOGRAPHY_STYLES)
            if not selected_photo_style:
                selected_photo_style = "photography"
            components.append(selected_photo_style)
            if kwargs.get("photography_style", "") != "disabled" and kwargs.get("default_tags", "") != "disabled" or subject != "":
                components.append(" of")
        
        default_tags = kwargs.get("default_tags", "random")
        body_type = kwargs.get("body_types", "")
        if not subject:
            if default_tags == "random":
                if body_type != "disabled" and body_type != "random":
                    selected_subject = self.get_choice(kwargs.get("default_tags", ""), DEFAULT_TAGS).replace("a ", "").replace("an ", "")
                    components.append("a ")
                    components.append(body_type)
                    components.append(selected_subject)
                elif body_type == "disabled":
                    selected_subject = self.get_choice(kwargs.get("default_tags", ""), DEFAULT_TAGS)
                    components.append(selected_subject)
                else:
                    body_type = self.get_choice(body_type, BODY_TYPES)
                    components.append("a ")
                    components.append(body_type)
                    selected_subject = self.get_choice(kwargs.get("default_tags", ""), DEFAULT_TAGS).replace("a ", "").replace("an ", "")
                    components.append(selected_subject)
            elif default_tags == "disabled":
                pass
            else:
                components.append(default_tags)
        else:
            if body_type != "disabled" and body_type != "random":
                components.append("a ")
                components.append(body_type)
            elif body_type == "disabled":
                pass
            else:
                body_type = self.get_choice(body_type, BODY_TYPES)
                components.append("a ")
                components.append(body_type)
            components.append(subject)

        params = [
            ("roles", ROLES),
            ("hairstyles", HAIRSTYLES),
            ("additional_details", ADDITIONAL_DETAILS),
        ]
        for param in params:
            components.append(self.get_choice(kwargs.get(param[0], ""), param[1]))
        for i in reversed(range(len(components))):
            if components[i] in PLACE:
                components[i] += ","
                break
        if kwargs.get("clothing", "") != "disabled" and kwargs.get("clothing", "") != "random":
            components.append(", dressed in ")
            clothing = kwargs.get("clothing", "")
            components.append(clothing)
        elif kwargs.get("clothing", "") == "random":
            components.append(", dressed in ")
            clothing = self.get_choice(kwargs.get("clothing", ""), CLOTHING)
            components.append(clothing)

        if kwargs.get("composition", "") != "disabled" and kwargs.get("composition", "") != "random":
            components.append(",")
            composition = kwargs.get("composition", "")
            components.append(composition)
        elif kwargs.get("composition", "") == "random": 
            components.append(",")
            composition = self.get_choice(kwargs.get("composition", ""), COMPOSITION)
            components.append(composition)
        
        if kwargs.get("pose", "") != "disabled" and kwargs.get("pose", "") != "random":
            components.append(",")
            pose = kwargs.get("pose", "")
            components.append(pose)
        elif kwargs.get("pose", "") == "random":
            components.append(",")
            pose = self.get_choice(kwargs.get("pose", ""), POSE)
            components.append(pose)
        components.append("BREAK_CLIPG")
        if kwargs.get("background", "") != "disabled" and kwargs.get("background", "") != "random":
            components.append(",")
            background = kwargs.get("background", "")
            components.append(background)
        elif kwargs.get("background", "") == "random": 
            components.append(",")
            background = self.get_choice(kwargs.get("background", ""), BACKGROUND)
            components.append(background)

        if kwargs.get("place", "") != "disabled" and kwargs.get("place", "") != "random":
            components.append(",")
            place = kwargs.get("place", "")
            components.append(place)
        elif kwargs.get("place", "") == "random": 
            components.append(",")
            place = self.get_choice(kwargs.get("place", ""), PLACE)
            components.append(place + ",")

        lighting = kwargs.get("lighting", "").lower()
        if lighting == "random":
            selected_lighting = ", ".join(self.rng.sample(LIGHTING, self.rng.randint(2, 5)))
            components.append(",")
            components.append(selected_lighting)
        elif lighting == "disabled":
            pass
        else:
            components.append(", ")
            components.append(lighting)
        components.append("BREAK_CLIPG")
        components.append("BREAK_CLIPL")
        if is_photographer:
            if kwargs.get("photo_type", "") != "disabled":
                photo_type_choice = self.get_choice(kwargs.get("photo_type", ""), PHOTO_TYPE)
                if photo_type_choice and photo_type_choice != "random" and photo_type_choice != "disabled":
                    random_value = round(self.rng.uniform(1.1, 1.5), 1)
                    components.append(f", ({photo_type_choice}:{random_value}), ")

            params = [
                ("device", DEVICE),
                ("photographer", PHOTOGRAPHER),
            ]
            components.extend([self.get_choice(kwargs.get(param[0], ""), param[1]) for param in params])
            if kwargs.get("device", "") != "disabled":
                components[-2] = f", shot on {components[-2]}"
            if kwargs.get("photographer", "") != "disabled":
                components[-1] = f", photo by {components[-1]}"
        else:
            digital_artform_choice = self.get_choice(kwargs.get("digital_artform", ""), DIGITAL_ARTFORM)
            if digital_artform_choice:
                components.append(f"{digital_artform_choice}")
            if kwargs.get("artist", "") != "disabled":
                components.append(f"by {self.get_choice(kwargs.get('artist', ''), ARTIST)}")
        components.append("BREAK_CLIPL")

        prompt = " ".join(components)
        prompt = re.sub(" +", " ", prompt)
        replaced = prompt.replace("of as", "of")
        replaced = self.clean_consecutive_commas(replaced)

        return self.process_string(replaced, seed)

class GPT4MiniNode:
    def __init__(self):
        self.client = None
        self.prompts_dir = "./prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    def save_prompt(self, prompt):
        filename_text = "mini_" + prompt.split(',')[0].strip()
        filename_text = re.sub(r'[^\w\-_\. ]', '_', filename_text)
        filename_text = filename_text[:30]  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)
        
        with open(filename, "w") as file:
            file.write(prompt)
        
        print(f"Prompt saved to {filename}")

    def generate(self, api_key, input_text, happy_talk, compress, compression_level, poster, custom_base_prompt=""):
        try:
            self.client = OpenAI(api_key=api_key)

            default_happy_prompt = """Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like "create an image").Remove any mention of digital artwork or artwork style. Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style (e.g. Wes Anderson, Christopher Nolan, Quentin Tarantino) and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Always frame the scene as a screen grab from a 35mm film still, including details about the film grain, color grading, and any artifacts or characteristics specific to 35mm film photography."""

            default_simple_prompt = """Create a brief, straightforward caption for this description, suitable for a text-to-image AI system. Focus on the main elements, key characters, and overall scene without elaborate details. Provide a clear and concise description in one or two sentences."""

            poster_prompt = """Analyze the provided description and extract key information to create a movie poster style description. Format the output as follows:

Title: A catchy, intriguing title that captures the essence of the scene, place the title in "".
Main character: Give a description of the main character.
Background: Describe the background in detail.
Supporting characters: Describe the supporting characters
Branding type: Describe the branding type
Tagline: Include a tagline that captures the essence of the movie.
Visual style: Ensure that the visual style fits the branding type and tagline.
You are allowed to make up film and branding names, and do them like 80's, 90's or modern movie posters.

Never add ** in front of title, main character, etc.
Here is an example of a prompt, make the output like a movie poster: 
Title: Display the title "Verdant Spirits" in elegant and ethereal text, placed centrally at the top of the poster.

Main Character: Depict a serene and enchantingly beautiful woman with an aura of nature, her face partly adorned and encased with vibrant green foliage and delicate floral arrangements. She exudes an ethereal and mystical presence.

Background: The background should feature a dreamlike enchanted forest with lush greenery, vibrant flowers, and an ethereal glow emanating from the foliage. The scene should feel magical and otherworldly, suggesting a hidden world within nature.

Supporting Characters: Add an enigmatic skeletal figure entwined with glowing, bioluminescent leaves and plants, subtly blending with the dark, verdant background. This figure should evoke a sense of ancient wisdom and mysterious energy.

Studio Ghibli Branding: Incorporate the Studio Ghibli logo at the bottom center of the poster to establish it as an official Ghibli film.

Tagline: Include a tagline that reads: "Where Nature's Secrets Come to Life" prominently on the poster.

Visual Style: Ensure the overall visual style is consistent with Studio Ghibli s signature look   rich, detailed backgrounds, and characters imbued with a touch of whimsy and mystery. The colors should be lush and inviting, with an emphasis on the enchanting and mystical aspects of nature."""

            if poster:
                base_prompt = poster_prompt
            elif custom_base_prompt.strip():
                base_prompt = custom_base_prompt
            else:
                base_prompt = default_happy_prompt if happy_talk else default_simple_prompt

            if compress and not poster:
                compression_chars = {
                    "soft": 600 if happy_talk else 300,
                    "medium": 400 if happy_talk else 200,
                    "hard": 200 if happy_talk else 100
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"{base_prompt}\nDescription: {input_text}"}],
            )
            
            result = response.choices[0].message.content
            self.save_prompt(result)
            return result
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"Error occurred while processing the request: {str(e)}"

def create_interface():
    prompt_generator = PromptGenerator()
    gpt4_mini_node = GPT4MiniNode()

    with gr.Blocks() as demo:
        gr.Markdown("# AI Prompt Generator and Text Generator")

        with gr.Tab("Prompt Generator"):
            with gr.Row():
                with gr.Column():
                    seed = gr.Number(label="Seed", value=0)
                    custom = gr.Textbox(label="Custom")
                    subject = gr.Textbox(label="Subject")
                    artform = gr.Dropdown(["disabled", "random"] + ARTFORM, label="Artform", value="photography")
                    photo_type = gr.Dropdown(["disabled", "random"] + PHOTO_TYPE, label="Photo Type", value="random")
                    body_types = gr.Dropdown(["disabled", "random"] + BODY_TYPES, label="Body Types", value="random")
                    default_tags = gr.Dropdown(["disabled", "random"] + DEFAULT_TAGS, label="Default Tags", value="random")
                with gr.Column():
                    roles = gr.Dropdown(["disabled", "random"] + ROLES, label="Roles", value="random")
                    hairstyles = gr.Dropdown(["disabled", "random"] + HAIRSTYLES, label="Hairstyles", value="random")
                    additional_details = gr.Dropdown(["disabled", "random"] + ADDITIONAL_DETAILS, label="Additional Details", value="random")
                    photography_styles = gr.Dropdown(["disabled", "random"] + PHOTOGRAPHY_STYLES, label="Photography Styles", value="random")
                    device = gr.Dropdown(["disabled", "random"] + DEVICE, label="Device", value="random")
                    photographer = gr.Dropdown(["disabled", "random"] + PHOTOGRAPHER, label="Photographer", value="random")
                with gr.Column():
                    artist = gr.Dropdown(["disabled", "random"] + ARTIST, label="Artist", value="random")
                    digital_artform = gr.Dropdown(["disabled", "random"] + DIGITAL_ARTFORM, label="Digital Artform", value="random")
                    place = gr.Dropdown(["disabled", "random"] + PLACE, label="Place", value="random")
                    lighting = gr.Dropdown(["disabled", "random"] + LIGHTING, label="Lighting", value="random")
                    clothing = gr.Dropdown(["disabled", "random"] + CLOTHING, label="Clothing", value="random")
                    composition = gr.Dropdown(["disabled", "random"] + COMPOSITION, label="Composition", value="random")
                    pose = gr.Dropdown(["disabled", "random"] + POSE, label="Pose", value="random")
                    background = gr.Dropdown(["disabled", "random"] + BACKGROUND, label="Background", value="random")

            generate_button = gr.Button("Generate Prompt")
            output = gr.Textbox(label="Generated Prompt")
            t5xxl_output = gr.Textbox(label="T5XXL Output")
            clip_l_output = gr.Textbox(label="CLIP L Output")
            clip_g_output = gr.Textbox(label="CLIP G Output")

            generate_button.click(
                prompt_generator.generate_prompt,
                inputs=[seed, custom, subject, artform, photo_type, body_types, default_tags, roles, hairstyles,
                        additional_details, photography_styles, device, photographer, artist, digital_artform,
                        place, lighting, clothing, composition, pose, background],
                outputs=[output, gr.Number(visible=False), t5xxl_output, clip_l_output, clip_g_output]
            )

        with gr.Tab("GPT4 Mini Text Generator"):
            api_key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="Enter your OpenAI API key")
            input_text = gr.Textbox(label="Input Text", lines=5)
            happy_talk = gr.Checkbox(label="Happy Talk", value=True)
            compress = gr.Checkbox(label="Compress", value=False)
            compression_level = gr.Radio(["soft", "medium", "hard"], label="Compression Level", value="medium")
            poster = gr.Checkbox(label="Poster", value=False)
            custom_base_prompt = gr.Textbox(label="Custom Base Prompt", lines=5)

            generate_text_button = gr.Button("Generate Text")
            text_output = gr.Textbox(label="Generated Text", lines=10)

            generate_text_button.click(
                gpt4_mini_node.generate,
                inputs=[api_key, input_text, happy_talk, compress, compression_level, poster, custom_base_prompt],
                outputs=text_output
            )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()