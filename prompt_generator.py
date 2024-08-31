import os
import json
import random
import re

# Load JSON files
def load_json_file(file_name):
    file_path = os.path.join("data", file_name)
    with open(file_path, "r") as file:
        return json.load(file)

# Load gender-specific JSON files
FEMALE_DEFAULT_TAGS = load_json_file("female_default_tags.json")
MALE_DEFAULT_TAGS = load_json_file("male_default_tags.json")
FEMALE_BODY_TYPES = load_json_file("female_body_types.json")
MALE_BODY_TYPES = load_json_file("male_body_types.json")
FEMALE_CLOTHING = load_json_file("female_clothing.json")
MALE_CLOTHING = load_json_file("male_clothing.json")
FEMALE_ADDITIONAL_DETAILS = load_json_file("female_additional_details.json")
MALE_ADDITIONAL_DETAILS = load_json_file("male_additional_details.json")

# Load non-gender-specific JSON files
ARTFORM = load_json_file("artform.json")
PHOTO_TYPE = load_json_file("photo_type.json")
ROLES = load_json_file("roles.json")
HAIRSTYLES = load_json_file("hairstyles.json")
PLACE = load_json_file("place.json")
LIGHTING = load_json_file("lighting.json")
COMPOSITION = load_json_file("composition.json")
POSE = load_json_file("pose.json")
BACKGROUND = load_json_file("background.json")
PHOTOGRAPHY_STYLES = load_json_file("photography_styles.json")
DEVICE = load_json_file("device.json")
PHOTOGRAPHER = load_json_file("photographer.json")
ARTIST = load_json_file("artist.json")
DIGITAL_ARTFORM = load_json_file("digital_artform.json")

class PromptGenerator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.next_data = self.load_next_data()

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
        cleaned_string = re.sub(r',\s*,', ', ', input_string)
        return cleaned_string

    def process_string(self, replaced, seed):
        replaced = re.sub(r'\s*,\s*', ', ', replaced)
        replaced = re.sub(r',+', ', ', replaced)
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
        original = re.sub(r'\s*,\s*', ', ', original)
        original = re.sub(r',+', ', ', original)
        clip_l = re.sub(r'\s*,\s*', ', ', clip_l)
        clip_l = re.sub(r',+', ', ', clip_l)
        clip_g = re.sub(r'\s*,\s*', ', ', clip_g)
        clip_g = re.sub(r',+', ', ', clip_g)
        if clip_l.startswith(", "):
            clip_l = clip_l[2:]
        if clip_g.startswith(", "):
            clip_g = clip_g[2:]
        if original.startswith(", "):
            original = original[2:]
        if t5xxl.startswith(", "):
            t5xxl = t5xxl[2:]

        # Add spaces after commas
        replaced = re.sub(r',(?!\s)', ', ', replaced)
        original = re.sub(r',(?!\s)', ', ', original)
        clip_l = re.sub(r',(?!\s)', ', ', clip_l)
        clip_g = re.sub(r',(?!\s)', ', ', clip_g)
        t5xxl = re.sub(r',(?!\s)', ', ', t5xxl)

        return original, seed, t5xxl, clip_l, clip_g

    def load_next_data(self):
        next_data = {}
        next_path = os.path.join("data", "next")
        for category in os.listdir(next_path):
            category_path = os.path.join(next_path, category)
            if os.path.isdir(category_path):
                next_data[category] = {}
                for file in os.listdir(category_path):
                    if file.endswith(".json"):
                        file_path = os.path.join(category_path, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            json_data = json.load(f)
                            next_data[category][file[:-5]] = json_data
        return next_data

    def process_next_data(self, prompt, separator, category, field, value, attributes=False):
        if category in self.next_data and field in self.next_data[category]:
            field_data = self.next_data[category][field]
            items = field_data.get("items", [])
            preprompt = str(field_data.get("preprompt", "")).strip()
            field_separator = f" {str(field_data.get('separator', ', ')).strip()} "
            endprompt = str(field_data.get("endprompt", "")).strip()

            if value == "None":
                return prompt
            elif value == "Random":
                selected_items = [self.rng.choice(items)]
            elif value == "Multiple Random":
                count = self.rng.randint(1, 3)
                selected_items = self.rng.sample(items, min(count, len(items)))
            else:
                selected_items = [value]

            formatted_items = []
            for item in selected_items:
                item_str = str(item)
                if attributes and "attributes" in field_data and item_str in field_data["attributes"]:
                    item_attributes = field_data["attributes"].get(item_str, [])
                    if item_attributes:
                        selected_attributes = self.rng.sample(item_attributes, min(3, len(item_attributes)))
                        formatted_items.append(f"{item_str} ({', '.join(map(str, selected_attributes))})")
                    else:
                        formatted_items.append(item_str)
                else:
                    formatted_items.append(item_str)

            formatted_values = field_separator.join(formatted_items)
            formatted_addition = []
            if preprompt:
                formatted_addition.append(preprompt)
            formatted_addition.append(formatted_values)
            if endprompt:
                formatted_addition.append(endprompt)

            formatted_output = " ".join(formatted_addition).strip()
            prompt += f" {formatted_output}"

        return prompt

    def generate_prompt(self, seed, custom, subject, gender, artform, photo_type, body_types, default_tags, roles, hairstyles,
                        additional_details, photography_styles, device, photographer, artist, digital_artform,
                        place, lighting, clothing, composition, pose, background, input_image, **next_params):
        kwargs = locals()
        del kwargs['self']
        
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
        gender = kwargs.get("gender", "female")

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
                    selected_subject = self.get_choice(kwargs.get("default_tags", ""), FEMALE_DEFAULT_TAGS if gender == "female" else MALE_DEFAULT_TAGS).replace("a ", "").replace("an ", "")
                    components.append("a ")
                    components.append(body_type)
                    components.append(selected_subject)
                elif body_type == "disabled":
                    selected_subject = self.get_choice(kwargs.get("default_tags", ""), FEMALE_DEFAULT_TAGS if gender == "female" else MALE_DEFAULT_TAGS)
                    components.append(selected_subject)
                else:
                    body_type = self.get_choice(body_type, FEMALE_BODY_TYPES if gender == "female" else MALE_BODY_TYPES)
                    components.append("a ")
                    components.append(body_type)
                    selected_subject = self.get_choice(kwargs.get("default_tags", ""), FEMALE_DEFAULT_TAGS if gender == "female" else MALE_DEFAULT_TAGS).replace("a ", "").replace("an ", "")
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
                body_type = self.get_choice(body_type, FEMALE_BODY_TYPES if gender == "female" else MALE_BODY_TYPES)
                components.append("a ")
                components.append(body_type)
            components.append(subject)

        params = [
            ("roles", ROLES),
            ("hairstyles", HAIRSTYLES),
            ("additional_details", FEMALE_ADDITIONAL_DETAILS if gender == "female" else MALE_ADDITIONAL_DETAILS),
        ]
        for param in params:
            components.append(self.get_choice(kwargs.get(param[0], ""), param[1]))
        for i in reversed(range(len(components))):
            if components[i] in PLACE:
                components[i] += ", "
                break
        if kwargs.get("clothing", "") != "disabled" and kwargs.get("clothing", "") != "random":
            components.append(", dressed in ")
            clothing = kwargs.get("clothing", "")
            components.append(clothing)
        elif kwargs.get("clothing", "") == "random":
            components.append(", dressed in ")
            clothing = self.get_choice(kwargs.get("clothing", ""), FEMALE_CLOTHING if gender == "female" else MALE_CLOTHING)
            components.append(clothing)

        if kwargs.get("composition", "") != "disabled" and kwargs.get("composition", "") != "random":
            components.append(", ")
            composition = kwargs.get("composition", "")
            components.append(composition)
        elif kwargs.get("composition", "") == "random": 
            components.append(", ")
            composition = self.get_choice(kwargs.get("composition", ""), COMPOSITION)
            components.append(composition)
        
        if kwargs.get("pose", "") != "disabled" and kwargs.get("pose", "") != "random":
            components.append(", ")
            pose = kwargs.get("pose", "")
            components.append(pose)
        elif kwargs.get("pose", "") == "random":
            components.append(", ")
            pose = self.get_choice(kwargs.get("pose", ""), POSE)
            components.append(pose)
        components.append("BREAK_CLIPG")
        if kwargs.get("background", "") != "disabled" and kwargs.get("background", "") != "random":
            components.append(", ")
            background = kwargs.get("background", "")
            components.append(background)
        elif kwargs.get("background", "") == "random": 
            components.append(", ")
            background = self.get_choice(kwargs.get("background", ""), BACKGROUND)
            components.append(background)

        if kwargs.get("place", "") != "disabled" and kwargs.get("place", "") != "random":
            components.append(", ")
            place = kwargs.get("place", "")
            components.append(place)
        elif kwargs.get("place", "") == "random": 
            components.append(", ")
            place = self.get_choice(kwargs.get("place", ""), PLACE)
            components.append(place + ", ")

        lighting = kwargs.get("lighting", "").lower()
        if lighting == "random":
            selected_lighting = ", ".join(self.rng.sample(LIGHTING, self.rng.randint(2, 5)))
            components.append(", ")
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

        # Process next_params
        for category, fields in next_params.items():
            for field, value in fields.items():
                prompt = self.process_next_data(prompt, ", ", category, field, value)

        return self.process_string(replaced, seed)
    
    def add_caption_to_prompt(self, prompt, caption):
        if caption:
            return f"{prompt}, {caption}"
        return prompt