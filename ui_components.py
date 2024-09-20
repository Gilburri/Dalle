import gradio as gr
from prompt_generator import PromptGenerator
from huggingface_inference_node import LLMInferenceNode
from caption_models import florence_caption, qwen_caption, joycaption
import random
from prompt_generator import ARTFORM, PHOTO_TYPE, FEMALE_BODY_TYPES, MALE_BODY_TYPES, FEMALE_DEFAULT_TAGS, MALE_DEFAULT_TAGS, ROLES, HAIRSTYLES, FEMALE_CLOTHING, MALE_CLOTHING, PLACE, LIGHTING, COMPOSITION, POSE, BACKGROUND, FEMALE_ADDITIONAL_DETAILS, MALE_ADDITIONAL_DETAILS, PHOTOGRAPHY_STYLES, DEVICE, PHOTOGRAPHER, ARTIST, DIGITAL_ARTFORM


title = """<h1 align="center">FLUX Prompt Generator</h1>
<p><center>
<a href="https://x.com/gokayfem" target="_blank">[X gokaygokay]</a>
<a href="https://github.com/gokayfem" target="_blank">[Github gokayfem]</a>
<a href="https://github.com/dagthomas/comfyui_dagthomas" target="_blank">[comfyui_dagthomas]</a>
<p align="center">Create long prompts from images or simple words. Enhance your short prompts with prompt enhancer.</p>
</center></p>
"""

# Add this global variable
selected_prompt_type = "happy"  # Default value

def create_interface():
    prompt_generator = PromptGenerator()
    llm_node = LLMInferenceNode()

    with gr.Blocks(theme='bethecloud/storj_theme') as demo:
        
        gr.HTML(title)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Basic Settings"):
                    custom = gr.Textbox(label="Custom Input Prompt (optional)")
                    subject = gr.Textbox(label="Subject (optional)")
                    gender = gr.Radio(["female", "male"], label="Gender", value="female")
                    
                    global_option = gr.Radio(
                        ["Disabled", "Random", "No Figure Rand"],
                        label="Set all options to:",
                        value="Disabled"
                    )
                
                with gr.Accordion("Artform and Photo Type", open=False):
                    artform = gr.Dropdown(["disabled", "random"] + ARTFORM, label="Artform", value="disabled")
                    photo_type = gr.Dropdown(["disabled", "random"] + PHOTO_TYPE, label="Photo Type", value="disabled")
            
                with gr.Accordion("Character Details", open=False):
                    body_types = gr.Dropdown(["disabled", "random"] + FEMALE_BODY_TYPES + MALE_BODY_TYPES, label="Body Types", value="disabled")
                    default_tags = gr.Dropdown(["disabled", "random"] + FEMALE_DEFAULT_TAGS + MALE_DEFAULT_TAGS, label="Default Tags", value="disabled")
                    roles = gr.Dropdown(["disabled", "random"] + ROLES, label="Roles", value="disabled")
                    hairstyles = gr.Dropdown(["disabled", "random"] + HAIRSTYLES, label="Hairstyles", value="disabled")
                    clothing = gr.Dropdown(["disabled", "random"] + FEMALE_CLOTHING + MALE_CLOTHING, label="Clothing", value="disabled")
            
                with gr.Accordion("Scene Details", open=False):
                    place = gr.Dropdown(["disabled", "random"] + PLACE, label="Place", value="disabled")
                    lighting = gr.Dropdown(["disabled", "random"] + LIGHTING, label="Lighting", value="disabled")
                    composition = gr.Dropdown(["disabled", "random"] + COMPOSITION, label="Composition", value="disabled")
                    pose = gr.Dropdown(["disabled", "random"] + POSE, label="Pose", value="disabled")
                    background = gr.Dropdown(["disabled", "random"] + BACKGROUND, label="Background", value="disabled")
            
                with gr.Accordion("Style and Artist", open=False):
                    additional_details = gr.Dropdown(["disabled", "random"] + FEMALE_ADDITIONAL_DETAILS + MALE_ADDITIONAL_DETAILS, label="Additional Details", value="disabled")
                    photography_styles = gr.Dropdown(["disabled", "random"] + PHOTOGRAPHY_STYLES, label="Photography Styles", value="disabled")
                    device = gr.Dropdown(["disabled", "random"] + DEVICE, label="Device", value="disabled")
                    photographer = gr.Dropdown(["disabled", "random"] + PHOTOGRAPHER, label="Photographer", value="disabled")
                    artist = gr.Dropdown(["disabled", "random"] + ARTIST, label="Artist", value="disabled")
                    digital_artform = gr.Dropdown(["disabled", "random"] + DIGITAL_ARTFORM, label="Digital Artform", value="disabled")

                # Add Next components
                with gr.Accordion("More Detailed Prompt Options", open=False):
                    next_components = {}
                    for category, fields in prompt_generator.next_data.items():
                        with gr.Accordion(f"{category.capitalize()} Options", open=False):
                            category_components = {}
                            for field, data in fields.items():
                                if isinstance(data, list):
                                    options = ["None", "Random", "Multiple Random"] + data
                                elif isinstance(data, dict):
                                    options = ["None", "Random", "Multiple Random"] + data.get("items", [])
                                else:
                                    options = ["None", "Random", "Multiple Random"]
                                category_components[field] = gr.Dropdown(options, label=field.capitalize(), value="None")
                            next_components[category] = category_components
                
                

            with gr.Column(scale=2):
                generate_button = gr.Button("Generate Prompt")

                with gr.Accordion("Image and Caption", open=False):
                    input_image = gr.Image(label="Input Image (optional)")
                    caption_output = gr.Textbox(label="Generated Caption", lines=3, show_copy_button=True)
                    caption_model = gr.Radio(["Florence-2", "Qwen2-VL", "JoyCaption"], label="Caption Model", value="Florence-2")
                    create_caption_button = gr.Button("Create Caption")
                    add_caption_button = gr.Button("Add Caption to Prompt")

                with gr.Accordion("Prompt Generation", open=True):
                    output = gr.Textbox(label="Generated Prompt / Input Text", lines=4, show_copy_button=True)
                    t5xxl_output = gr.Textbox(label="T5XXL Output", visible=True, show_copy_button=True)
                    clip_l_output = gr.Textbox(label="CLIP L Output", visible=True, show_copy_button=True)
                    clip_g_output = gr.Textbox(label="CLIP G Output", visible=True, show_copy_button=True)
            
            with gr.Column(scale=2):
                with gr.Accordion("""Prompt Generation with LLM 
                                (You need to use Generate Prompt first)""", open=False):
                    happy_talk = gr.Checkbox(label="Happy Talk", value=True)
                    compress = gr.Checkbox(label="Compress", value=True)
                    compression_level = gr.Dropdown(
                        choices=["soft", "medium", "hard"],
                        label="Compression Level",
                        value="hard"
                    )

                    custom_base_prompt = gr.Textbox(label="Custom Base Prompt", lines=5)

                prompt_type = gr.Dropdown(
                    choices=["happy", "simple", "poster", "only_objects", "no_figure", "landscape", "fantasy"],
                    label="Prompt Type",
                    value="happy",
                    interactive=True
                )
                    
                # Add the missing update_prompt_type function
                def update_prompt_type(value):
                    global selected_prompt_type
                    selected_prompt_type = value
                    print(f"Updated prompt type: {selected_prompt_type}")
                    return value
                
                # Connect the update_prompt_type function to the prompt_type dropdown
                prompt_type.change(update_prompt_type, inputs=[prompt_type], outputs=[prompt_type])                   
                    
                    # Add new components for LLM provider selection
                llm_provider = gr.Dropdown(
                    choices=["Hugging Face", "OpenAI", "Anthropic", "Groq"],
                    label="LLM Provider",
                    value="Hugging Face"
                )
                api_key = gr.Textbox(label="API Key", type="password", visible=False)
                model = gr.Dropdown(label="Model", choices=["meta-llama/Meta-Llama-3.1-70B-Instruct"], value="meta-llama/Meta-Llama-3.1-70B-Instruct")

                generate_text_button = gr.Button("Generate Prompt with LLM")
                text_output = gr.Textbox(label="Generated Text", lines=10, show_copy_button=True)

        def create_caption(image, model):
            if image is not None:
                if model == "Florence-2":
                    return florence_caption(image)
                elif model == "Qwen2-VL":
                    return qwen_caption(image)
                elif model == "JoyCaption":
                    return joycaption(image)
            return ""

        create_caption_button.click(
            create_caption,
            inputs=[input_image, caption_model],
            outputs=[caption_output]
        )

        

        def generate_prompt_with_dynamic_seed(*args, **kwargs):
            dynamic_seed = random.randint(0, 1000000)
            
            # Extract the main arguments
            main_args = args[:22]  # Assuming there are 22 main arguments before the next_params
            
            # Extract next_params
            next_params = {}
            next_args = args[22:]  # All arguments after the main ones are for next_params
            next_arg_index = 0
            for category, fields in prompt_generator.next_data.items():
                category_params = {}
                for field in fields:
                    value = next_args[next_arg_index]
                    # Include all values, even "None", "Random", and "Multiple Random"
                    category_params[field] = value
                    next_arg_index += 1
                if category_params:
                    next_params[category] = category_params
            # Call generate_prompt with the correct arguments
            result = prompt_generator.generate_prompt(dynamic_seed, *main_args, next_params=next_params)
            
            return [dynamic_seed] + list(result)

        generate_button.click(
            generate_prompt_with_dynamic_seed,
            inputs=[custom, subject, gender, artform, photo_type, body_types, default_tags, roles, hairstyles,
                    additional_details, photography_styles, device, photographer, artist, digital_artform,
                    place, lighting, clothing, composition, pose, background, input_image] + 
                    [component for category in next_components.values() for component in category.values()],
            outputs=[gr.Number(label="Used Seed", visible=False), output, gr.Number(visible=False), t5xxl_output, clip_l_output, clip_g_output]
        )

        add_caption_button.click(
            prompt_generator.add_caption_to_prompt,
            inputs=[output, caption_output],
            outputs=[output]
        )

        def update_model_choices(provider):
            provider_models = {
                "Hugging Face": ["meta-llama/Meta-Llama-3.1-70B-Instruct"],
                "Groq": ["llama-3.1-70b-versatile"],
                "OpenAI": ["gpt-4o", "gpt-4o-mini"],
                "Anthropic": ["claude-3-5-sonnet-20240620"],
            }
            models = provider_models[provider]
            return gr.Dropdown(choices=models, value=models[0])

        def update_api_key_visibility(provider):
            return gr.update(visible=(provider in ["OpenAI", "Anthropic"]))

        llm_provider.change(update_model_choices, inputs=[llm_provider], outputs=[model])
        llm_provider.change(update_api_key_visibility, inputs=[llm_provider], outputs=[api_key])

        def generate_text_with_llm(output, happy_talk, compress, compression_level, custom_base_prompt, provider, api_key, model):
            global selected_prompt_type
            result = llm_node.generate(output, happy_talk, compress, compression_level, False, selected_prompt_type, custom_base_prompt, provider, api_key, model)
            selected_prompt_type = "happy"  # Reset to "happy" after generation
            return result, "happy"  # Return the result and the new prompt type value

        generate_text_button.click(
            generate_text_with_llm,
            inputs=[output, happy_talk, compress, compression_level, custom_base_prompt, llm_provider, api_key, model],
            outputs=[text_output, prompt_type],
            api_name="generate_text"
        )

        # Add this line to disable caching for the generate_text_with_llm function
        generate_text_with_llm.cache_examples = False

        def update_all_options(choice):
            updates = {}
            if choice == "Disabled":
                for dropdown in [
                    artform, photo_type, body_types, default_tags, roles, hairstyles, clothing,
                    place, lighting, composition, pose, background, additional_details,
                    photography_styles, device, photographer, artist, digital_artform
                ]:
                    updates[dropdown] = gr.update(value="disabled")
            elif choice == "Random":
                for dropdown in [
                    artform, photo_type, body_types, default_tags, roles, hairstyles, clothing,
                    place, lighting, composition, pose, background, additional_details,
                    photography_styles, device, photographer, artist, digital_artform
                ]:
                    updates[dropdown] = gr.update(value="random")
            else:  # No Figure Random
                for dropdown in [photo_type, body_types, default_tags, roles, hairstyles, clothing, pose, additional_details]:
                    updates[dropdown] = gr.update(value="disabled")
                for dropdown in [artform, place, lighting, composition, background, photography_styles, device, photographer, artist, digital_artform]:
                    updates[dropdown] = gr.update(value="random")
            return updates
        
        global_option.change(
            update_all_options,
            inputs=[global_option],
            outputs=[
                artform, photo_type, body_types, default_tags, roles, hairstyles, clothing,
                place, lighting, composition, pose, background, additional_details,
                photography_styles, device, photographer, artist, digital_artform
            ]
        )

    return demo