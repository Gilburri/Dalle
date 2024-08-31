from ui_components import create_interface

import subprocess

subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()