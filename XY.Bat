@echo off
REM General Configs
set "server_ip=localhost"
set "run_job_url=http://%server_ip%:5300/xyplot"

REM X/Y candidates
set "x_values=8k,best quality,digital art,highest detail,highly detailed,intricate detail,masterpiece,photo realistic,trending on artstation"
set "y_values=1,2,3,4,5"

REM Make sure X/Y values are non-empty
if "%x_values%" == "" (
    echo Error: x_values not provided or empty.
    pause
    exit /b 1
)

if "%y_values%" == "" (
    echo Error: y_values not provided or empty.
    pause
    exit /b 1
)

REM Probably Fixed Parameters
set "text_prompts=a cat ***"
set "negative_prompts=mutated, deformed, amateur drawing, lowres, worst quality, low quality, jpeg artifacts, text, error, signature, watermark, username, blurry, censorship"
set "init_image="
set "mask_image="

REM General Configs, probably just set once:
set "models_dir=E:\Model_Weights"
set "image_save_path=E:\Artroom\ArtroomOuputs"
set "show_intermediates=False"
set "highres_fix=True"
set "long_save_path=False"
set "save_grid=False"
set "invert=False"
set "use_removed_background=False"
set "use_preprocessed_controlnet=False"

REM Define the available samplers and their corresponding values here
set "sampler=euler_a"  REM "ddim", "dpmpp_2m", "dpmpp_2s_ancestral", "euler", "euler_a", "dpm_2", "dpm_a", "lms", "heun", "plms"
set "remove_background=none"  REM "face", "u2net", "u2net_human_seg"
set "controlnet=none"  REM "none", "canny", "pose", "depth", "hed", "normal", "scribble"
set "vae=none"  REM "none"
set "loras="  REM Leave this empty if no loras are needed, else add the JSON structure with lora values

REM Fixed Parameters
set "clip_skip=1"
set "cfg_scale=7.5"
set "steps=30"
set "ckpt=sd_xl_base_1.0.safetensors"
set "width=1024"
set "height=1024"
set "strength=0.65"
set "n_iter=1"  REM Probably makes the most sense to keep this at 1
set "seed=5"   REM Probably makes sense to keep this fixed

REM Split the X and Y values into an array
setlocal enabledelayedexpansion
for %%x in (%x_values%) do (
    for %%y in (%y_values%) do (
        REM Generate the payload for this iteration
        set "payload={\"clip_skip\":%clip_skip%,\"cfg_scale\":%cfg_scale%,\"steps\":%steps%,\"sampler\":\"%sampler%\",\"remove_background\":\"%remove_background%\",\"controlnet\":\"%controlnet%\",\"vae\":\"%vae%\",\"loras\":%loras%,\"ckpt\":\"%ckpt%\",\"width\":%width%,\"height\":%height%,\"strength\":%strength%,\"text_prompts\":\"%text_prompts%\",\"negative_prompts\":\"%negative_prompts%\",\"seed\":%seed%,\"init_image\":\"%init_image%\",\"mask_image\":\"%mask_image%\",\"invert\":%invert%,\"use_removed_background\":%use_removed_background%,\"use_preprocessed_controlnet\":%use_preprocessed_controlnet%,\"image_save_path\":\"%image_save_path%\",\"n_iter\":%n_iter%,\"save_grid\":%save_grid%,\"long_save_path\":%long_save_path%,\"highres_fix\":%highres_fix%,\"show_intermediates\":%show_intermediates%,\"models_dir\":\"%models_dir%\",\"id\":\"!random!\"}"
        REM Send the payload as an HTTP POST request
        curl -X POST -H "Content-Type: application/json" -d !payload! %run_job_url%

        REM Check if curl command returned an error
        if not "%errorlevel%"=="0" (
            echo Error: Failed to send HTTP request for x_value=%%x and y_value=%%y.
            pause
            exit /b 1
        )
    )
)

REM No errors, exit successfully
exit /b 0
