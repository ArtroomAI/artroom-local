@echo off
:: ..\..\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_turbo_sf_ddim.json
:: C:\Users\Nick\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_turbo_sf_ddim.json

::Optimized txt2img regression tests
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_noturbo_nosf_ddim.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_noturbo_nosf_dpm.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_noturbo_nosf_dpm_a.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_noturbo_nosf_euler.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_noturbo_nosf_euler_a.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_noturbo_nosf_heun.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_noturbo_nosf_lms.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_noturbo_nosf_plms.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_turbo_nosf_ddim.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_turbo_nosf_plms.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_turbo_sf_ddim.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_opt_turbo_sf_plms.json

::Optimized img2img regression tests
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_noturbo_nosf_ddim.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_noturbo_nosf_dpm.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_noturbo_nosf_dpm_a.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_noturbo_nosf_euler.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_noturbo_nosf_euler_a.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_noturbo_nosf_heun.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_noturbo_nosf_lms.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_noturbo_nosf_plms.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_turbo_nosf_ddim.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_turbo_nosf_plms.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_turbo_sf_ddim.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_opt_turbo_sf_plms.json

::non-Optimized txt2img regression tests
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_noopt_ddim.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_noopt_dpm.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_noopt_dpm_a.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_noopt_euler.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_noopt_euler_a.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_noopt_heun.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_noopt_lms.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_noopt_plms.json

::non-optimized img2img regression tests
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_noopt_ddim.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_noopt_dpm.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_noopt_dpm_a.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_noopt_euler.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_noopt_euler_a.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_noopt_heun.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_noopt_lms.json
%userprofile%\artroom\miniconda3\envs\artroom-ldm\python runner.py unit_tests\sd_img_noopt_plms.json
pause