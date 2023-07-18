import path from 'path';
import os from 'os';
import React from 'react';
import { atom, selector } from "recoil";
import { recoilPersist } from 'recoil-persist';
import { Text, UseToastOptions } from '@chakra-ui/react';
import { ExifValidation } from '../interfaces/imageData';
import { DEFAULT_EXIF } from '../interfaces/ExifDataType';
import type { IValidation } from "typia";

const { persistAtom } = recoilPersist();

export const artroomPathState = atom<string>({
    key: "artroomPath",
    default: os.homedir() || "",
    effects_UNSTABLE: [persistAtom]
});

export const modelsDirState = atom<string>({
    key: "models_dir",
    default: path.join(os.homedir() || "", 'artroom', 'model_weights'),
    effects_UNSTABLE: [persistAtom]
});

export const modelsState = atom<InternalModelsType>({
    key: 'modelsState',
    default: {
        ckpts: [],
        vaes: [],
        loras: []
    }
})

export const textPromptsState = atom<string>({
    key: "text_prompts",
    default: "",
    effects_UNSTABLE: [persistAtom]
});



export const DEFAULT_NEGATIVE_PROMPT = "mutated, deformed, amateur drawing, lowres, worst quality, low quality, jpeg artifacts, text, error, signature, watermark, username, blurry, censorship";
export const negativePromptsState = atom<string>({
    key: "negative_prompts",
    default: DEFAULT_NEGATIVE_PROMPT,
    effects_UNSTABLE: [persistAtom]
});

export const batchNameState = atom<string>({
    key: "batch_name",
    default: "ArtroomOutputs",
    effects_UNSTABLE: [persistAtom]
});

export const stepsState = atom<string>({
    key: "steps",
    default: "30",
    effects_UNSTABLE: [persistAtom]
});

export const widthState = atom<number>({
    key: "width",
    default: 512,
    effects_UNSTABLE: [persistAtom]
});

export const heightState = atom<number>({
    key: "height",
    default: 512,
    effects_UNSTABLE: [persistAtom]
});

export const aspectRatioState = atom<string>({
    key: "aspect_ratio",
    default: "None",
    effects_UNSTABLE: [persistAtom]
});

export const seedState = atom<number>({
    key: "seed",
    default: 5,
    effects_UNSTABLE: [persistAtom]
});

export const randomSeedState = atom<boolean>({
    key: "use_random_seed",
    default: true,
    effects_UNSTABLE: [persistAtom]
});

export const iterationsState = atom<string>({
    key: "n_iter",
    default: "3",
    effects_UNSTABLE: [persistAtom]
});

export const cfgState = atom<number>({
    key: "cfg_scale",
    default: 7.5,
    effects_UNSTABLE: [persistAtom]
});

export const clipSkipState = atom<string>({
    key: "clip_skip",
    default: "1",
    effects_UNSTABLE: [persistAtom]
});

export const samplerState = atom<string>({
    key: "sampler",
    default: "euler_a",
    effects_UNSTABLE: [persistAtom]
});

export const initImageState = atom<string>({
    key: "init_image",
    default: "",
    effects_UNSTABLE: [persistAtom]
});

export const strengthState = atom<number>({
    key: "strength",
    default: 0.75,
    effects_UNSTABLE: [persistAtom]
});

export const invertState = atom<boolean>({
    key: "invert",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const paletteFixState = atom<boolean>({
    key: "palette_fix",
    default: false
});

export const imageSavePathState = atom<string>({
    key: "image_save_path",
    default: path.join(os.homedir(), 'Desktop'),
    effects_UNSTABLE: [persistAtom]
});

export const longSavePathState = atom<boolean>({
    key: "long_save_path",
    default: false,
    effects_UNSTABLE: [persistAtom]
});


export const showIntermediatesState = atom<boolean>({
    key: "show_intermediates",
    default: true,
    effects_UNSTABLE: [persistAtom]
});

export const highresFixState = atom<boolean>({
    key: "highres_fix",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const saveGridState = atom<boolean>({
    key: "save_grid",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const ckptState = atom<string>({
    key: "ckpt",
    default: "model.ckpt",
    effects_UNSTABLE: [persistAtom]
});

export const vaeState = atom<string>({
    key: "vae",
    default: "",
    effects_UNSTABLE: [persistAtom]
});

export const loraState = atom<Lora[]>({
    key: "lora",
    default: [],
    effects_UNSTABLE: [persistAtom]
});

export const controlnetState = atom<string>({
    key: "controlnet",
    default: "none",
    effects_UNSTABLE: [persistAtom]
});

export const usePreprocessedControlnetState = atom<boolean>({
    key: "use_preprocessed_controlnet",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const removeBackgroundState = atom<string>({
    key: "remove_background",
    default: "face",
    effects_UNSTABLE: [persistAtom]
});

export const useRemovedBackgroundState = atom<boolean>({
    key: "use_removed_background",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const runTypeState = atom<string>({
    key: "run_type",
    default: "regular",
    effects_UNSTABLE: [persistAtom]
});

export const debugModeState = atom<boolean>({
    key: 'debug_mode',
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const cloudOnlyState = atom<boolean>({
    key: 'cloudOnly',
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const connectedToServerState = atom<boolean>({
    key: 'connected_to_server',
    default: false
});

export const highresfixOnlyState = atom({
    key: 'highresfix_only',
    default: false
});

export const highresStepsState = atom({
    key: 'highresfix_steps',
    default: '10'
});

export const highresStrengthState = atom({
    key: 'highresfix_strength',
    default: 0.15
});

const parseAndCheckFloat = (num: string, def: number) => {
    const parsed = parseFloat(num);
    return isFinite(parsed) ? parsed : def; 
}

const floatBetween = (num: number, min: number, max: number) => {
    return Math.max(Math.min(num, max), min);
}

const normalizeString = (input: string) => input.replace('\n', ' ').trim();

export const queueSettingsSelector = selector<QueueType>({
    key: "queue.settings",
    get: ({ get }) => {
        const settings: QueueType = {

            text_prompts: normalizeString(get(textPromptsState)),
            negative_prompts: normalizeString(get(negativePromptsState)),

            // image to image options
            strength: get(strengthState),
            init_image: get(initImageState), // replaced in Paint.tsx

            width: get(widthState),
            height: get(heightState),
            
            // models
            models_dir: get(modelsDirState),
            ckpt: get(ckptState),
            vae: get(vaeState),
            loras: get(loraState),
            controlnet: get(controlnetState),
            use_preprocessed_controlnet: get(usePreprocessedControlnetState),
            remove_background: get(removeBackgroundState),
            use_removed_background: get(useRemovedBackgroundState),
            // sampler options
            sampler: get(samplerState),
            steps: floatBetween(parseAndCheckFloat(get(stepsState), 30), 5, 500),
            cfg_scale: get(cfgState),
            clip_skip: floatBetween(parseAndCheckFloat(get(clipSkipState), 1), 1, 12),

            seed: get(seedState),

            // inpainting options
            mask_image: '', // handled in Paint.tsx
            invert: get(invertState), // handled in Paint.tsx
            palette_fix: get(paletteFixState),

            //
            image_save_path: path.join(
                normalizeString(get(imageSavePathState)),
                normalizeString(get(batchNameState))
            ), // absolute path
            
            // highres fix options
            highres_steps: floatBetween(parseAndCheckFloat(get(highresStepsState), 10), 5, 100),
            highres_strength: get(highresStrengthState),

            // generation options
            n_iter: parseAndCheckFloat(get(iterationsState), 1),
            save_grid: get(saveGridState),
            device: undefined, // ? CPU / GPU
            long_save_path: get(longSavePathState),
            highres_fix: get(highresFixState),
            show_intermediates: get(showIntermediatesState),
            id: "",
            generation_mode: ''
        }
        return settings;
    }
});

// SET ONLY
export const exifDataSelector = selector<Partial<ExifDataType>>({
    key: "exif.settings",
    get: () => {
        return {};
    },
    set: ({ set }, queue) => {
        const exif = queue as ExifDataType;

        set(widthState, exif.width);
        set(heightState, exif.height);

        set(cfgState, exif.cfg_scale);
        set(controlnetState, exif.controlnet ?? "none");
        set(loraState, exif.loras);

        set(negativePromptsState, exif.negative_prompts);

        set(samplerState, exif.sampler);
        set(seedState, exif.seed);
        set(stepsState, `${exif.steps}`);
        set(strengthState, exif.strength);
        set(textPromptsState, exif.text_prompts);
        set(vaeState, exif.vae);
        set(clipSkipState, `${exif.clip_skip ?? 1}`);

        // load specific
        set(randomSeedState, false);
        set(aspectRatioState, 'None');
    }
});

const parseExifErrors = (errors: IValidation.IError[]) => {
    const parsePath = (path: string) => path.replace('$input.', '');

    const parseType = (type: string) => {
        if (type.includes('@type int')) return 'integer';
        return type;
    };

    const parseValue = (value: any) => typeof value === 'object' ? JSON.stringify(value) : String(value);
    return errors.map((err) => {
        const path = parsePath(err.path);
        const type = parseType(err.expected);
        const value = parseValue(err.value);
        return <Text>In {path} expected {type}, but got {value}</Text>
    });
}

export type CheckSettingsType = [ExifDataType, UseToastOptions];

export const checkSettings = (exif: ExifValidation | null, models?: InternalModelsType): CheckSettingsType => {
    if(exif === null) {
        return [DEFAULT_EXIF, {
            title: `Settings were not loaded`,
            status: "error",
            duration: 5000,
            isClosable: true,
            position: 'top'
        }];
    }
    if(exif.success) {
        if(models) {
            const missing: string[] = [];
            const hasModel = models.ckpts.includes(exif.data.ckpt);
            const hasVae = models.vaes.includes(exif.data.vae);
            const hasLoras = exif.data.loras
                .map(lora => models.loras.includes(lora.name));

            if (!hasModel) {
                missing.push(`Missing model: ${exif.data.ckpt}`);
            }
            if (!hasVae) {
                missing.push(`Missing vae: ${exif.data.vae}`);
            }
            if (hasLoras.find(e => e === false) ?? false) {
                hasLoras.forEach((err, index) => {
                    if(err) {
                        missing.push(`Missing lora: ${exif.data.loras[index]}`);
                    }
                })
            }

            if (missing.length) {
                return [exif.data, {
                    title: "Settings loaded partially",
                    description: missing.map(text => <Text>{text}</Text>),
                    status: "warning",
                    duration: 5000,
                    position: 'top'
                }];
            }
        }
        return [exif.data, {
            title: "Settings loaded successfully",
            status: "success",
            duration: 500,
            position: 'top'
        }];
    }
    return [DEFAULT_EXIF, {
        title: `Error during loading settings`,
        description: parseExifErrors(exif.errors),
        status: "error",
        duration: 5000,
        isClosable: true,
        position: 'top'
    }];
}
