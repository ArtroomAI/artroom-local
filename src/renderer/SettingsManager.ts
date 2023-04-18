import path from 'path';
import os from 'os';
import { atom, selector } from "recoil";
import { recoilPersist } from 'recoil-persist';
import { UseToastOptions } from '@chakra-ui/react';

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

export const cfgState = atom<string>({
    key: "cfg_scale",
    default: "7.5",
    effects_UNSTABLE: [persistAtom]
});

export const clipSkipState = atom<string>({
    key: "clip_skip",
    default: "1",
    effects_UNSTABLE: [persistAtom]
});

export const samplerState = atom<string>({
    key: "sampler",
    default: "ddim",
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
    default: false,
    effects_UNSTABLE: [persistAtom]
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

export const speedState = atom<string>({
    key: "speed",
    default: "High",
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
    default: "None",
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
    default: true,
    effects_UNSTABLE: [persistAtom]
});

const parseAndCheckFloat = (num: string, def: number) => {
    const parsed = parseFloat(num);
    return isFinite(parsed) ? parsed : def; 
}

const floatBetween = (num: number, min: number, max: number) => {
    return Math.max(Math.min(num, max), min);
}

export const queueSettingsSelector = selector<QueueType>({
    key: "queue.settings",
    get: ({ get }) => {
        const settings: QueueType = {

            text_prompts: get(textPromptsState),
            negative_prompts: get(negativePromptsState),

            // image to image options
            strength: floatBetween(get(strengthState), 0.03, 0.96),
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
            cfg_scale: floatBetween(parseAndCheckFloat(get(cfgState), 7.5), 1.01, 50),
            clip_skip: floatBetween(parseAndCheckFloat(get(clipSkipState), 1), 1, 12),

            seed: get(seedState),

            // inpainting options
            mask_image: '', // handled in Paint.tsx
            invert: get(invertState), // handled in Paint.tsx
            palette_fix: get(paletteFixState),

            //
            image_save_path: path.join(get(imageSavePathState), get(batchNameState)), // absolute path

            // generation options
            n_iter: parseAndCheckFloat(get(iterationsState), 1),
            save_grid: get(saveGridState),
            speed: get(speedState),
            device: undefined, // ? CPU / GPU
            long_save_path: get(longSavePathState),
            highres_fix: get(highresFixState),
            show_intermediates: get(showIntermediatesState),
            id: ""
        }
        return settings;
    }
});


// @DEPRECATE: LEGACY LORAS DON'T HAVE NAME ONLY PATH, IT WILL BE REMOVED 
export const parseLoras = (loras: Lora[]) => {
    return loras.map(el => {
        return { name: el.name ?? path.basename((el as any).path), weight: el.weight };
    });
}
// @DEPRECATE: CHANGE 'W' INTO 'WIDTH' AND 'H' INTO 'HEIGHT'
export const parseWidth = (exif: Partial<ExifDataType>) => {
    if('W' in exif) {
        return exif.W;
    } else if ('width' in exif) {
        return exif.width;
    }
    return "Error";
}
// @DEPRECATE: CHANGE 'W' INTO 'WIDTH' AND 'H' INTO 'HEIGHT'
export const parseHeigth = (exif: Partial<ExifDataType>) => {
    if('H' in exif) {
        return exif.H;
    } else if ('height' in exif) {
        return exif.height;
    }
    return "Error";
}

// SET ONLY
export const exifDataSelector = selector<Partial<ExifDataType>>({
    key: "exif.settings",
    get: () => {
        return {};
    },
    set: ({ set }, queue) => {
        const exif = queue as Partial<ExifDataType>;

        set(widthState, parseWidth(exif));
        set(heightState, parseHeigth(exif));

        set(cfgState, `${exif.cfg_scale}`);
        set(controlnetState, exif.controlnet ?? "none");
        set(loraState, parseLoras(exif.loras));

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

export const checkSettings = (clipboard: string): [Partial<ExifDataType>, UseToastOptions] => {
    try {
        if(clipboard === '') {
            return [null, {
                title: "Settings not loaded",
                status: "info",
                duration: 500,
                position: 'top'
            }]
        }
        const json = JSON.parse(clipboard);
        return [json, {
            title: "Settings loaded successfully",
            status: "success",
            duration: 500,
            position: 'top'
        }];
    } catch(err) {
        return [null, {
            title: `Error during loading settings`,
            status: "error",
            duration: 5000,
            isClosable: true,
            position: 'top'
        }];
    }
}
