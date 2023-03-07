import path from 'path';
import os from 'os';
import { atom, selector } from "recoil";
import { recoilPersist } from 'recoil-persist';

const { persistAtom } = recoilPersist();

export const textPromptsState = atom<string>({
    key: "text_prompts",
    default: "",
    effects_UNSTABLE: [persistAtom]
});

export const negativePromptsState = atom<string>({
    key: "negative_prompts",
    default: "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    effects_UNSTABLE: [persistAtom]
});

export const batchNameState = atom<string>({
    key: "batch_name",
    default: "ArtroomOutputs",
    effects_UNSTABLE: [persistAtom]
});

export const stepsState = atom<number>({
    key: "steps",
    default: 30,
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

export const iterationsState = atom<number>({
    key: "n_iter",
    default: 3,
    effects_UNSTABLE: [persistAtom]
});

export const cfgState = atom<string>({
    key: "cfg_scale",
    default: "7.5",
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
    default: "vae.ckpt",
    effects_UNSTABLE: [persistAtom]
});

export const loraState = atom<{ name: string, weight: number }[]>({
    key: "lora",
    default: [],
    effects_UNSTABLE: [persistAtom]
});

export const controlnetState = atom<string>({
    key: "controlnet",
    default: "None",
    effects_UNSTABLE: [persistAtom]
});

export const modelsDirState = atom<string>({
    key: "models_dir",
    default: path.resolve('/'),
    effects_UNSTABLE: [persistAtom]
});

export const runTypeState = atom<string>({
    key: "run_type",
    default: "regular",
    effects_UNSTABLE: [persistAtom]
});

export const queueSettingsSelector = selector<QueueType>({
    key: "queue.settings",
    get: ({ get }) => {
        const settings: QueueType = {
            text_prompts: get(textPromptsState),
            negative_prompts: get(negativePromptsState),

            // image to image options
            strength: get(strengthState),
            init_image: get(initImageState), // replaced in Paint.tsx

            width: get(widthState),
            height: get(heightState),
            
            // models
            models_dir: get(modelsDirState),
            ckpt: get(ckptState),
            vae: get(vaeState),
            lora: get(loraState),
            controlnet: get(controlnetState),

            // sampler options
            sampler: get(samplerState),
            steps: get(stepsState),
            cfg_scale: isNaN(parseFloat(get(cfgState))) ? 7.5 : parseFloat(get(cfgState)),
            seed: get(seedState),

            // inpainting options
            mask_image: '', // handled in Paint.tsx
            invert: get(invertState), // handled in Paint.tsx
            palette_fix: get(paletteFixState),

            //
            image_save_path: path.join(get(imageSavePathState), get(batchNameState)), // absolute path

            // generation options
            n_iter: get(iterationsState),
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

export const loadSettingsFromFile = (json: QueueType) => {
}
