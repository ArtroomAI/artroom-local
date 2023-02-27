import path from 'path';
import os from 'os';
import { atom, selector } from "recoil";
import { recoilPersist } from 'recoil-persist';

const { persistAtom } = recoilPersist();

export const textPromptsState = atom({
    key: "text_prompts",
    default: "",
    effects_UNSTABLE: [persistAtom]
});

export const negativePromptsState = atom({
    key: "negative_prompts",
    default: "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    effects_UNSTABLE: [persistAtom]
});

export const batchNameState = atom({
    key: "batch_name",
    default: "ArtroomOutputs",
    effects_UNSTABLE: [persistAtom]
});

export const stepsState = atom({
    key: "steps",
    default: 30,
    effects_UNSTABLE: [persistAtom]
});

export const widthState = atom({
    key: "width",
    default: 512,
    effects_UNSTABLE: [persistAtom]
});

export const heightState = atom({
    key: "height",
    default: 512,
    effects_UNSTABLE: [persistAtom]
});

export const aspectRatioState = atom({
    key: "aspect_ratio",
    default: "None",
    effects_UNSTABLE: [persistAtom]
});

export const seedState = atom({
    key: "seed",
    default: 5,
    effects_UNSTABLE: [persistAtom]
});

export const randomSeedState = atom({
    key: "use_random_seed",
    default: true,
    effects_UNSTABLE: [persistAtom]
});

export const iterationsState = atom({
    key: "n_iter",
    default: 3,
    effects_UNSTABLE: [persistAtom]
});

export const cfgState = atom({
    key: "cfg_scale",
    default: 7.5,
    effects_UNSTABLE: [persistAtom]
});

export const samplerState = atom({
    key: "sampler",
    default: "ddim",
    effects_UNSTABLE: [persistAtom]
});

export const initImageState = atom({
    key: "init_image",
    default: "",
    effects_UNSTABLE: [persistAtom]
});

export const strengthState = atom({
    key: "strength",
    default: 0.75,
    effects_UNSTABLE: [persistAtom]
});

export const invertState = atom({
    key: "invert",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const paletteFixState = atom({
    key: "palette_fix",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const imageSavePathState = atom({
    key: "image_save_path",
    default: path.join(os.homedir(), 'Desktop'),
    effects_UNSTABLE: [persistAtom]
});

export const longSavePathState = atom({
    key: "long_save_path",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const highresFixState = atom({
    key: "highres_fix",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const speedState = atom({
    key: "speed",
    default: "High",
    effects_UNSTABLE: [persistAtom]
});

export const saveGridState = atom({
    key: "save_grid",
    default: false,
    effects_UNSTABLE: [persistAtom]
});

export const ckptState = atom({
    key: "ckpt",
    default: "model.ckpt",
    effects_UNSTABLE: [persistAtom]
});

export const vaeState = atom({
    key: "vae",
    default: "vae.ckpt",
    effects_UNSTABLE: [persistAtom]
});

export const modelsDirState = atom({
    key: "models_dir",
    default: path.resolve('/'),
    effects_UNSTABLE: [persistAtom]
});

export const runTypeState = atom({
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
            init_image: get(initImageState),

            width: get(widthState),
            height: get(heightState),
            
            // models
            models_dir: get(modelsDirState),
            ckpt: get(ckptState),
            vae: get(vaeState),

            // sampler options
            sampler: get(samplerState),
            steps: get(stepsState),
            cfg_scale: get(cfgState),
            seed: get(seedState),

            // inpainting options
            mask_image: get(textPromptsState),
            invert: get(invertState),
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
            id: ""
        }
        return settings;
    }
});

export const loadSettingsFromFile = (json: QueueType) => {
}
