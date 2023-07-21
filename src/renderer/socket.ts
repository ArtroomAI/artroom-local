import React from 'react';
import { io, Socket } from 'socket.io-client';
import { ImageState } from './atoms/atoms.types';
import { UseToastOptions } from '@chakra-ui/react';

interface WithStatus {
    status: 'Success' | 'Failure';
    status_message?: string;
}

export interface SocketOnEvents {
    get_images: (res: ImageState) => void;
    get_controlnet_preview: (res: {controlnetPreview: string}) => void;
    get_remove_background_preview: (res: {removeBackgroundPreview: string}) => void;
    intermediate_image: (res: ImageState) => void;
    get_progress: (res: {
        current_step: number;
        total_steps: number;
        current_num: number;
        total_num: number;
        eta: string; // mm:ss
        iterations_per_sec: string; // x.xx it/s s/it
        time_spent: string; // mm:ss
    }) => void;
    stop_queue: (res: WithStatus) => void;
    job_done: () => void;
    status: (res: UseToastOptions) => void;
}

export interface SocketEmitEvents {
    stop_queue: () => void;
    generate: (data: QueueType) => void;
    upscale: (data: {
        upscale_images: string[];
        upscaler: string;
        upscale_factor: number;
        upscale_strength: number;
        upscale_dest: string;
        image_save_path: string;
    }) => void;
    train: (data: {
        images: string[];
        name: string,
        trigger_word: string;
        image_save_path: string;
        modelsDir : string,
        resolution: string,
        networkAlpha: string,
        maxTrainSteps: string,
        clipSkip: string,
        textEncoderLr: string,
        unetLr: string,
        networkDim: string,
        lrSchedulerNumCycles: string,
        learningRate: string,
        lrScheduler: string,
        trainBatchSize: string,
        saveEveryNEpochs: string,
        optimizerType: string,
        bucketResoSteps: string,
        minBucketReso: string,
        maxBucketReso: string
    }) => void;
    remove_from_queue: (data: { id: string }) => void;
    merge_models: (data: {
        model_0: string;
        model_1: string;
        model_2: string;
        method: string;
        alpha: number;
        output: string;
        steps: number;
        start_steps: number;
        end_steps: number;
    }) => void;
    remove_background: (data: {
        initImage: string;
        batchName: string;
        imageSavePath: string;
        model: string;
    }) => void;
    preview_remove_background: (data: {
        initImage: string;
        remove_background: string;
    }) => void;
    preview_controlnet: (data: {
        initImage: string;
        controlnet: string;
        models_dir: string;
    }) => void;
}

export const socket: Socket<SocketOnEvents, SocketEmitEvents> = io('http://localhost:5300');
export const SocketContext = React.createContext(socket);
