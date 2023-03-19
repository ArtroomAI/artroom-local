import React from 'react';
import { io, Socket } from 'socket.io-client';
import { ImageState } from './atoms/atoms.types';

interface WithStatus {
    status: 'Success' | 'Failure';
    status_message?: string;
}

export interface SocketOnEvents {
    get_images: (res: ImageState) => void;
    get_controlnet_preview: (res: {controlnetPreview: string}) => void;
    get_remove_background_preview: (res: {removeBackgroundPreview: string}) => void;
    intermediate_image: (res: ImageState) => void;
    get_server_status: (res: { server_running: boolean }) => void;
    get_progress: (res: { current_step: number; total_steps: number; current_num: number; total_num: number }) => void;
    get_status: (res: { status: 'Loading Model' | 'Finished Loading Model' | 'Generating' }) => void;
    stop_queue: (res: WithStatus) => void;
    upscale: (res: WithStatus) => void;
    job_done: () => void;
}

export interface SocketEmitEvents {
    get_server_status: () => void;
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
    }) => void;
}

export const socket: Socket<SocketOnEvents, SocketEmitEvents> = io('http://localhost:5300');
export const SocketContext = React.createContext(socket);
