import React from 'react';
import { io, Socket } from 'socket.io-client';
import { ImageSettings, ImageState } from './atoms/atoms.types';

interface WithStatus {
    status: 'Success' | 'Failure';
    status_message?: string;
}

export interface SocketOnEvents {
    get_images: (res: ImageState) => void;
    get_server_status: (res: { server_running: boolean }) => void;
    add_to_queue: (res: WithStatus & { queue_size?: number }) => void;
    get_progress: (res: { current_step: number; total_steps: number; current_num: number; total_num: number }) => void;
    get_queue: (res: { queue: QueueType[] }) => void;
    get_status: (res: { status: 'Loading Model' | 'Finished Loading Model' | 'Generating' }) => void;
    start_queue: (res: WithStatus) => void;
    pause_queue: (res: WithStatus) => void;
    stop_queue: (res: WithStatus) => void;
    clear_queue: (res: WithStatus) => void;
    update_settings: (res: WithStatus) => void;
    update_settings_with_restart: (res: WithStatus) => void;
    upscale: (res: WithStatus) => void;
    remove_from_queue: (res: WithStatus & { queue: QueueType[] }) => void;
}

export interface SocketEmitEvents {
    get_server_status: () => void;
    add_to_queue: (data: ImageSettings & { init_image?: string; mask_image?: string }) => void;
    get_queue: () => void;
    start_queue: () => void;
    pause_queue: () => void;
    stop_queue: () => void;
    clear_queue: () => void;
    update_settings: (data: {
        long_save_path: boolean;
        highres_fix: boolean;
        debug_mode: boolean;
        delay: number;
        speed: string;
        image_save_path: string;
        save_grid: boolean;
        vae: string;
        ckpt_dir: string;
    }) => void;
    update_settings_with_restart: (data: {
        long_save_path: boolean;
        highres_fix: boolean;
        debug_mode: boolean;
        delay: number;
        speed: string;
        image_save_path: string;
        save_grid: boolean;
        vae: string;
        ckpt_dir: string;
    }) => void;
    upscale: (data: {
        upscale_images: string[];
        upscaler: string;
        upscale_factor: number;
        upscale_strength: number;
        upscale_dest: string;
    }) => void;
    remove_from_queue: (data: { id: string }) => void;
}

export const socket: Socket<SocketOnEvents, SocketEmitEvents> = io('http://localhost:5300');
export const SocketContext = React.createContext(socket);
