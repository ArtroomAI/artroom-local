import { atom } from 'recoil';
import { ImageState } from './atoms.types';

export const serverStatusState = atom({
    key: 'serverStatus',
    default: ''
});

export const aspectRatioSelectionState = atom({
    key: 'aspectRatioSelection',
    default: 'None'
});

export const initImagePathState = atom({
    key: 'initImagePath',
    default: ''
});

export const ckptsState = atom({
    key: 'ckpts',
    default: []
});

export const navSizeState = atom({
    key: 'navSize',
    default: 'small'
});

export const queueRunningState = atom({
    key: 'queueRunning',
    default: false
});


export const queueState = atom<QueueType[]>({
    key: 'queue',
    default: []
});

export const queuePausedState = atom({
    key: 'queue_paused',
    default: false
});

export const runningState = atom({
    key: 'running',
    default: false
});

export const cloudRunningState = atom({
    key: 'cloud_running',
    default: false
});


export const initImageThumbnailState = atom({
    key: 'initImageThumbnail',
    default: 
    { 'ImagePath': '',
        'b64': '' }
});

export const addToQueueState = atom({
    key: 'addToQueue',
    default: false
});

export const mainImageState = atom<Partial<ImageState>>({
    key: 'mainImage',
    default: { 'b64': '', 'path': '', 'batch_id': 0 }
});

export const latestImageState = atom<Array<Partial<ImageState>>>({
    key: 'latestImage',
    default: []
});

export const imageViewPathState = atom({
    key: 'imageViewPath',
    default: ''
});

export * from './atoms.modals'
export * from './atoms.login'
