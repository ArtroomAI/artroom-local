import { atom } from 'recoil';

export const cloudModeState = atom({
    key: 'cloudMode',
    default: false
});

export const emailState = atom({
    key: 'email',
    default: ''
});