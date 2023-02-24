import { atom } from 'recoil';

export const cloudModeState = atom({
    key: 'cloudMode',
    default: false
});

export const emailState = atom({
    key: 'email',
    default: ''
});

export const usernameState = atom({
    key: 'username',
    default: 'My Profile'
});

export const shardState = atom({
    key: 'shard',
    default: 0.0
});